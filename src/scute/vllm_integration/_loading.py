"""Carry ``lora_magnitude_vector`` through vLLM's adapter loader.

Three things in the loader assume an adapter tensor is either ``lora_A`` or
``lora_B``.

``parse_fine_tuned_lora_name`` recognises ``lora_A``, ``lora_B`` and the two
embedding names, and **raises** on anything else.  A DoRA checkpoint carries one
``lora_magnitude_vector`` per adapted module, so every one of them hits that raise.

``LoRAModel.from_lora_tensors`` branches on a single ``is_lora_a`` boolean, so there
is no third place for the magnitude to go.

``PackedLoRALayerWeights.pack`` fuses ``q,k,v`` into one entry and ``gate,up`` into
another.  The magnitude has to be packed the same way or it will not line up with the
stacked buffer it eventually scales.

The parser's contract changes from ``(module_name, is_lora_a: bool)`` to
``(module_name, kind: str)`` with ``kind`` in ``{"a", "b", "m"}``.  Both of vLLM's
call sites are patched with it: the one in ``from_lora_tensors`` is replaced outright,
and the one in ``from_local_checkpoint`` discards the second element, so it only needs
the parser to stop raising.

**The name is bound at import time.**  ``vllm.lora.lora_model`` does
``from vllm.lora.utils import parse_fine_tuned_lora_name``, so patching
``vllm.lora.utils`` alone leaves the loader calling the original.  Both bindings are
replaced below.
"""

from __future__ import annotations

import torch

from ._util import patch_attr

MAGNITUDE_SUFFIX = "lora_magnitude_vector"

#: Normal-form vectors written by :mod:`scute.compile`: the output scale ``s`` and the
#: diagonals ``d``, ``e`` of a shared-factor term.  Each becomes an attribute of the
#: module's ``LoRALayerWeights`` and is packed like the magnitude.
VECTOR_KINDS = {"scute_s": "s", "scute_d": "d", "scute_e": "e", "scute_R": "R"}
VECTOR_ATTRS = {"s": "scute_s", "d": "scute_d", "e": "scute_e", "R": "scute_R"}


def _parse_fine_tuned_lora_name(name: str, weights_mapper=None) -> tuple[str, str]:
    """As vLLM's, plus the magnitude, and returning a three-valued kind."""
    if name.startswith("base_model.model."):
        stripped = name.replace("base_model.model.", "")
        stripped = weights_mapper._map_name(stripped) if weights_mapper else stripped
        name = "base_model.model." + stripped
    else:
        name = weights_mapper._map_name(name) if weights_mapper else name

    start_index = 2 if name.startswith("base_model.model.") else 0
    parts = name.split(".")

    if parts[-1] in VECTOR_KINDS:
        return ".".join(parts[start_index:-1]), VECTOR_KINDS[parts[-1]]

    # PEFT writes the magnitude without a `.weight` suffix.
    if parts[-1] == MAGNITUDE_SUFFIX:
        return ".".join(parts[start_index:-1]), "m"
    if parts[-1] == "weight" and parts[-2] == MAGNITUDE_SUFFIX:
        return ".".join(parts[start_index:-2]), "m"

    if parts[-1] == "weight" and parts[-2] in ("lora_A", "lora_B"):
        return ".".join(parts[start_index:-2]), "a" if parts[-2] == "lora_A" else "b"
    if parts[-1] in ("lora_embedding_A", "lora_embedding_B"):
        return ".".join(parts[start_index:-1]), (
            "a" if parts[-1] == "lora_embedding_A" else "b"
        )
    raise ValueError(f"{name} is unsupported LoRA weight")


def _from_lora_tensors(
    cls,
    lora_model_id: int,
    tensors: dict[str, torch.Tensor],
    peft_helper,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    model_vocab_size: int | None = None,
    weights_mapper=None,
    skip_prefixes: list[str] | None = None,
):
    """vLLM's own, with a third branch for the magnitude."""
    from vllm.lora.lora_weights import LoRALayerWeights
    from vllm.lora.utils import is_base_embedding_weights
    from vllm.utils.platform_utils import is_pin_memory_available

    pin_memory = str(device) == "cpu" and is_pin_memory_available()
    loras: dict[str, LoRALayerWeights] = {}

    for tensor_name, tensor in tensors.items():
        if is_base_embedding_weights(tensor_name):
            continue
        if skip_prefixes and cls._should_skip_module(tensor_name, skip_prefixes):
            continue
        module_name, kind = _parse_fine_tuned_lora_name(tensor_name, weights_mapper)
        if module_name not in loras:
            loras[module_name] = LoRALayerWeights.from_config(module_name, peft_helper)

        if kind == "m":
            # Kept in float32 whatever the serving dtype is: it is divided by a
            # norm at fold time, and the fold is the one place where an fp16
            # rounding would be baked into every subsequent token.
            loras[module_name].magnitude = tensor.to(device=device,
                                                     dtype=torch.float32)
            continue

        if kind in VECTOR_ATTRS:
            # float32 like the magnitude: these are written into fp16 pools once,
            # at admission, and nowhere else.
            setattr(loras[module_name], VECTOR_ATTRS[kind],
                    tensor.to(device=device, dtype=torch.float32))
            continue

        if kind == "a":
            if (
                "lora_embedding_A" in tensor_name
                and model_vocab_size is not None
                and model_vocab_size != tensor.shape[1]
            ):
                raise RuntimeError(
                    f"The embedding LoRA size({tensor.shape[1]}) must be consistent"
                    f" with the base model's vocabulary size({model_vocab_size})."
                )
            loras[module_name].lora_a = tensor.to(device=device, dtype=dtype)
            if pin_memory:
                loras[module_name].lora_a = loras[module_name].lora_a.pin_memory()
        else:
            loras[module_name].lora_b = tensor.to(device=device, dtype=dtype)
            if pin_memory:
                loras[module_name].lora_b = loras[module_name].lora_b.pin_memory()

    return cls(lora_model_id, peft_helper.r, loras)


def _pack(cls, loras):
    """vLLM's ``PackedLoRALayerWeights.pack``, carrying the magnitude list along.

    The list is positional and keeps ``None`` for an unadapted sub-module, because
    the packed layer's output slices are positional too.  An adapter of ``up_proj``
    without ``gate_proj`` has such a ``None``.
    """
    first_lora = next(lora for lora in loras if lora is not None)
    for lora in loras:
        if lora is not None:
            lora.optimize()
    obj = cls(
        first_lora.module_name,
        first_lora.rank,
        [lora.lora_alpha if lora is not None else None for lora in loras],
        [lora.lora_a if lora is not None else None for lora in loras],
        [lora.lora_b if lora is not None else None for lora in loras],
        scaling=[1 if lora is not None else None for lora in loras],
    )
    obj.magnitude = [
        getattr(lora, "magnitude", None) if lora is not None else None
        for lora in loras
    ]
    for attr in VECTOR_ATTRS.values():
        setattr(obj, attr, [getattr(lora, attr, None) if lora is not None else None
                            for lora in loras])
    return obj


def apply() -> None:
    import vllm.lora.lora_model as lora_model_mod
    import vllm.lora.utils as utils_mod
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights

    # Default so that every code path can ask for `.magnitude` without a getattr.
    patch_attr(LoRALayerWeights, "magnitude", None)
    for attr in VECTOR_ATTRS.values():
        patch_attr(LoRALayerWeights, attr, None)

    patch_attr(utils_mod, "parse_fine_tuned_lora_name", _parse_fine_tuned_lora_name)
    # The binding the loader actually calls.  See the module docstring.
    patch_attr(lora_model_mod, "parse_fine_tuned_lora_name",
               _parse_fine_tuned_lora_name)

    patch_attr(LoRAModel, "from_lora_tensors", classmethod(_from_lora_tensors))
    patch_attr(PackedLoRALayerWeights, "pack", classmethod(_pack))
