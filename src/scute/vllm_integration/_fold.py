"""The load-time fold, executed inside the engine at adapter admission.

DoRA's serving form needs ``g = m / ||W + AB||_c``, computed once when an adapter is
admitted rather than per token.  This module hooks adapter activation and writes ``g``
into a per-module pool that the expand reads.

**Why the fold reads the stacked buffers rather than the checkpoint.**  By the time an
adapter activates, vLLM has already packed ``q,k,v`` into one layer and ``gate,up``
into another, and has already sharded every tensor for tensor parallelism.  Folding
from the stacked buffers therefore inherits both, instead of re-deriving the packing
and the sharding from the checkpoint and hoping the two agree.  It also means the fold
sees exactly the weights the kernel will use.

**Scaling is already folded in.**  ``LoRALayerWeights.optimize()`` multiplies
``lora_b`` by ``alpha/r`` during packing, which happens before activation, and sets
``scaling`` to 1.  So ``B @ A`` here already carries the scale and the norm is taken
over ``W + scaling*BA`` as PEFT defines it, with no second application.

**Tensor parallelism.**  A column-parallel shard splits ``d_out``, so the norm is
local and only the magnitude has to be sliced to match.  A row-parallel shard splits
``d_in``, so each shard holds a *partial* sum of each squared row norm: the reduction
must happen over the squares, before the square root.  In the other order, the
shards disagree.

**Precision, and why the engine must not run in bfloat16.**  The norm is accumulated
in float32 and ``g`` is stored in the model's compute dtype, which is what a deployed
engine does and what the expand then reads.  That storage dtype is not a free
choice.  A trained ``g`` sits very close to 1, typically within ``[0.98, 1.02]``,
and bfloat16 has only 8 mantissa bits, so it resolves that whole range into **9 distinct values**, a relative error of $3.9\\times 10^{-3}$ against a
signal of $\\pm 2\\%$.  Roughly a fifth of the adaptation would be quantisation.
float16 gives 62 levels over the same range and float32 is exact for this purpose.

vLLM defaults Llama-3.1-8B to bfloat16, so run the engine with ``dtype="float16"``.
:func:`check_dtype` refuses bfloat16 rather than letting it degrade DoRA quietly.
"""

from __future__ import annotations

import os

import torch

from ._util import patch_attr

#: Output rows folded per chunk.  ``B @ A`` is a full ``[d_out, d_in]`` matrix --
#: 235 MB in float32 for a Llama-3.1-8B ``up_proj`` -- and it is only ever reduced to
#: one number per row, so it never needs to exist all at once.
ROW_CHUNK = 1024

#: Set ``SCUTE_FOLD_LOG=<path-prefix>`` to record a checksum of every folded ``g``.
#: Each rank writes its own file, ``<prefix>.rank<N>.jsonl``.
_FOLD_LOG = os.environ.get("SCUTE_FOLD_LOG", "")
#: ``SCUTE_FOLD_VEC=1`` also dumps the full ``g`` vector per shard, for an
#: element-wise diff across tensor-parallel degrees.
_FOLD_VEC = os.environ.get("SCUTE_FOLD_VEC", "0") == "1"


def _slice_magnitude(m: torch.Tensor, out_rows: int, tp_rank: int) -> torch.Tensor:
    """Take this shard's part of a full-width magnitude vector.

    A column-parallel shard holds a contiguous block of output rows, and each packed
    slice is split the same way, so the shard's magnitude is the matching contiguous
    block.  A row-parallel layer does not split ``d_out`` at all, in which case the
    lengths already agree and nothing is taken.
    """
    if m.numel() == out_rows:
        return m
    if m.numel() % out_rows != 0:
        raise ValueError(
            f"magnitude of {m.numel()} does not divide into shards of {out_rows}"
        )
    return m[tp_rank * out_rows: (tp_rank + 1) * out_rows]


def _to_device(m: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """vLLM stages adapter tensors on CPU before the stacked buffers reach the GPU."""
    return m if m.device == like.device else m.to(like.device)


def fold_slice(
    w_slice: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    magnitude: torch.Tensor,
    reduce_squares: bool = False,
) -> torch.Tensor:
    """``g = m / ||W + BA||`` for one output slice, in vLLM's row-major layout.

    Args:
        w_slice: ``[out, in]`` base weight rows for this slice.
        lora_a: ``[r, in]``.
        lora_b: ``[out, r]``, with the scale already folded in.
        magnitude: ``[out]``.
        reduce_squares: all-reduce the squared norms before the square root, which
            a row-parallel shard requires because it holds only part of ``in``.

    Returns:
        ``g`` of shape ``[out]``, float32.
    """
    out_rows = w_slice.shape[0]
    sq = torch.empty(out_rows, dtype=torch.float32, device=w_slice.device)
    for start in range(0, out_rows, ROW_CHUNK):
        stop = min(start + ROW_CHUNK, out_rows)
        delta = lora_b[start:stop].float() @ lora_a.float()
        merged = w_slice[start:stop].float() + delta
        sq[start:stop] = merged.pow(2).sum(dim=1)
    if reduce_squares:
        from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
        sq = tensor_model_parallel_all_reduce(sq)
    return magnitude.float() / sq.sqrt()


def _ensure_pool(module, max_loras: int) -> torch.Tensor:
    """Per-module ``[max_loras + 1, total_out]`` pool of ``g``, defaulting to ones.

    The extra row is the point of the layout: vLLM marks a token with no adapter as
    ``-1``, and ``-1`` indexes the last row, so a base token multiplies by ones
    without any masking or branch.  Adapters that are plain LoRA also leave their row
    at ones, so every token takes the same code path.

    The pool is a buffer, not a plain attribute. At TP>1 the layer ``apply`` is
    compiled with ``fullgraph=True`` before any adapter is admitted, and a plain
    tensor attribute can be baked into that graph as a constant. A buffer is a
    graph input, so the fold's later in-place writes are what the expand reads.
    """
    total_out = sum(module.output_slices)
    if not hasattr(module, "_scute_main_mask"):
        # Slots whose adapter has an owned term or an s on this layer; the main expand
        # skips the rest (VeRA, OFT), so their rows are not read and rewritten twice.
        module.register_buffer(
            "_scute_main_mask",
            torch.ones(max_loras + 1, dtype=torch.int32,
                       device=module.lora_b_stacked[0].device), persistent=False)
    needed = (max_loras + 1, total_out)
    pool = getattr(module, "_scute_g_pool", None)
    if pool is None or tuple(pool.shape) != needed:
        pool = torch.ones(
            *needed,
            dtype=module.lora_b_stacked[0].dtype,
            device=module.lora_b_stacked[0].device,
        )
        # Replacing an existing buffer of the wrong shape: drop it first.
        bufs = getattr(module, "_buffers", None)
        if bufs is not None and "_scute_g_pool" in bufs:
            del module._buffers["_scute_g_pool"]
        module.register_buffer("_scute_g_pool", pool, persistent=False)
    return module._scute_g_pool


#: ``SCUTE_SHARED_FACTORS=<path>``: a ``scute_shared.safetensors`` written by
#: :mod:`scute.compile` (VeRA's ``vera_A`` [r, max d_in] and ``vera_B`` [max d_out, r]).
#: When set, every adapted layer gets views of the shared factors and the pools for the
#: shared-factor term's diagonals, at weight creation, so the compiled forward reads them.
_SHARED_PATH = os.environ.get("SCUTE_SHARED_FACTORS", "")
_SHARED: dict[str, torch.Tensor] = {}


def _shared_factors(device, dtype) -> dict[str, torch.Tensor]:
    if not _SHARED:
        from safetensors.torch import load_file
        t = load_file(_SHARED_PATH)
        _SHARED["A"] = t["vera_A"].to(device=device, dtype=dtype).contiguous()
        _SHARED["B"] = t["vera_B"].to(device=device, dtype=dtype).contiguous()
    return _SHARED


def _ensure_shared(module, max_loras: int) -> None:
    """Views of the shared factors and the ``d``/``e`` pools for one layer.

    ``d`` is per slice, ``[max_loras + 1, n_slices, r]``, zero by default, so a token
    whose adapter has no shared-factor term (and a token with no adapter, which
    indexes the last row) adds nothing.  ``e`` is ``[max_loras + 1, total_out]``,
    ones by default.  Each slice reads the first ``out_i`` rows of ``vera_B`` and every
    slice reads the first ``d_in`` columns of ``vera_A``, as PEFT slices them.
    """
    if not _SHARED_PATH or hasattr(module, "_scute_d_pool"):
        return
    from vllm.model_executor.layers.linear import RowParallelLinear

    ref = module.lora_b_stacked[0]
    sh = _shared_factors(ref.device, ref.dtype)
    r = sh["A"].shape[0]
    d_in = module.lora_a_stacked[0].shape[-1]
    rank = module.tp_rank
    row = isinstance(module.base_layer, RowParallelLinear) and module.tp_size > 1
    # Row-parallel: this rank holds columns [rank*d_in, (rank+1)*d_in) of the input, so
    # it reads the same columns of A; its partial term is summed by the layer's
    # all-reduce, like its partial base output.  Column-parallel: the full input, and
    # each slice's rows of B for this rank's block of the output.
    module.register_buffer("_scute_shared_A",
                           sh["A"][:, rank * d_in:(rank + 1) * d_in] if row
                           else sh["A"][:, :d_in], persistent=False)
    for i, out in enumerate(module.output_slices):
        lo = 0 if (row or module.tp_size == 1) else rank * out
        module.register_buffer(f"_scute_shared_B{i}", sh["B"][lo: lo + out], persistent=False)
    n = len(module.output_slices)
    module.register_buffer(
        "_scute_shared_mask", torch.zeros(max_loras + 1, dtype=torch.int32, device=ref.device),
        persistent=False)
    module.register_buffer(
        "_scute_d_pool", torch.zeros(max_loras + 1, n, r, dtype=ref.dtype, device=ref.device),
        persistent=False)
    module.register_buffer(
        "_scute_e_pool",
        torch.ones(max_loras + 1, sum(module.output_slices), dtype=ref.dtype,
                   device=ref.device), persistent=False)


#: ``SCUTE_OFT_BLOCK=<b>``: allocate each layer's pool of OFT blocks ``R`` (identity by
#: default), so the base GEMM takes a per-slice pre-transformed input.
_OFT_BLOCK = int(os.environ.get("SCUTE_OFT_BLOCK", "0") or 0)


def _ensure_oft(module, max_loras: int) -> None:
    """``[max_loras + 1, n_slices, d_in / b, b, b]`` identity blocks per layer.

    Every slice of a packed layer has its own ``R`` (PEFT adapts ``q``, ``k`` and ``v``
    as separate modules), so the pool is per slice.  The last row is the identity, which
    a token with no adapter indexes; but the kernel never visits those rows anyway.
    """
    if not _OFT_BLOCK or hasattr(module, "_scute_R_pool"):
        return
    ref = module.lora_b_stacked[0]
    d_in = module.lora_a_stacked[0].shape[-1]
    b = _OFT_BLOCK
    assert d_in % b == 0, f"OFT block {b} does not divide d_in {d_in}"
    eye = torch.eye(b, dtype=ref.dtype, device=ref.device)
    pool = eye.expand(max_loras + 1, len(module.output_slices), d_in // b, b, b).contiguous()
    module.register_buffer("_scute_R_pool", pool, persistent=False)


#: Slots whose admitted adapter has OFT blocks ("R") or a shared-factor term ("shared").
#: The per-step flags derived from them let a step whose batch has no such adapter skip
#: the split base GEMM and the shared term (see ``_update_flags``).
SLOT_KINDS: dict[str, set] = {"R": set(), "shared": set()}
#: Per-step flags take effect only after the engine has captured its CUDA graphs.  Until
#: then every step takes the full path, so each kernel is compiled in the warm-up and
#: profiling runs, never inside a capture (where a Triton compile is not permitted).
_ARMED = [False]


def _ensure_flags(module) -> None:
    w = module.punica_wrapper
    if not hasattr(w, "_scute_flags"):
        # [any OFT adapter in the step, any shared-factor adapter in the step]; True by
        # default, so a step whose flags were never computed runs the full path.
        w._scute_flags = torch.ones(2, dtype=torch.bool)


def _as_list(v):
    if v is None:
        return None
    return list(v) if isinstance(v, (list, tuple)) else [v]


def write_vectors(module, index: int, module_lora) -> None:
    """Write the normal-form vectors ``s``, ``d``, ``e`` of slot ``index``.

    Runs after :func:`fold_into_module`, which has reset ``s`` to ones (or filled it
    from a DoRA magnitude).  ``d`` and ``e`` are reset here on every activation,
    because vLLM reuses slots across adapters.
    """
    if hasattr(module, "_scute_d_pool"):
        module._scute_d_pool[index] = 0.0
        module._scute_e_pool[index] = 1.0
        module._scute_shared_mask[index] = 0
    if hasattr(module, "_scute_R_pool"):
        b = module._scute_R_pool.shape[-1]
        module._scute_R_pool[index] = torch.eye(b, dtype=module._scute_R_pool.dtype,
                                                device=module._scute_R_pool.device)
    if module_lora is None:
        if hasattr(module, "_scute_main_mask"):
            # The adapter does not touch this layer: s = 1 and no term.
            module._scute_main_mask[index] = 0
        return
    if hasattr(module, "_scute_main_mask"):
        live = any(v is not None for v in (_as_list(getattr(module_lora, "scute_s", None)) or [])
                   + (_as_list(getattr(module_lora, "magnitude", None)) or []))
        owned = any(module.lora_b_stacked[i][index].abs().max().item() > 0
                    for i in range(len(module.output_slices)))
        module._scute_main_mask[index] = 1 if (live or owned) else 0
    r_list = _as_list(getattr(module_lora, "scute_R", None))
    if r_list and any(r is not None for r in r_list):
        if not hasattr(module, "_scute_R_pool"):
            from ._guards import UnsupportedDoRAPath
            raise UnsupportedDoRAPath("adapter has OFT blocks but the engine was started "
                                      "without SCUTE_OFT_BLOCK")
        nb = module._scute_R_pool.shape[2]
        for i, rv in enumerate(r_list):
            if rv is not None:
                # A row-parallel shard holds a contiguous block of the input, so its
                # blocks of R are the matching contiguous run.
                if rv.shape[0] != nb:
                    rv = rv[module.tp_rank * nb:(module.tp_rank + 1) * nb]
                module._scute_R_pool[index, i] = _to_device(rv, module._scute_R_pool).to(
                    module._scute_R_pool.dtype)
    s_list = _as_list(getattr(module_lora, "scute_s", None))
    d_list = _as_list(getattr(module_lora, "scute_d", None))
    e_list = _as_list(getattr(module_lora, "scute_e", None))
    has_shared = any(v is not None for v in (d_list or []) + (e_list or []))
    if has_shared and not hasattr(module, "_scute_d_pool"):
        from ._guards import UnsupportedDoRAPath
        raise UnsupportedDoRAPath(
            "adapter has a shared-factor term but the engine was started without "
            "SCUTE_SHARED_FACTORS")
    pool = module._scute_g_pool
    mags = _as_list(getattr(module_lora, "magnitude", None)) or []
    off = 0
    for i, out in enumerate(module.output_slices):
        sv = s_list[i] if s_list and i < len(s_list) else None
        if sv is not None:
            if i < len(mags) and mags[i] is not None:
                raise ValueError("a slice with both a DoRA magnitude and an s vector")
            pool[index, off: off + out] = _slice_magnitude(
                _to_device(sv, pool), out, module.tp_rank).to(pool.dtype)
        dv = d_list[i] if d_list and i < len(d_list) else None
        if dv is not None:
            module._scute_shared_mask[index] = 1
            module._scute_d_pool[index, i, : dv.numel()] = _to_device(dv, pool).to(pool.dtype)
        ev = e_list[i] if e_list and i < len(e_list) else None
        if ev is not None:
            module._scute_e_pool[index, off: off + out] = _slice_magnitude(
                _to_device(ev, pool), out, module.tp_rank).to(pool.dtype)
        off += out


def fold_into_module(
    module, index: int, module_lora, fold_e_into_b: bool = False, lora_id: int | None = None,
) -> bool:
    """Write ``g`` for adapter slot ``index`` into ``module``'s pool.

    DoRA activates two slots with the same vector: ``s`` on the base output and ``e``
    on the term, so the served operator is ``g \\odot h + (xA)(B\\,\\mathrm{diag}(g))``.
    The ``e`` half obeys the fold rule -- ``B`` is per-adapter -- so it is pushed into
    ``B`` here, once, and the kernel never learns it existed.  The ``s`` half cannot
    fold, because the tensor it scales is the shared base output, and that is the slot
    the expand has to carry.

    Args:
        fold_e_into_b: multiply this adapter's ``B`` by ``g`` in place.  Required by
            the patched expand, which computes ``h \\odot s + VB`` and therefore
            relies on ``B`` already carrying ``e``.

    Returns True if a magnitude was found and folded.
    """
    from vllm.model_executor.layers.linear import RowParallelLinear

    from ._guards import check_dtype

    max_loras = module.lora_b_stacked[0].shape[0]
    pool = _ensure_pool(module, max_loras)
    check_dtype(pool.dtype)

    magnitudes = getattr(module_lora, "magnitude", None)
    if magnitudes is None:
        pool[index] = 1.0
        return False
    if not isinstance(magnitudes, (list, tuple)):
        magnitudes = [magnitudes]

    w = module.weight
    is_row_parallel = isinstance(module.base_layer, RowParallelLinear)
    reduce_squares = is_row_parallel and module.tp_size > 1

    row_off = 0        # into the concatenated output
    w_off = 0          # into the base weight's rows
    folded_any = False
    for i, out_rows in enumerate(module.output_slices):
        m = magnitudes[i] if i < len(magnitudes) else None
        if m is None:
            # An unadapted sub-module of a packed layer keeps identity, which is the
            # `gate_proj` case in the pool used here.
            pool[index, row_off: row_off + out_rows] = 1.0
        else:
            g = fold_slice(
                w[w_off: w_off + out_rows],
                module.lora_a_stacked[i][index, 0],
                module.lora_b_stacked[i][index, 0, :out_rows],
                _slice_magnitude(_to_device(m, w), out_rows, module.tp_rank),
                reduce_squares=reduce_squares,
            )
            pool[index, row_off: row_off + out_rows] = g.to(pool.dtype)
            if _FOLD_LOG:
                # A per-shard checksum shows that the fold ran on the right block of
                # the magnitude under tensor parallelism.  Records are keyed by the
                # adapter id, because slot indices change from run to run.
                import json as _json
                _name = getattr(module, "_scute_name", "?")
                _ident = f"lora{lora_id}" if lora_id is not None else f"slot{index}"
                if _FOLD_VEC:
                    import os as _os
                    _d = f"{_FOLD_LOG}.vec"
                    _os.makedirs(_d, exist_ok=True)
                    _key = f"tp{module.tp_size}.rank{module.tp_rank}.{_ident}.{_name}.s{i}"
                    torch.save(g.detach().float().cpu(), f"{_d}/{_key}.pt")
                with open(f"{_FOLD_LOG}.rank{module.tp_rank}.jsonl", "a") as _f:
                    _f.write(_json.dumps(dict(
                        rank=module.tp_rank, tp=module.tp_size, slot=index,
                        lora_id=lora_id,
                        module=_name, slice=i,
                        rows=out_rows, row_parallel=bool(reduce_squares),
                        g_min=g.min().item(), g_max=g.max().item(),
                        g_sum=g.double().sum().item())) + "\n")
            if fold_e_into_b:
                b = module.lora_b_stacked[i][index, 0, :out_rows]
                b.mul_(g.to(b.dtype).unsqueeze(1))
            folded_any = True
        row_off += out_rows
        w_off += out_rows
    return folded_any


def _has_magnitude(module_lora) -> bool:
    m = getattr(module_lora, "magnitude", None)
    if m is None:
        return False
    if isinstance(m, (list, tuple)):
        return any(x is not None for x in m)
    return True


def _wrap_create_lora_weights() -> None:
    """Allocate the ``g`` pool when LoRA weights are created, before compile.

    The expand reads ``layer._scute_g_pool`` inside the layer's ``apply``, and vLLM
    traces that ``apply`` during its profiling run, before any adapter is admitted.
    Whatever the pool is at trace time is what the compiled graph does: if it does
    not exist yet, the graph is traced without the scale and never applies it, even
    after a fold fills the pool.  Ones are the identity scale, so a trace with no
    adapter admitted is a LoRA forward.
    """
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

    # Importing the layer modules registers every subclass ``__subclasses__`` sees.
    import vllm.lora.layers.column_parallel_linear  # noqa: F401
    import vllm.lora.layers.row_parallel_linear  # noqa: F401

    seen: set[type] = set()

    def walk(cls) -> None:
        if cls in seen:
            return
        seen.add(cls)
        if "create_lora_weights" in cls.__dict__:
            original = cls.create_lora_weights

            def create_lora_weights(self, *args, _orig=original, **kwargs):
                _orig(self, *args, **kwargs)
                stacked = getattr(self, "lora_b_stacked", None)
                if stacked and getattr(self, "output_slices", None):
                    _ensure_pool(self, stacked[0].shape[0])
                    _ensure_shared(self, stacked[0].shape[0])
                    _ensure_oft(self, stacked[0].shape[0])
                    if getattr(self, "punica_wrapper", None) is not None:
                        _ensure_flags(self)

            patch_attr(cls, "create_lora_weights", create_lora_weights)
        for sub in cls.__subclasses__():
            walk(sub)

    walk(BaseLinearLayerWithLoRA)


def apply() -> None:
    # The expand computes h * s + VB, so B must carry e.
    fold_e = True
    _wrap_create_lora_weights()
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    from vllm.lora.model_manager import LoRAModelManager

    from ._guards import UnsupportedDoRAPath

    # ``LRUCacheLoRAModelManager`` overrides this and delegates with ``super()``, so
    # patching the base class covers both managers.
    original = LoRAModelManager.activate_adapter

    def activate_adapter(self, lora_id: int) -> bool:
        # Let vLLM fill the stacked buffers first; the fold reads them.
        activated = original(self, lora_id)
        if not activated:
            return activated
        index = self.lora_index_to_id.index(lora_id)
        lora_model = self._registered_adapters[lora_id]
        for kind in SLOT_KINDS.values():
            kind.discard(index)
        for module_name, module in self.modules.items():
            module_lora = self._get_lora_layer_weights(lora_model, module_name)

            # Embedding and logits-processor adapters carry no ``output_slices``
            # and are not served through the linear path this integration folds
            # into. Skipping them silently would be fine for a LoRA adapter and
            # wrong for a DoRA one, so only the latter raises.
            if not isinstance(module, BaseLinearLayerWithLoRA):
                if module_lora is not None and _has_magnitude(module_lora):
                    raise UnsupportedDoRAPath(
                        f"{module_name} is a {type(module).__name__}, which this "
                        "integration does not fold into, but its adapter carries a "
                        "DoRA magnitude. It would serve as plain LoRA."
                    )
                continue

            if module_lora is None:
                max_loras = module.lora_b_stacked[0].shape[0]
                _ensure_pool(module, max_loras)[index] = 1.0
                write_vectors(module, index, None)
                continue
            module._scute_name = module_name
            fold_into_module(
                module, index, module_lora, fold_e_into_b=fold_e, lora_id=lora_id,
            )
            write_vectors(module, index, module_lora)
            if any(v is not None for v in (_as_list(getattr(module_lora, "scute_R", None)) or [])):
                SLOT_KINDS["R"].add(index)
            if any(v is not None for v in (_as_list(getattr(module_lora, "scute_d", None)) or [])):
                SLOT_KINDS["shared"].add(index)
        return activated

    patch_attr(LoRAModelManager, "activate_adapter", activate_adapter)

    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU

    original_update = PunicaWrapperGPU.update_metadata

    def update_metadata(self, mapping, lora_index_to_id, *args, **kwargs):
        """vLLM's, plus two CPU flags for the step: is any OFT or shared-factor adapter in
        the batch.  Computed from the CPU-side mapping, so no device sync."""
        original_update(self, mapping, lora_index_to_id, *args, **kwargs)
        if not hasattr(self, "_scute_flags"):
            self._scute_flags = torch.ones(2, dtype=torch.bool)
        if not _ARMED[0]:
            self._scute_flags.fill_(True)
            return
        ids = {i for i in set(mapping.index_mapping) if i and i > 0}
        slots = {lora_index_to_id.index(i) for i in ids if i in lora_index_to_id}
        self._scute_flags[0] = bool(slots & SLOT_KINDS["R"])
        self._scute_flags[1] = bool(slots & SLOT_KINDS["shared"])

    patch_attr(PunicaWrapperGPU, "update_metadata", update_metadata)

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_capture = GPUModelRunner.capture_model

    def capture_model(self, *args, **kwargs):
        out = original_capture(self, *args, **kwargs)
        _ARMED[0] = True
        return out

    patch_attr(GPUModelRunner, "capture_model", capture_model)
