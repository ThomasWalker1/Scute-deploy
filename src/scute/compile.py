"""Admission compiler: a PEFT checkpoint of any supported method, in normal form.

The runtime (:mod:`scute.vllm_integration`) serves one operator per hook,

    h' = (h ⊙ s) + Σ_k c_k ((x A_k) ⊙ d_k) B_k ⊙ e_k ,

and loads its tensors under vLLM's LoRA loader.  This module writes an adapter
directory that the patched loader reads, whatever PEFT method produced the
checkpoint.  Each method is mapped by the fold rule:

* **LoRA**: an owned term; the scale folds into ``B`` (vLLM already does this).
* **DoRA**: an owned term plus the magnitude, copied unchanged; the engine folds
  ``e`` into ``B`` and fills ``s`` at admission, because ``g`` needs the base weight
  as the engine shards it.
* **(IA)^3**: the live slot ``s`` and nothing else.  On a feed-forward module PEFT
  scales the *input* of ``down_proj``; in a gated MLP that input is
  ``act(gate) * up``, so the same vector is ``s`` on the ``up`` slice of ``up_proj``.
* **VeRA**: a term whose factors are shared by every VeRA adapter of the same key,
  with live inner and outer diagonals ``d`` and ``e``.  The factors are written once,
  to ``scute_shared.safetensors``, and the engine reads them once per batch.

Vector-only modules still need a term for vLLM to admit them, so they get a
zero rank-1 term; it adds nothing to the output.

Tensor names, per module ``<m>`` (vLLM's naming, without ``base_model.model.``):

    <m>.lora_A.weight, <m>.lora_B.weight   owned term (or zero rank 1)
    <m>.lora_magnitude_vector              DoRA magnitude (folded by the engine)
    <m>.scute_s                            output scale s, [d_out]
    <m>.scute_d, <m>.scute_e               shared-factor term diagonals, [r], [d_out]

``adapter_config.json`` stays a LoRA config for vLLM's checks, plus a ``scute``
entry naming the source method and, for VeRA, the shared-factor key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass

import torch
from safetensors.torch import load_file, save_file

PREFIX = "base_model.model."
SHARED_FILE = "scute_shared.safetensors"

# Llama-family module geometry, used to size zero terms and to place (IA)^3 vectors.
_LINEAR_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                    "down_proj")


@dataclass
class Compiled:
    method: str
    out_dir: str
    modules: int
    shared_key: str | None = None


def _read_peft(path: str) -> tuple[dict, dict[str, torch.Tensor]]:
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    st = os.path.join(path, "adapter_model.safetensors")
    if os.path.exists(st):
        tensors = load_file(st)
    else:
        tensors = torch.load(os.path.join(path, "adapter_model.bin"), map_location="cpu")
    return cfg, tensors


def _method(cfg: dict) -> str:
    t = str(cfg.get("peft_type", "")).upper()
    if t == "LORA":
        return "dora" if cfg.get("use_dora") else "lora"
    if t == "IA3":
        return "ia3"
    if t == "VERA":
        return "vera"
    if t == "OFT":
        return "oft"
    raise ValueError(f"unsupported PEFT type {t!r}")


def _lora_config(cfg: dict, targets: list[str], r: int, alpha: float, method: str,
                 extra: dict | None = None) -> dict:
    out = dict(
        peft_type="LORA", task_type=cfg.get("task_type", "CAUSAL_LM"),
        base_model_name_or_path=cfg.get("base_model_name_or_path"),
        r=r, lora_alpha=alpha, target_modules=sorted(set(targets)), bias="none",
        use_dora=False, fan_in_fan_out=False, modules_to_save=None,
        scute=dict(method=method, **(extra or {})),
    )
    return out


def _strip(name: str) -> str:
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def _module_of(key: str, suffix_re: str) -> str:
    return re.sub(suffix_re, "", _strip(key))


def _zero_term(d_in: int, d_out: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(1, d_in, dtype=torch.float16), torch.zeros(d_out, 1, dtype=torch.float16)


def _dims(module: str, shapes: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """(d_in, d_out) of a module, from the base model's config geometry."""
    return shapes[module.rsplit(".", 1)[-1]]


def load_base_config(model: dict | str) -> dict:
    """The base model's config: a dict, a ``config.json``, a model directory, or a
    Hugging Face model id."""
    if isinstance(model, dict):
        return model
    if os.path.isfile(model):
        return json.load(open(model))
    if os.path.isdir(model):
        return json.load(open(os.path.join(model, "config.json")))
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(model, "config.json")))


def base_shapes(model) -> dict[str, tuple[int, int]]:
    """(d_in, d_out) per projection name for a Llama-family config."""
    c = load_base_config(model)
    h, i = c["hidden_size"], c["intermediate_size"]
    hd = c.get("head_dim", h // c["num_attention_heads"])
    q, kv = c["num_attention_heads"] * hd, c["num_key_value_heads"] * hd
    return dict(q_proj=(h, q), k_proj=(h, kv), v_proj=(h, kv), o_proj=(q, h),
                gate_proj=(h, i), up_proj=(h, i), down_proj=(i, h))


def _copy_through(src: str, out: str, cfg: dict, method: str, tensors: dict) -> Compiled:
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith((".safetensors", ".bin", ".json")):
            shutil.copy(os.path.join(src, f), os.path.join(out, f))
    c = json.load(open(os.path.join(out, "adapter_config.json")))
    c["scute"] = dict(method=method)
    json.dump(c, open(os.path.join(out, "adapter_config.json"), "w"), indent=1)
    n = sum(1 for k in tensors if k.endswith("lora_A.weight"))
    return Compiled(method, out, n)


def _compile_ia3(cfg: dict, tensors: dict, shapes: dict, out: str) -> Compiled:
    ff = set(cfg.get("feedforward_modules") or [])
    res: dict[str, torch.Tensor] = {}
    targets = []
    n = 0
    for key, t in tensors.items():
        if not key.endswith("ia3_l"):
            continue
        module = _module_of(key, r"\.ia3_l$")
        name = module.rsplit(".", 1)[-1]
        vec = t.flatten().to(torch.float32)
        if name in ff or any(module.endswith(f) for f in ff):
            # Input scale of down_proj == output scale of the up slice (gated MLP).
            if name != "down_proj":
                raise ValueError(f"feed-forward (IA)^3 on {name} has no output-side form")
            module = module.rsplit(".", 1)[0] + ".up_proj"
            name = "up_proj"
        d_in, d_out = shapes[name]
        if vec.numel() != d_out:
            raise ValueError(f"{module}: (IA)^3 vector of {vec.numel()} != d_out {d_out}")
        a, b = _zero_term(d_in, d_out)
        res[f"{PREFIX}{module}.lora_A.weight"] = a
        res[f"{PREFIX}{module}.lora_B.weight"] = b
        res[f"{PREFIX}{module}.scute_s"] = vec
        targets.append(name)
        n += 1
    os.makedirs(out, exist_ok=True)
    save_file(res, os.path.join(out, "adapter_model.safetensors"))
    json.dump(_lora_config(cfg, targets, 1, 1, "ia3"),
              open(os.path.join(out, "adapter_config.json"), "w"), indent=1)
    return Compiled("ia3", out, n)


def shared_key(a: torch.Tensor, b: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in (a, b):
        h.update(t.contiguous().to(torch.float32).numpy().tobytes())
    return h.hexdigest()[:16]


def _compile_vera(cfg: dict, tensors: dict, shapes: dict, out: str) -> Compiled:
    va = next((t for k, t in tensors.items() if k.endswith("vera_A")), None)
    vb = next((t for k, t in tensors.items() if k.endswith("vera_B")), None)
    if va is None or vb is None:
        raise ValueError("VeRA checkpoint without vera_A/vera_B: train with "
                         "save_projection=True")
    va, vb = va.to(torch.float32), vb.to(torch.float32)
    key = shared_key(va, vb)
    res: dict[str, torch.Tensor] = {}
    targets = []
    n = 0
    for k, t in tensors.items():
        if not k.endswith("vera_lambda_d"):
            continue
        module = _module_of(k, r"\.vera_lambda_d$")
        name = module.rsplit(".", 1)[-1]
        lam_b = tensors[k.replace("vera_lambda_d", "vera_lambda_b")]
        d_in, d_out = shapes[name]
        if lam_b.numel() != d_out:
            raise ValueError(f"{module}: lambda_b of {lam_b.numel()} != d_out {d_out}")
        a, b = _zero_term(d_in, d_out)
        res[f"{PREFIX}{module}.lora_A.weight"] = a
        res[f"{PREFIX}{module}.lora_B.weight"] = b
        res[f"{PREFIX}{module}.scute_d"] = t.flatten().to(torch.float32)
        res[f"{PREFIX}{module}.scute_e"] = lam_b.flatten().to(torch.float32)
        targets.append(name)
        n += 1
    os.makedirs(out, exist_ok=True)
    save_file(res, os.path.join(out, "adapter_model.safetensors"))
    save_file(dict(vera_A=va.contiguous(), vera_B=vb.contiguous()),
              os.path.join(out, SHARED_FILE))
    json.dump(_lora_config(cfg, targets, 1, 1, "vera",
                           dict(shared_key=key, shared_rank=int(va.shape[0]))),
              open(os.path.join(out, "adapter_config.json"), "w"), indent=1)
    return Compiled("vera", out, n, key)


def _compile_oft(cfg: dict, tensors: dict, shapes: dict, out: str) -> Compiled:
    """OFT's rotation folded to explicit blocks ``R`` at admission (the Cayley map).

    The blocks are what PEFT's ``OFTRotationModule`` builds in its forward, in the same
    parametrization the adapter was trained with (exact or truncated Neumann).
    """
    from scute.ops.pretransform import cayley

    if cfg.get("coft") or cfg.get("block_share"):
        raise NotImplementedError("COFT and block-shared OFT are not compiled")
    b = int(cfg["oft_block_size"])
    neumann = bool(cfg.get("use_cayley_neumann", True))
    terms = int(cfg.get("num_cayley_neumann_terms", 5))
    res: dict[str, torch.Tensor] = {}
    targets = []
    for key, t in tensors.items():
        if not re.search(r"\.oft_R(\.weight)?$", key):
            continue
        module = _module_of(key, r"\.oft_R(\.weight)?$")
        name = module.rsplit(".", 1)[-1]
        d_in, d_out = shapes[name]
        r_blocks = cayley(t.to(torch.float64), b, neumann, terms).to(torch.float32)
        if r_blocks.shape != (d_in // b, b, b):
            raise ValueError(f"{module}: OFT blocks {tuple(r_blocks.shape)} for d_in {d_in}")
        a0, b0 = _zero_term(d_in, d_out)
        res[f"{PREFIX}{module}.lora_A.weight"] = a0
        res[f"{PREFIX}{module}.lora_B.weight"] = b0
        res[f"{PREFIX}{module}.scute_R"] = r_blocks.contiguous()
        targets.append(name)
    os.makedirs(out, exist_ok=True)
    save_file(res, os.path.join(out, "adapter_model.safetensors"))
    json.dump(_lora_config(cfg, targets, 1, 1, "oft", dict(block=b)),
              open(os.path.join(out, "adapter_config.json"), "w"), indent=1)
    return Compiled("oft", out, len(targets))


def compile_adapter(src: str, out: str, base_config: dict | str) -> Compiled:
    """Compile the PEFT adapter at ``src`` into normal form at ``out``.

    ``base_config`` is the base model's config: a dict, a ``config.json``, a model
    directory, or a Hugging Face model id.
    """
    cfg, tensors = _read_peft(src)
    method = _method(cfg)
    shapes = base_shapes(base_config)
    if method in ("lora", "dora"):
        return _copy_through(src, out, cfg, method, tensors)
    if method == "ia3":
        return _compile_ia3(cfg, tensors, shapes, out)
    if method == "vera":
        return _compile_vera(cfg, tensors, shapes, out)
    if method == "oft":
        return _compile_oft(cfg, tensors, shapes, out)
    raise NotImplementedError(f"{method} is not compiled yet")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--base", required=True,
                    help="base model: Hugging Face id, model directory or config.json")
    a = ap.parse_args()
    print(compile_adapter(a.src, a.out, a.base))


if __name__ == "__main__":
    main()
