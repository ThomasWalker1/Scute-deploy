"""Make the paths this integration does not cover fail closed.

The integration covers dense linear layers on the default (non-fully-sharded) LoRA
path.  It does not cover quantized weights, MoE experts,
embedding adapters, or the fully-sharded LoRA variants.

A silently wrong number is the failure mode that matters here.  Every uncovered path
below would otherwise run and produce plausible tokens with the magnitude applied to
the wrong tensor, or not applied at all.  So each one raises at setup, where it is attributable, rather than at a
number nobody can trace.
"""

from __future__ import annotations


class UnsupportedDoRAPath(RuntimeError):
    """A DoRA adapter reached a path this integration has not implemented."""


def check_config(lora_config, model_config=None) -> None:
    """Refuse configurations whose serving path is not covered."""
    if getattr(lora_config, "fully_sharded_loras", False):
        raise UnsupportedDoRAPath(
            "fully_sharded_loras=True routes through _mcp_apply and the sharded "
            "layer variants, which this integration does not patch. Run with "
            "fully_sharded_loras=False, which is vLLM's default."
        )
    if getattr(lora_config, "bias_enabled", False):
        raise UnsupportedDoRAPath(
            "LoRA bias is enabled. PEFT's DoRA does not scale the bias, and this "
            "integration does not separate it from the scaled output."
        )


def check_dtype(dtype) -> None:
    """Refuse bfloat16, which cannot represent a trained DoRA magnitude.

    ``g`` sits within about 1% of unity, and bfloat16's 8 mantissa bits resolve that
    band into 9 distinct values -- a 3.9e-3 relative error against a 2% signal. The
    adapter would still serve, and about a fifth of its effect would be quantisation
    noise.
    """
    import torch

    if dtype == torch.bfloat16:
        raise UnsupportedDoRAPath(
            "dtype=bfloat16 cannot represent a DoRA magnitude: g spans ~[0.98, 1.02] "
            "and bfloat16 resolves that into 9 values (3.9e-3 relative error against "
            "a 2% signal). Run with dtype='float16' or float32."
        )


def check_module(module) -> None:
    """Refuse a module whose weight the fold cannot read as ``[out, in]`` fp16/bf16."""
    base = module.base_layer
    quant = getattr(base, "quant_method", None)
    quant_name = type(quant).__name__ if quant is not None else ""
    if quant is not None and "Unquantized" not in quant_name:
        raise UnsupportedDoRAPath(
            f"{type(base).__name__} uses {quant_name}. The fold needs the dense base "
            "weight to compute ||W + AB||, and a packed or quantized weight is not "
            "that tensor."
        )
    if not hasattr(base, "weight"):
        raise UnsupportedDoRAPath(
            f"{type(base).__name__} exposes no dense .weight for the fold."
        )


def check_magnitude_coverage(lora_model, expected_modules: set[str]) -> None:
    """Refuse a DoRA checkpoint whose magnitudes did not all reach a covered module.

    A magnitude that is parsed but never folded is the quiet failure: the adapter
    serves as a plain LoRA.
    """
    missing = sorted(
        name for name, w in lora_model.loras.items()
        if getattr(w, "magnitude", None) is not None and name not in expected_modules
    )
    if missing:
        raise UnsupportedDoRAPath(
            f"{len(missing)} module(s) carry a DoRA magnitude but were not folded, "
            f"e.g. {missing[:3]}. They would serve as plain LoRA."
        )
