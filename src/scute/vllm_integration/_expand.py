"""Carry the output scale ``s`` inside vLLM's own expand.

The layer runs vLLM's shrink, then :func:`scute.ops.vllm_expand_s.lora_expand_s`,
which is vLLM's shipped ``lora_expand`` with one change to its epilogue: the base
output it already loads is multiplied by ``s`` before the term is added.  The grid,
the tiling, the configs and the metadata are vLLM's, so the patched layer differs from
stock LoRA serving by the slot and by nothing else.  It issues the same two launches per
adapted layer as stock LoRA: one shrink and one expand covering every output slice.

``s`` lives in a ``[max_loras + 1, total_out]`` pool per layer, allocated with the
LoRA weights (``_fold._wrap_create_lora_weights``) so that vLLM's compiled forward
reads it as a buffer from the first trace on.  The last row is ones, so a token with
no adapter, and a plain LoRA adapter, are scaled by one on the same path.  The ``e``
half of DoRA's magnitude is folded into ``B`` at admission, which is why the expand
only has to carry ``s``.
"""

from __future__ import annotations

import torch

from ._util import patch_attr


def expand_with_slot(layer, output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``add_lora_linear``, with the expand replaced by the one that carries ``s``."""
    from scute.ops.vllm_expand_s import lora_expand_s

    wrapper = layer.punica_wrapper
    y = output.view(-1, output.shape[-1])
    x2 = x.view(-1, x.shape[-1])
    rank = layer.lora_b_stacked[0].size(-1)
    # fp32, as vLLM allocates it; the shrink zeroes it.
    buffer = torch.empty(
        (len(layer.output_slices), x2.size(0), rank), dtype=torch.float32, device=x2.device
    )
    mask = (getattr(layer, "_scute_main_mask", None)
            if _os.environ.get("SCUTE_MAIN_MASK", "1") == "1" else None)
    if mask is None:
        wrapper.add_shrink(buffer, x2, layer.lora_a_stacked, 1.0)
    else:
        from scute.ops.vllm_shrink_m import lora_shrink_m
        lora_shrink_m(x2, layer.lora_a_stacked, buffer,
                      *wrapper.token_mapping_meta.meta_args(
                          x2.size(0), wrapper.lora_config.specialize_active_lora),
                      1.0, slot_mask=mask)
    lora_expand_s(
        buffer,
        layer.lora_b_stacked,
        y,
        layer._scute_g_pool,
        *wrapper.token_mapping_meta.meta_args(
            x2.size(0), wrapper.lora_config.specialize_active_lora
        ),
        offset_start=0,
        add_inputs=True,
        slot_mask=mask,
    )
    return output


def shared_term(layer, y: torch.Tensor, x: torch.Tensor) -> None:
    """Add ``((x A) ⊙ d) B ⊙ e`` for terms whose factors every adapter shares (VeRA).

    ``A`` and ``B`` are the same for every adapter of the key, so ``xA`` is one dense
    GEMM over the whole batch and each slice's ``B`` is one more: the factors are read
    once per step however many adapters use them.  ``d`` and ``e`` are gathered per
    token; their default rows (zero ``d``) make the term vanish for every other token.
    """
    idx = layer.punica_wrapper._token_lora_indices[: x.shape[0]]
    v = x @ layer._scute_shared_A.t()
    d = layer._scute_d_pool[idx]
    e = layer._scute_e_pool[idx]
    off = 0
    for i, out in enumerate(layer.output_slices):
        b = getattr(layer, f"_scute_shared_B{i}")
        y[:, off: off + out] += ((v * d[:, i]) @ b.t()) * e[:, off: off + out]
        off += out


import os as _os

#: ``SCUTE_SHARED_IMPL`` picks how the shared-factor term runs:
#:
#: ``dense`` (default): ``x A^T`` by cuBLAS, then one Triton expand per slice that is
#:   dense over row tiles (``B`` read once per row tile), gathers ``d`` and ``e`` per
#:   row, and loads and stores only rows whose adapter has the term.  The factors are
#:   read once per step, whatever the number of tenants.
#: ``torch``: the same dense form in PyTorch, over every row of the batch.
#: ``fused``: two kernels over the tenant segments only; each segment reads the factors.
#: ``hybrid``: a dense shrink, then an expand over the tenant segments.
#: ``auto``: the dense form below ``SCUTE_SHARED_SEG_ROWS`` rows (256) and the segmented
#:   expand above, decided inside one opaque op.
_SHARED_IMPL = _os.environ.get("SCUTE_SHARED_IMPL", "dense")


def shared_term_fused(layer, y: torch.Tensor, x: torch.Tensor) -> None:
    """``((x A) ⊙ d) B ⊙ e`` in two launches over the tokens whose adapter has the term."""
    from scute.ops.vllm_expand_s import lora_expand_s
    from scute.ops.vllm_shared import vllm_shared_shrink

    wrapper = layer.punica_wrapper
    meta = wrapper.token_mapping_meta.meta_args(
        x.size(0), wrapper.lora_config.specialize_active_lora)
    r = layer._scute_shared_A.shape[0]
    buf = torch.empty((len(layer.output_slices), x.size(0), r), dtype=torch.float32,
                      device=x.device)
    vllm_shared_shrink(buf, x, layer._scute_shared_A, layer._scute_d_pool,
                       layer._scute_shared_mask, *meta)
    lora_expand_s(buf, layer._scute_shared_Bx, y, None, *meta, offset_start=0,
                  add_inputs=True, e_pool=layer._scute_e_pool,
                  slot_mask=layer._scute_shared_mask)


def shared_term_dense(layer, y: torch.Tensor, x: torch.Tensor) -> None:
    """The default: ``x A^T`` once, then one dense-over-rows expand per slice that reads
    each tile of ``B`` once for all rows and touches only rows that have the term."""
    from scute.ops.vllm_shared import shared_dense_op

    idx = layer.punica_wrapper._token_lora_indices[: x.size(0)]
    b_list = [getattr(layer, f"_scute_shared_B{i}") for i in range(len(layer.output_slices))]
    shared_dense_op(y, x, layer._scute_shared_A, b_list, layer._scute_d_pool,
                    layer._scute_e_pool, layer._scute_shared_mask, idx,
                    layer.punica_wrapper._scute_flags)


def shared_term_auto(layer, y: torch.Tensor, x: torch.Tensor) -> None:
    """One opaque op that runs the dense form for small batches and the
    segmented expand for large ones (:func:`scute.ops.vllm_shared._shared_term`)."""
    from scute.ops.vllm_shared import shared_term_op

    wrapper = layer.punica_wrapper
    meta = wrapper.token_mapping_meta.meta_args(
        x.size(0), wrapper.lora_config.specialize_active_lora)
    idx = wrapper._token_lora_indices[: x.size(0)]
    b_list = [getattr(layer, f"_scute_shared_B{i}") for i in range(len(layer.output_slices))]
    shared_term_op(y, x, layer._scute_shared_A, b_list, layer._scute_shared_Bx,
                   layer._scute_d_pool, layer._scute_e_pool, layer._scute_shared_mask, idx,
                   *meta)


def shared_term_hybrid(layer, y: torch.Tensor, x: torch.Tensor) -> None:
    """Dense shrink over every row (cuBLAS reads ``A`` once per step), then the masked
    expand with ``e`` over the rows whose adapter has the term."""
    from scute.ops.vllm_expand_s import lora_expand_s

    wrapper = layer.punica_wrapper
    meta = wrapper.token_mapping_meta.meta_args(
        x.size(0), wrapper.lora_config.specialize_active_lora)
    idx = wrapper._token_lora_indices[: x.size(0)]
    v = (x @ layer._scute_shared_A.t()).float()
    buf = (v.unsqueeze(0) * layer._scute_d_pool[idx].transpose(0, 1).float()).contiguous()
    lora_expand_s(buf, layer._scute_shared_Bx, y, None, *meta, offset_start=0,
                  add_inputs=True, e_pool=layer._scute_e_pool,
                  slot_mask=layer._scute_shared_mask)


def oft_base(layer, x: torch.Tensor, bias) -> torch.Tensor:
    """The base GEMM with OFT's pre-transform: ``(x R_s) W_s`` per output slice.

    A packed layer (``qkv``, ``gate_up``) shares one input, but each of its slices may
    carry its own ``R``, so the GEMM is split by slice: the weight bytes read are the
    same, each slice reads its own transformed copy of ``x``.  One opaque op
    (:func:`scute.ops.vllm_pretransform._oft_base`).
    """
    from scute.ops.vllm_pretransform import oft_base_op

    wrapper = layer.punica_wrapper
    meta = wrapper.token_mapping_meta.meta_args(
        x.size(0), wrapper.lora_config.specialize_active_lora)
    w = layer.base_layer.weight
    out = torch.empty(x.size(0), w.shape[0], dtype=x.dtype, device=x.device)
    oft_base_op(out, x, w, layer._scute_R_pool, list(layer.output_slices),
                wrapper._scute_flags, *meta)
    if bias is not None:
        out += bias
    return out


def apply() -> None:
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

    from scute.ops.vllm_expand_s import _register
    from scute.ops.vllm_pretransform import _register as _register_pre
    from scute.ops.vllm_shared import _register as _register_shared

    # Register the ops before vLLM traces anything, so the trace sees opaque ops.
    _register()
    _register_pre()
    from scute.ops.vllm_pretransform import _register_base
    _register_base()
    from scute.ops.vllm_shrink_m import _register as _register_shrink
    _register_shrink()
    _register_shared()
    from scute.ops.vllm_shared import _register_dense, _register_term
    _register_term()
    _register_dense()

    def _apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if hasattr(self, "_scute_R_pool"):
            output = oft_base(self, x.view(-1, x.shape[-1]), bias).view(
                *x.shape[:-1], -1)
        else:
            output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        original_shape = output.shape if output.ndim == 3 else None
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)
        output = expand_with_slot(self, output, x)
        if hasattr(self, "_scute_d_pool"):
            {"fused": shared_term_fused, "hybrid": shared_term_hybrid,
             "torch": shared_term, "auto": shared_term_auto}.get(
                _SHARED_IMPL, shared_term_dense)(
                self, output.view(-1, output.shape[-1]), x.view(-1, x.shape[-1]))
        if original_shape is not None:
            output = output.reshape(original_shape)
        return output

    patch_attr(BaseLinearLayerWithLoRA, "apply", _apply)
