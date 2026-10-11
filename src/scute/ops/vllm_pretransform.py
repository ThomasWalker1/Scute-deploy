"""OFT's input-side slot ``R`` inside vLLM: ``x <- x R_a`` per token, before the base GEMM.

The same pass as :mod:`scute.ops.pretransform`, indexed by vLLM's own token-to-adapter
metadata (``token_indices_sorted_by_lora_ids``, ``num_tokens_per_lora``,
``lora_token_start_loc``, ``lora_ids``) exactly as vLLM's ``lora_expand`` is, so it is
safe under CUDA graphs: the grid and the tables are fixed buffers.

Out of place: each program reads a ``[BLOCK_M, b]`` block of ``x`` and writes the same
block of ``y``; rows with no adapter (vLLM's ``-1`` group) are copied.

Needs the vLLM environment (it registers a custom op the way vLLM registers its own).
"""


import torch
import triton
import triton.language as tl

__all__ = ["vllm_pretransform"]

BLOCK_M = 32
G = 4


@triton.jit
def _vllm_pretransform_kernel(
    x_ptr,
    y_ptr,
    r_ptr,
    M,
    n_blocks,
    stride_x_row,
    stride_r_slot,
    stride_r_slice,
    stride_y_slice,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
    G: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)
    r_ptr += slice_id * stride_r_slice
    y_ptr += slice_id * stride_y_slice
    cta_m_num = tl.cdiv(M, BLOCK_M)
    pid_m = pid % cta_m_num
    pid_g = pid // cta_m_num

    lora_id = tl.load(lora_ids + lora_idx)
    m_size = tl.load(num_tokens_per_lora + lora_idx)
    m_off = pid_m * BLOCK_M
    if m_off >= m_size:
        return
    start = tl.load(lora_token_start_loc + lora_idx)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < (m_size - m_off)
    tok = tl.load(token_indices_sorted_by_lora_ids + start + m_off + offs_m, mask=m_mask,
                  other=0)
    offs_b = tl.arange(0, BLOCK)
    for gi in tl.static_range(G):
        j = pid_g * G + gi
        if j < n_blocks:
            cols = j * BLOCK + offs_b
            x_ptrs = x_ptr + tok[:, None] * stride_x_row + cols[None, :]
            y_ptrs = y_ptr + tok[:, None] * stride_x_row + cols[None, :]
            xt = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            if lora_id == -1:
                # A row with no adapter: the identity, copied.
                tl.store(y_ptrs, xt, mask=m_mask[:, None])
            else:
                rt = tl.load(r_ptr + lora_id * stride_r_slot + j * BLOCK * BLOCK
                             + offs_b[:, None] * BLOCK + offs_b[None, :])
                out = tl.dot(xt, rt, out_dtype=tl.float32)
                tl.store(y_ptrs, out.to(y_ptr.dtype.element_ty), mask=m_mask[:, None])


@torch.inference_mode()
def _vllm_pretransform(
    x: torch.Tensor,
    y: torch.Tensor,
    r_pool: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
) -> None:
    """``y[t] = x[t] blockdiag(R[slot(t)])``, and ``y[t] = x[t]`` for rows with no adapter.

    Out of place, in one pass: a separate copy followed by an in-place transform reads
    and writes ``x`` twice.

    ``r_pool`` is ``[max_loras + 1, n_blocks, b, b]`` (a slice of the layer's pool may be
    passed; only its slot stride is used), each block row-major.
    """
    if no_lora_flag_cpu.item():
        y.copy_(x)
        return
    assert x.stride(1) == 1 and y.stride() == x.stride()
    _, n_blocks, b, b2 = r_pool.shape
    assert b == b2 and n_blocks * b == x.shape[1] and b >= 16
    assert r_pool.stride(3) == 1 and r_pool.stride(2) == b and r_pool.stride(1) == b * b
    M = x.shape[0]
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(n_blocks, G), 1, num_active_loras.item())
    _vllm_pretransform_kernel[grid](
        x, y, r_pool, M, n_blocks, x.stride(0), r_pool.stride(0), 0, 0,
        token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc,
        lora_ids, BLOCK_M=BLOCK_M, BLOCK=b, G=G, num_warps=4, num_stages=2,
    )


def _fake(x, y, r_pool, *args) -> None:
    return None


_LIB = None


def _register() -> None:
    global _LIB
    if hasattr(getattr(torch.ops, "scute", None), "vllm_pretransform"):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    _LIB = torch.library.Library("scute", "FRAGMENT")
    direct_register_custom_op(
        op_name="vllm_pretransform",
        op_func=_vllm_pretransform,
        mutates_args=["y"],
        fake_impl=_fake,
        target_lib=_LIB,
    )


def vllm_pretransform(*args, **kwargs) -> None:
    _register()
    torch.ops.scute.vllm_pretransform(*args, **kwargs)


def _pretransform_slices(x, ys, r_pool, meta) -> None:
    """All slices of a layer in one launch: ``ys[s] = x blockdiag(R[slot, s])``.

    ``ys`` is ``[n_slices, N, d_in]``, ``r_pool`` ``[max_loras + 1, n_slices, nb, b, b]``.
    One launch per layer instead of one per slice: at prefill sizes above the largest
    captured CUDA graph each launch is paid on the host.
    """
    (token_lora_mapping, token_indices_sorted_by_lora_ids, num_tokens_per_lora,
     lora_token_start_loc, lora_ids, no_lora_flag_cpu, num_active_loras) = meta
    if no_lora_flag_cpu.item():
        ys.copy_(x.unsqueeze(0).expand_as(ys))
        return
    n_slices, M, d_in = ys.shape
    _, _, n_blocks, b, _ = r_pool.shape
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(n_blocks, G), n_slices, num_active_loras.item())
    _vllm_pretransform_kernel[grid](
        x, ys, r_pool, M, n_blocks, x.stride(0), r_pool.stride(0), r_pool.stride(1),
        ys.stride(0), token_indices_sorted_by_lora_ids, num_tokens_per_lora,
        lora_token_start_loc, lora_ids, BLOCK_M=BLOCK_M, BLOCK=b, G=G, num_warps=4,
        num_stages=2)


def _oft_base(
    out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    r_pool: torch.Tensor,
    rows: list[int],
    flags: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
) -> None:
    """The base GEMM with OFT's pre-transform, ``out[:, s] = (x R_s) W_s^T`` per slice.

    Opaque to the compiler, so each slice's GEMM writes straight into its columns of
    ``out`` (cuBLAS takes the row stride as the leading dimension) instead of a
    temporary plus a copy, one read and write of ``[N, rows]`` less per slice.
    """
    if not flags[0].item() and not torch.cuda.is_current_stream_capturing():
        # No OFT adapter in this step: the fused base GEMM.  A captured graph always
        # takes the split path, since it is replayed for any batch.
        torch.mm(x, w.t(), out=out)
        return
    meta = (token_lora_mapping, token_indices_sorted_by_lora_ids, num_tokens_per_lora,
            lora_token_start_loc, lora_ids, no_lora_flag_cpu, num_active_loras)
    xs = torch.empty((len(rows),) + tuple(x.shape), dtype=x.dtype, device=x.device)
    _pretransform_slices(x, xs, r_pool, meta)
    off = 0
    for i, n in enumerate(rows):
        torch.mm(xs[i], w[off: off + n].t(), out=out[:, off: off + n])
        off += n


def _oft_base_fake(out, x, w, r_pool, rows, flags, *meta) -> None:
    return None


_LIB_BASE = None


def _register_base() -> None:
    global _LIB_BASE
    if hasattr(getattr(torch.ops, "scute", None), "oft_base"):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    _LIB_BASE = torch.library.Library("scute", "FRAGMENT")
    direct_register_custom_op(op_name="oft_base", op_func=_oft_base, mutates_args=["out"],
                              fake_impl=_oft_base_fake, target_lib=_LIB_BASE)


def oft_base_op(*args) -> None:
    _register_base()
    torch.ops.scute.oft_base(*args)
