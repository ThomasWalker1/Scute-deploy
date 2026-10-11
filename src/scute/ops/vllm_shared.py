"""The shrink of a term whose factors every adapter shares (VeRA), with its inner diagonal.

``W[s, t] = (x[t] A^T) ⊙ d[slot(t), s]`` for every slice ``s`` of the layer and every
token ``t`` whose adapter has a shared-factor term.  ``A`` is the same for every adapter,
so a program computes ``x A^T`` once for its tile and writes one scaled copy per slice.
Tokens are visited through vLLM's own sorted metadata, as its shrink does, and slots
without the term (``slot_mask == 0``) return at once: the FLOPs are spent only on the
tokens that need them, and ``A`` is read once per tile from L2.

The expand that follows is :func:`scute.ops.vllm_expand_s.lora_expand_s` with the
shared ``B`` aliased across slots (slot stride 0), ``e`` in its epilogue and the same
mask.  Needs the vLLM environment.
"""

import torch
import triton
import triton.language as tl

__all__ = ["vllm_shared_shrink"]

BLOCK_M = 32
BLOCK_R = 64
BLOCK_K = 64


@triton.jit
def _shared_shrink_kernel(
    x_ptr,
    a_ptr,
    d_ptr,
    out_ptr,
    slot_mask_ptr,
    M,
    K,
    R,
    stride_x_row,
    stride_a_row,
    stride_d_slot,
    stride_d_slice,
    stride_o_slice,
    stride_o_row,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    N_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    lora_idx = tl.program_id(axis=2)
    cta_m_num = tl.cdiv(M, BLOCK_M)
    pid_m = pid % cta_m_num
    pid_r = pid // cta_m_num

    slot = tl.load(lora_ids + lora_idx)
    if slot == -1:
        return
    if tl.load(slot_mask_ptr + slot) == 0:
        return
    m_size = tl.load(num_tokens_per_lora + lora_idx)
    m_off = pid_m * BLOCK_M
    if m_off >= m_size:
        return
    start = tl.load(lora_token_start_loc + lora_idx)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < (m_size - m_off)
    tok = tl.load(token_indices_sorted_by_lora_ids + start + m_off + offs_m, mask=m_mask,
                  other=0)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = offs_r < R
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        xt = tl.load(x_ptr + tok[:, None] * stride_x_row + kk[None, :],
                     mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        at = tl.load(a_ptr + offs_r[None, :] * stride_a_row + kk[:, None],
                     mask=r_mask[None, :] & k_mask[:, None], other=0.0)
        acc += tl.dot(xt, at, out_dtype=tl.float32)
    for s in tl.static_range(N_SLICES):
        d = tl.load(d_ptr + slot * stride_d_slot + s * stride_d_slice + offs_r, mask=r_mask,
                    other=0.0).to(tl.float32)
        o_ptrs = (out_ptr + s * stride_o_slice + tok[:, None] * stride_o_row
                  + offs_r[None, :])
        tl.store(o_ptrs, acc * d[None, :], mask=m_mask[:, None] & r_mask[None, :])


@torch.inference_mode()
def _vllm_shared_shrink(
    out: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    d_pool: torch.Tensor,
    slot_mask: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
) -> None:
    """``out[s, t] = (x[t] a^T) * d_pool[slot(t), s]`` for masked-in slots.

    Shapes: ``out`` ``[n_slices, N, R]`` float32, ``x`` ``[N, K]``, ``a`` ``[R, K]`` (any
    row stride), ``d_pool`` ``[max_loras + 1, n_slices, R]``.
    """
    if no_lora_flag_cpu.item():
        return
    n_slices, M, R = out.shape
    K = x.shape[1]
    assert x.stride(1) == 1 and a.stride(1) == 1 and out.stride(2) == 1
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(R, BLOCK_R), 1, num_active_loras.item())
    _shared_shrink_kernel[grid](
        x, a, d_pool, out, slot_mask, M, K, R,
        x.stride(0), a.stride(0), d_pool.stride(0), d_pool.stride(1),
        out.stride(0), out.stride(1),
        token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc,
        lora_ids, N_SLICES=n_slices, BLOCK_M=BLOCK_M, BLOCK_R=BLOCK_R, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )


def _fake(out, x, a, d_pool, slot_mask, *args) -> None:
    return None


_LIB = None


def _register() -> None:
    global _LIB
    if hasattr(getattr(torch.ops, "scute", None), "vllm_shared_shrink"):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    _LIB = torch.library.Library("scute", "FRAGMENT")
    direct_register_custom_op(
        op_name="vllm_shared_shrink",
        op_func=_vllm_shared_shrink,
        mutates_args=["out"],
        fake_impl=_fake,
        target_lib=_LIB,
    )


def vllm_shared_shrink(*args, **kwargs) -> None:
    _register()
    torch.ops.scute.vllm_shared_shrink(*args, **kwargs)


#: Rows at and above which the shared-factor term takes the segmented expand.  Below it
#: (decode), the term is dense GEMMs over the batch, reading the factors once per step;
#: above it (prefill), the segments are long enough that the expand's per-segment read
#: of ``B`` is amortised, and ``e`` is applied in its epilogue instead of through a
#: gathered ``[N, d_out]`` tensor.
SEGMENTED_MIN_ROWS = int(__import__("os").environ.get("SCUTE_SHARED_SEG_ROWS", "256"))


def _shared_term(
    y: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    b_list: list[torch.Tensor],
    bx_list: list[torch.Tensor],
    d_pool: torch.Tensor,
    e_pool: torch.Tensor,
    slot_mask: torch.Tensor,
    idx: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
) -> None:
    """``y += ((x A^T) * d) B^T * e`` per slice, for rows whose adapter has the term.

    The choice between the dense and the segmented form is made here, at run time, on
    the row count: it is inside an opaque op, so a compiled graph neither specialises on
    it nor guards on it, and each captured CUDA graph records the form of its size.
    """
    if no_lora_flag_cpu.item():
        return
    v = x @ a.t()
    if x.shape[0] < SEGMENTED_MIN_ROWS:
        d = d_pool[idx]
        e = e_pool[idx]
        off = 0
        for i, b in enumerate(b_list):
            out = b.shape[0]
            y[:, off: off + out] += ((v * d[:, i]) @ b.t()) * e[:, off: off + out]
            off += out
        return
    from scute.ops.vllm_expand_s import _lora_expand_s
    buf = (v.unsqueeze(0).float() * d_pool[idx].transpose(0, 1).float()).contiguous()
    _lora_expand_s(buf, bx_list, y, None, token_lora_mapping, token_indices_sorted_by_lora_ids,
                   num_tokens_per_lora, lora_token_start_loc, lora_ids, no_lora_flag_cpu,
                   num_active_loras, offset_start=0, add_inputs=True, e_pool=e_pool,
                   slot_mask=slot_mask)


def _shared_term_fake(y, x, a, b_list, bx_list, d_pool, e_pool, slot_mask, idx, *meta) -> None:
    return None


_LIB2 = None


def _register_term() -> None:
    global _LIB2
    if hasattr(getattr(torch.ops, "scute", None), "shared_term"):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    _LIB2 = torch.library.Library("scute", "FRAGMENT")
    direct_register_custom_op(
        op_name="shared_term",
        op_func=_shared_term,
        mutates_args=["y"],
        fake_impl=_shared_term_fake,
        target_lib=_LIB2,
    )


def shared_term_op(*args) -> None:
    _register_term()
    torch.ops.scute.shared_term(*args)


@triton.jit
def _shared_expand_kernel(
    v_ptr, b_ptr, e_ptr, y_ptr, idx_ptr, mask_ptr,
    N, OUT, R, n_rows_pool,
    stride_v, stride_b, stride_e, stride_y,
    col0,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """``y[t, col0 + j] += sum_k v[t, k] B_s[j, k] * e[slot(t), col0 + j]``, ``v`` already
    scaled by ``d``.

    A dense GEMM over row tiles: each program reads its tile of ``B_s`` once for all its
    rows, whatever their adapters, so ``B`` is read once per row tile (once per step at
    decode).  ``d`` and ``e`` are gathered per row; rows whose adapter has no such term
    (``mask == 0``) are neither loaded from nor stored to ``y``.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_ok = rows < N
    c_ok = cols < OUT
    slot = tl.load(idx_ptr + rows, mask=r_ok, other=-1)
    slot = tl.where(slot < 0, slot + n_rows_pool, slot)
    live = tl.load(mask_ptr + slot, mask=r_ok, other=0) != 0
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, R, BLOCK_K):
        kk = k0 + ks
        k_ok = kk < R
        # A per-row gather inside the K loop (all rows naming one adapter) ran 4x slower
        # than the same work spread over 16 adapters; d is applied before the kernel.
        vt = tl.load(v_ptr + rows[:, None] * stride_v + kk[None, :],
                     mask=r_ok[:, None] & k_ok[None, :], other=0.0)
        bt = tl.load(b_ptr + cols[None, :] * stride_b + kk[:, None],
                     mask=c_ok[None, :] & k_ok[:, None], other=0.0)
        acc += tl.dot(vt, bt, out_dtype=tl.float32)
    et = tl.load(e_ptr + slot[:, None] * stride_e + col0 + cols[None, :],
                 mask=(r_ok & live)[:, None] & c_ok[None, :], other=0.0).to(tl.float32)
    y_ptrs = y_ptr + rows[:, None] * stride_y + col0 + cols[None, :]
    st_mask = (r_ok & live)[:, None] & c_ok[None, :]
    yt = tl.load(y_ptrs, mask=st_mask, other=0.0)
    tl.store(y_ptrs, (yt.to(tl.float32) + acc * et).to(yt.dtype), mask=st_mask)


def _shared_dense(
    y: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    b_list: list[torch.Tensor],
    d_pool: torch.Tensor,
    e_pool: torch.Tensor,
    slot_mask: torch.Tensor,
    idx: torch.Tensor,
) -> None:
    """The shared-factor term in two launches per slice set: one dense GEMM ``x A^T`` (cuBLAS
    reads ``A`` once) and :func:`_shared_expand_kernel` per slice."""
    n = x.shape[0]
    if n == 0:
        return
    v = x @ a.t()
    r = v.shape[1]
    vd = (v.unsqueeze(0) * d_pool[idx].transpose(0, 1)).contiguous()   # [S, N, R]
    col0 = 0
    for s, b in enumerate(b_list):
        out = b.shape[0]
        bm = 16 if n <= 64 else 64
        grid = (triton.cdiv(n, bm), triton.cdiv(out, 64))
        _shared_expand_kernel[grid](
            vd[s], b, e_pool, y, idx, slot_mask, n, out, r, d_pool.shape[0],
            vd.stride(1), b.stride(0), e_pool.stride(0), y.stride(0), col0,
            BLOCK_M=bm, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3)
        col0 += out


def _shared_dense_fake(y, x, a, b_list, d_pool, e_pool, slot_mask, idx, flags) -> None:
    return None


_LIB3 = None


def _register_dense() -> None:
    global _LIB3
    if hasattr(getattr(torch.ops, "scute", None), "shared_dense"):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    _LIB3 = torch.library.Library("scute", "FRAGMENT")
    direct_register_custom_op(op_name="shared_dense", op_func=_shared_dense_multi,
                              mutates_args=["y"], fake_impl=_shared_dense_fake,
                              target_lib=_LIB3)


def shared_dense_op(*args) -> None:
    _register_dense()
    torch.ops.scute.shared_dense(*args)


@triton.jit
def _shared_expand_multi_kernel(
    vd_ptr, b0_ptr, b1_ptr, b2_ptr, e_ptr, y_ptr, idx_ptr, mask_ptr,
    N, R, n_rows_pool, stride_vs, stride_v, stride_b, stride_e, stride_y,
    OUT0: tl.constexpr, OUT1: tl.constexpr, OUT2: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """:func:`_shared_expand_kernel` for every slice of a layer in one launch (axis 2)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    s = tl.program_id(2)
    if s == 0:
        b_ptr = b0_ptr
        out_n = OUT0
        col0 = 0
    elif s == 1:
        b_ptr = b1_ptr
        out_n = OUT1
        col0 = OUT0
    else:
        b_ptr = b2_ptr
        out_n = OUT2
        col0 = OUT0 + OUT1
    if pid_n * BLOCK_N >= out_n:
        return
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_ok = rows < N
    c_ok = cols < out_n
    slot = tl.load(idx_ptr + rows, mask=r_ok, other=-1)
    slot = tl.where(slot < 0, slot + n_rows_pool, slot)
    live = tl.load(mask_ptr + slot, mask=r_ok, other=0) != 0
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    v_base = vd_ptr + s * stride_vs
    for k0 in range(0, R, BLOCK_K):
        kk = k0 + ks
        k_ok = kk < R
        vt = tl.load(v_base + rows[:, None] * stride_v + kk[None, :],
                     mask=r_ok[:, None] & k_ok[None, :], other=0.0)
        bt = tl.load(b_ptr + cols[None, :] * stride_b + kk[:, None],
                     mask=c_ok[None, :] & k_ok[:, None], other=0.0)
        acc += tl.dot(vt, bt, out_dtype=tl.float32)
    st_mask = (r_ok & live)[:, None] & c_ok[None, :]
    et = tl.load(e_ptr + slot[:, None] * stride_e + col0 + cols[None, :], mask=st_mask,
                 other=0.0).to(tl.float32)
    y_ptrs = y_ptr + rows[:, None] * stride_y + col0 + cols[None, :]
    yt = tl.load(y_ptrs, mask=st_mask, other=0.0)
    tl.store(y_ptrs, (yt.to(tl.float32) + acc * et).to(yt.dtype), mask=st_mask)


def _shared_dense_multi(
    y: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    b_list: list[torch.Tensor],
    d_pool: torch.Tensor,
    e_pool: torch.Tensor,
    slot_mask: torch.Tensor,
    idx: torch.Tensor,
    flags: torch.Tensor,
) -> None:
    """:func:`_shared_dense` in four launches per layer, whatever the number of slices:
    ``x A^T``, the gather of ``d``, its product, and one expand over all slices."""
    n = x.shape[0]
    if n == 0:
        return
    if not flags[1].item() and not torch.cuda.is_current_stream_capturing():
        return  # no shared-factor adapter in this step
    assert len(b_list) <= 3
    v = x @ a.t()
    vd = torch.empty((len(b_list),) + tuple(v.shape), dtype=v.dtype, device=v.device)
    torch.mul(v.unsqueeze(0), d_pool[idx].transpose(0, 1), out=vd)
    outs = [b.shape[0] for b in b_list] + [0] * (3 - len(b_list))
    bs = list(b_list) + [b_list[0]] * (3 - len(b_list))
    bm = 16 if n <= 64 else 64
    grid = (triton.cdiv(n, bm), triton.cdiv(max(outs), 64), len(b_list))
    _shared_expand_multi_kernel[grid](
        vd, bs[0], bs[1], bs[2], e_pool, y, idx, slot_mask, n, v.shape[1], d_pool.shape[0],
        vd.stride(0), vd.stride(1), b_list[0].stride(0), e_pool.stride(0), y.stride(0),
        OUT0=outs[0], OUT1=outs[1], OUT2=outs[2], BLOCK_M=bm, BLOCK_N=64, BLOCK_K=64,
        num_warps=4, num_stages=3)
