"""The shared-factor term of VeRA, ``y += ((x A^T) ⊙ d) B^T ⊙ e``, as GEMMs over row tiles.

``A`` and ``B`` are the same for every adapter, so ``x A^T`` is one cuBLAS GEMM over the
batch, and one Triton expand covers every slice of the layer.  The expand is dense over
row tiles: each program reads its tile of ``B`` once for all its rows, whatever their
adapters, so the factors are read once per step however many adapters use them.
``d`` and ``e`` are gathered per row, and rows whose adapter has no shared-factor term
(``slot_mask == 0``) are neither loaded from nor stored to ``y``.  Needs the vLLM
environment.
"""

import torch
import triton
import triton.language as tl

__all__ = ["shared_dense_op"]


@triton.jit
def _shared_expand_multi_kernel(
    vd_ptr, b0_ptr, b1_ptr, b2_ptr, e_ptr, y_ptr, idx_ptr, mask_ptr,
    N, R, n_rows_pool, stride_vs, stride_v, stride_b, stride_e, stride_y,
    OUT0: tl.constexpr, OUT1: tl.constexpr, OUT2: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """``y[t, col0_s + j] += sum_k vd[s, t, k] B_s[j, k] * e[slot(t), col0_s + j]`` for
    every slice ``s`` of a layer in one launch (axis 2).  ``vd`` is already scaled by
    ``d``: a per-row gather of ``d`` inside the K loop is slow when many rows name the
    same adapter."""
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
    """The shared-factor term in four launches per layer, whatever the number of
    slices: ``x A^T``, the gather of ``d``, its product, and one expand over all slices."""
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


def _shared_dense_fake(y, x, a, b_list, d_pool, e_pool, slot_mask, idx, flags) -> None:
    return None


_LIB = None


def _register_dense() -> None:
    global _LIB
    if hasattr(getattr(torch.ops, "scute", None), "shared_dense"):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    _LIB = torch.library.Library("scute", "FRAGMENT")
    direct_register_custom_op(op_name="shared_dense", op_func=_shared_dense_multi,
                              mutates_args=["y"], fake_impl=_shared_dense_fake,
                              target_lib=_LIB)


def shared_dense_op(*args) -> None:
    _register_dense()
    torch.ops.scute.shared_dense(*args)
