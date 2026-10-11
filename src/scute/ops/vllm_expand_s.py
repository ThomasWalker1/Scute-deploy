# SPDX-License-Identifier: Apache-2.0
# The kernel and host code below are vLLM 0.19.1's ``lora_expand``
# (vllm/lora/ops/triton_ops/lora_expand_op.py and kernel_utils.do_expand_kernel),
# Copyright contributors to the vLLM project, with the DoRA ``s`` slot added.
"""vLLM's shipped ``lora_expand``, carrying the ``s`` slot.  Needs the vLLM environment.

The expand already reads the base output ``h`` when it adds the term
(``ADD_INPUTS``): its epilogue is ``tiled_c += tiled_out``.  Carrying ``s`` changes
that line to ``tiled_c += tiled_out * s``, in the same dtype, so the epilogue holds
no more live registers than the shipped one, with ``s`` gathered once per tile from a
``[max_loras + 1, total_out]`` pool, the row index being the slot the tile already
uses for ``B``.  Nothing else differs from the shipped kernel: the grid, the tiling,
the configs from ``get_lora_op_configs``, the pointer tables, and the metadata.  So
the cost this kernel adds over vLLM's is the cost of the slot and of nothing else.

With ``USE_S=False`` the kernel is the shipped one, line for line.
``tests/test_vllm_expand_s.py`` checks that, and checks ``s`` against the reference
expand.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from vllm.lora.ops.triton_ops.kernel_utils import mm_k

__all__ = ["lora_expand_s"]


@triton.jit
def _do_expand_s(
    pid_n,
    lora_index,
    slice_id,
    input_ptr,
    lora_ptr,
    out_ptr,
    s_ptr,
    s_d0_stride,
    e_ptr,
    e_d0_stride,
    N,
    K,
    M_LEN,
    ram,
    slice_start_loc,
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,
    output_d0_stride,
    output_d1_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    EVEN_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    USE_S: tl.constexpr,
    USE_E: tl.constexpr,
):
    if SAME_STRIDE:
        cur_lora_d0_stride = ls_d0_ptr
        cur_lora_d1_stride = ls_d1_ptr
        cur_lora_d2_stride = ls_d2_ptr
    else:
        cur_lora_d0_stride = tl.load(ls_d0_ptr + slice_id)
        cur_lora_d1_stride = tl.load(ls_d1_ptr + slice_id)
        cur_lora_d2_stride = tl.load(ls_d2_ptr + slice_id)

    if SLICE_NUM == 1:
        cur_input_ptr = input_ptr
        cur_lora_ptr = lora_ptr
    else:
        cur_input_ptr = input_ptr + slice_id * input_d0_stride
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(out_ptr.dtype.element_ty)
        )

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)
    a_ptr = (
        cur_input_ptr
        + ram[:, None] * input_d1_stride
        + offset_k[None, :] * input_d2_stride
    )
    b_ptr = (
        cur_lora_ptr
        + cur_lora_d0_stride * lora_index
        + offset_k[:, None] * cur_lora_d2_stride
        + rbn[None, :] * cur_lora_d1_stride
    )
    accumulator = mm_k(
        a_ptr, b_ptr, input_d2_stride, cur_lora_d2_stride, offset_k, K,
        BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, 1, CAST_TYPE,
        cur_lora_ptr.dtype.element_ty, False, base_k=0,
    )

    tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)
    if SLICE_NUM == 1:
        cur_slice_start = slice_start_loc
    else:
        cur_slice_start = tl.load(slice_start_loc + slice_id)

    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    offset_cm = tl.arange(0, BLOCK_M)
    c_ptr = (
        out_ptr
        + ram[:, None] * output_d0_stride
        + offset_cn[None, :] * output_d1_stride
    )
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < (cur_slice_start + N))

    if USE_E:
        # The outer diagonal of a term whose B is shared (VeRA): it scales the term,
        # and cannot fold into B because B belongs to every adapter.
        e_mask = offset_cn < (cur_slice_start + N)
        e_tile = tl.load(e_ptr + lora_index * e_d0_stride + offset_cn, mask=e_mask,
                         other=1.0)
        tiled_c = tiled_c * e_tile[None, :]
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        if USE_S:
            # The slot: one gather of s per tile, one multiply on the h already loaded.
            # Done in the output dtype, as the shipped add is: an fp32 epilogue keeps
            # the accumulator live across the load and spills under tuned configs.
            n_mask = offset_cn < (cur_slice_start + N)
            s_tile = tl.load(s_ptr + lora_index * s_d0_stride + offset_cn, mask=n_mask,
                             other=1.0)
            tiled_c += tiled_out * s_tile[None, :]
        else:
            tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


@triton.jit
def _lora_expand_s_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    s_ptr,
    s_d0_stride,
    e_ptr,
    e_d0_stride,
    slot_mask_ptr,
    M,
    N,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    slice_start_loc,
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,
    output_d0_stride,
    output_d1_stride,
    output_hs_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    USE_S: tl.constexpr,
    USE_E: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)
    pid_mn = tl.program_id(axis=0)
    pid_m = pid_mn % cta_m_num
    pid_n = (pid_mn // cta_m_num) % cta_n_num
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return
    if USE_MASK:
        # Only the slots whose adapter has this term (a shared-factor pass skips the rest).
        if tl.load(slot_mask_ptr + lora_id) == 0:
            return
    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return
    curr_N = N if SAME_STRIDE else tl.load(output_hs_ptr + slice_id)
    if pid_n * BLOCK_N >= curr_N:
        return
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (
        token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    )
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    _do_expand_s(
        pid_n, lora_id, slice_id, input_ptr, lora_ptr, out_ptr, s_ptr, s_d0_stride,
        e_ptr, e_d0_stride, curr_N, K, cta_m_len, ram, slice_start_loc,
        input_d0_stride, input_d1_stride, input_d2_stride,
        ls_d0_ptr, ls_d1_ptr, ls_d2_ptr,
        output_d0_stride, output_d1_stride,
        BLOCK_M, BLOCK_N, BLOCK_K, SAME_STRIDE, SLICE_NUM, EVEN_K, CAST_TYPE,
        ADD_INPUTS, USE_S, USE_E,
    )


_B_PTR: dict = {}


def _b_ptr_table(lora_weights: list[torch.Tensor], offset_start: int, device):
    """vLLM's ``_get_lora_b_ptr``, allowing a slot stride of 0.

    A shared ``B`` (VeRA) is one tensor read by every slot, passed as a view expanded
    over the slot dimension.  vLLM's builder asserts contiguity, which such a view is not;
    everything else here is its logic.  The cache key includes sizes and strides,
    because shared views of different layers start at the same address.
    """
    key = tuple((w.data_ptr(), tuple(w.shape), tuple(w.stride())) for w in lora_weights)
    if key in _B_PTR:
        return _B_PTR[key]
    starts, ptrs, d0, d1, d2, sizes = [], [], [], [], [], []
    off = offset_start
    for w in lora_weights:
        if w.ndim == 4:
            assert w.size(1) == 1
            w = w.squeeze(1)
        assert w.ndim == 3 and w.stride(2) == 1
        ptrs.append(w.data_ptr())
        d0.append(w.stride(0))
        d1.append(w.stride(1))
        d2.append(w.stride(2))
        starts.append(off)
        off += w.size(1)
        sizes.append(w.size(1))
    if len(lora_weights) > 1:
        ptr_t = torch.tensor(ptrs, device=device, dtype=torch.uint64)
        start_t = torch.tensor(starts, device=device, dtype=torch.uint64)
    else:
        start_t, ptr_t = starts[0], w[0]
    if len(set(d0)) == 1 and len(set(d1)) == 1 and len(set(d2)) == 1 and len(set(sizes)) == 1:
        vals = (start_t, ptr_t, d0[0], d1[0], d2[0], sizes[0], True, max(sizes))
    else:
        vals = (start_t, ptr_t, torch.tensor(d0, device=device), torch.tensor(d1, device=device),
                torch.tensor(d2, device=device), torch.tensor(sizes, device=device), False,
                max(sizes))
    _B_PTR[key] = vals
    return vals


@torch.inference_mode()
def _lora_expand_s(
    inputs: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    s_pool: Optional[torch.Tensor],
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
    offset_start: int = 0,
    add_inputs: bool = False,
    e_pool: Optional[torch.Tensor] = None,
    slot_mask: Optional[torch.Tensor] = None,
) -> None:
    """vLLM's ``_lora_expand`` with the ``s`` pool (or ``None``), and optionally the outer
    diagonal ``e`` and a per-slot mask of the slots to visit."""
    from vllm.lora.ops.triton_ops.utils import _get_lora_b_ptr, get_lora_op_configs

    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        return
    assert inputs.size(0) == len(lora_b_weights)
    assert output_tensor.is_contiguous()
    use_s = s_pool is not None
    if use_s:
        # s multiplies the h the kernel loads, so it needs that load.
        assert add_inputs, "the s slot scales h, which only an ADD_INPUTS expand reads"
        assert s_pool.stride(1) == 1

    M = inputs.size(1)
    (
        slice_start_tensor,
        lora_ptr_tensor,
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        hidden_sizes_tensor,
        same_stride,
        MAX_N,
    ) = (_b_ptr_table(lora_b_weights, offset_start, inputs.device) if slot_mask is not None
         else _get_lora_b_ptr(lora_b_weights, offset_start, inputs.device))

    K = lora_b_weights[0].shape[-1]
    MAX_LORAS = lora_ids.size(0)
    NUM_SLICES = len(lora_b_weights)
    kernel_config = get_lora_op_configs(
        op_type="expand", max_loras=MAX_LORAS, batch=M, hidden_size=MAX_N, rank=K,
        num_slices=NUM_SLICES, add_inputs=add_inputs,
    )
    BLOCK_M = kernel_config["block_m"]
    BLOCK_N = kernel_config["block_n"]
    BLOCK_K = kernel_config["block_k"]
    EVEN_K = K % BLOCK_K == 0
    CAST_TYPE = inputs.dtype == torch.float32 and lora_b_weights[0].dtype in (
        torch.float16, torch.bfloat16)

    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(MAX_N, BLOCK_N),
        NUM_SLICES,
        num_active_loras.item(),
    )
    _lora_expand_s_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        s_pool if use_s else output_tensor,
        s_pool.stride(0) if use_s else 0,
        e_pool if e_pool is not None else output_tensor,
        e_pool.stride(0) if e_pool is not None else 0,
        slot_mask if slot_mask is not None else lora_ids,
        M,
        MAX_N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        slice_start_tensor,
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        hidden_sizes_tensor,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        add_inputs,
        CAST_TYPE,
        NUM_SLICES,
        same_stride,
        use_s,
        e_pool is not None,
        slot_mask is not None,
        num_warps=kernel_config["num_warps"],
        num_ctas=kernel_config["num_ctas"],
        num_stages=kernel_config["num_stages"],
    )


def _lora_expand_s_fake(
    inputs: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    s_pool: Optional[torch.Tensor],
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
    offset_start: int = 0,
    add_inputs: bool = False,
    e_pool: Optional[torch.Tensor] = None,
    slot_mask: Optional[torch.Tensor] = None,
) -> None:
    return None


_LIB = None


def _register() -> None:
    """Register ``scute::lora_expand_s`` the way vLLM registers ``vllm::lora_expand``."""
    global _LIB
    if hasattr(getattr(torch.ops, "scute", None), "lora_expand_s"):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    _LIB = torch.library.Library("scute", "FRAGMENT")
    direct_register_custom_op(
        op_name="lora_expand_s",
        op_func=_lora_expand_s,
        mutates_args=["output_tensor"],
        fake_impl=_lora_expand_s_fake,
        target_lib=_LIB,
    )


def lora_expand_s(*args, **kwargs) -> None:
    """The registered op: ``lora_expand`` with an ``s`` pool after the output tensor."""
    _register()
    torch.ops.scute.lora_expand_s(*args, **kwargs)
