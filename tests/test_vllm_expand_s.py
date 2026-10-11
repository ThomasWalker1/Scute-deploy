"""vLLM's ``lora_expand`` with the ``s`` slot, against vLLM's own and against the formula.

Requires the vLLM environment::

    python -m pytest tests/test_vllm_expand_s.py
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("vllm.lora.ops.triton_ops", reason="needs the vLLM environment")

from vllm.lora.ops.triton_ops import lora_expand  # noqa: E402
from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta  # noqa: E402

from scute.ops.vllm_expand_s import lora_expand_s  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

MAX_LORAS, RANK = 4, 16


def _case(n, widths, base_rows, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bs = [torch.randn(MAX_LORAS, w, RANK, device="cuda", dtype=torch.float16,
                      generator=gen) * 0.1 for w in widths]
    buf = torch.randn(len(widths), n, RANK, device="cuda", dtype=torch.float32, generator=gen)
    h = torch.randn(n, sum(widths), device="cuda", dtype=torch.float16, generator=gen)
    pool = torch.ones(MAX_LORAS + 1, sum(widths), device="cuda", dtype=torch.float16)
    pool[:MAX_LORAS] = 1 + 0.3 * torch.randn(MAX_LORAS, sum(widths), device="cuda",
                                             dtype=torch.float16, generator=gen)
    mapping = torch.arange(n, device="cuda", dtype=torch.int32) % MAX_LORAS
    if base_rows:
        mapping[::3] = -1
    meta = LoRAKernelMeta.make(max_loras=MAX_LORAS, max_num_tokens=max(n, 8), device="cuda")
    meta.prepare_tensors(mapping)
    return bs, buf, h, pool, mapping, meta.meta_args(n, specialize_active_lora=False)


SHAPES = [(4096,), (4096, 1024, 1024), (14336, 14336), (40, 24)]


@pytest.mark.parametrize("widths", SHAPES)
@pytest.mark.parametrize("n", [1, 8, 300])
@pytest.mark.parametrize("base_rows", [False, True])
def test_without_s_is_the_shipped_kernel(widths, n, base_rows):
    bs, buf, h, pool, mapping, meta = _case(n, widths, base_rows)
    ref, got = h.clone(), h.clone()
    lora_expand(buf, bs, ref, *meta, offset_start=0, add_inputs=True)
    lora_expand_s(buf, bs, got, None, *meta, offset_start=0, add_inputs=True)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("widths", SHAPES)
@pytest.mark.parametrize("n", [1, 8, 300])
@pytest.mark.parametrize("base_rows", [False, True])
def test_with_s_is_h_times_s_plus_the_term(widths, n, base_rows):
    bs, buf, h, pool, mapping, meta = _case(n, widths, base_rows, seed=1)
    got = h.clone()
    lora_expand_s(buf, bs, got, pool, *meta, offset_start=0, add_inputs=True)

    want = h.float().clone()
    live = mapping >= 0
    rows = mapping[live].long()
    term = torch.cat([torch.einsum("nr,ndr->nd", buf[i][live].half().float(),
                                   b[rows].float()) for i, b in enumerate(bs)], dim=1)
    want[live] = h[live].float() * pool[rows].float() + term
    torch.testing.assert_close(got.float(), want, rtol=2e-3, atol=2e-3)
    # Rows with no adapter are not visited: they keep h exactly.
    torch.testing.assert_close(got[~live], h[~live], rtol=0, atol=0)


def test_with_s_is_one_launch():
    bs, buf, h, pool, mapping, meta = _case(256, (4096, 1024, 1024), True)
    y = h.clone()
    lora_expand_s(buf, bs, y, pool, *meta, offset_start=0, add_inputs=True)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        lora_expand_s(buf, bs, y, pool, *meta, offset_start=0, add_inputs=True)
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert len(kernels) == 1, [e.name for e in kernels]
