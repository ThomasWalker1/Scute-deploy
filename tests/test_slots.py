"""The fold rule: a slot folds iff an owned tensor sits next to it.

Sufficiency is checked by applying each fold and comparing the operator.  Necessity is
checked numerically: with the neighbouring tensors shared, the best operator that sets
the slot to identity, fitted by least squares over everything the adapter owns, still
misses.  The cost of folding ``s`` is checked as a rank.
"""

from __future__ import annotations

import pytest
import torch

from scute.slots import METHOD_SPECS, compose_specs, live_slot_bytes, plan_slots

torch.manual_seed(0)
D_IN, D_OUT, R = 96, 80, 6


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.float64)


def _operator(A, d, B, e, c):
    """The term as a matrix: c * A diag(d) B diag(e)."""
    return c * (A * d) @ B * e


# --- sufficiency -----------------------------------------------------------------


def test_each_fold_preserves_the_operator():
    A, B = _rand(D_IN, R), _rand(R, D_OUT)
    d, e, c = 1 + 0.3 * _rand(R), 1 + 0.3 * _rand(D_OUT), 0.7
    ref = _operator(A, d, B, e, c)
    one_r, one_o = torch.ones(R, dtype=torch.float64), torch.ones(D_OUT, dtype=torch.float64)
    # e into B; d into A; d into B; c into A.
    torch.testing.assert_close(_operator(A, d, B * e, one_o, c), ref)
    torch.testing.assert_close(_operator(A * d, one_r, B, e, c), ref)
    torch.testing.assert_close(_operator(A, one_r, d[:, None] * B, e, c), ref)
    torch.testing.assert_close(_operator(c * A, d, B, e, 1.0), ref)
    # VeRA's case: A and B shared, c into the owned d (or e).
    torch.testing.assert_close(_operator(A, c * d, B, e, 1.0), ref)
    torch.testing.assert_close(_operator(A, d, B, c * e, 1.0), ref)


# --- necessity -------------------------------------------------------------------


def _relative_residual(design: torch.Tensor, target: torch.Tensor) -> float:
    """Least-squares fit of ``target`` (flattened) in the column span of ``design``."""
    sol = torch.linalg.lstsq(design, target.reshape(-1, 1)).solution
    fit = (design @ sol).reshape(target.shape)
    return float((fit - target).norm() / target.norm())


def test_e_does_not_fold_when_b_is_shared():
    """VeRA: with A and B shared, no choice of the owned d reproduces a non-trivial e."""
    A, B = _rand(D_IN, R), _rand(R, D_OUT)
    d, e = 1 + 0.3 * _rand(R), 1 + 0.3 * _rand(D_OUT)
    target = _operator(A, d, B, e, 1.0)
    # With e = 1 the operator is linear in d: A diag(d') B = sum_j d'_j A[:, j] B[j, :].
    design = torch.stack([torch.outer(A[:, j], B[j]).reshape(-1) for j in range(R)], dim=1)
    assert _relative_residual(design, target) > 0.1


def test_d_does_not_fold_when_a_and_b_are_shared():
    A, B = _rand(D_IN, R), _rand(R, D_OUT)
    d = 1 + 0.3 * _rand(R)
    target = _operator(A, d, B, torch.ones(D_OUT, dtype=torch.float64), 1.0)
    # With d = 1 the operator is linear in e': (A B) diag(e').
    AB = A @ B
    design = torch.stack([torch.outer(AB[:, n], torch.eye(D_OUT, dtype=torch.float64)[n])
                          .reshape(-1) for n in range(D_OUT)], dim=1)
    assert _relative_residual(design, target) > 0.1


@pytest.mark.parametrize("d_in,d_out", [(4096, 4096), (4096, 1024), (4096, 14336)])
def test_s_folds_only_into_a_full_rank_term(d_in, d_out):
    """W diag(g - 1) has full rank, so folding s needs a rank ~min(d) term."""
    if not torch.cuda.is_available():
        pytest.skip("rank of a Llama-sized matrix wants a GPU")
    g = torch.Generator(device="cuda").manual_seed(0)
    W = torch.randn(d_in, d_out, device="cuda", generator=g) / d_in ** 0.5
    s = 1 + 0.02 * torch.randn(d_out, device="cuda", generator=g)
    rank = int(torch.linalg.matrix_rank((W * (s - 1)).double()))
    assert rank == min(d_in, d_out)
    # A rank-16 term cannot get close: the best rank-16 approximation leaves most of it
    # (87-97% on these shapes; bench/fold_rank.py measures it on the real weights).
    sv = torch.linalg.svdvals((W * (s - 1)).double())
    left = float(sv[16:].pow(2).sum() / sv.pow(2).sum())
    assert left > 0.5


# --- the decision procedure ------------------------------------------------------


@pytest.mark.parametrize(
    "method,live",
    [
        ("lora", ()),
        ("dora", ("s",)),
        ("ia3", ("s",)),
        ("ssf", ("s", "b")),
        ("vera", ("e0", "d0")),
        ("oft", ("R",)),
    ],
)
def test_plan_slots_live_sets(method, live):
    plan = plan_slots(METHOD_SPECS[method])
    assert plan.in_family
    assert set(plan.live) == set(live)


def test_dora_folds_e_into_b_and_keeps_s():
    plan = plan_slots(METHOD_SPECS["dora"])
    assert "e0 -> B0" in plan.folds and plan.live == ("s",)


def test_vera_folds_its_scalar_into_an_owned_diagonal():
    assert "c0 -> d0" in plan_slots(METHOD_SPECS["vera"]).folds


def test_a_dense_output_operator_is_outside_the_family():
    assert not plan_slots(METHOD_SPECS["full_finetune"]).in_family


def test_stacking_makes_e_live_only_when_the_outer_member_scales():
    """Stacking: (IA)^3 after LoRA makes LoRA's e live; LoRA after (IA)^3 does not."""
    ia3_after_lora = plan_slots(compose_specs(METHOD_SPECS["lora"], METHOD_SPECS["ia3"]))
    lora_after_ia3 = plan_slots(compose_specs(METHOD_SPECS["ia3"], METHOD_SPECS["lora"]))
    assert "e0" in ia3_after_lora.live
    assert "e0" not in lora_after_ia3.live
    # Without factor sharing the new e folds into the inner adapter's own B.
    copied = plan_slots(compose_specs(METHOD_SPECS["lora"], METHOD_SPECS["ia3"],
                                      share_factors=False))
    assert "e0" not in copied.live


def test_live_slot_bytes_match_the_byte_counts():
    n, d, k = 4096, 4096, 4
    b = live_slot_bytes(plan_slots(METHOD_SPECS["dora"]), METHOD_SPECS["dora"], n, d, d, k)
    assert b["carried"] == k * d * 2
    assert b["outside"] == 2 * n * d * 2 + k * d * 2
    assert b["outside_materialised"] == 4 * n * d * 2
    assert live_slot_bytes(plan_slots(METHOD_SPECS["lora"]), METHOD_SPECS["lora"],
                           n, d, d, k) == dict(carried=0, outside=0, outside_materialised=0)
