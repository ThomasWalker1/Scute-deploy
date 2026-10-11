"""OFT's load-time fold, and a float64 reference for the input transform.

:func:`cayley` turns the skew parameters of a PEFT OFT checkpoint into explicit
block-diagonal rotation blocks ``R``.  The engine applies them with
:mod:`scute.ops.vllm_pretransform`.
"""

from __future__ import annotations

import torch

__all__ = ["cayley", "reference_pretransform"]


def reference_pretransform(x: torch.Tensor, r_pool: torch.Tensor,
                           mapping: torch.Tensor) -> torch.Tensor:
    """float64 reference: ``x' = x blockdiag(R[mapping])``, ``-1`` rows unchanged."""
    out = x.double().clone()
    _, n_blocks, b, _ = r_pool.shape
    for i, slot in enumerate(mapping.tolist()):
        if slot < 0:
            continue
        xb = x[i].double().view(n_blocks, b)
        out[i] = torch.einsum("jk,jkc->jc", xb, r_pool[slot].double()).reshape(-1)
    return out


def cayley(skew_params: torch.Tensor, block: int, use_neumann: bool = True,
           neumann_terms: int = 5) -> torch.Tensor:
    """The load-time fold: PEFT's packed skew parameters to explicit ``R`` blocks.

    Mirrors ``OFTRotationModule._cayley_batch`` in PEFT 0.20 term for term, so the
    served ``R`` is the one PEFT trains and evaluates with.  Each row of
    ``skew_params`` is a block's upper triangle (``torch.triu_indices(b, b, 1)``
    order).  The exact form is ``(I - Q)(I + Q)^{-1}``.  PEFT's default is the
    truncated Neumann series ``I + 2Q + 2Q^2 + ... + Q^{n-1}``, which is what a
    checkpoint trained with the default was trained under.  Run once per adapter at
    admission.
    """
    n = skew_params.shape[0]
    dtype = skew_params.dtype
    q = torch.zeros(n, block, block, dtype=dtype, device=skew_params.device)
    rows, cols = torch.triu_indices(block, block, 1)
    q[:, rows, cols] = skew_params
    q = q - q.transpose(-2, -1)
    eye = torch.eye(block, dtype=dtype, device=skew_params.device).repeat(n, 1, 1)
    if not use_neumann:
        return torch.linalg.solve(eye + q, eye - q, left=False)
    r = eye.clone()
    if neumann_terms > 1:
        r.add_(q, alpha=2.0)
        if neumann_terms > 2:
            q2 = torch.bmm(q, q)
            r.add_(q2, alpha=2.0)
            power = q2
            for _ in range(3, neumann_terms - 1):
                power = torch.bmm(power, q)
                r.add_(power, alpha=2.0)
            power = torch.bmm(power, q)
            r.add_(power)
    return r
