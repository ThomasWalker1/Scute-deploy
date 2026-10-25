"""The normal form's slots and the fold rule, as a decision procedure.

Every supported method is one operator at each adapted projection,

.. code::

    h' = ((x R) W) * s + b + sum_k c_k * ( ( (x A_k) * d_k ) B_k * e_k )

where ``*`` multiplies element by element, and each slot (``R``, ``s``, ``b``,
``d_k``, ``e_k``, ``c_k``) is the identity when a method does not use it.

A slot *folds*, that is, it merges into the adapter's own weights once at load,
exactly when the adapter owns a tensor next to it in the product that applies it.
``s``, ``b`` and ``R`` sit next to the shared ``W`` only, so they never fold.  A slot
that does not fold is *live*, and the engine computes it for every token.

:func:`plan_slots` applies this rule to a method given as an :class:`AdapterSpec`,
and :func:`live_slot_bytes` counts the bytes that each live slot adds to a pass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TermSpec:
    """One term of a specification: which slots are non-identity, who owns the factors."""

    rank: int = 16
    d: bool = False
    e: bool = False
    c: bool = False
    a_owned: bool = True
    b_owned: bool = True


@dataclass(frozen=True)
class AdapterSpec:
    """An adapter method as slots of the extended form, with factor ownership.

    ``r`` is an input-side pre-transform (OFT's rotation) with blocks of ``r_block``.
    ``output_dense`` marks a per-adapter dense operator on the output side of the
    shared GEMM, which no member of the family can express.
    """

    name: str
    s: bool = False
    b: bool = False
    r: bool = False
    r_block: int = 0
    terms: tuple[TermSpec, ...] = ()
    output_dense: bool = False


@dataclass(frozen=True)
class SlotPlan:
    """What :func:`plan_slots` decides for one specification."""

    in_family: bool
    folds: tuple[str, ...]
    live: tuple[str, ...]


def plan_slots(spec: AdapterSpec) -> SlotPlan:
    """Fold every slot that the fold rule allows.  The rest stay live."""
    if spec.output_dense:
        return SlotPlan(in_family=False, folds=(), live=())
    folds: list[str] = []
    live: list[str] = []
    if spec.r:
        live.append("R")
    if spec.s:
        live.append("s")
    if spec.b:
        live.append("b")
    for k, t in enumerate(spec.terms):
        if t.e:
            if t.b_owned:
                folds.append(f"e{k} -> B{k}")
            else:
                live.append(f"e{k}")
        if t.d:
            if t.a_owned:
                folds.append(f"d{k} -> A{k}")
            elif t.b_owned:
                folds.append(f"d{k} -> B{k}")
            else:
                live.append(f"d{k}")
        if t.c:
            # Any owned tensor of the term absorbs a scalar; a live d or e is owned.
            for owned, name in ((t.a_owned, "A"), (t.b_owned, "B"),
                                (t.d, "d"), (t.e, "e")):
                if owned:
                    folds.append(f"c{k} -> {name}{k}")
                    break
            else:
                live.append(f"c{k}")
    return SlotPlan(in_family=True, folds=tuple(folds), live=tuple(live))


def compose_specs(inner: AdapterSpec, outer: AdapterSpec, share_factors: bool = True
                  ) -> AdapterSpec:
    """``outer ∘ inner`` at one hook: stacking two adapters gives another member.

    The inner member's terms acquire an outer diagonal when the outer member scales
    the output.  ``share_factors`` models a stack that reuses the inner adapter's ``B``
    rather than copying it; ``B`` is then shared, and the new outer diagonal stays
    live.
    """
    if inner.r or outer.r:
        raise ValueError("closure is stated for R = I; compose pre-transforms separately")
    carried = tuple(
        TermSpec(rank=t.rank, d=t.d, e=t.e or outer.s, c=t.c, a_owned=t.a_owned,
                 b_owned=t.b_owned and not (share_factors and outer.s))
        for t in inner.terms
    )
    return AdapterSpec(
        name=f"{outer.name}∘{inner.name}",
        s=inner.s or outer.s,
        b=inner.b or outer.b,
        terms=carried + outer.terms,
        output_dense=inner.output_dense or outer.output_dense,
    )


def live_slot_bytes(plan: SlotPlan, spec: AdapterSpec, n: int, d_in: int, d_out: int,
                    n_adapters: int, itemsize: int = 2) -> dict[str, int]:
    """Extra bytes per pass for each live slot, carried and applied outside.

    ``carried`` is the slot read inside
    a pass that already touches the tensor it scales; ``outside`` is a pass of its
    own; ``outside_materialised`` also writes and re-reads a per-row copy of the slot.
    """
    carried = outside = materialised = 0
    for slot in plan.live:
        if slot in ("s", "b") or slot.startswith("e"):
            carried += n_adapters * d_out * itemsize
            outside += 2 * n * d_out * itemsize + n_adapters * d_out * itemsize
            materialised += 4 * n * d_out * itemsize
        elif slot.startswith("d"):
            rank = spec.terms[int(slot[1:])].rank
            carried += n_adapters * rank * itemsize
            outside += 2 * n * rank * 4 + n_adapters * rank * itemsize
            materialised += 4 * n * rank * 4
        elif slot == "R":
            # No pass of the base GEMM can carry R, so it is always a pass of its own.
            pass_bytes = 2 * n * d_in * itemsize + n_adapters * d_in * spec.r_block * itemsize
            carried += pass_bytes
            outside += pass_bytes
            materialised += pass_bytes
    return dict(carried=carried, outside=outside, outside_materialised=materialised)


#: The methods, as specifications.
METHOD_SPECS: dict[str, AdapterSpec] = {
    "lora": AdapterSpec("lora", terms=(TermSpec(c=True),)),
    "dora": AdapterSpec("dora", s=True, terms=(TermSpec(e=True, c=True),)),
    "ia3": AdapterSpec("ia3", s=True),
    "ssf": AdapterSpec("ssf", s=True, b=True),
    "vera": AdapterSpec("vera", terms=(TermSpec(d=True, e=True, c=True,
                                                a_owned=False, b_owned=False),)),
    "oft": AdapterSpec("oft", r=True, r_block=32),
    "full_finetune": AdapterSpec("full_finetune", output_dense=True),
}
