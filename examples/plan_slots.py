"""Show which slots of each method fold at load and which stay live, and what they cost.

A slot folds when the adapter owns a tensor next to it; a live slot is computed for
every token.  For each live slot, the script gives the bytes that it adds to one pass
over a batch, when the expand carries it and when it runs in a pass of its own.

Usage::

    python examples/plan_slots.py --tokens 256 --adapters 8
"""

from __future__ import annotations

import argparse

from scute.slots import METHOD_SPECS, AdapterSpec, TermSpec, live_slot_bytes, plan_slots


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=256, help="tokens in the batch")
    ap.add_argument("--adapters", type=int, default=8, help="adapters in the batch")
    ap.add_argument("--d", type=int, default=4096, help="d_in = d_out of the projection")
    a = ap.parse_args()

    print(f"{'method':14s} {'folds at load':24s} {'live':10s} {'carried':>10s} {'own pass':>10s}")
    for name, spec in METHOD_SPECS.items():
        plan = plan_slots(spec)
        if not plan.in_family:
            print(f"{name:14s} not expressible: a dense per-adapter operator on the output")
            continue
        b = live_slot_bytes(plan, spec, a.tokens, a.d, a.d, a.adapters)
        print(f"{name:14s} {', '.join(plan.folds) or '-':24s} {', '.join(plan.live) or '-':10s} "
              f"{b['carried'] / 1e3:8.1f}kB {b['outside'] / 1e3:8.1f}kB")

    # A new method: a scale on the output and a LoRA term with an inner diagonal.
    spec = AdapterSpec("my-method", s=True, terms=(TermSpec(d=True, c=True),))
    plan = plan_slots(spec)
    print(f"\n{spec.name}: folds {plan.folds}, live {plan.live}")


if __name__ == "__main__":
    main()
