"""End to end: adapters of all five methods in one vLLM batch, against the PEFT library.

Starts a vLLM engine on SmolLM2-360M (about a minute with downloads cached).  Run with::

    python -m pytest tests/test_engine.py -m engine
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

#: The largest error as a share of an adapter's effect.  float16 rounding gives about
#: 1-2%; an adapter whose slot is dropped or applied twice gives 50% or more.
MAX_SHARE = 0.05


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    pytest.importorskip("peft")
    from check_against_peft import PROMPTS, compare, engine_logprobs, peft_logprobs
    from make_adapters import BASE, make_adapters
    from serve_mixed_batch import compile_all

    root = tmp_path_factory.mktemp("adapters")
    entries = make_adapters(str(root / "peft"))
    compiled = compile_all(str(root / "peft"), str(root / "compiled"), BASE)
    engine = engine_logprobs(compiled, BASE, PROMPTS)
    peft = peft_logprobs([(e["name"], e["path"]) for e in entries], BASE, PROMPTS)
    out = compare(engine, peft, [n for n, _ in compiled], len(PROMPTS))
    print(json.dumps(out, indent=1))
    return {r["adapter"]: r for r in out}


@pytest.mark.parametrize("method", ["lora", "dora", "ia3", "vera", "oft"])
def test_every_method_matches_peft_in_one_batch(rows, method):
    for seed in (0, 1):
        r = rows[f"{method}-{seed}"]
        # The adapter changes the model, so the check can tell a correct engine from
        # one that drops the method.
        assert r["effect"] > 10 * rows["base"]["error"], r
        assert r["share"] < MAX_SHARE, r


def test_base_requests_are_unchanged(rows):
    # Requests without an adapter share the batch; they see only float16 rounding.
    assert rows["base"]["error"] < 0.02, rows["base"]
