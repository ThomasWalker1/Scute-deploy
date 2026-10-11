"""Check the engine against the PEFT library, for every adapter in one batch.

For each adapter and prompt, the engine scores the prompt's tokens (their
log-probabilities) with all adapters in one batch.  The PEFT library scores the same
tokens in float32, one adapter at a time.  For each adapter the script reports:

* the effect: the mean change in log-probability per token that the adapter makes to
  the base model, as the PEFT library computes it;
* the error: the mean difference between the engine and the PEFT library, per token;
* the error as a share of the effect.

The engine runs in float16, so a share of a few percent is the floating-point error
of float16, the same as for a plain LoRA adapter.

Usage::

    python examples/check_against_peft.py --adapters adapters/peft
"""

from __future__ import annotations

import argparse
import json

from make_adapters import BASE
from serve_mixed_batch import compile_all

PROMPTS = [
    "The capital of France is Paris, and the capital of Germany is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Photosynthesis converts light energy into chemical energy. The main product is",
    "Q: If a train travels 60 miles in 1.5 hours, what is its average speed?\nA:",
]


def engine_logprobs(adapters: list[tuple[str, str]], base: str, prompts: list[str]) -> dict:
    """{(adapter, prompt index): [log p of each prompt token after the first]}."""
    import scute

    scute.enable([d for _, d in adapters])
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=base, enable_lora=True, max_loras=len(adapters), max_lora_rank=16,
              dtype="float16", max_model_len=1024, gpu_memory_utilization=0.4)
    reqs = [None] + [LoRARequest(n, i + 1, d) for i, (n, d) in enumerate(adapters)]
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    batch, keys = [], []
    for r in reqs:
        for j, p in enumerate(prompts):
            batch.append((p, r))
            keys.append((r.lora_name if r else "base", j))
    # One call: every adapter and the base model in the same batch.
    outs = llm.generate([p for p, _ in batch], sp, lora_request=[r for _, r in batch])
    res = {}
    for k, o in zip(keys, outs):
        res[k] = [pos[t].logprob for t, pos in zip(o.prompt_token_ids[1:], o.prompt_logprobs[1:])]
    del llm
    return res


def peft_logprobs(peft_dirs: list[tuple[str, str]], base: str, prompts: list[str]) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).cuda().eval()

    @torch.no_grad()
    def score(m, name):
        for j, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids.cuda()
            logp = torch.log_softmax(m(ids).logits.float()[0, :-1], dim=-1)
            res[(name, j)] = logp.gather(1, ids[0, 1:, None])[:, 0].tolist()

    res: dict = {}
    score(model, "base")
    for name, path in peft_dirs:
        m = PeftModel.from_pretrained(model, path).eval()
        score(m, name)
        model = m.unload()
    return res


def compare(engine: dict, peft: dict, names: list[str], n_prompts: int) -> list[dict]:
    rows = []
    for name in ["base"] + names:
        err = eff = n = 0.0
        for j in range(n_prompts):
            e, p, b = engine[(name, j)], peft[(name, j)], peft[("base", j)]
            assert len(e) == len(p), (name, j, len(e), len(p))
            err += sum(abs(x - y) for x, y in zip(e, p))
            eff += sum(abs(x - y) for x, y in zip(p, b))
            n += len(p)
        rows.append(dict(adapter=name, effect=eff / n, error=err / n,
                         share=(err / eff) if eff else None))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapters", default="adapters/peft")
    ap.add_argument("--compiled", default="adapters/compiled")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--json", help="also write the rows to this file")
    a = ap.parse_args()

    import os
    compiled = compile_all(a.adapters, a.compiled, a.base)
    peft_dirs = [(n, os.path.join(a.adapters, n)) for n, _ in compiled]
    engine = engine_logprobs(compiled, a.base, PROMPTS)
    peft = peft_logprobs(peft_dirs, a.base, PROMPTS)
    rows = compare(engine, peft, [n for n, _ in compiled], len(PROMPTS))
    print(f"\n{'adapter':8s} {'effect':>8s} {'error':>8s} {'error/effect':>13s}")
    for r in rows:
        share = f"{100 * r['share']:.2f}%" if r["share"] is not None else "-"
        print(f"{r['adapter']:8s} {r['effect']:8.4f} {r['error']:8.4f} {share:>13s}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
