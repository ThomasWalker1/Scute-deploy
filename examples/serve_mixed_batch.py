"""Compile adapters of five methods and serve them all in one vLLM batch.

Every request in the batch names a different adapter, and the engine serves them
together, each exactly as PEFT computes it.

Usage::

    python examples/make_adapters.py --out adapters/peft
    python examples/serve_mixed_batch.py --adapters adapters/peft
"""

from __future__ import annotations

import argparse
import os

from make_adapters import BASE

PROMPT = "The three most important ideas in computer science are"


def compile_all(peft_root: str, out_root: str, base: str) -> list[tuple[str, str]]:
    """Compile every PEFT adapter under ``peft_root``.  Returns ``[(name, compiled_dir)]``."""
    import scute

    out = []
    for name in sorted(os.listdir(peft_root)):
        src = os.path.join(peft_root, name)
        if not os.path.exists(os.path.join(src, "adapter_config.json")):
            continue
        c = scute.compile_adapter(src, os.path.join(out_root, name), base)
        print(f"compiled {name:8s} ({c.method}, {c.modules} modules)")
        out.append((name, os.path.abspath(c.out_dir)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapters", default="adapters/peft")
    ap.add_argument("--compiled", default="adapters/compiled")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--max-tokens", type=int, default=24)
    a = ap.parse_args()

    import scute

    adapters = compile_all(a.adapters, a.compiled, a.base)
    settings = scute.enable([d for _, d in adapters])
    print("engine settings:", settings)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=a.base, enable_lora=True, max_loras=len(adapters), max_lora_rank=16,
              dtype="float16", max_model_len=1024, gpu_memory_utilization=0.5)
    requests = [None] + [LoRARequest(n, i + 1, d) for i, (n, d) in enumerate(adapters)]
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    # One call, one batch: a request for the base model and one for every adapter.
    outs = llm.generate([PROMPT] * len(requests), sp, lora_request=requests)
    print(f"\nprompt: {PROMPT!r}\n")
    for req, o in zip(requests, outs):
        name = req.lora_name if req else "base"
        print(f"{name:8s} {o.outputs[0].text!r}")


if __name__ == "__main__":
    main()
