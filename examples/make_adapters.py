"""Build PEFT adapters of all five methods for a base model, to try the engine on.

The adapters are untrained.  PEFT initialises each method to the identity, so each
adapter's trainable tensors are set to random values that change the model's output
by a clear margin.  Each method gets two adapters with different seeds.  The two VeRA
adapters share one pair of factors, as VeRA requires.

Usage::

    python examples/make_adapters.py --out adapters/peft
"""

from __future__ import annotations

import argparse
import os

import torch

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
METHODS = ("lora", "dora", "ia3", "vera", "oft")


def peft_config(method: str):
    from peft import IA3Config, LoraConfig, OFTConfig, VeraConfig

    if method in ("lora", "dora"):
        return LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=16, target_modules=TARGETS,
                          use_dora=method == "dora")
    if method == "ia3":
        return IA3Config(task_type="CAUSAL_LM", target_modules=["k_proj", "v_proj", "down_proj"],
                         feedforward_modules=["down_proj"])
    if method == "vera":
        return VeraConfig(task_type="CAUSAL_LM", r=256, target_modules=TARGETS,
                          save_projection=True)
    if method == "oft":
        return OFTConfig(task_type="CAUSAL_LM", r=0, oft_block_size=32, target_modules=TARGETS)
    raise ValueError(method)


@torch.no_grad()
def randomise(model, seed: int) -> None:
    """Move each trainable tensor away from the identity."""
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        z = torch.randn(p.shape, generator=g)
        if name.endswith("ia3_l.default"):
            p.copy_(1 + 0.3 * z)
        elif name.endswith("vera_lambda_d.default"):
            p.copy_(0.1 * (1 + 0.3 * z))
        elif name.endswith("vera_lambda_b.default"):
            p.copy_(1.0 * z)
        elif ".oft_R." in name:
            p.copy_(0.02 * z)
        elif "lora_B" in name:
            p.copy_(0.08 * z)
        elif "lora_magnitude_vector" in name:
            p.mul_(1 + 0.05 * z)


def make_adapters(out: str, base: str = BASE, methods=METHODS, per_method: int = 2
                  ) -> list[dict]:
    """Write ``per_method`` adapters of each method under ``out``.

    Returns ``[{"name", "method", "path"}]``.
    """
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    entries = []
    for method in methods:
        for seed in range(per_method):
            name = f"{method}-{seed}"
            path = os.path.abspath(os.path.join(out, name))
            if not os.path.exists(os.path.join(path, "adapter_config.json")):
                model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
                model = get_peft_model(model, peft_config(method))
                randomise(model, seed)
                model.save_pretrained(path)
                del model
            entries.append(dict(name=name, method=method, path=path))
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="adapters/peft")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--per-method", type=int, default=2)
    a = ap.parse_args()
    for e in make_adapters(a.out, a.base, per_method=a.per_method):
        print(f"{e['name']:8s} {e['method']:5s} {e['path']}")


if __name__ == "__main__":
    main()
