# Scute

Serve **LoRA, DoRA, (IA)³, VeRA and OFT** adapters exactly, from their PEFT checkpoints,
in one batch of an unmodified vLLM.

vLLM batches requests for many LoRA adapters through one schedule. Its LoRA path
computes the base layer once for the whole batch, then a small *shrink* (`x A`) and
*expand* (`· B`) for each adapter. It refuses adapters of other methods. Scute writes
every method as one formula with six *slots*:

```
h' = ((x R) W) ⊙ s + b + Σ_k c_k · ( ((x A_k) ⊙ d_k) B_k ⊙ e_k )
```

Each slot is the identity when a method does not use it. A slot *folds* into the
adapter's own weights at load when the adapter owns a tensor next to it. The other slots
stay *live*, and the engine computes them inside the LoRA schedule:

| method | live slots | how the engine computes them |
|---|---|---|
| LoRA | none | vLLM's own shrink and expand |
| DoRA | output scale `s` | the expand multiplies each tile by `s` before it writes it |
| (IA)³ | output scale `s` | the same |
| VeRA | vectors `d`, `e` next to shared factors | two GEMMs over the whole batch, which read the shared factors once per step |
| OFT | input rotation `R` | one pass before the base GEMM, which stays shared |

All tenants still share one base GEMM, whatever their method, so one engine serves
all five methods in one batch.

## Install

Scute needs a CUDA GPU and **vLLM 0.19.1** exactly, because it patches private vLLM
symbols at run time. vLLM itself is not modified.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[test]"
source .venv/bin/activate
```

The `test` extra adds PEFT 0.20.0 and pytest, which the examples and tests use.

## Quick start

The examples use `HuggingFaceTB/SmolLM2-360M-Instruct`, a small Llama-type model that
needs no access request. Any Llama-type model works (projections named `q_proj`,
`k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj`).

```bash
python examples/make_adapters.py --out adapters/peft      # 2 adapters of each method
python examples/serve_mixed_batch.py --adapters adapters/peft
```

`serve_mixed_batch.py` compiles the ten adapters, starts one engine, and generates
for the base model and every adapter in one call. In your own code:

```python
import scute
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

base = "HuggingFaceTB/SmolLM2-360M-Instruct"

# 1. Compile each PEFT checkpoint once. LoRA and DoRA are copied through.
scute.compile_adapter("adapters/peft/ia3-0", "adapters/compiled/ia3-0", base)
scute.compile_adapter("adapters/peft/vera-0", "adapters/compiled/vera-0", base)

# 2. Patch vLLM before you create the engine. The settings come from the adapters.
scute.enable(["adapters/compiled/ia3-0", "adapters/compiled/vera-0"])

# 3. Serve the compiled directories as LoRA adapters.
llm = LLM(model=base, enable_lora=True, max_loras=2, max_lora_rank=16, dtype="float16")
outs = llm.generate(
    ["Hello, my name is"] * 2,
    SamplingParams(max_tokens=16),
    lora_request=[LoRARequest("ia3-0", 1, "adapters/compiled/ia3-0"),
                  LoRARequest("vera-0", 2, "adapters/compiled/vera-0")],
)
```

## OpenAI-compatible server

`scute-compile` compiles one adapter. `scute-serve` takes the arguments of
`vllm serve`, reads the engine settings from the adapters in `--lora-modules`, and
starts vLLM's server with the patches on in every vLLM process.

```bash
scute-compile adapters/peft/oft-0 adapters/compiled/oft-0 --base $BASE
scute-serve $BASE --dtype float16 --enable-lora --max-lora-rank 16 \
    --lora-modules oft-0=$PWD/adapters/compiled/oft-0
```

`examples/openai_server.sh` does this for all ten adapters and sends one request per
adapter at the same time.

## Check it against PEFT

```bash
python examples/check_against_peft.py --adapters adapters/peft
```

The engine scores four prompts for every adapter, all in one batch, in float16. The
PEFT library scores the same tokens in float32, one adapter at a time. For each
adapter the script prints its *effect* (the mean change in log-probability per token
that it makes to the base model) and the engine's *error* against PEFT:

```
adapter    effect    error  error/effect
base       0.0000   0.0028             -
dora-0     0.3393   0.0047         1.39%
dora-1     0.3148   0.0056         1.79%
ia3-0      1.5561   0.0091         0.58%
ia3-1      1.6808   0.0071         0.42%
lora-0     0.1583   0.0030         1.88%
lora-1     0.2002   0.0036         1.80%
oft-0      2.0015   0.0052         0.26%
oft-1      4.2728   0.0140         0.33%
vera-0     0.2001   0.0045         2.27%
vera-1     0.1646   0.0032         1.95%
```

(RTX A6000.) Each adapter's error is close to the error of the base model, which has
no adapter, so it is float16 rounding. It is at most a few percent of each adapter's
effect, while an engine that drops a slot or applies it twice is wrong by a large
share of the effect.

## See which slots a method needs

```bash
python examples/plan_slots.py
```

prints, for each method, the slots that fold at load, the slots that stay live, and
the bytes that each live slot adds to one pass. `scute.slots.plan_slots` applies the
same rule to a new method that you describe with `AdapterSpec`.

## Tests

```bash
python -m pytest                 # unit tests: compiler, fold rule, kernels (about 2 min)
python -m pytest -m engine       # end to end: five methods in one vLLM batch against PEFT
```

## Limits

- **vLLM 0.19.1** only. `scute.enable` refuses another release.
- **One pair of VeRA shared factors and one OFT block size per engine.** `scute.enable`
  refuses adapters that differ in either.
- **DoRA needs `dtype="float16"` or float32.** bfloat16 cannot resolve DoRA's scale,
  which is within about 2% of one, so the engine refuses it for DoRA.
- No quantized base weights, no `fully_sharded_loras`, no LoRA bias, no COFT or
  block-shared OFT. Each of these is refused at load, with a message.
- PEFT has no SSF method, so the compiler does not read SSF checkpoints, although
  SSF is in the normal form.

## Layout

```
src/scute/
  compile.py            PEFT checkpoint -> normal form (scute-compile)
  engine.py             scute.enable: engine settings, patches
  plugin.py             vLLM plugin: patches every vLLM process when enabled
  serve.py              scute-serve: vllm serve with the patches
  slots.py              the fold rule as a decision procedure
  ops/                  Triton kernels for the routes
  vllm_integration/     the run-time patches of vLLM
examples/               make adapters, serve a mixed batch, check against PEFT, server
tests/
```

`scute.vllm_integration.applied_patches()` lists every vLLM symbol that is patched.

## License

Apache License 2.0 (`LICENSE`). Two kernels are modified versions of vLLM's LoRA
kernels, which are also Apache-2.0; `NOTICE` lists them.
