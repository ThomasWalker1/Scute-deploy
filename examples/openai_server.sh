#!/usr/bin/env bash
# Serve compiled adapters of every method behind vLLM's OpenAI-compatible server.
#
#   python examples/make_adapters.py --out adapters/peft
#   bash examples/openai_server.sh
#
# Each adapter is a model name in the API. Requests for different adapters are
# batched together.
set -euo pipefail

BASE=${BASE:-HuggingFaceTB/SmolLM2-360M-Instruct}
PORT=${PORT:-8000}
PEFT=${PEFT:-adapters/peft}
OUT=${OUT:-adapters/compiled}

modules=()
for src in "$PEFT"/*/; do
    name=$(basename "$src")
    scute-compile "$src" "$OUT/$name" --base "$BASE" > /dev/null
    modules+=("$name=$(realpath "$OUT/$name")")
done

scute-serve "$BASE" --port "$PORT" --dtype float16 --max-model-len 1024 \
    --gpu-memory-utilization 0.5 --enable-lora --max-lora-rank 16 \
    --max-loras "${#modules[@]}" --lora-modules "${modules[@]}" &
server=$!
trap 'kill $server' EXIT

until curl -sf "localhost:$PORT/v1/models" > /dev/null; do sleep 2; done

# Send one request per adapter at the same time, so the engine batches them.
tmp=$(mktemp -d)
pids=()
for name in base "${modules[@]%%=*}"; do
    model=$([ "$name" = base ] && echo "$BASE" || echo "$name")
    curl -s "localhost:$PORT/v1/completions" -H "Content-Type: application/json" \
        -d "{\"model\": \"$model\", \"prompt\": \"The three most important ideas in computer science are\", \"max_tokens\": 16, \"temperature\": 0}" \
        > "$tmp/$name.json" &
    pids+=($!)
done
wait "${pids[@]}"
for name in base "${modules[@]%%=*}"; do
    printf '%-8s %s\n' "$name" \
        "$(python -c "import json, sys; print(repr(json.load(open(sys.argv[1]))['choices'][0]['text']))" "$tmp/$name.json")"
done
