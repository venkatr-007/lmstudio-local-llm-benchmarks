# lmstudio-local-llm-benchmarks

A small, practical benchmark suite for **LM Studio local inference** that focuses on the real operational questions:
- Which decoding settings are stable and performant?
- How does performance degrade as **context length grows**?
- What happens under **concurrent load** on CPU?
- Can we make **structured JSON output** reliable enough for tooling?

Tested setup (example):
- LM Studio local server (`lms server start --port 1234`)
- Model: `qwen3-1.7b` (GGUF)
- Hardware: CPU-only (no GPU), constrained RAM

## What’s inside

### Phase 1 — Sampler sweep (baseline decoding)
Benchmarks different sampler settings (temperature/top_p/top_k) and logs:
- TTFT (time to first token)
- total latency
- tokens/sec

Scripts:
- `scripts/bench_chat_v2.py`

### Phase 2 — Context scaling
Measures how latency and TTFT change as prompt size increases (prompt multiplier approach).

Scripts:
- `scripts/bench_chat_v2a.py`

### Phase 3 — Concurrency saturation
Runs multiple simultaneous requests and reports:
- mean latency
- P95 latency (tail latency)
- mean TTFT
- requests/sec
- aggregate tokens/sec

Scripts:
- `scripts/bench_concurrency.py`

### Phase 4 — Structured output reliability (JSON)
#### Phase 4.0 (raw)
Measures whether the model returns **strict JSON only** (no extra text).
Result observed: raw strict JSON validity can be effectively 0% due to `<think>` blocks and multiple JSON objects.

Scripts:
- `scripts/bench_json_reliability.py`

#### Phase 4.1/4.2 (auto-repair + trials)
Adds a safe post-processor:
- strips `<think>...</think>`
- extracts the first balanced JSON object `{...}`
- validates with `json.loads`

Then runs multiple trials across prompt sizes + concurrency.
Result observed in this environment: repaired JSON validity reached 100% over 90 requests.

Scripts:
- `scripts/bench_json_reliability_v2.py`

## Quickstart

### 1) Start LM Studio server and load model
In one terminal:

```bash
lms server start --port 1234
lms load qwen3-1.7b --context-length 1024 --gpu off
