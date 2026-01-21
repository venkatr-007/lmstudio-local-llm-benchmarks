# lmstudio-local-llm-benchmarks

A small, practical benchmark suite for **LM Studio local inference** focused on the operational questions that matter when you run a model locally:

- Which decoding settings are stable and performant?
- How does performance degrade as **context length grows**?
- What happens under **concurrent load** (CPU-only, constrained RAM)?
- Can we make **structured JSON output** reliable enough for tooling?

This repo targets LM Studio’s **OpenAI-compatible local server** (`/v1/chat/completions`) and produces simple CSV/JSON outputs you can graph later.

---

## Tested setup (example)

- LM Studio local server (`lms server start --port 1234`)
- Model: `qwen3-1.7b` (GGUF)
- Hardware: CPU-only (no GPU), constrained RAM

> You can use any model available in LM Studio; just update the `--model` argument (and context length) accordingly.

---

## What’s inside

### Phase 1 — Sampler sweep (baseline decoding)
Benchmarks different sampler settings (`temperature`, `top_p`, `top_k`) and logs:
- TTFT (time to first token)
- total latency
- tokens/sec

Script:
- `scripts/bench_chat_v2.py`

### Phase 2 — Context scaling
Measures how latency and TTFT change as prompt size increases (prompt multiplier approach).

Script:
- `scripts/bench_chat_v2a.py`

### Phase 3 — Concurrency saturation
Runs multiple simultaneous requests and reports:
- mean latency
- P95 latency (tail latency)
- mean TTFT
- requests/sec
- aggregate tokens/sec

Script:
- `scripts/bench_concurrency.py`

### Phase 4 — Structured output reliability (JSON)
#### Phase 4.0 (raw)
Measures whether the model returns **strict JSON only** (no extra text).

Script:
- `scripts/bench_json_reliability.py`

#### Phase 4.1/4.2 (auto-repair + trials)
Adds a safe post-processor that:
- strips `<think>...</think>`
- extracts the first balanced JSON object `{...}`
- validates with `json.loads`

Then runs multiple trials across prompt sizes + concurrency.

Script:
- `scripts/bench_json_reliability_v2.py`

---

## Quickstart

### 0) Prereqs
- Windows + PowerShell (or equivalent)
- Python 3.9+ recommended
- LM Studio installed with `lms` CLI available in PATH

### 1) Create a venv and install deps
```powershell
cd D:\code\lmstudio-local-llm-benchmarks
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

### 2) Start LM Studio server and load a model
In one terminal:
```bash
lms server start --port 1234
lms load qwen3-1.7b --context-length 1024 --gpu off
```

> If you’re using a different model name or context length, update the `lms load ...` command and/or script arguments.

### 3) Run benchmarks
In another terminal (venv activated):

**Phase 1 (sampler sweep):**
```powershell
python .\scripts\bench_chat_v2.py --model qwen3-1.7b --base-url http://localhost:1234/v1
```

**Phase 2 (context scaling):**
```powershell
python .\scripts\bench_chat_v2a.py --model qwen3-1.7b --base-url http://localhost:1234/v1
```

**Phase 3 (concurrency):**
```powershell
python .\scripts\bench_concurrency.py --model qwen3-1.7b --base-url http://localhost:1234/v1
```

**Phase 4 (JSON reliability):**
```powershell
python .\scripts\bench_json_reliability.py --model qwen3-1.7b --base-url http://localhost:1234/v1
python .\scripts\bench_json_reliability_v2.py --model qwen3-1.7b --base-url http://localhost:1234/v1
```

---

## Outputs

Most scripts write timestamped outputs (typically under a `results/` directory).  
If you want results tracked in git, remove `results/` from `.gitignore`. Otherwise, keep results ignored and commit only **code**.

---

## Security / hygiene

- Do **not** commit `.env` files or any API keys/tokens.
- This repo is intended to run against *local* LM Studio; you generally should not need any external API keys.

---

## License

MIT (see `LICENSE`).
