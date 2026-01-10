## Phase 2 — Context scaling
- Latency and TTFT increase sharply as prompt size grows.
- On constrained CPU/RAM, prompt evaluation dominates end-to-end latency.

Recommendation:
- Treat long context as a “budget.” Trim prompt aggressively.
- Prefer smaller context defaults unless the user explicitly needs long history.

## Phase 3 — Concurrency saturation
Observed behavior:
- Throughput (aggregate tokens/sec) plateaus early.
- Increasing concurrency beyond a small number increases **mean latency** and dramatically increases **tail latency** (P95).
- TTFT degrades severely as concurrency increases, indicating queueing and contention during prompt evaluation.

Recommendation:
- Cap in-flight requests to a small number (e.g., 2 on CPU-only).
- Queue requests beyond that limit instead of running them concurrently.

## Phase 4 — Structured output (strict JSON)
Raw behavior:
- “Strict JSON only” frequently fails due to:
  - `<think>...</think>` blocks before JSON
  - multiple JSON objects returned in one response
  - occasional truncation when the model over-produces

Auto-repair:
- A safe post-processor that:
  1) strips `<think>...</think>`
  2) extracts the first balanced `{...}`
  3) validates via `json.loads`
  can convert many “formatting failures” into valid JSON.

Recommendation:
- Never trust raw LLM output for tool/JSON pipelines.
- Always apply validation + normalization.
- Add retry logic for unrecoverable cases (e.g., truncated JSON).

## Operational defaults (CPU-only, constrained RAM)
- Max concurrency: 2 (queue beyond)
- Context: keep small by default; avoid large multipliers
- Structured output: enforce validation/repair step before downstream parsing
