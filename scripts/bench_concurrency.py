import asyncio
import time
import json
import statistics
import csv
import os
from datetime import datetime
import requests

URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen3-1.7b"

SYSTEM = "Answer concisely. No <think> tags."
BASE_PROMPT = (
    "Explain KV-cache in transformer inference. "
    "Focus on inference-time behavior.\n"
)
PROMPT = BASE_PROMPT * 16  # from Phase 2 safe range

MAX_TOKENS = 128
TIMEOUT_S = 300

CONCURRENCY_LEVELS = [1, 2, 4, 6, 8]

PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
}

RESULTS_DIR = "results"


def wait_for_server(base_url="http://localhost:1234", timeout_s=15):
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/api/v0/models", timeout=2)
            if r.status_code == 200:
                return True
        except Exception as e:
            last_err = e
        time.sleep(0.5)

    raise RuntimeError(
        f"LM Studio server not reachable at {base_url}. "
        f"Start it with: lms server start --port 1234. Last error: {last_err}"
    )


def make_payload():
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT},
        ],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
        **PARAMS,
    }


def one_request(session_id: int):
    payload = make_payload()
    t0 = time.perf_counter()
    first_token_t = None
    completion_tokens = 0

    with requests.post(URL, json=payload, stream=True, timeout=TIMEOUT_S) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data = line[6:].strip()
            if data == "[DONE]":
                break

            evt = json.loads(data)

            # usage may arrive near the end
            if isinstance(evt, dict) and evt.get("usage"):
                completion_tokens = evt["usage"].get("completion_tokens", completion_tokens)

            choices = evt.get("choices")
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content and first_token_t is None:
                first_token_t = time.perf_counter()

    t1 = time.perf_counter()

    return {
        "session_id": session_id,
        "ttft_s": (first_token_t - t0) if first_token_t else None,
        "total_s": t1 - t0,
        "completion_tokens": completion_tokens,
    }


def safe_p95(values):
    if len(values) < 2:
        return None
    # statistics.quantiles returns cut points; n=20 => 5th percentile steps; index 18 => 95th
    return statistics.quantiles(values, n=20)[18]


async def warmup():
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Warmup OK"}],
        "max_tokens": 8,
        "temperature": 0.2,
        "stream": False,
    }
    requests.post(URL, json=payload, timeout=TIMEOUT_S)


async def run_level(n: int):
    print(f"\n=== Concurrency: {n} ===")
    start = time.perf_counter()

    tasks = [asyncio.to_thread(one_request, i) for i in range(n)]
    results = await asyncio.gather(*tasks)

    end = time.perf_counter()

    latencies = [r["total_s"] for r in results]
    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    tokens = sum(r["completion_tokens"] for r in results)

    mean_latency = statistics.mean(latencies)
    p95_latency = safe_p95(latencies)
    mean_ttft = statistics.mean(ttfts) if ttfts else None
    wall_s = (end - start)
    rps = n / wall_s if wall_s > 0 else None
    tps = tokens / wall_s if wall_s > 0 else None

    print(f"Requests completed: {n}")
    print(f"Mean latency: {mean_latency:.2f}s")
    if p95_latency is None:
        print("P95 latency: N/A (need >= 2 samples)")
    else:
        print(f"P95 latency: {p95_latency:.2f}s")

    if mean_ttft is None:
        print("Mean TTFT: N/A")
    else:
        print(f"Mean TTFT: {mean_ttft:.2f}s")

    print(f"Requests/sec: {rps:.2f}" if rps is not None else "Requests/sec: N/A")
    print(f"Tokens/sec (aggregate): {tps:.2f}" if tps is not None else "Tokens/sec (aggregate): N/A")

    return {
        "concurrency": n,
        "wall_s": wall_s,
        "requests_completed": n,
        "mean_latency_s": mean_latency,
        "p95_latency_s": p95_latency,
        "mean_ttft_s": mean_ttft,
        "completion_tokens_sum": tokens,
        "requests_per_s": rps,
        "tokens_per_s_aggregate": tps,
        "per_request": results,
    }


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def write_csv(ts: str, all_levels):
    ensure_results_dir()

    summary_path = os.path.join(RESULTS_DIR, f"phase3_concurrency_summary_{ts}.csv")
    perreq_path = os.path.join(RESULTS_DIR, f"phase3_concurrency_requests_{ts}.csv")

    # Summary CSV
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "model",
                "temperature",
                "top_p",
                "top_k",
                "max_tokens",
                "prompt_multiplier",
                "concurrency",
                "wall_s",
                "requests_completed",
                "mean_latency_s",
                "p95_latency_s",
                "mean_ttft_s",
                "completion_tokens_sum",
                "requests_per_s",
                "tokens_per_s_aggregate",
            ],
        )
        w.writeheader()

        for lvl in all_levels:
            w.writerow(
                {
                    "timestamp": ts,
                    "model": MODEL,
                    "temperature": PARAMS.get("temperature"),
                    "top_p": PARAMS.get("top_p"),
                    "top_k": PARAMS.get("top_k"),
                    "max_tokens": MAX_TOKENS,
                    "prompt_multiplier": 16,
                    "concurrency": lvl["concurrency"],
                    "wall_s": round(lvl["wall_s"], 6),
                    "requests_completed": lvl["requests_completed"],
                    "mean_latency_s": round(lvl["mean_latency_s"], 6),
                    "p95_latency_s": round(lvl["p95_latency_s"], 6) if lvl["p95_latency_s"] is not None else None,
                    "mean_ttft_s": round(lvl["mean_ttft_s"], 6) if lvl["mean_ttft_s"] is not None else None,
                    "completion_tokens_sum": lvl["completion_tokens_sum"],
                    "requests_per_s": round(lvl["requests_per_s"], 6) if lvl["requests_per_s"] is not None else None,
                    "tokens_per_s_aggregate": round(lvl["tokens_per_s_aggregate"], 6)
                    if lvl["tokens_per_s_aggregate"] is not None
                    else None,
                }
            )

    # Per-request CSV
    with open(perreq_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "model",
                "concurrency",
                "session_id",
                "ttft_s",
                "total_s",
                "completion_tokens",
            ],
        )
        w.writeheader()

        for lvl in all_levels:
            for r in lvl["per_request"]:
                w.writerow(
                    {
                        "timestamp": ts,
                        "model": MODEL,
                        "concurrency": lvl["concurrency"],
                        "session_id": r["session_id"],
                        "ttft_s": round(r["ttft_s"], 6) if r["ttft_s"] is not None else None,
                        "total_s": round(r["total_s"], 6),
                        "completion_tokens": r["completion_tokens"],
                    }
                )

    print(f"\nSaved CSV:\n- {summary_path}\n- {perreq_path}")


def main():
    wait_for_server("http://localhost:1234", timeout_s=15)
    asyncio.run(warmup())

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_levels = []
    for n in CONCURRENCY_LEVELS:
        lvl = asyncio.run(run_level(n))
        all_levels.append(lvl)

    write_csv(ts, all_levels)


if __name__ == "__main__":
    main()
