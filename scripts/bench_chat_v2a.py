import csv
import json
import time
import re
import requests
from datetime import datetime

URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen3-1.7b"

BASE_PROMPT = (
    "Explain KV-cache in transformer inference. "
    "Provide a concise, technical explanation.\n"
)
#prompt varies by multiplier; no fixed prompt
#PROMPT = "Explain KV-cache in simple terms with exactly 5 bullet points."

SYSTEM = "Be concise. Do not include <think> tags or internal reasoning. Output only the final answer."

MAX_TOKENS = 220
TIMEOUT_S = 300

PROMPT_MULTIPLIERS = [1, 4, 8, 16, 32, 64]

# Keep your same grid, but you can expand later.
GRID = [
    {"temperature": 0.2, "top_p": 0.9, "top_k": 40},
    {"temperature": 0.7, "top_p": 0.9, "top_k": 40},
    {"temperature": 1.0, "top_p": 0.9, "top_k": 40},
    {"temperature": 0.7, "top_p": 0.95, "top_k": 40},
    {"temperature": 0.7, "top_p": 0.9, "top_k": 0},  # try disabling top_k
]

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

def strip_think(s: str) -> str:
    return THINK_RE.sub("", s)

def build_payload(params: dict, stream: bool, prompt: str):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "stream": stream,
        **params,
    }

    # Best-effort: some OpenAI-compatible servers support include_usage in stream.
    if stream:
        payload["stream_options"] = {"include_usage": True}

    # Best-effort determinism if supported (ignored if not supported).
    payload["seed"] = 42

    return payload

def stream_once(params:dict, prompt: str):
    """
    Returns:
      text (str), ttft_s (float|None), total_s (float), usage (dict|None)
    """
    payload = build_payload(params, stream=True, prompt=prompt)

    t0 = time.perf_counter()
    first_token_t = None
    chunks = []
    usage = None

    with requests.post(URL, json=payload, stream=True, timeout=TIMEOUT_S) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue

            data = line[6:].strip()
            if data == "[DONE]":
                break

            evt = json.loads(data)

            # Some servers attach usage at the end (or occasionally mid-stream).
            if isinstance(evt, dict) and "usage" in evt and evt["usage"]:
                usage = evt["usage"]

            choice0 = (evt.get("choices") or [{}])[0]
            delta = choice0.get("delta") or {}
            txt = delta.get("content")
            if txt:
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                chunks.append(txt)

    t1 = time.perf_counter()

    text = strip_think("".join(chunks))
    ttft_s = (first_token_t - t0) if first_token_t else None
    total_s = t1 - t0

    return text, ttft_s, total_s, usage

def nonstream_usage_only(params: dict, prompt: str):
    """
    Fallback if streaming doesn't include usage.
    Returns usage dict (may be None if server doesn't provide).
    """
    payload = build_payload(params, stream=False, prompt=prompt)

    t0 = time.perf_counter()
    r = requests.post(URL, json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    t1 = time.perf_counter()

    obj = r.json()
    usage = obj.get("usage")
    # We return usage plus total duration of this fallback run (optional)
    return usage, (t1 - t0)

def warmup():
    # Small warm-up (discard results). Keeps it quick.
    params = {"temperature": 0.2, "top_p": 0.9, "top_k": 40}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Warmup: reply with OK."}],
        "max_tokens": 16,
        "temperature": 0.2,
        "stream": False,
        "seed": 42,
        **params,
    }
    r = requests.post(URL, json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/phase2_context_scaling_{ts}.csv"

    print("Warming up (discarded)...")
    warmup()
    print("Warm-up done.\n")

    fieldnames = [
        "temperature", "top_p", "top_k",
        "ttft_s", "total_s",
        "completion_tokens", "prompt_tokens", "total_tokens",
        "tok_per_s_total", "tok_per_s_gen",
        "usage_source"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        # Phase 2: fix sampling; vary prompt length
        fixed_params = {"temperature": 0.7, "top_p": 0.9, "top_k": 40}

        # Update CSV fields to include multiplier
        fieldnames = [
            "prompt_multiplier",
            "temperature", "top_p", "top_k",
            "ttft_s", "total_s",
            "completion_tokens", "prompt_tokens", "total_tokens",
            "tok_per_s_total", "tok_per_s_gen",
            "usage_source"
        ]

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for m in PROMPT_MULTIPLIERS:
            prompt = BASE_PROMPT * m
            params = fixed_params

            print(f"\n=== Prompt multiplier: {m}x | params: {params} ===")

            text, ttft_s, total_s, usage = stream_once(params, prompt=prompt)

            # Verbose output (what you asked for)
            print(text)

            usage_source = "stream"
            fallback_total_s = None

            if not usage:
                # Fallback to non-stream to get usage (tokens)
                usage, fallback_total_s = nonstream_usage_only(params, prompt=prompt)
                usage_source = "nonstream_fallback"

            prompt_tokens = usage.get("prompt_tokens") if usage else None
            completion_tokens = usage.get("completion_tokens") if usage else None
            total_tokens = usage.get("total_tokens") if usage else None

            tok_per_s_total = (completion_tokens / total_s) if (completion_tokens and total_s) else None
            # tokens/sec excluding TTFT (generation-only)
            gen_s = (total_s - ttft_s) if (ttft_s is not None) else None
            tok_per_s_gen = (completion_tokens / gen_s) if (completion_tokens and gen_s and gen_s > 0) else None

            row = {
                "prompt_multiplier": m,
                **params,
                "ttft_s": round(ttft_s, 3) if ttft_s is not None else None,
                "total_s": round(total_s, 3),
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "tok_per_s_total": round(tok_per_s_total, 3) if tok_per_s_total is not None else None,
                "tok_per_s_gen": round(tok_per_s_gen, 3) if tok_per_s_gen is not None else None,
                "usage_source": usage_source,
            }

            w.writerow(row)
            f.flush()

            # Optional: show quick metrics summary after each run
            print("\n--- metrics ---")
            print(f"TTFT: {row['ttft_s']} s")
            print(f"Total: {row['total_s']} s")
            print(f"Completion tokens: {completion_tokens} (usage: {usage_source})")
            print(f"tok/s (total): {row['tok_per_s_total']}")
            print(f"tok/s (gen-only): {row['tok_per_s_gen']}")

            # If usage came from fallback, note that it was a second run
            if usage_source == "nonstream_fallback":
                print("Note: tokens came from a non-stream fallback call (separate run).")

    print(f"\nSaved CSV: {csv_path}")

if __name__ == "__main__":
    main()
