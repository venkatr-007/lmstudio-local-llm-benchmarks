import json
import asyncio
import requests
import csv
import os
from datetime import datetime
from collections import defaultdict
import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TRIALS = 10

URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen3-1.7b"

SYSTEM = (
    "You are a strict JSON generator. "
    "Output ONLY valid JSON. "
    "Do not include explanations, markdown, or extra text."
)

BASE_PROMPT = (
    "Return a JSON object with keys: "
    "name (string), version (number), features (array of strings). "
    "Use simple dummy values."
)

PROMPT_MULTIPLIERS = [1, 8, 16]
CONCURRENCY_LEVELS = [1, 2]

MAX_TOKENS = 200
TIMEOUT_S = 300

PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
}

RESULTS_DIR = "results"


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def make_payload(prompt):
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "stream": False,
        **PARAMS,
    }


def run_once(prompt):
    r = requests.post(URL, json=make_payload(prompt), timeout=TIMEOUT_S)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    return text


def safe_run_once(prompt):
    try:
        return True, run_once(prompt)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except Exception:
        return False


async def run_concurrent(prompt, n):
    tasks = [asyncio.to_thread(safe_run_once, prompt) for _ in range(n)]
    results = await asyncio.gather(*tasks)

    rows = []
    for i, (ok, payload) in enumerate(results):
        if not ok:
            rows.append(
                {
                    "request_index": i,
                    "ok": False,
                    "raw_text": None,
                    "raw_error": payload,
                    "raw_valid": False,
                }
            )
        else:
            txt = payload
            rows.append(
                {
                    "request_index": i,
                    "ok": True,
                    "raw_text": txt,
                    "raw_error": None,
                    "raw_valid": is_valid_json(txt),
                }
            )
    return rows


def warmup():
    requests.post(
        URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Warmup OK"}],
            "max_tokens": 8,
            "stream": False,
        },
        timeout=TIMEOUT_S,
    )


def main():
    ensure_results_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows_csv = os.path.join(RESULTS_DIR, f"phase4_json_raw_trials_{ts}.csv")
    summary_csv = os.path.join(RESULTS_DIR, f"phase4_json_raw_summary_{ts}.csv")

    print("Warming up...")
    warmup()
    print("Warm-up done.\n")

    summary = defaultdict(lambda: {"pass": 0, "fail": 0, "req_fail": 0})

    with open(rows_csv, "w", newline="", encoding="utf-8") as f_rows:
        w_rows = csv.DictWriter(
            f_rows,
            fieldnames=[
                "timestamp",
                "model",
                "prompt_multiplier",
                "concurrency",
                "trial_index",
                "request_index",
                "request_ok",
                "raw_valid",
                "raw_error",
            ],
        )
        w_rows.writeheader()

        for m in PROMPT_MULTIPLIERS:
            prompt = BASE_PROMPT * m
            for c in CONCURRENCY_LEVELS:
                print(f"=== Prompt {m}x | Concurrency {c} | Trials {TRIALS} ===")
                example_printed = False

                for t in range(TRIALS):
                    batch = asyncio.run(run_concurrent(prompt, c))

                    for r in batch:
                        if not r["ok"]:
                            summary[(m, c)]["req_fail"] += 1
                            summary[(m, c)]["fail"] += 1
                        else:
                            if r["raw_valid"]:
                                summary[(m, c)]["pass"] += 1
                            else:
                                summary[(m, c)]["fail"] += 1

                        w_rows.writerow(
                            {
                                "timestamp": ts,
                                "model": MODEL,
                                "prompt_multiplier": m,
                                "concurrency": c,
                                "trial_index": t,
                                "request_index": r["request_index"],
                                "request_ok": r["ok"],
                                "raw_valid": r["raw_valid"],
                                "raw_error": r["raw_error"],
                            }
                        )

                    f_rows.flush()

                    p = summary[(m, c)]["pass"]
                    f = summary[(m, c)]["fail"]
                    rf = summary[(m, c)]["req_fail"]
                    print(f"Progress: {p}/{p+f} valid JSON | ReqFail {rf}")

                    if not example_printed:
                        first_fail = next((x for x in batch if x["ok"] and not x["raw_valid"]), None)
                        if first_fail is not None:
                            print("Example failure output (first failure for this config):")
                            print(first_fail["raw_text"])
                            print("---")
                            example_printed = True

    with open(summary_csv, "w", newline="", encoding="utf-8") as f_sum:
        w_sum = csv.DictWriter(
            f_sum,
            fieldnames=[
                "timestamp",
                "model",
                "prompt_multiplier",
                "concurrency",
                "pass",
                "fail",
                "rate_percent",
                "request_failures",
            ],
        )
        w_sum.writeheader()

        for (m, c), stats in sorted(summary.items()):
            total = stats["pass"] + stats["fail"]
            rate = (stats["pass"] / total * 100.0) if total else 0.0
            w_sum.writerow(
                {
                    "timestamp": ts,
                    "model": MODEL,
                    "prompt_multiplier": m,
                    "concurrency": c,
                    "pass": stats["pass"],
                    "fail": stats["fail"],
                    "rate_percent": round(rate, 3),
                    "request_failures": stats["req_fail"],
                }
            )

    print("\n=== Final Summary ===")
    for (m, c), stats in sorted(summary.items()):
        total = stats["pass"] + stats["fail"]
        rate = (stats["pass"] / total) * 100 if total else 0
        print(
            f"Prompt {m}x | Concurrency {c} -> "
            f"{rate:.1f}% valid JSON ({stats['pass']}/{total}) | ReqFail {stats['req_fail']}"
        )

    print(f"\nSaved CSV:\n- {rows_csv}\n- {summary_csv}")


if __name__ == "__main__":
    main()
