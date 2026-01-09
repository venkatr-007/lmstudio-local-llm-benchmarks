import json
import re
import asyncio
import requests
import csv
import os
from datetime import datetime
from collections import defaultdict, Counter
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

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def extract_first_json_object(text: str):
    """
    Finds the first balanced JSON object in the text and returns it as a string.
    Returns None if no balanced object is found.
    """
    s = text
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return None


def try_parse_json(text: str):
    """
    Returns (ok: bool, obj_or_err)
    """
    try:
        obj = json.loads(text)
        return True, obj
    except Exception as e:
        return False, e


def repair_to_single_json(text: str):
    """
    Returns (repaired_text_or_None, reason_string)
    """
    t = strip_think(text)

    ok, _ = try_parse_json(t)
    if ok:
        return t, "already_valid_after_strip_think"

    first = extract_first_json_object(t)
    if first is None:
        return None, "no_json_object_found"

    ok, _ = try_parse_json(first)
    if ok:
        return first, "extracted_first_json_object"

    return None, "json_object_found_but_invalid"


def make_payload(prompt: str):
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


def run_once(prompt: str):
    r = requests.post(URL, json=make_payload(prompt), timeout=TIMEOUT_S)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    return text


def is_valid_json(text: str) -> bool:
    ok, _ = try_parse_json(text)
    return ok


def safe_run_once(prompt: str):
    """
    Never raises; returns (ok, text_or_errstr)
    """
    try:
        return True, run_once(prompt)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


async def run_concurrent(prompt: str, n: int):
    tasks = [asyncio.to_thread(safe_run_once, prompt) for _ in range(n)]
    results = await asyncio.gather(*tasks)

    rows = []
    for i, (ok, payload) in enumerate(results):
        if not ok:
            rows.append(
                {
                    "request_index": i,
                    "raw_text": None,
                    "raw_valid": False,
                    "raw_error": payload,
                    "repaired_text": None,
                    "repaired_valid": False,
                    "repair_reason": "request_failed",
                }
            )
            continue

        out = payload
        raw_valid = is_valid_json(out)

        repaired_text, reason = repair_to_single_json(out)
        repaired_valid = (repaired_text is not None) and is_valid_json(repaired_text)

        rows.append(
            {
                "request_index": i,
                "raw_text": out,
                "raw_valid": raw_valid,
                "raw_error": None,
                "repaired_text": repaired_text,
                "repaired_valid": repaired_valid,
                "repair_reason": reason,
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

    rows_csv = os.path.join(RESULTS_DIR, f"phase4_json_repaired_trials_{ts}.csv")
    summary_csv = os.path.join(RESULTS_DIR, f"phase4_json_repaired_summary_{ts}.csv")

    print("Warming up...")
    warmup()
    print("Warm-up done.\n")

    summary = defaultdict(lambda: {"raw_pass": 0, "raw_fail": 0, "rep_pass": 0, "rep_fail": 0, "req_fail": 0})
    reason_counts = defaultdict(Counter)

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
                "raw_valid",
                "repaired_valid",
                "repair_reason",
                "raw_error",
            ],
        )
        w_rows.writeheader()

        for m in PROMPT_MULTIPLIERS:
            prompt = BASE_PROMPT * m
            for c in CONCURRENCY_LEVELS:
                print(f"\n=== Prompt {m}x | Concurrency {c} | Trials {TRIALS} ===")
                example_printed = False

                for t in range(TRIALS):
                    batch = asyncio.run(run_concurrent(prompt, c))

                    for r in batch:
                        if r["repair_reason"] == "request_failed":
                            summary[(m, c)]["req_fail"] += 1
                            summary[(m, c)]["raw_fail"] += 1
                            summary[(m, c)]["rep_fail"] += 1
                            reason_counts[(m, c)][r["repair_reason"]] += 1
                        else:
                            if r["raw_valid"]:
                                summary[(m, c)]["raw_pass"] += 1
                            else:
                                summary[(m, c)]["raw_fail"] += 1

                            if r["repaired_valid"]:
                                summary[(m, c)]["rep_pass"] += 1
                            else:
                                summary[(m, c)]["rep_fail"] += 1

                            reason_counts[(m, c)][r["repair_reason"]] += 1

                        w_rows.writerow(
                            {
                                "timestamp": ts,
                                "model": MODEL,
                                "prompt_multiplier": m,
                                "concurrency": c,
                                "trial_index": t,
                                "request_index": r["request_index"],
                                "raw_valid": r["raw_valid"],
                                "repaired_valid": r["repaired_valid"],
                                "repair_reason": r["repair_reason"],
                                "raw_error": r.get("raw_error"),
                            }
                        )

                    f_rows.flush()

                    raw_pass = summary[(m, c)]["raw_pass"]
                    raw_fail = summary[(m, c)]["raw_fail"]
                    rep_pass = summary[(m, c)]["rep_pass"]
                    rep_fail = summary[(m, c)]["rep_fail"]
                    req_fail = summary[(m, c)]["req_fail"]

                    print(
                        f"Progress: Raw {raw_pass}/{raw_pass+raw_fail} OK | "
                        f"Repaired {rep_pass}/{rep_pass+rep_fail} OK | "
                        f"ReqFail {req_fail}"
                    )

                    if not example_printed:
                        first_raw_fail = next((x for x in batch if x["repair_reason"] != "request_failed" and not x["raw_valid"]), None)
                        if first_raw_fail is not None:
                            print("\nExample RAW failure output (first failure for this config):")
                            print(first_raw_fail["raw_text"])
                            print("---")
                            if first_raw_fail["repaired_valid"]:
                                print(f"Repaired OK ({first_raw_fail['repair_reason']}):")
                                print(first_raw_fail["repaired_text"])
                            else:
                                print(f"Repaired FAILED ({first_raw_fail['repair_reason']})")
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
                "raw_pass",
                "raw_fail",
                "raw_rate_percent",
                "repaired_pass",
                "repaired_fail",
                "repaired_rate_percent",
                "request_failures",
                "top_repair_reasons",
            ],
        )
        w_sum.writeheader()

        for (m, c), stats in sorted(summary.items()):
            raw_total = stats["raw_pass"] + stats["raw_fail"]
            rep_total = stats["rep_pass"] + stats["rep_fail"]

            raw_rate = (stats["raw_pass"] / raw_total * 100.0) if raw_total else 0.0
            rep_rate = (stats["rep_pass"] / rep_total * 100.0) if rep_total else 0.0

            top_reasons = reason_counts[(m, c)].most_common(5)
            top_reasons_str = "; ".join([f"{k}:{v}" for k, v in top_reasons])

            w_sum.writerow(
                {
                    "timestamp": ts,
                    "model": MODEL,
                    "prompt_multiplier": m,
                    "concurrency": c,
                    "raw_pass": stats["raw_pass"],
                    "raw_fail": stats["raw_fail"],
                    "raw_rate_percent": round(raw_rate, 3),
                    "repaired_pass": stats["rep_pass"],
                    "repaired_fail": stats["rep_fail"],
                    "repaired_rate_percent": round(rep_rate, 3),
                    "request_failures": stats["req_fail"],
                    "top_repair_reasons": top_reasons_str,
                }
            )

    print("\n=== Final Summary (Raw vs Repaired) ===")
    for (m, c), stats in sorted(summary.items()):
        raw_total = stats["raw_pass"] + stats["raw_fail"]
        rep_total = stats["rep_pass"] + stats["rep_fail"]
        raw_rate = (stats["raw_pass"] / raw_total) * 100 if raw_total else 0
        rep_rate = (stats["rep_pass"] / rep_total) * 100 if rep_total else 0

        print(
            f"Prompt {m}x | Concurrency {c} -> "
            f"Raw: {raw_rate:.1f}% ({stats['raw_pass']}/{raw_total}) | "
            f"Repaired: {rep_rate:.1f}% ({stats['rep_pass']}/{rep_total}) | "
            f"ReqFail: {stats['req_fail']}"
        )

    print(f"\nSaved CSV:\n- {rows_csv}\n- {summary_csv}")


if __name__ == "__main__":
    main()
