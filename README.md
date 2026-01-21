# lmstudio-local-llm-benchmarks

Benchmark scripts for **LM Studio’s OpenAI-compatible local server**.

Primary metrics:
- **TTFT** (time to first token)
- **tokens/sec**
- **aggregate throughput** (overall tokens/sec across runs)

> This repo targets LM Studio’s `/v1/chat/completions` endpoint (OpenAI-compatible). Adjust the base URL/port if your server differs.

---

## Requirements

- Python 3.9+ (recommended)
- LM Studio running locally with **OpenAI-compatible server** enabled

---

## LM Studio setup (one-time)

1. Open **LM Studio**
2. Start the **Local Server (OpenAI compatible)**
3. Confirm the base URL (commonly):
   - `http://localhost:1234/v1`

---

## Python setup

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

If this repo does not include `requirements.txt`, install the packages imported by the scripts (commonly `requests`, `rich`, etc.).

---

## Configure endpoint + model

Most scripts typically require (names may differ per script):
- Server URL, e.g. `http://localhost:1234/v1/chat/completions`
- Model id, e.g. `qwen3-1.7b` (whatever you loaded in LM Studio)

Search the repo for these constants/args and update them as needed:

```powershell
git grep -n "localhost:1234|/v1/chat/completions|MODEL\s*=|URL\s*="
```

---

## Run a benchmark

This repo may contain one or more scripts under `scripts/`.

List what’s available:

```powershell
Get-ChildItem .\scripts -File
```

Run the script you want (example):

```powershell
python .\scripts\<your_script>.py --help
python .\scripts\<your_script>.py
```

> Replace `<your_script>.py` with the actual filename(s) in your repo.

---

## Outputs

Bench outputs often go into a folder like `results/` (CSV/JSON/logs).  
This repo’s `.gitignore` should ignore `results/` so you don’t accidentally commit large or noisy artifacts.

If your scripts write outputs elsewhere, update `.gitignore` accordingly.

---

## Metric notes

- **TTFT (s)**: time from request start → first streamed token received
- **tokens/sec**: completion tokens divided by generation time (excluding TTFT if computed that way in the script)
- **aggregate tokens/sec**: total completion tokens across runs divided by total wall time

---

## Safety / secrets

Before making a repo public:
- do not commit `.env`, API keys, or private credentials
- keep `.venv/` out of git

---

## License

MIT License
