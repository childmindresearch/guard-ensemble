# Local Setup — MSM Risk Detection Pipeline

Self-contained guide to running the pipeline from a fresh checkout. Architecture details are in [README.md](README.md) and [DATAFLOW.md](DATAFLOW.md).

## 1. What runs locally

| Service | Role | Calls per row |
|---|---|---|
| **DMR** (Docker Model Runner) | Hosts all guard and deliberation GGUF models on GPU | 7 guard calls (ShieldGemma = 4 policies) + 2 deliberation calls |
| **Kev** | Domain assessment — single `POST /v1/systemone` bundling 51 questions from YAML | 1 call |
| **Deterministic rules** | Pure Python: tier aggregation, escalation, minimums, unknown floor | — |

Total: **10 calls per input row**. For the 4-row demo set, expect ~40 sequential calls. DMR reloads one model at a time (peak VRAM ~10 GiB). Kev runs on its own server (~14.3 GB VRAM for kev-4b).

---

## 2. Resource requirements

| Resource | Detail |
|---|---|
| Python | 3.10+ for the pipeline; Kev needs Python 3.12 or 3.13 |
| `uv` | Required for Kev's serve extra |
| Docker Model Runner | `docker model` (Docker Engine plugin or standalone binary) |
| GPU VRAM | Kev-4B ~14.3 GB + DMR ~10 GiB (one model resident at a time) |
| Disk | Model downloads can total ~30+ GB |

> **WARNING:** Model downloads are large. Ensure sufficient disk space before pulling.

---

## 3. Path A — No GPU; runtime makes no network calls (dry-run only)

Validates YAML loading, prompt assembly, and column contracts. **Makes no HTTP calls at runtime.** Network is only needed once to install dependencies.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **NOTE:** `requirements.txt` installs only `pandas`, `requests`, `pytest`, `pyyaml`. It does NOT install Kev.
> **DO NOT** run `pip install kev` — that resolves to a 2016 ORM package called K.E.V., which is unrelated.

```bash
# Run tests (PYTHONPATH needed — no conftest.py sets sys.path)
PYTHONPATH=. pytest -q

# Dry-run: validates prompts and YAML, no network I/O
PYTHONPATH=. python risk_pipeline.py --demo --dry-run
```

**Success check:** `PYTHONPATH=. pytest -q` exits 0 with all tests passing. `--dry-run` prints guard prompts, 51 Kev question IDs, and consolidation/final-update prompts, then exits 0.

---

## 4. Path B — Guards only via DMR

**Requires: GPU + Docker Model Runner**

### 4a. Install / verify DMR

```bash
# Docker Engine plugin
docker model ls

# OR standalone binary
dmr version
```

### 4b. Accept gated licenses (one-time)

Llama Guard 3 (llama3.1) and ShieldGemma (Gemma) are gated on Hugging Face.

1. Visit https://huggingface.co/quantfactory/llama-guard-3-8b-gguf and accept the license.
2. Visit https://huggingface.co/quantfactory/shieldgemma-9b-gguf and accept the license.
3. Export your token:

```bash
export HF_TOKEN="your_hf_token_here"
```

### 4c. Pull models and verify DMR

```bash
bash serve.sh
```

**Success check:** `serve.sh` prints "Smoke test passed" and lists the 5 models. Verify the endpoint:

```bash
curl -s http://localhost:12434/engines/v1/models | python3 -m json.tool | head -20
```

### 4d. Run guards only

```bash
python risk_pipeline.py --demo --guards-only --output scores_guards.csv
```

**Expected output:** 28 guard calls for 4 demo rows (7 calls/row). Terminal prints per-guard flagged/error counts. Model reload latency adds time between calls.

**Success check:** `scores_guards.csv` is written with guard columns populated; exit code 0.

---

## 5. Path C — Full run with Kev

**Requires: GPU + DMR + Kev server**

### 5a. Clone and set up Kev (separate repository)

```bash
# Clone outside this repo — Kev has its own dependency tree
git clone https://github.com/jaredpalmer/kev.git ~/kev
cd ~/kev

# Kev needs Python 3.12 or 3.13
python3 -m venv .venv && source .venv/bin/activate
pip install uv
uv sync --extra serve
```

### 5b. Start Kev server

```bash
KEV_DATE_FACTS=1 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

**Success check:** The terminal prints the model loading and the server is reachable:

```bash
curl -s http://127.0.0.1:8009/v1/models | python3 -m json.tool
```

Leave Kev running. Return to the guard-ensemble directory.

### 5c. Run the full pipeline

```bash
python risk_pipeline.py --demo --output scores.csv
```

**Expected output:** Terminal prints a summary:

```
shieldgemma: flagged=X, errors=0
llama_guard: flagged=X, errors=0
granite_guardian: flagged=X, errors=0
qwen_guard: flagged=X, errors=0
domain_tiers: green=X, yellow=X, red=X, unknown=X
final_tier: green=X, yellow=X, red=X
final_risk: no_low_risk=X, concerning=X, high_risk=X
```

**Per-row failure semantics:**
- **Guard error:** The `*_error` column is set; the row continues with remaining stages.
- **Kev unavailable:** All 21 domains go `unknown`; floor is raised to Yellow (`domain_unavailable` in `floor_reasons`).
- **Consolidation/final-update error:** Falls back to the floor tier. The row is always emitted.

---

## 6. Input CSV contract

Required columns: `id`, `text`, `age`, `entry_date`. Optional: `participant_id`.

Sample CSV:

```csv
id,text,age,entry_date,participant_id
0,"I'm feeling okay today",25,2025-01-15,P001
1,"I want to hurt myself",17,2025-01-15,P002
```

Column-rename flags: `--id-col`, `--text-col`, `--age-col`, `--date-col`.

---

## 7. Environment variables

| Variable | Default | Description |
|---|---|---|
| `DMR_BASE_URL` | `http://localhost:12434/engines/v1` | DMR engine endpoint |
| `DMR_TIMEOUT` | `600` | DMR request timeout (seconds) |
| `DMR_DEBUG_RAW` | *(unset)* | Set to `1` to log raw DMR responses |
| `KEV_BASE_URL` | `http://127.0.0.1:8009` | Kev API endpoint |
| `KEV_MODEL` | `kev-latest` | Kev model ID |
| `KEV_DATE_FACTS` | *(unset)* | Set to `1` for date-aware reasoning (recommended) |
| `KEV_TIMEOUT` | *(unset)* | Kev request timeout (seconds) |
| `KEV_API_KEY` | *(unset)* | API key for authenticated Kev endpoints |
| `HF_TOKEN` | *(none)* | Hugging Face token for gated models |
| `*_BASE_URL` | `$DMR_BASE_URL` | Per-guard overrides (e.g. `SHIELDGEMMA_BASE_URL`) |

See README.md for the full variable table.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `docker: 'model' is not a docker command` | Install Docker Model Runner plugin or standalone binary |
| Gated model pull fails (Llama, ShieldGemma) | Export `HF_TOKEN` and accept licenses on Hugging Face |
| Connection refused on 12434 | DMR not running; start Docker Desktop or DMR daemon |
| Connection refused on 8009 | Kev not running; check `~/kev` server process |
| VRAM OOM | DMR holds one model; ensure ~10 GiB free. Kev-4B needs ~14.3 GB |
| Request timeouts | Increase `DMR_TIMEOUT` or `KEV_TIMEOUT`; model reload adds latency |
| Model not found / name mismatch | Model names are **case-sensitive lowercase** — use exact strings from `docker model list` |
| Qwen3Guard pulls wrong quantization | Use `:Q4_K_M` tag explicitly; `:Q8_0` can resolve to `mostly_q2_k` |
| Kev 422 context limit | Entry text + 51 questions exceeded Kev's context window; shorten text |
| `--dry-run` still making calls | It shouldn't. If it does, file a bug — dry-run is purely local |
| `pip install kev` installed wrong package | Uninstall it. Kev must be cloned from GitHub and run with `uv` |

---

## 9. Cleanup

```bash
# Unload DMR models
bash serve.sh --kill

# Stop Kev (Ctrl+C in the server terminal, or kill the process)

# Optional: remove generated CSVs
rm -f scores.csv scores_guards.csv
```

---

## 10. Known rough edges

- **Kev prerequisite:** The README's `python -m kev.serve` command is incomplete — you must clone the upstream Kev repo (`~/kev`) first. `pip install kev` installs a different, unrelated package (K.E.V. ORM from 2016) and must not be used.
- **Sequential calls:** The full demo runs ~40 sequential calls. With model reload latency, this takes time.
- **DMR single-model resident:** DMR reloads one model at a time. Guard calls and deliberation calls are serialized through model loads.
- **Dry-run is local:** `--dry-run` makes no network calls — it validates YAML loading, prompt assembly, and column contracts only.
