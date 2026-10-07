# MSM Risk Detection Pipeline

Two-stage risk pipeline: guard fan-out across four models, then a deliberation call per row, with results emitted as CSV.

## Architecture

Two stages per input row:

1. **Guard fan-out** — four guard models are called (serial HTTP calls to a single Docker Model Runner instance). Each model parses its response independently.
2. **Deliberation** — a general-purpose LLM (the deliberator) is called once per row, receiving the four guard verdicts and producing a final risk assessment.

Results are written as a single CSV row per input text.

DMR hosts one resident model at a time. Each guard or deliberation call triggers a model reload, which adds latency but keeps peak VRAM low.

## Models

All four guards plus the deliberator are pulled as GGUF models. Model names are **case-sensitive lowercase** — use the exact strings reported by `docker model list`.

| Role | Model (`docker model list` name) | Quantization | Size | License |
|---|---|---|---|---|
| ShieldGemma | `huggingface.co/quantfactory/shieldgemma-9b-gguf:Q8_0` | Q8_0 | 9 B | Gemma |
| Llama Guard 3 | `huggingface.co/quantfactory/llama-guard-3-8b-gguf:Q8_0` | Q8_0 | 8 B | llama3.1 |
| Granite Guardian | `huggingface.co/ibm-granite/granite-guardian-3.3-8b-gguf:Q8_0` | Q8_0 | 8 B | Apache-2.0 |
| Qwen3Guard | `huggingface.co/geoffmunn/Qwen3Guard-Gen-8B-GGUF:Q4_K_M` | Q4_K_M | 8 B | Apache-2.0 |
| Deliberator | `huggingface.co/qwen/qwen3-4b-gguf:Q8_0` | Q8_0 | 4 B | Apache-2.0 |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DMR_BASE_URL` | `http://localhost:12434/engines/v1` | Base URL for the DMR engine API |
| `DMR_TIMEOUT` | `600` | Request timeout in seconds |
| `SHIELDGEMMA_BASE_URL` | `$DMR_BASE_URL` | Override ShieldGemma endpoint (vLLM migration) |
| `LLAMA_GUARD_BASE_URL` | `$DMR_BASE_URL` | Override Llama Guard endpoint (vLLM migration) |
| `GRANITE_GUARDIAN_BASE_URL` | `$DMR_BASE_URL` | Override Granite Guardian endpoint (vLLM migration) |
| `QWEN_GUARD_BASE_URL` | `$DMR_BASE_URL` | Override Qwen3Guard endpoint (vLLM migration) |
| `DELIBERATION_BASE_URL` | `$DMR_BASE_URL` | Override Deliberator endpoint (vLLM migration) |

Per-guard `*_BASE_URL` overrides let you route individual models to separate inference servers while leaving the rest on DMR.

## Prerequisites

- Python 3.10+
- `pip install -r requirements.txt`
- GPU with at least ~10 GiB VRAM (DMR loads one model at a time; peak VRAM stays ~10 GiB)

## Gated models

Llama Guard 3 8B (llama3.1 license) and ShieldGemma 9B (Gemma license) require an `HF_TOKEN` and prior license acceptance on Hugging Face. Granite Guardian, Qwen3Guard, and the Deliberator are Apache-2.0 and ungated.

## Run

```bash
bash serve.sh                    # probe DMR, pull models, smoke-test
python risk_pipeline.py --input texts.csv --output scores.csv
```

To unload the models:

```bash
bash serve.sh --kill
```

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--input` | *(none)* | Path to input CSV |
| `--text-col` | `text` | Column name containing the text to score |
| `--output` | `scores.csv` | Path to output CSV |
| `--dry-run` | `false` | Print built prompts only, no network calls |
| `--demo` | `false` | Use built-in demo texts (auto-enabled when `--input` is omitted) |
| `--skip-deliberation` | `false` | Skip the deliberation stage; emit only guard scores |
| `--deliberation-prompt` | `prompts/deliberator.txt` | Path to custom deliberation prompt template file |

## Deliberator prompt

The deliberator prompt is loaded from `prompts/deliberator.txt` by default. Override with `--deliberation-prompt path/to/custom.txt`.

The prompt file is a `string.Template` with two placeholders:

| Placeholder | Substituted with |
|---|---|
| `$text` | The original input text |
| `$scores_block` | Guard model ratings (bullet list, or "none available" if all guards failed) |

The file must contain the three risk level definitions (`high_risk`, `concerning`, `no_low_risk`) or the parser will reject the deliberator's output.

**Example**:

```bash
python risk_pipeline.py --input texts.csv --deliberation-prompt my_prompt.txt
```

## Column dictionary

| Column | Description |
|---|---|
| `id` | Zero-based index of the input text |
| `text` | The original input text |
| `shieldgemma_unsafe` | 1 if any ShieldGemma policy flagged the text, else 0 |
| `shieldgemma_label` | `safe` or `unsafe` |
| `shieldgemma_category` | Pipe-separated list of violated policies, or `None` |
| `shieldgemma_p_dangerous_content` | 1 if dangerous_content policy violated, else 0 |
| `shieldgemma_p_harassment` | 1 if harassment policy violated, else 0 |
| `shieldgemma_p_hate_speech` | 1 if hate_speech policy violated, else 0 |
| `shieldgemma_p_sexually_explicit` | 1 if sexually_explicit policy violated, else 0 |
| `llama_guard_unsafe` | 1 or 0 |
| `llama_guard_label` | `safe` or `unsafe` |
| `llama_guard_category` | Comma-separated S-codes (e.g. `S1,S3`), or `None` |
| `granite_guardian_unsafe` | 1 or 0 |
| `granite_guardian_label` | `safe` or `unsafe` |
| `granite_guardian_category` | `harm` or `None` |
| `qwen_guard_unsafe` | 1 or 0 |
| `qwen_guard_label` | `Safe`, `Unsafe`, or `Controversial` |
| `qwen_guard_category` | Comma-separated categories, or `None` |
| `<guard>_error` | Error message if that guard failed (one column per guard) |
| `errors` | Semicolon-joined `model:message` pairs for any failed guards |
| `deliberation_risk` | Final risk assessment from the deliberator, conditioned on all guard verdicts |

## vLLM migration

For production Linux deployments, replace DMR with per-model vLLM servers. No code changes are needed — set the five `*_BASE_URL` environment variables to point at the corresponding vLLM ports:

```bash
export SHIELDGEMMA_BASE_URL=http://localhost:8001/v1
export LLAMA_GUARD_BASE_URL=http://localhost:8002/v1
export GRANITE_GUARDIAN_BASE_URL=http://localhost:8003/v1
export QWEN_GUARD_BASE_URL=http://localhost:8004/v1
export DELIBERATION_BASE_URL=http://localhost:8005/v1
```

Each vLLM server hosts a single model. Start them independently (e.g. via systemd or docker-compose) and the pipeline routes automatically.

## Caveats

- **DMR reload cost.** DMR keeps one model resident. Each guard or deliberation call reloads the target model, adding latency proportional to model size. This is accepted for the POC.
- **VRAM comparison.** DMR peaks at roughly 10 GiB VRAM (one model resident). Five concurrent vLLM servers hold all models simultaneously, peaking at roughly 32 GiB.
- **Qwen3Guard quantization.** The `geoffmunn/Qwen3Guard-Gen-8B-GGUF:Q8_0` tag can resolve to `mostly_q2_k` instead of Q8_0. The pipeline pulls `:Q4_K_M` explicitly and gates on the reported quantization in `docker model list`.
- **ShieldGemma costs 4 calls/text.** ShieldGemma evaluates one policy per request, and the pipeline queries all 4 policies (dangerous content, harassment, hate speech, sexually explicit) for every input.
- **Qwen Controversial preserved.** The `Controversial` verdict is kept in `qwen_guard_label` and is not collapsed into safe/unsafe (the `_unsafe` flag is 0 for Controversial).
- **Deliberation adds runtime.** The per-row deliberation call adds ~25% overhead per row on a 4B model. Use `--skip-deliberation` if guard-only scores suffice.

## License

Apache-2.0
