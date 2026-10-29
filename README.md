# MSM Risk Detection Pipeline

Five-stage risk pipeline: guard fan-out across four models, 21 domain prompts
(3 runs each), deterministic rule phases, then two LLM consolidation steps.
Results emitted as CSV.

## Architecture

Five stages per input row:

1. **Guard fan-out** — four guard models are called (serial HTTP calls to a single Docker Model Runner instance). Each model parses its response independently.
2. **Domain prompts** — 21 risk-domain prompts are rendered and called N times each (default N=3). Results aggregate via severity-max tier, timing tie-break, and flag union.
3. **Deterministic rules** — five phases run: highest domain tier, escalation (Yellow-to-Red modifiers), minimums (minor abuse, lifetime history), unknown-domain floor, and floor resolution.
4. **Consolidation** — an LLM consolidates guard scores, domain tiers, and rule results into a single tier. Monotonic clamp enforces the floor tier.
5. **Final update** — a second LLM pass may raise the tier further. Monotonic clamp enforces the consolidated tier as floor.

Results are written as a single CSV row per input text.

DMR hosts one resident model at a time. Each guard or LLM call triggers a model reload, which adds latency but keeps peak VRAM low.

## Models

All four guards plus the consolidation/final-update model are pulled as GGUF models. Model names are **case-sensitive lowercase** — use the exact strings reported by `docker model list`.

| Role | Model (`docker model list` name) | Quantization | Size | License |
|---|---|---|---|---|
| ShieldGemma | `huggingface.co/quantfactory/shieldgemma-9b-gguf:Q8_0` | Q8_0 | 9 B | Gemma |
| Llama Guard 3 | `huggingface.co/quantfactory/llama-guard-3-8b-gguf:Q8_0` | Q8_0 | 8 B | llama3.1 |
| Granite Guardian | `huggingface.co/ibm-granite/granite-guardian-3.3-8b-gguf:Q8_0` | Q8_0 | 8 B | Apache-2.0 |
| Qwen3Guard | `huggingface.co/geoffmunn/Qwen3Guard-Gen-8B-GGUF:Q4_K_M` | Q4_K_M | 8 B | Apache-2.0 |
| Consolidation/Final | `huggingface.co/qwen/qwen3-4b-gguf:Q8_0` | Q8_0 | 4 B | Apache-2.0 |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DMR_BASE_URL` | `http://localhost:12434/engines/v1` | Base URL for the DMR engine API |
| `DMR_TIMEOUT` | `600` | Request timeout in seconds |
| `SHIELDGEMMA_BASE_URL` | `$DMR_BASE_URL` | Override ShieldGemma endpoint (vLLM migration) |
| `LLAMA_GUARD_BASE_URL` | `$DMR_BASE_URL` | Override Llama Guard endpoint (vLLM migration) |
| `GRANITE_GUARDIAN_BASE_URL` | `$DMR_BASE_URL` | Override Granite Guardian endpoint (vLLM migration) |
| `QWEN_GUARD_BASE_URL` | `$DMR_BASE_URL` | Override Qwen3Guard endpoint (vLLM migration) |
| `DELIBERATION_BASE_URL` | `$DMR_BASE_URL` | Override consolidation/final-update endpoint |
| `DELIBERATION_MAX_TOKENS` | `2048` | Max tokens for consolidation/final-update responses |
| `DELIBERATION_DISABLE_THINKING` | `1` | Disable thinking mode (set to `0` to enable) |
| `CUSTOM_PROMPT_MAX_TOKENS` | `512` | Max tokens for domain-prompt responses |

Per-guard `*_BASE_URL` overrides let you route individual models to separate inference servers while leaving the rest on DMR.

## Prerequisites

- Python 3.10+
- `pip install -r requirements.txt`
- GPU with at least ~10 GiB VRAM (DMR loads one model at a time; peak VRAM stays ~10 GiB)

## Gated models

Llama Guard 3 8B (llama3.1 license) and ShieldGemma 9B (Gemma license) require an `HF_TOKEN` and prior license acceptance on Hugging Face. Granite Guardian, Qwen3Guard, and the consolidation model are Apache-2.0 and ungated.

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
| `--id-col` | `id` | Column name for entry ID |
| `--text-col` | `text` | Column name containing the text to score |
| `--age-col` | `age` | Column name for participant age |
| `--date-col` | `entry_date` | Column name for entry date |
| `--output` | `scores.csv` | Path to output CSV |
| `--dry-run` | `false` | Print built prompts only, no network calls |
| `--demo` | `false` | Use built-in demo texts (auto-enabled when `--input` is omitted) |
| `--guards-only` | `false` | Run guards only, skip domain prompts through final update |
| `--custom-prompts-dir` | `prompts/domains` | Directory for domain prompt files |
| `--custom-runs` | `3` | Number of runs per domain prompt |
| `--custom-temperature` | `0.3` | Temperature for custom-prompt calls |
| `--consolidation-prompt` | `prompts/consolidation.txt` | Path to custom consolidation prompt template |
| `--final-update-prompt` | `prompts/final_update.txt` | Path to custom final-update prompt template |

## Prompt contracts

### Domain prompts (`prompts/domains/*.txt`)

Each domain prompt file is a `string.Template` with placeholders `$text`, `$age`, `$entry_date`. At runtime the prelude (`prompts/_shared/rubric_prelude.txt`) is prepended and the output schema (`prompts/_shared/output_schema.txt`) is appended.

The model must respond with strict JSON:

```json
{
  "domain": "suicidality",
  "tier": "green",
  "timing": "unclear",
  "flags": [],
  "evidence": "brief justification"
}
```

- **tier**: one of `green`, `yellow`, `red`.
- **timing**: one of `historical_resolved`, `historical_unresolved`, `recent`, `current`, `imminent`, `unclear`.
- **flags**: zero or more from the flag vocabulary (escalation flags: `lethal_means_access`, `current_intoxication`, `alone_no_support`; minimum flags: `minor_abuse_disclosure`, `past_suicide_attempt`, `past_self_harm`, `past_sexual_assault`, `past_psychosis`, `past_eating_disorder`).

### Consolidation prompt (`prompts/consolidation.txt`)

Placeholders: `$text`, `$context_block`, `$guard_block`, `$domain_block`, `$rule_block`.

The model responds with:

```json
{
  "tier": "yellow",
  "rationale": "brief justification"
}
```

### Final-update prompt (`prompts/final_update.txt`)

Placeholders: `$text`, `$context_block`, `$guard_block`, `$domain_block`, `$rule_block`, `$consolidation_block`.

Same JSON response schema as consolidation.

**Example** — override consolidation and final-update prompts:

```bash
python risk_pipeline.py --input texts.csv \
  --consolidation-prompt my_consolidation.txt \
  --final-update-prompt my_final.txt
```

## Column dictionary

| Column | Description |
|---|---|
| `id` | Zero-based index of the input text |
| `text` | The original input text |
| `participant_id` | Participant identifier (if present in input) |
| `age` | Participant age |
| `entry_date` | Date the entry was written |
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
| `domain_results` | JSON object: per-domain tier, timing, flags, evidence |
| `highest_domain_tier` | Highest tier among known domain ratings (uses enum ordering green < yellow < red, not string ordering). Set to `unknown` when no domains returned a tier. |
| `escalated_tier` | Tier after escalation rules applied |
| `escalation_reasons` | Comma-separated escalation triggers |
| `minimum_tier` | Tier after minimum-rating rules applied |
| `minimum_reasons` | Comma-separated minimum triggers |
| `floor_tier` | Final deterministic floor before LLM consolidation |
| `floor_reasons` | Comma-separated reasons the floor was raised (e.g. `domain_unavailable`, escalation flags, minimum triggers). Empty string when no rule raised the floor. |
| `consolidated_tier` | Tier after consolidation LLM pass |
| `consolidation_rationale` | Rationale from consolidation LLM |
| `final_tier` | Tier after final-update LLM pass |
| `final_rationale` | Rationale from final-update LLM |
| `final_adjusted` | 1 if final-update tier was monotonic-clamped |
| `final_risk` | Legacy label: `no_low_risk`, `concerning`, or `high_risk` |
| `domain_incomplete` | 1 if any domain returned unknown |
| `custom_prompt_errors` | Semicolon-joined domain-prompt error messages |
| `consolidation_error` | Error from consolidation phase (empty on success) |
| `final_update_error` | Error from final-update phase (empty on success) |
| `errors` | Semicolon-joined `model:message` pairs for any failed guards |

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

- **DMR reload cost.** DMR keeps one model resident. Each guard or LLM call reloads the target model, adding latency proportional to model size. This is accepted for the POC.
- **VRAM comparison.** DMR peaks at roughly 10 GiB VRAM (one model resident). Five concurrent vLLM servers hold all models simultaneously, peaking at roughly 32 GiB.
- **Qwen3Guard quantization.** The `geoffmunn/Qwen3Guard-Gen-8B-GGUF:Q8_0` tag can resolve to `mostly_q2_k` instead of Q8_0. The pipeline pulls `:Q4_K_M` explicitly and gates on the reported quantization in `docker model list`.
- **ShieldGemma costs 4 calls/text.** ShieldGemma evaluates one policy per request, and the pipeline queries all 4 policies (dangerous content, harassment, hate speech, sexually explicit) for every input.
- **Qwen Controversial preserved.** The `Controversial` verdict is kept in `qwen_guard_label` and is not collapsed into safe/unsafe (the `_unsafe` flag is 0 for Controversial).
- **Custom-prompt runtime.** The domain-prompt stage dominates runtime: 21 domains × N runs (default N=3) = 63 calls per entry. Each call reloads the model via DMR. Lower `--custom-runs` to reduce cost.
- **JSON-column size.** The `domain_results` column stores a full JSON object per row (21 domains with tier, timing, flags, evidence). CSV row sizes can be large.
- **Unknown-domain Yellow floor.** If any domain prompt fails (network error, parse failure), that domain is marked `unknown` and the deterministic floor phase raises to at least Yellow, emitting `domain_unavailable` in `floor_reasons`. This is conservative: uncertainty cannot be safely treated as low risk. The same floor applies when the custom-prompt stage returns empty results (all domains unavailable).
- **highest_domain_tier uses enum ordering.** Tier comparison uses green < yellow < red (enum order), not alphabetical string ordering. When no domains return a known tier, this column is set to `unknown`.
- **Empty custom-prompt stage is not safe.** If the custom-prompt stage fails entirely (exception, empty domain results), `domain_incomplete` is set and the floor is raised to Yellow. An empty stage is never treated as safe coverage.
- **14-day cross-entry rule.** The rubric requires escalating a domain to Red if the same domain is Yellow in 2+ entries within 14 days for a participant. This rule is applied downstream (after the pipeline), across all rows — the pipeline processes entries independently.

## License

Apache-2.0
