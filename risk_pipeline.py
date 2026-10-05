"""Linear risk-detection pipeline: iterate texts x guards, write CSV."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import requests

from guards import (
    DELIBERATION_BASE_URL,
    DELIBERATION_MAX_TOKENS,
    DELIBERATION_MODEL_ID,
    DELIBERATION_PATH,
    GUARDS,
    RISK_LEVELS,
    deliberation_build,
    deliberation_parse,
    shieldgemma_aggregate,
)

# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

DEMO_TEXTS = [
    "What's the weather like today?",
    "How do I make a bomb at home?",
    "Write a poem about the sunset",
    "I hate everyone in this town, they're all terrible people and deserve to suffer.",
]


# ---------------------------------------------------------------------------
# Network seam
# ---------------------------------------------------------------------------

DMR_TIMEOUT = int(os.environ.get("DMR_TIMEOUT", "600"))
_DMR_DEBUG_RAW = os.environ.get("DMR_DEBUG_RAW") == "1"


def _log_raw(guard: str, text: str, raw, finish_reason: str | None = None) -> None:
    """Append a raw-response record to raw_responses.jsonl (fire-and-forget)."""
    if not _DMR_DEBUG_RAW:
        return
    try:
        with open("raw_responses.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"guard": guard, "text": text, "raw": raw, "finish_reason": finish_reason}) + "\n")
    except Exception:
        pass


def _join_url(base: str, path: str) -> str:
    """Join base URL and path, stripping duplicate /v1 segment."""
    if base.endswith("/v1") and path.startswith("/v1"):
        base = base[:-3]  # strip trailing /v1 from base
    return f"{base}{path}"


def call_llm(guard, text):
    """Seam to replace for local models.

    The only function in this module that performs network I/O.
    """
    prompt = guard.build(text)
    url = _join_url(guard.base_url, guard.path)

    if isinstance(prompt, list):
        # ShieldGemma — one request per policy prompt
        results = []
        for p in prompt:
            body = {
                "model": guard.model_id,
                "temperature": 0,
                "max_tokens": guard.max_tokens,
                "prompt": p,
            }
            resp = requests.post(url, json=body, timeout=DMR_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            _log_raw(guard.name, text, data, data["choices"][0].get("finish_reason"))
            results.append(data["choices"][0]["text"])
        return results

    is_chat = guard.path == "/v1/chat/completions"

    if is_chat:
        body = {
            "model": guard.model_id,
            "temperature": 0,
            "max_tokens": guard.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        body = {
            "model": guard.model_id,
            "temperature": 0,
            "max_tokens": guard.max_tokens,
            "prompt": prompt,
        }

    resp = requests.post(url, json=body, timeout=DMR_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    _log_raw(guard.name, text, data, data["choices"][0].get("finish_reason"))
    if is_chat:
        return data["choices"][0]["message"]["content"]
    return data["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Deliberation stage — runs after all guards, one call per row
# ---------------------------------------------------------------------------

def call_deliberator(text: str, guard_scores: dict) -> str:
    """Call the general-purpose deliberator model for a single row.

    Network function, same seam discipline as call_llm.
    """
    prompt = deliberation_build(text, guard_scores)
    body = {
        "model": DELIBERATION_MODEL_ID,
        "temperature": 0,
        "max_tokens": DELIBERATION_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(
        _join_url(DELIBERATION_BASE_URL, DELIBERATION_PATH), json=body, timeout=DMR_TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()
    _log_raw("deliberator", text, data, data["choices"][0].get("finish_reason"))
    return data["choices"][0]["message"]["content"]


def _guard_scores_from_row(row) -> dict:
    """Pull guard rating columns out of a DataFrame row, dropping errors/NaN."""
    return {
        k: v
        for k, v in row.items()
        if k not in ("id", "text", "errors", "deliberation_risk")
        and not k.endswith("_error")
        and pd.notna(v)
        and v != ""
    }


def run_deliberation(df):
    """Add a `deliberation_risk` column: one deliberator call per row."""
    risks = []
    for _, row in df.iterrows():
        scores = _guard_scores_from_row(row)
        failed = [k for k in ("shieldgemma", "llama_guard", "granite_guardian", "qwen_guard")
                  if pd.notna(row.get(f"{k}_error"))]
        if failed:
            scores["_guard_models_unavailable"] = ", ".join(failed)
        try:
            risks.append(deliberation_parse(call_deliberator(row["text"], scores)))
        except Exception as e:
            risks.append(f"error: {e}")
    df = df.copy()
    df["deliberation_risk"] = risks
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(texts, output_path, skip_deliberation=False):
    rows = []
    for i, text in enumerate(texts):
        row = {"id": i, "text": text}
        for guard in GUARDS:
            try:
                raw = call_llm(guard, text)
                if isinstance(raw, list):  # ShieldGemma
                    parsed = [guard.parse(r) for r in raw]
                    row.update(shieldgemma_aggregate(parsed))
                else:
                    row.update(guard.parse(raw))
            except Exception as e:
                row[f"{guard.name}_error"] = str(e)
        rows.append(row)

    df = pd.DataFrame(rows)

    # errors column: semicolon-joined model:message pairs
    error_cols = [c for c in df.columns if c.endswith("_error")]
    if error_cols:
        df["errors"] = df[error_cols].apply(
            lambda r: ";".join(
                f"{col.replace('_error', '')}:{val}"
                for col, val in r.items()
                if val
            ),
            axis=1,
        )

    if not skip_deliberation:
        df = run_deliberation(df)

    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(texts):
    for text in texts:
        for guard in GUARDS:
            prompt = guard.build(text)
            if isinstance(prompt, list):
                for p in prompt:
                    print(f"[{guard.name}] {p}")
            else:
                print(f"[{guard.name}] {prompt}")
        print(f"[deliberator] {deliberation_build(text, {'_note': 'dry-run placeholder scores'})}")


# ---------------------------------------------------------------------------
# Per-model summary
# ---------------------------------------------------------------------------

def _summary(df):
    for guard in GUARDS:
        unsafe_key = guard.name + "_unsafe"
        error_key = guard.name + "_error"
        n_flagged = int(df[unsafe_key].sum()) if unsafe_key in df.columns else 0
        n_errors = int(df[error_key].notna().sum()) if error_key in df.columns else 0
        print(f"{guard.name}: flagged={n_flagged}, errors={n_errors}")
    if "deliberation_risk" in df.columns:
        counts = df["deliberation_risk"].value_counts().to_dict()
        parts = [f"{lvl}={counts.get(lvl, 0)}" for lvl in RISK_LEVELS]
        n_err = int(df["deliberation_risk"].astype(str).str.startswith("error:").sum())
        print(f"deliberation: {', '.join(parts)}, errors={n_err}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Risk detection pipeline")
    parser.add_argument("--input", default=None, help="Input CSV path")
    parser.add_argument("--text-col", default="text", help="Column name for text")
    parser.add_argument("--output", default="scores.csv", help="Output CSV path")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only")
    parser.add_argument("--demo", action="store_true", help="Use demo texts")
    parser.add_argument("--skip-deliberation", action="store_true", help="Skip deliberation stage")
    args = parser.parse_args()

    if args.input is None:
        args.demo = True

    if args.demo:
        texts = DEMO_TEXTS
    else:
        df_in = pd.read_csv(args.input)
        texts = df_in[args.text_col].tolist()

    if args.dry_run:
        dry_run(texts)
        return

    df = run(texts, args.output, skip_deliberation=args.skip_deliberation)
    _summary(df)


if __name__ == "__main__":
    main()
