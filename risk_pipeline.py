"""Linear risk-detection pipeline: guards -> Kev domains -> deterministic phases -> consolidation -> final update -> CSV."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests

from guards import (
    DELIBERATION_BASE_URL,
    DELIBERATION_MAX_TOKENS,
    DELIBERATION_MODEL_ID,
    DELIBERATION_PATH,
    GUARDS,
    _shieldgemma_parse_all,
    shieldgemma_aggregate,
)
from rubric import (
    CANONICAL_DOMAINS,
    Tier,
    TIER_UNKNOWN,
)
from rules import (
    ESCALATION_FLAGS,
    MINIMUM_FLAGS,
    RiskDecision,
    floor_tier,
    highest_tier,
    resolve,
    to_legacy_risk,
    _escalate,
    _minimums,
    _unknown_floor,
)
from kev_client import (
    KEV_FLAG_THRESHOLD,
    load_kev_questions,
    run_kev_domains,
)
from deliberation import (
    consolidation_build,
    final_update_build,
    load_consolidation_prompt,
    load_final_update_prompt,
    parse_deliberation_response,
    run_phase_with_retry,
    VALID_TIERS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

DEMO_ROWS = [
    {"id": 0, "text": "What's the weather like today?", "age": 25, "entry_date": "2025-01-15", "participant_id": "P001"},
    {"id": 1, "text": "How do I make a bomb at home?", "age": 30, "entry_date": "2025-01-15", "participant_id": "P002"},
    {"id": 2, "text": "Write a poem about the sunset", "age": 17, "entry_date": "2025-02-01", "participant_id": "P003"},
    {"id": 3, "text": "I hate everyone in this town, they're all terrible people and deserve to suffer.", "age": 19, "entry_date": "2025-02-10", "participant_id": "P004"},
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
# Input validation
# ---------------------------------------------------------------------------

REQUIRED_COLS = ["id", "text", "age", "entry_date"]


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and data types before any network calls.

    Returns the validated dataframe. Raises SystemExit on failure with row indices.
    """
    # Check required columns exist
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    errors: List[str] = []

    # Validate age: must be integer >= 0
    for idx in df.index:
        age_val = df.at[idx, "age"]
        try:
            age_int = int(age_val)
            if age_int < 0:
                errors.append(f"row {idx}: age={age_val} is negative")
        except (ValueError, TypeError):
            errors.append(f"row {idx}: age={age_val!r} is not a valid integer")

    # Validate entry_date: must be parseable as a date string
    for idx in df.index:
        date_val = df.at[idx, "entry_date"]
        if pd.isna(date_val) or str(date_val).strip() == "":
            errors.append(f"row {idx}: entry_date is missing")
        else:
            try:
                pd.to_datetime(str(date_val))
            except (ValueError, TypeError):
                errors.append(f"row {idx}: entry_date={date_val!r} is not a valid date")

    if errors:
        print("ERROR: Input validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    # Normalize age to int
    df = df.copy()
    df["age"] = df["age"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Guard stage
# ---------------------------------------------------------------------------

def run_guards(texts: List[str]) -> List[dict]:
    """Run all guards for each text. Returns list of row dicts with guard results."""
    rows = [{"id": i, "text": t} for i, t in enumerate(texts)]
    for guard in GUARDS:
        for i, text in enumerate(texts):
            try:
                raw = call_llm(guard, text)
                if isinstance(raw, list):  # ShieldGemma
                    parsed = _shieldgemma_parse_all(raw)
                    rows[i].update(shieldgemma_aggregate(parsed))
                else:
                    rows[i].update(guard.parse(raw))
            except Exception as e:
                rows[i][f"{guard.name}_error"] = str(e)
    return rows


# ---------------------------------------------------------------------------
# Kev domain stage
# ---------------------------------------------------------------------------

def run_kev_domain_stage(
    df: pd.DataFrame,
    keq: dict,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Run Kev domain assessment for each row.

    Adds domain_results, domain_incomplete, custom_prompt_errors,
    global_flags, and global_flag_probabilities columns.
    """
    domain_results_list: List[dict] = []
    domain_incomplete_list: List[bool] = []
    kev_errors_list: List[str] = []
    global_flags_list: List[list] = []
    global_flag_probabilities_list: List[dict] = []

    for _, row in df.iterrows():
        text = row["text"]
        age = int(row["age"])
        entry_date = str(row["entry_date"])

        try:
            kev_result = run_kev_domains(
                text=text,
                age=age,
                entry_date=entry_date,
                keq=keq,
                threshold=threshold if threshold is not None else KEV_FLAG_THRESHOLD,
            )

            # Serialize domain results (no evidence, no per-domain flags)
            serializable = {}
            all_errors: List[str] = []
            for domain, result in kev_result["domains"].items():
                serializable[domain] = {
                    "tier": result.get("tier", TIER_UNKNOWN),
                    "timing": result.get("timing"),
                    "tier_score": result.get("mean_score"),
                    "tier_probabilities": result.get("probabilities", {}),
                    "tier_confidence": result.get("confidence"),
                    "tier_mode_probability": result.get("mode_probability"),
                    "timing_probabilities": result.get("timing_probabilities", {}),
                    "timing_confidence": result.get("timing_confidence"),
                }
                if "error" in result:
                    all_errors.append(f"domain {domain}: {result['error']}")

            domain_results_list.append(serializable)
            domain_incomplete_list.append(
                any(r.get("tier") == TIER_UNKNOWN for r in kev_result["domains"].values())
                or len(kev_result["errors"]) > 0
            )
            kev_errors_list.append(";".join(kev_result["errors"]) if kev_result["errors"] else "")
            global_flags_list.append(kev_result.get("global_flags", []))
            global_flag_probabilities_list.append(kev_result.get("flag_probabilities", {}))

        except Exception as e:
            domain_results_list.append({})
            domain_incomplete_list.append(True)
            kev_errors_list.append(f"kev_domain_stage_error: {e}")
            global_flags_list.append([])
            global_flag_probabilities_list.append({})

    df = df.copy()
    df["domain_results"] = domain_results_list
    df["domain_incomplete"] = domain_incomplete_list
    df["custom_prompt_errors"] = kev_errors_list
    df["global_flags"] = global_flags_list
    df["global_flag_probabilities"] = global_flag_probabilities_list
    return df


# ---------------------------------------------------------------------------
# Deterministic phases
# ---------------------------------------------------------------------------

def _tier_to_int(tier_str: str) -> Optional[int]:
    """Convert tier string to int for comparison; None for unknown."""
    if tier_str == TIER_UNKNOWN:
        return None
    try:
        return Tier[tier_str]
    except KeyError:
        return None


def _int_to_tier_str(tier_int: Optional[int]) -> str:
    """Convert tier int to string; 'unknown' for None."""
    if tier_int is None:
        return TIER_UNKNOWN
    return Tier(tier_int).name


def run_deterministic_phases(df: pd.DataFrame) -> pd.DataFrame:
    """Run deterministic rule phases: highest -> escalation -> minimums -> unknown floor -> floor.

    Uses domain_results JSON to extract domain tiers and flags.
    Also reads global_flags and unions them with any per-domain flags.
    Computes and writes domain_incomplete, highest_domain_tier, and floor_reasons.
    """
    escalated_tiers: List[str] = []
    escalation_reasons_list: List[str] = []
    minimum_tiers: List[str] = []
    minimum_reasons_list: List[str] = []
    floor_tiers: List[str] = []
    domain_incomplete_list: List[bool] = []
    highest_tiers: List[str] = []
    floor_reasons_list: List[str] = []

    for _, row in df.iterrows():
        domain_results = row.get("domain_results", {})
        if isinstance(domain_results, str):
            try:
                domain_results = json.loads(domain_results)
            except (json.JSONDecodeError, TypeError):
                domain_results = {}

        # Read stage-level incompleteness from Kev domain stage
        stage_domain_incomplete = bool(row.get("domain_incomplete", False))

        # Build domain_tiers dict: domain -> Tier or None
        domain_tiers: Dict[str, Optional[int]] = {}
        all_flags: set = set()

        for domain, data in domain_results.items():
            if isinstance(data, dict):
                tier_str = data.get("tier", TIER_UNKNOWN)
                domain_tiers[domain] = _tier_to_int(tier_str)
                # Per-domain flags if present (legacy compatibility)
                flags = data.get("flags", [])
                if isinstance(flags, list):
                    all_flags.update(flags)
            else:
                domain_tiers[domain] = None

        # Union global_flags with per-domain flags
        global_flags = row.get("global_flags", [])
        if isinstance(global_flags, list):
            for gf in global_flags:
                if isinstance(gf, dict) and gf.get("flagged"):
                    flag_name = gf.get("name", "")
                    if flag_name:
                        all_flags.add(flag_name)

        # Phase 1: highest_tier
        ht_result = highest_tier(domain_tiers)

        # Combine stage incompleteness with highest_tier incompleteness
        incomplete = stage_domain_incomplete or ht_result.incomplete

        # Phase 2: escalation
        yellow_count = sum(1 for t in domain_tiers.values() if t is Tier.yellow)
        esc = _escalate(ht_result.tier, frozenset(all_flags), yellow_count)

        # Phase 3: minimums
        age = int(row.get("age", 0)) if pd.notna(row.get("age")) else None
        min_ = _minimums(age, frozenset(all_flags))

        # Phase 4: unknown floor
        unk = _unknown_floor(incomplete)

        # Phase 5: floor
        ft = floor_tier(
            ht_result.tier, esc.tier, min_.tier, unk.tier,
            esc.reasons, min_.reasons, unk.reasons,
        )

        escalated_tiers.append(_int_to_tier_str(esc.tier))
        escalation_reasons_list.append(",".join(esc.reasons) if esc.reasons else "")
        minimum_tiers.append(_int_to_tier_str(min_.tier) if min_.tier is not None else "")
        minimum_reasons_list.append(",".join(min_.reasons) if min_.reasons else "")
        floor_tiers.append(_int_to_tier_str(ft.tier))
        domain_incomplete_list.append(incomplete)
        highest_tiers.append(TIER_UNKNOWN if ht_result.tier is None else Tier(ht_result.tier).name)
        floor_reasons_list.append(",".join(ft.reasons) if ft.reasons else "")

    df = df.copy()
    df["escalated_tier"] = escalated_tiers
    df["escalation_reasons"] = escalation_reasons_list
    df["minimum_tier"] = minimum_tiers
    df["minimum_reasons"] = minimum_reasons_list
    df["floor_tier"] = floor_tiers
    df["domain_incomplete"] = domain_incomplete_list
    df["highest_domain_tier"] = highest_tiers
    df["floor_reasons"] = floor_reasons_list
    return df


# ---------------------------------------------------------------------------
# Consolidation and final update
# ---------------------------------------------------------------------------

def _build_context_block(row) -> str:
    """Build context block from row metadata."""
    lines = [f"Entry ID: {row.get('id', 'unknown')}"]
    if pd.notna(row.get("participant_id")):
        lines.append(f"Participant: {row['participant_id']}")
    lines.append(f"Age: {row.get('age', 'unknown')}")
    lines.append(f"Entry date: {row.get('entry_date', 'unknown')}")
    return "\n".join(lines)


def _build_guard_block(row) -> str:
    """Build guard block from guard columns."""
    lines = []
    for guard in GUARDS:
        unsafe_key = f"{guard.name}_unsafe"
        label_key = f"{guard.name}_label"
        category_key = f"{guard.name}_category"
        error_key = f"{guard.name}_error"
        if unsafe_key in row and pd.notna(row.get(unsafe_key)):
            lines.append(f"{guard.name}: label={row.get(label_key, 'unknown')}, unsafe={row[unsafe_key]}, category={row.get(category_key, 'unknown')}")
        elif error_key in row and pd.notna(row.get(error_key)):
            lines.append(f"{guard.name}: error={row[error_key]}")
    return "\n".join(lines) if lines else "No guard data available"


def _build_domain_block(row) -> str:
    """Build domain block from domain_results with tier, timing, confidence."""
    domain_results = row.get("domain_results", {})
    if isinstance(domain_results, str):
        try:
            domain_results = json.loads(domain_results)
        except (json.JSONDecodeError, TypeError):
            domain_results = {}

    lines = []
    for domain, data in domain_results.items():
        if isinstance(data, dict):
            tier = data.get("tier", "unknown")
            timing = data.get("timing", "unknown")
            confidence = data.get("tier_confidence")
            conf_str = f"{confidence}" if confidence is not None else "None"
            lines.append(f"{domain}: tier={tier}, timing={timing}, confidence={conf_str}")
    return "\n".join(lines) if lines else "No domain data available"


def _build_rule_block(row) -> str:
    """Build rule context block from deterministic phase results."""
    lines = [
        f"Highest domain tier: {row.get('highest_domain_tier', 'unknown')}",
        f"Domain coverage incomplete: {row.get('domain_incomplete', False)}",
        f"Escalated tier: {row.get('escalated_tier', 'unknown')}",
        f"Escalation reasons: {row.get('escalation_reasons', '')}",
        f"Minimum tier: {row.get('minimum_tier', 'none')}",
        f"Minimum reasons: {row.get('minimum_reasons', '')}",
        f"Floor tier: {row.get('floor_tier', 'unknown')}",
        f"Floor reasons: {row.get('floor_reasons', '')}",
    ]
    return "\n".join(lines)


def _call_deliberation(prompt: str, url: str, model_id: str, max_tokens: int) -> str:
    """Call the LLM for deliberation (consolidation/final-update)."""
    body = {
        "model": model_id,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(url, json=body, timeout=DMR_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    _log_raw("deliberation", prompt[:200], data)
    return data["choices"][0]["message"]["content"]


def run_consolidation_and_final_update(
    df: pd.DataFrame,
    consolidation_prompt: Optional[str] = None,
    final_update_prompt: Optional[str] = None,
    url: Optional[str] = None,
    model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> pd.DataFrame:
    """Run consolidation and final-update LLM phases.

    Falls back to floor_tier on errors, preserving the row.
    """
    if model_id is None:
        model_id = DELIBERATION_MODEL_ID
    if url is None:
        url = _join_url(DELIBERATION_BASE_URL, DELIBERATION_PATH)
    if max_tokens is None:
        max_tokens = DELIBERATION_MAX_TOKENS
    consolidated_tiers: List[str] = []
    consolidation_rationales: List[str] = []
    consolidation_errors: List[str] = []
    final_tiers: List[str] = []
    final_rationales: List[str] = []
    final_adjusted: List[bool] = []
    final_errors: List[str] = []

    for _, row in df.iterrows():
        text = row["text"]
        floor = row.get("floor_tier", "green")

        context_block = _build_context_block(row)
        guard_block = _build_guard_block(row)
        domain_block = _build_domain_block(row)
        rule_block = _build_rule_block(row)

        # Consolidation phase
        try:
            cons_prompt = consolidation_build(
                text=text,
                context_block=context_block,
                guard_block=guard_block,
                domain_block=domain_block,
                rule_block=rule_block,
                prompt=consolidation_prompt,
            )

            def call_fn_consolidation(p: str) -> str:
                return _call_deliberation(p, url, model_id, max_tokens)

            cons_tier, cons_error = run_phase_with_retry(call_fn_consolidation, cons_prompt, floor)
            cons_rationale = ""
            consolidated_tiers.append(cons_tier)
            consolidation_errors.append(cons_error)
        except Exception as e:
            consolidated_tiers.append(floor)
            consolidation_errors.append(f"consolidation_error: {e}")
            cons_rationale = ""

        consolidation_rationales.append(cons_rationale)

        # Final update phase
        try:
            cons_block = f"Consolidated tier: {consolidated_tiers[-1]}"
            final_prompt = final_update_build(
                text=text,
                context_block=context_block,
                guard_block=guard_block,
                domain_block=domain_block,
                rule_block=rule_block,
                consolidation_block=cons_block,
                prompt=final_update_prompt,
            )

            def call_fn_final(p: str) -> str:
                return _call_deliberation(p, url, model_id, max_tokens)

            final_tier, final_error = run_phase_with_retry(call_fn_final, final_prompt, consolidated_tiers[-1])
            final_rationale_str = ""
            adjusted = False

            final_tiers.append(final_tier)
            final_rationales.append(final_rationale_str)
            final_adjusted.append(adjusted)
            final_errors.append(final_error)
        except Exception as e:
            final_tiers.append(consolidated_tiers[-1])
            final_rationales.append("")
            final_adjusted.append(False)
            final_errors.append(f"final_update_error: {e}")

    df = df.copy()
    df["consolidated_tier"] = consolidated_tiers
    df["consolidation_rationale"] = consolidation_rationales
    df["consolidation_error"] = consolidation_errors
    df["final_tier"] = final_tiers
    df["final_rationale"] = final_rationales
    df["final_adjusted"] = final_adjusted
    df["final_error"] = final_errors
    return df


# ---------------------------------------------------------------------------
# Pipeline output assembly
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "id", "text", "participant_id", "age", "entry_date",
    # Guard columns added dynamically
    "domain_results",
    "global_flags",
    "global_flag_probabilities",
    "highest_domain_tier",
    "escalated_tier",
    "escalation_reasons",
    "minimum_tier",
    "minimum_reasons",
    "floor_tier",
    "floor_reasons",
    "consolidated_tier",
    "consolidation_rationale",
    "final_tier",
    "final_rationale",
    "final_adjusted",
    "final_risk",
    "domain_incomplete",
    "custom_prompt_errors",
    "consolidation_error",
    "final_update_error",
    "errors",
]


def _guard_col_order() -> List[str]:
    """Return guard columns in deterministic order."""
    cols = []
    for guard in GUARDS:
        cols.extend([
            f"{guard.name}_label",
            f"{guard.name}_unsafe",
            f"{guard.name}_category",
            f"{guard.name}_p_dangerous_content",
            f"{guard.name}_p_harassment",
            f"{guard.name}_p_hate_speech",
            f"{guard.name}_p_sexually_explicit",
            f"{guard.name}_error",
        ])
    return cols


def _assemble_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder and finalize output columns, adding legacy mapping and errors column."""
    # Add legacy risk mapping
    tier_to_legacy = {Tier.green: "no_low_risk", Tier.yellow: "concerning", Tier.red: "high_risk"}
    legacy_labels = []
    for _, row in df.iterrows():
        ft = row.get("final_tier", "green")
        try:
            legacy_labels.append(tier_to_legacy[Tier[ft]])
        except (KeyError, ValueError):
            legacy_labels.append("unknown")
    df = df.copy()
    df["final_risk"] = legacy_labels

    # Build errors column: semicolon-joined guard errors
    guard_error_cols = [c for c in df.columns if c.endswith("_error") and c != "final_error"]
    if guard_error_cols:
        df["errors"] = df[guard_error_cols].apply(
            lambda r: ";".join(
                f"{col.replace('_error', '')}:{val}"
                for col, val in r.items()
                if pd.notna(val) and val
            ),
            axis=1,
        )
    else:
        df["errors"] = ""

    # Rename final_error to final_update_error
    if "final_error" in df.columns:
        df = df.rename(columns={"final_error": "final_update_error"})

    # participant_id: include only if present
    base_cols = ["id", "text"]
    if "participant_id" in df.columns:
        base_cols.append("participant_id")
    base_cols.extend(["age", "entry_date"])

    # Guard columns that actually exist
    guard_cols = [c for c in _guard_col_order() if c in df.columns]

    # Final column order
    final_cols = base_cols + guard_cols + [
        "domain_results",
        "global_flags",
        "global_flag_probabilities",
        "highest_domain_tier",
        "escalated_tier",
        "escalation_reasons",
        "minimum_tier",
        "minimum_reasons",
        "floor_tier",
        "floor_reasons",
        "consolidated_tier",
        "consolidation_rationale",
        "final_tier",
        "final_rationale",
        "final_adjusted",
        "final_risk",
        "domain_incomplete",
        "custom_prompt_errors",
        "consolidation_error",
        "final_update_error",
        "errors",
    ]

    # Only include columns that exist
    final_cols = [c for c in final_cols if c in df.columns]
    return df[final_cols]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    input_df: pd.DataFrame,
    output_path: str,
    keq: dict,
    consolidation_prompt: Optional[str] = None,
    final_update_prompt: Optional[str] = None,
    guards_only: bool = False,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Run the full pipeline: guards -> Kev domains -> deterministic -> consolidation -> final -> CSV.

    Args:
        input_df: DataFrame with columns id, text, age, entry_date (optional participant_id).
        output_path: Path for output CSV.
        keq: Pre-loaded Kev question config from kev_client.load_kev_questions().
        consolidation_prompt: Custom consolidation prompt template.
        final_update_prompt: Custom final update prompt template.
        guards_only: If True, skip Kev, deterministic, consolidation, final stages.
    """
    # Validate input
    input_df = validate_input(input_df)

    # Resolve LLM endpoint for deliberation stages
    llm_url = _join_url(DELIBERATION_BASE_URL, DELIBERATION_PATH)

    # Stage 1: Guards
    guard_rows = run_guards(input_df["text"].tolist())

    # Merge guard results back with original data
    df = pd.DataFrame(guard_rows)
    for col in input_df.columns:
        if col not in df.columns:
            df[col] = input_df[col].values

    # If guards-only, skip remaining stages
    if guards_only:
        # Fill in empty values for remaining columns
        for col in ["domain_results", "global_flags", "global_flag_probabilities",
                     "highest_domain_tier", "escalated_tier", "escalation_reasons",
                     "minimum_tier", "minimum_reasons",
                     "floor_tier", "floor_reasons", "consolidated_tier", "consolidation_rationale",
                     "final_tier", "final_rationale", "final_adjusted",
                     "final_risk", "domain_incomplete", "custom_prompt_errors",
                     "consolidation_error", "final_update_error"]:
            if col not in df.columns:
                df[col] = ""

        # Add errors column
        error_cols = [c for c in df.columns if c.endswith("_error")]
        if error_cols:
            df["errors"] = df[error_cols].apply(
                lambda r: ";".join(
                    f"{col.replace('_error', '')}:{val}"
                    for col, val in r.items()
                    if pd.notna(val) and val
                ),
                axis=1,
            )
        else:
            df["errors"] = ""

        df = _assemble_output_columns(df)
        df.to_csv(output_path, index=False)
        return df

    # Stage 2: Kev domains
    df = run_kev_domain_stage(df, keq=keq, threshold=threshold)

    # Stage 3: Deterministic phases
    df = run_deterministic_phases(df)

    # Stage 4: Consolidation and final update
    df = run_consolidation_and_final_update(
        df, consolidation_prompt, final_update_prompt,
        url=llm_url, model_id=DELIBERATION_MODEL_ID, max_tokens=DELIBERATION_MAX_TOKENS
    )

    # Assemble output
    df = _assemble_output_columns(df)
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(
    input_df: pd.DataFrame,
    kev_questions_dir: Optional[str] = None,
    consolidation_prompt_path: Optional[str] = None,
    final_update_prompt_path: Optional[str] = None,
) -> None:
    """Print all prompts without making any network calls.

    Prints guard prompts, assembled Kev questions, consolidation/final prompts.
    """
    # Guard prompts (first row only for brevity)
    for guard in GUARDS:
        prompt = guard.build(input_df.iloc[0]["text"])
        if isinstance(prompt, list):
            for p in prompt:
                print(f"[{guard.name}] {p}")
        else:
            print(f"[{guard.name}] {prompt}")

    # Load and print Kev questions
    try:
        keq = load_kev_questions()
        from kev_client import build_kev_questions
        questions = build_kev_questions(keq)
        print(f"[kev] {len(questions)} questions loaded:")
        for q in questions:
            print(f"  - {q['id']} ({q['type']})")
    except Exception as e:
        print(f"[kev] ERROR loading questions: {e}")

    # Consolidation prompt (build with placeholder blocks)
    if consolidation_prompt_path:
        with open(consolidation_prompt_path) as f:
            cons_template = f.read().rstrip("\n")
    else:
        cons_template = load_consolidation_prompt()
    cons_rendered = consolidation_build(
        text=input_df.iloc[0]["text"],
        context_block=f"Entry ID: {input_df.iloc[0].get('id', 0)}, Age: {input_df.iloc[0]['age']}",
        guard_block="Guard results placeholder",
        domain_block="Domain results placeholder",
        rule_block="Rule context placeholder",
        prompt=cons_template,
    )
    print(f"[consolidation] {cons_rendered}")

    # Final update prompt
    if final_update_prompt_path:
        with open(final_update_prompt_path) as f:
            final_template = f.read().rstrip("\n")
    else:
        final_template = load_final_update_prompt()
    final_rendered = final_update_build(
        text=input_df.iloc[0]["text"],
        context_block=f"Entry ID: {input_df.iloc[0].get('id', 0)}, Age: {input_df.iloc[0]['age']}",
        guard_block="Guard results placeholder",
        domain_block="Domain results placeholder",
        rule_block="Rule context placeholder",
        consolidation_block="Consolidated tier: yellow",
        prompt=final_template,
    )
    print(f"[final_update] {final_rendered}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summary(df: pd.DataFrame) -> None:
    """Print summary: guard flags/errors, domain tier dist, final tier/legacy dist, phase errors."""
    # Guard flags/errors
    for guard in GUARDS:
        unsafe_key = guard.name + "_unsafe"
        error_key = guard.name + "_error"
        n_flagged = int(df[unsafe_key].sum()) if unsafe_key in df.columns else 0
        n_errors = int(df[error_key].notna().sum()) if error_key in df.columns else 0
        print(f"{guard.name}: flagged={n_flagged}, errors={n_errors}")

    # Domain tier distribution
    if "domain_results" in df.columns:
        tier_counts = {}
        for _, row in df.iterrows():
            dr = row.get("domain_results", {})
            if isinstance(dr, str):
                try:
                    dr = json.loads(dr)
                except (json.JSONDecodeError, TypeError):
                    dr = {}
            for domain, data in dr.items():
                if isinstance(data, dict):
                    t = data.get("tier", TIER_UNKNOWN)
                    tier_counts[t] = tier_counts.get(t, 0) + 1
        parts = [f"{tier}={tier_counts.get(tier, 0)}" for tier in ("green", "yellow", "red", TIER_UNKNOWN)]
        print(f"domain_tiers: {', '.join(parts)}")

    # Final tier distribution
    if "final_tier" in df.columns:
        counts = df["final_tier"].value_counts().to_dict()
        parts = [f"{tier}={counts.get(tier, 0)}" for tier in VALID_TIERS]
        print(f"final_tier: {', '.join(parts)}")

    # Final risk (legacy) distribution
    if "final_risk" in df.columns:
        counts = df["final_risk"].value_counts().to_dict()
        parts = [f"{lvl}={counts.get(lvl, 0)}" for lvl in [to_legacy_risk(Tier.green), to_legacy_risk(Tier.yellow), to_legacy_risk(Tier.red)]]
        print(f"final_risk: {', '.join(parts)}")

    # Phase errors
    phase_error_cols = ["custom_prompt_errors", "consolidation_error", "final_update_error"]
    for col in phase_error_cols:
        if col in df.columns:
            n_errors = int(df[col].apply(lambda x: bool(x) and x != "").sum())
            print(f"{col}: errors={n_errors}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Risk detection pipeline")
    parser.add_argument("--input", default=None, help="Input CSV path")
    parser.add_argument("--id-col", default="id", help="Column name for id")
    parser.add_argument("--text-col", default="text", help="Column name for text")
    parser.add_argument("--age-col", default="age", help="Column name for age")
    parser.add_argument("--date-col", default="entry_date", help="Column name for entry date")
    parser.add_argument("--output", default="scores.csv", help="Output CSV path")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only")
    parser.add_argument("--demo", action="store_true", help="Use demo data")
    parser.add_argument("--guards-only", action="store_true", help="Run guards only, skip remaining stages")
    parser.add_argument("--consolidation-prompt", default=None, help="Path to custom consolidation prompt template")
    parser.add_argument("--final-update-prompt", default=None, help="Path to custom final update prompt template")
    # Kev-specific CLI flags
    parser.add_argument("--kev-base-url", default=None, help="Kev API base URL (default: from KEV_BASE_URL env)")
    parser.add_argument("--kev-model", default=None, help="Kev model ID (default: from KEV_MODEL env)")
    parser.add_argument("--kev-timeout", type=int, default=None, help="Kev request timeout in seconds")
    parser.add_argument("--kev-flag-threshold", type=float, default=None, help="Threshold for flagging global flags")
    parser.add_argument("--kev-questions-dir", default=None, help="Directory for Kev question YAML files")
    args = parser.parse_args()

    if args.input is None:
        args.demo = True

    if args.demo:
        input_df = pd.DataFrame(DEMO_ROWS)
    else:
        input_df = pd.read_csv(args.input)
        # Rename columns if needed
        col_map = {}
        if args.id_col != "id" and args.id_col in input_df.columns:
            col_map[args.id_col] = "id"
        if args.text_col != "text" and args.text_col in input_df.columns:
            col_map[args.text_col] = "text"
        if args.age_col != "age" and args.age_col in input_df.columns:
            col_map[args.age_col] = "age"
        if args.date_col != "entry_date" and args.date_col in input_df.columns:
            col_map[args.date_col] = "entry_date"
        if col_map:
            input_df = input_df.rename(columns=col_map)

    # Apply Kev env overrides from CLI
    if args.kev_base_url:
        import kev_client
        kev_client.KEV_BASE_URL = args.kev_base_url
    if args.kev_model:
        import kev_client
        kev_client.KEV_MODEL = args.kev_model
    if args.kev_timeout is not None:
        import kev_client
        kev_client.KEV_TIMEOUT = args.kev_timeout
    if args.kev_flag_threshold is not None:
        import kev_client
        kev_client.KEV_FLAG_THRESHOLD = args.kev_flag_threshold
    if args.kev_questions_dir:
        import kev_client
        kev_client.KEV_DIR = Path(args.kev_questions_dir)

    if args.dry_run:
        dry_run(
            input_df,
            kev_questions_dir=args.kev_questions_dir,
            consolidation_prompt_path=args.consolidation_prompt,
            final_update_prompt_path=args.final_update_prompt,
        )
        return

    # Load Kev questions
    keq = load_kev_questions()

    # Load custom prompts if paths provided
    consolidation_prompt = None
    if args.consolidation_prompt:
        try:
            with open(args.consolidation_prompt) as f:
                consolidation_prompt = f.read().rstrip("\n")
        except FileNotFoundError:
            parser.error(f"Consolidation prompt file not found: {args.consolidation_prompt}")

    final_update_prompt = None
    if args.final_update_prompt:
        try:
            with open(args.final_update_prompt) as f:
                final_update_prompt = f.read().rstrip("\n")
        except FileNotFoundError:
            parser.error(f"Final update prompt file not found: {args.final_update_prompt}")

    df = run(
        input_df,
        args.output,
        keq=keq,
        consolidation_prompt=consolidation_prompt,
        final_update_prompt=final_update_prompt,
        guards_only=args.guards_only,
        threshold=args.kev_flag_threshold,
    )
    _summary(df)


if __name__ == "__main__":
    main()
