"""Kev client: YAML loader, request builder, HTTP client, and response parser."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests
import yaml


# ── Configuration ─────────────────────────────────────────────────────────────

KEV_BASE_URL: str = os.environ.get("KEV_BASE_URL", "http://127.0.0.1:8009")
KEV_MODEL: str = os.environ.get("KEV_MODEL", "kev-latest")
KEV_API_KEY: str | None = os.environ.get("KEV_API_KEY") or None
KEV_TIMEOUT: int = int(os.environ.get("KEV_TIMEOUT", "600"))
KEV_FLAG_THRESHOLD: float = float(os.environ.get("KEV_FLAG_THRESHOLD", "0.5"))

KEV_DIR: Path = Path(__file__).parent / "prompts" / "kev"

TIER_NAMES = ["green", "yellow", "red"]

# ── YAML loader ───────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and return a YAML file via safe_load."""
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:
        raise ValueError(f"Failed to parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level in {path}, got {type(data).__name__}")
    return data


TIMING_VOCABULARY = frozenset({
    "historical_resolved",
    "historical_unresolved",
    "recent",
    "current",
    "imminent",
    "unclear",
})

FLAG_VOCABULARY = frozenset({
    "lethal_means_access",
    "current_intoxication",
    "alone_no_support",
    "minor_abuse_disclosure",
    "past_suicide_attempt",
    "past_self_harm",
    "past_sexual_assault",
    "past_psychosis",
    "past_eating_disorder",
})


def _validate_shared(shared: dict[str, Any]) -> None:
    """Validate shared YAML has all required keys."""
    for key in ("reading_rules", "timing", "flags"):
        if key not in shared:
            raise ValueError(f"Shared YAML missing required key: {key}")
    if not isinstance(shared["reading_rules"], str):
        raise ValueError("Shared reading_rules must be a string")
    timing = shared["timing"]
    if not isinstance(timing, dict):
        raise ValueError("Shared timing must be a mapping")
    if timing.get("type") != "choice":
        raise ValueError(f"Shared timing type must be 'choice', got {timing.get('type')}")
    if "instructions" not in timing:
        raise ValueError("Shared timing missing 'instructions'")
    if "criteria" not in timing or not isinstance(timing["criteria"], dict):
        raise ValueError("Shared timing must have a 'criteria' dict")
    timing_keys = set(timing["criteria"].keys())
    if timing_keys != TIMING_VOCABULARY:
        raise ValueError(
            f"Shared timing criteria keys must equal the expected 6 timing vocabulary. "
            f"Got {sorted(timing_keys)}, expected {sorted(TIMING_VOCABULARY)}"
        )
    flags = shared["flags"]
    if not isinstance(flags, dict):
        raise ValueError("Shared flags must be a mapping")
    flag_keys = set(flags.keys())
    if flag_keys != FLAG_VOCABULARY:
        raise ValueError(
            f"Shared flag keys must equal the expected 9 flags. "
            f"Got {sorted(flag_keys)}, expected {sorted(FLAG_VOCABULARY)}"
        )
    for fname, fdef in flags.items():
        if not isinstance(fdef, dict):
            raise ValueError(f"Flag '{fname}' must be a mapping")
        if fdef.get("type") != "noul":
            raise ValueError(f"Flag '{fname}' type must be 'noul', got {fdef.get('type')}")
        if "instructions" not in fdef:
            raise ValueError(f"Flag '{fname}' missing 'instructions'")


def _validate_domain(domain_data: dict[str, Any]) -> None:
    """Validate a single domain YAML has all required keys."""
    if "domain" not in domain_data:
        raise ValueError("Domain YAML missing 'domain' key")
    if not isinstance(domain_data["domain"], str):
        raise ValueError("Domain 'domain' must be a string")
    if "tier" not in domain_data:
        raise ValueError("Domain YAML missing 'tier' key")
    tier = domain_data["tier"]
    if not isinstance(tier, dict):
        raise ValueError(f"Domain '{domain_data['domain']}' tier must be a mapping")
    if tier.get("type") != "score":
        raise ValueError(
            f"Domain '{domain_data['domain']}' tier type must be 'score', got {tier.get('type')}"
        )
    if "instructions" not in tier:
        raise ValueError(f"Domain '{domain_data['domain']}' tier missing 'instructions'")
    if "criteria" not in tier:
        raise ValueError(f"Domain '{domain_data['domain']}' tier missing 'criteria'")
    criteria = tier["criteria"]
    if not isinstance(criteria, list) or len(criteria) != 3:
        raise ValueError(
            f"Domain '{domain_data['domain']}' tier.criteria must be a list of 3, got {len(criteria) if isinstance(criteria, list) else 'non-list'}"
        )


def load_kev_questions() -> dict[str, Any]:
    """Load and validate all Kev YAML sources, returning a single config dict.

    Returns:
        A dict with keys: reading_rules, timing, flags, domains.
    """
    shared_path = KEV_DIR / "_shared.yaml"
    shared = _load_yaml(shared_path)
    _validate_shared(shared)

    domain_files = sorted((KEV_DIR / "domains").glob("*.yaml"))
    if not domain_files:
        raise ValueError(f"No domain YAML files found in {KEV_DIR / 'domains'}")

    domains: list[dict[str, Any]] = []
    for df in domain_files:
        data = _load_yaml(df)
        _validate_domain(data)
        domains.append(data)

    return {
        "reading_rules": shared["reading_rules"],
        "timing": shared["timing"],
        "flags": shared["flags"],
        "domains": domains,
    }


# ── Question builder ──────────────────────────────────────────────────────────


def build_kev_questions(keq: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Assemble the Kev question dict from loaded YAML config.

    For each domain we emit a tier question and a timing question.
    Then we emit global flag questions.
    Returns a dict keyed by question ID (no ``id`` field in values).
    """
    questions: dict[str, dict[str, Any]] = {}
    reading_rules = keq["reading_rules"]
    shared_timing = keq["timing"]

    for domain in keq["domains"]:
        slug = domain["domain"]
        tier = domain["tier"]

        # Tier question
        tier_instructions = tier["instructions"].replace("<reading_rules>", reading_rules)
        questions[f"{slug}__tier"] = {
            "type": "score",
            "instructions": tier_instructions,
            "criteria": tier["criteria"],
        }

        # Timing question
        timing_instructions = shared_timing["instructions"]
        if "timing_guidance" in domain:
            timing_instructions += "\n\nDomain-specific guidance:\n" + domain["timing_guidance"]
        questions[f"{slug}__timing"] = {
            "type": "choice",
            "instructions": timing_instructions,
            "criteria": shared_timing["criteria"],
        }

    # Global flag questions
    for flag_name, flag_def in keq["flags"].items():
        questions[f"flag__{flag_name}"] = {
            "type": "noul",
            "instructions": flag_def["instructions"],
        }

    return questions


# ── HTTP client ───────────────────────────────────────────────────────────────


def _post_kev(payload: dict[str, Any], *, max_retries: int = 1) -> dict[str, Any]:
    """POST to Kev API with retry on transient failures."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if KEV_API_KEY:
        headers["Authorization"] = f"Bearer {KEV_API_KEY}"

    last_exc: Exception | None = None
    attempts = 1 + max_retries
    for attempt in range(attempts):
        try:
            resp = requests.post(
                f"{KEV_BASE_URL}/v1/systemone",
                json=payload,
                headers=headers,
                timeout=KEV_TIMEOUT,
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(0.5)
                continue
            raise

        if resp.status_code >= 500:
            last_exc = Exception(f"Server error {resp.status_code}: {resp.text}")
            if attempt < attempts - 1:
                time.sleep(0.5)
                continue
            raise

        if resp.status_code != 200:
            raise Exception(f"Request failed with {resp.status_code}: {resp.text}")

        try:
            return resp.json()
        except ValueError as exc:
            raise Exception(f"Invalid JSON response: {exc}") from exc

    raise last_exc  # type: ignore[misc]


# ── Response parser ───────────────────────────────────────────────────────────


def _decode_score(answer: dict[str, Any], errors: list[str] | None = None) -> dict[str, Any]:
    """Decode a score-type answer into tier name, mean, probabilities, etc."""
    probabilities = answer.get("probabilities", {})
    if not probabilities:
        if errors is not None:
            errors.append("Malformed score probabilities: empty or missing")
        return {
            "tier": "unknown",
            "mean_score": None,
            "probabilities": {},
            "confidence": None,
            "mode_probability": None,
        }

    max_prob = -1.0
    max_idx = -1
    for idx_str, prob in probabilities.items():
        try:
            idx = int(idx_str)
        except (ValueError, TypeError):
            continue
        if prob > max_prob:
            max_prob = prob
            max_idx = idx
        elif prob == max_prob and idx > max_idx:
            # Tie-break: higher index (conservative safety)
            max_idx = idx

    tier_name = TIER_NAMES[max_idx] if 0 <= max_idx < len(TIER_NAMES) else "unknown"

    return {
        "tier": tier_name,
        "mean_score": answer.get("score"),
        "probabilities": probabilities,
        "confidence": answer.get("confidence"),
        "mode_probability": max_prob,
    }


def _decode_choice(answer: dict[str, Any]) -> dict[str, Any]:
    """Decode a choice-type answer."""
    probabilities = answer.get("probabilities", {})
    return {
        "timing": answer.get("choice"),
        "timing_probabilities": probabilities,
        "timing_confidence": answer.get("confidence"),
    }


def _decode_noul(answer: dict[str, Any], threshold: float | None = None) -> dict[str, Any]:
    """Decode a noul-type answer, applying the threshold."""
    if threshold is None:
        threshold = KEV_FLAG_THRESHOLD
    prob = answer.get("noul")
    return {
        "flagged": prob is not None and prob >= threshold,
        "probability": prob,
    }


# ── Main entry point ─────────────────────────────────────────────────────────


def run_kev_domains(
    text: str,
    age: int,
    entry_date: str,
    keq: dict[str, Any],
    threshold: float | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Run a single Kev request covering all domains, timing, and flags.

    Args:
        text: The participant entry text.
        age: Participant age.
        entry_date: Entry date string.
        keq: Loaded Kev question config from ``load_kev_questions()``.
        **_kwargs: Ignored (reserved for future extensions).

    Returns:
        Dict with keys: domains, global_flags, flag_probabilities, errors.
    """
    questions = build_kev_questions(keq)

    payload = {
        "state": {
            "entry_text": text,
            "participant_age": age,
            "entry_date": entry_date,
        },
        "model": KEV_MODEL,
        "questions": questions,
    }

    try:
        response = _post_kev(payload)
    except Exception as exc:
        # All domains go to unknown on request failure
        domains = {
            d["domain"]: {"tier": "unknown", "timing": None, "error": str(exc)}
            for d in keq["domains"]
        }
        return {
            "domains": domains,
            "global_flags": [],
            "flag_probabilities": {},
            "errors": [str(exc)],
        }

    answers: dict[str, Any] = response.get("answers", {})
    errors: list[str] = []
    domains: dict[str, Any] = {}
    global_flags: list[dict[str, Any]] = []
    flag_probabilities: dict[str, float] = {}

    # Parse domain answers
    for domain_cfg in keq["domains"]:
        slug = domain_cfg["domain"]
        domain_result: dict[str, Any] = {}

        # Tier
        tier_ans = answers.get(f"{slug}__tier")
        if tier_ans is None or not isinstance(tier_ans, dict):
            domain_result["tier"] = "unknown"
            errors.append(f"Missing or malformed tier answer for domain '{slug}'")
        else:
            domain_result.update(_decode_score(tier_ans, errors=errors))

        # Timing
        timing_ans = answers.get(f"{slug}__timing")
        if timing_ans is None or not isinstance(timing_ans, dict):
            domain_result["timing"] = None
            errors.append(f"Missing or malformed timing answer for domain '{slug}'")
        else:
            domain_result.update(_decode_choice(timing_ans))

        domains[slug] = domain_result

    # Parse flag answers
    for flag_name in keq["flags"]:
        flag_ans = answers.get(f"flag__{flag_name}")
        if flag_ans is None or not isinstance(flag_ans, dict):
            errors.append(f"Missing or malformed flag answer for '{flag_name}'")
            global_flags.append({"name": flag_name, "flagged": False, "probability": None})
            flag_probabilities[flag_name] = -1.0
        else:
            decoded = _decode_noul(flag_ans, threshold=threshold)
            global_flags.append({
                "name": flag_name,
                "flagged": decoded["flagged"],
                "probability": decoded["probability"],
            })
            flag_probabilities[flag_name] = decoded["probability"] or 0.0

    return {
        "domains": domains,
        "global_flags": global_flags,
        "flag_probabilities": flag_probabilities,
        "errors": errors,
    }
