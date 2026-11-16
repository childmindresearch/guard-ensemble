"""Rubric core: tiers, domains, flags, prompt assembly, and response parsing."""

import glob
import json
import os
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


# ── Tier ──────────────────────────────────────────────────────────────────────

class Tier(IntEnum):
    green = 0
    yellow = 1
    red = 2


TIER_UNKNOWN = "unknown"


# ── Canonical domains ────────────────────────────────────────────────────────

CANONICAL_DOMAINS: Tuple[str, ...] = (
    "suicidality",
    "nonsuicidal_self_harm",
    "abuse_neglect_coercion_exploitation",
    "violence_harm_to_others",
    "psychosis_paranoia_reality_testing",
    "mania_agitation_dysregulation",
    "substance_use_intoxication_overdose_withdrawal",
    "eating_disorder_nutrition",
    "acute_medical_physical_danger",
    "environmental_basic_safety",
    "dangerous_risk_taking_accidental_harm",
    "severe_functional_incapacity",
    "ai_related_safety_risk",
    "other_serious_unclassifiable",
    "sextortion_image_based_sexual_abuse",
    "sexual_assault",
    "bullying_cyberbullying_peer_victimization",
    "risk_to_third_party",
    "help_seeking_directed_at_study_team",
    "threats_to_schools_groups_violent_extremism",
    "recent_traumatic_event_or_loss",
)

assert len(CANONICAL_DOMAINS) == 21, f"Expected 21 domains, got {len(CANONICAL_DOMAINS)}"


# ── Flag vocabulary ──────────────────────────────────────────────────────────

ESCALATION_FLAGS: Tuple[str, ...] = (
    "lethal_means_access",
    "current_intoxication",
    "alone_no_support",
)

MINIMUM_FLAGS: Tuple[str, ...] = (
    "minor_abuse_disclosure",
    "past_suicide_attempt",
    "past_self_harm",
    "past_sexual_assault",
    "past_psychosis",
    "past_eating_disorder",
)

TIMING_VOCABULARY: Tuple[str, ...] = (
    "historical_resolved",
    "historical_unresolved",
    "recent",
    "current",
    "imminent",
    "unclear",
)

FLAG_VOCABULARY: Dict[str, Any] = {
    "escalation": list(ESCALATION_FLAGS),
    "minimum": list(MINIMUM_FLAGS),
    "timing": list(TIMING_VOCABULARY),
}

_ALL_KNOWN_FLAGS = set(ESCALATION_FLAGS) | set(MINIMUM_FLAGS)


# ── Prompt assembly ──────────────────────────────────────────────────────────

_REQUIRED_PLACEHOLDERS = {"$text", "$age", "$entry_date"}
_OPTIONAL_PLACEHOLDERS = {"$scores_block"}
_VALID_PLACEHOLDERS = _REQUIRED_PLACEHOLDERS | _OPTIONAL_PLACEHOLDERS


def _validate_placeholders(content: str, source: str) -> None:
    """Reject files that use placeholders not in the allowed set."""
    found = re.findall(r"\$\w+", content)
    unknown = set(found) - _VALID_PLACEHOLDERS
    if unknown:
        raise ValueError(
            f"{source} contains unknown placeholder(s): {sorted(unknown)}"
        )


def load_domain_prompt_templates(
    dir: str,
    prelude_path: str,
    schema_path: str,
) -> Dict[str, str]:
    """Load domain prompt files, validate placeholders, and assemble templates.

    Returns a dict mapping domain slug -> composed prompt string.
    Files named ``*_shared*`` are skipped.
    """
    prelude = _read_file(prelude_path)
    schema = _read_file(schema_path)

    result: Dict[str, str] = {}
    for filepath in sorted(glob.glob(os.path.join(dir, "*.txt"))):
        basename = os.path.basename(filepath)
        if "_shared" in basename:
            continue
        domain = os.path.splitext(basename)[0]
        body = _read_file(filepath)
        _validate_placeholders(body, filepath)
        _validate_placeholders(prelude, "prelude")
        _validate_placeholders(schema, "schema")
        result[domain] = f"{prelude}\n\n{body}\n\n{schema}"

    return result


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ── Response parsing ─────────────────────────────────────────────────────────

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(.*?)```", re.DOTALL)


def _strip_decorations(raw: str) -> str:
    text = _THINK_BLOCK_RE.sub("", raw)
    text = _CODE_FENCE_RE.sub(r"\1", text)
    return text.strip()


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from free text."""
    # Try the whole string first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON object between braces
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    pass
                start = None
    raise ValueError("No valid JSON found in response")


@dataclass
class DomainParseError:
    kind: str
    message: str


@dataclass
class ParsedDomainResponse:
    domain: str
    tier: Optional[Tier]
    timing: Optional[str]
    flags: List[str] = field(default_factory=list)
    evidence: str = ""
    errors: List[DomainParseError] = field(default_factory=list)


def parse_domain_response(raw: str) -> ParsedDomainResponse:
    """Parse a single domain-model response into structured data."""
    text = _strip_decorations(raw)
    errors: List[DomainParseError] = []

    try:
        data = _extract_json(text)
    except ValueError as exc:
        return ParsedDomainResponse(
            domain="", tier=None, timing=None, errors=[DomainParseError("json", str(exc))]
        )

    domain = data.get("domain", "")
    tier_raw = data.get("tier")
    timing = data.get("timing")
    flags_raw = data.get("flags", [])
    evidence = data.get("evidence", "")

    # Validate tier
    tier: Optional[Tier] = None
    if tier_raw is not None:
        try:
            tier = Tier[tier_raw]
        except KeyError:
            errors.append(DomainParseError("unknown_tier", f"tier={tier_raw!r}"))
    else:
        errors.append(DomainParseError("unknown_tier", f"tier={tier_raw!r}"))

    # Validate timing
    if timing not in TIMING_VOCABULARY:
        errors.append(DomainParseError("unknown_timing", f"timing={timing!r}"))
        timing = None

    # Validate flags
    valid_flags: List[str] = []
    if isinstance(flags_raw, list):
        for flag in flags_raw:
            if flag in _ALL_KNOWN_FLAGS:
                valid_flags.append(flag)
            else:
                errors.append(DomainParseError("unknown_flag", f"flag={flag!r}"))
    else:
        errors.append(DomainParseError("bad_flags", "flags is not a list"))

    return ParsedDomainResponse(
        domain=domain,
        tier=tier,
        timing=timing,
        flags=valid_flags,
        evidence=evidence,
        errors=errors,
    )


# ── Aggregation ──────────────────────────────────────────────────────────────

@dataclass
class AggregatedDomainRun:
    domain: str
    tier: Optional[Tier]
    timing: Optional[str]
    flags: List[str] = field(default_factory=list)
    run_errors: List[str] = field(default_factory=list)


def aggregate_domain_runs(runs: List[ParsedDomainResponse]) -> AggregatedDomainRun:
    """Combine multiple parsed runs for a single domain.

    Rules:
    - tier: severity-max (highest tier value wins).
    - flags: union of all valid flags.
    - timing: taken from the run that produced the winning tier.
    - run_errors: flat list of per-run error messages.
    - If *all* runs failed (tier is None for every run), tier is None (unknown).
    """
    if not runs:
        return AggregatedDomainRun(domain="", tier=None, timing=None)

    domain = runs[0].domain
    all_errors: List[str] = []
    best_tier: Optional[Tier] = None
    best_timing: Optional[str] = None
    all_flags: set = set()

    for run in runs:
        for err in run.errors:
            all_errors.append(f"{err.kind}: {err.message}")
        all_flags.update(run.flags)
        if run.tier is not None and (best_tier is None or run.tier > best_tier):
            best_tier = run.tier
            best_timing = run.timing

    return AggregatedDomainRun(
        domain=domain,
        tier=best_tier,
        timing=best_timing,
        flags=sorted(all_flags),
        run_errors=all_errors,
    )
