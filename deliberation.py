"""Deliberation module: consolidation and final-update LLM phases.

Provides prompt loading, prompt building, strict JSON parsing, retry logic,
and monotonic-clamp enforcement (tiers may only be raised, never lowered).
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier constants
# ---------------------------------------------------------------------------

VALID_TIERS = ("green", "yellow", "red")
TIER_ORDER = {tier: i for i, tier in enumerate(VALID_TIERS)}


def _tier_index(tier: str) -> int:
    return TIER_ORDER.get(tier, -1)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_CONSOLIDATION_REQUIRED = {"text", "context_block", "guard_block", "domain_block", "rule_block"}
_CONSOLIDATION_KNOWN = set(_CONSOLIDATION_REQUIRED)

_FINAL_UPDATE_REQUIRED = _CONSOLIDATION_REQUIRED | {"consolidation_block"}
_FINAL_UPDATE_KNOWN = set(_FINAL_UPDATE_REQUIRED)


def _find_placeholder(content: str) -> set[str]:
    """Return all $name placeholders found in *content*."""
    return set(re.findall(r"\$(?!\$)([A-Za-z_]\w*)", content))


def _load_template(path: str, required: set[str], known: set[str]) -> str:
    """Load a prompt template, validate required placeholders, reject unknowns."""
    with open(path, "r") as fh:
        content = fh.read().rstrip("\n")
    found = _find_placeholder(content)
    missing = required - found
    if missing:
        raise ValueError(
            f"prompt template missing placeholder(s): {sorted(missing)}"
        )
    unknown = found - known
    if unknown:
        raise ValueError(
            f"prompt template contains unknown placeholder(s): {sorted(unknown)}"
        )
    return content


@lru_cache(maxsize=4)
def load_consolidation_prompt(path: str | None = None) -> str:
    """Load and validate the consolidation prompt template."""
    if path is None:
        path = str(Path(__file__).parent / "prompts" / "consolidation.txt")
    return _load_template(path, _CONSOLIDATION_REQUIRED, _CONSOLIDATION_KNOWN)


@lru_cache(maxsize=4)
def load_final_update_prompt(path: str | None = None) -> str:
    """Load and validate the final-update prompt template."""
    if path is None:
        path = str(Path(__file__).parent / "prompts" / "final_update.txt")
    return _load_template(path, _FINAL_UPDATE_REQUIRED, _FINAL_UPDATE_KNOWN)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def consolidation_build(
    text: str,
    context_block: str,
    guard_block: str,
    domain_block: str,
    rule_block: str,
    prompt: str | None = None,
) -> str:
    """Build the consolidation prompt by substituting all placeholders."""
    if prompt is None:
        prompt = load_consolidation_prompt()
    result = Template(prompt).substitute(
        text=text,
        context_block=context_block,
        guard_block=guard_block,
        domain_block=domain_block,
        rule_block=rule_block,
    )
    # Safety: no residual $name placeholders (must start with letter/underscore)
    if re.search(r"\$[A-Za-z_]", result):
        raise ValueError("prompt build residue: unreplaced $ found in result")
    return result


def final_update_build(
    text: str,
    context_block: str,
    guard_block: str,
    domain_block: str,
    rule_block: str,
    consolidation_block: str,
    prompt: str | None = None,
) -> str:
    """Build the final-update prompt by substituting all placeholders."""
    if prompt is None:
        prompt = load_final_update_prompt()
    result = Template(prompt).substitute(
        text=text,
        context_block=context_block,
        guard_block=guard_block,
        domain_block=domain_block,
        rule_block=rule_block,
        consolidation_block=consolidation_block,
    )
    # Safety: no residual $name placeholders (must start with letter/underscore)
    if re.search(r"\$[A-Za-z_]", result):
        raise ValueError("prompt build residue: unreplaced $ found in result")
    return result


# ---------------------------------------------------------------------------
# Strict JSON parser
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<antThinking>.*?</antThinking>", re.DOTALL)
_THINK_UNTERM_RE = re.compile(r"<antThinking>.*?\n", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(.*?)```", re.DOTALL)


def _strip_decorations(raw: str) -> str:
    text = _THINK_BLOCK_RE.sub("", raw)
    text = _THINK_UNTERM_RE.sub("", text)
    text = _CODE_FENCE_RE.sub(r"\1", text)
    return text.strip()


def parse_deliberation_response(raw: str) -> Tuple[str, str]:
    """Parse an LLM response into (tier, rationale).

    Returns a tuple of (tier_str, rationale_str).
    Raises ValueError on empty input, invalid JSON, or invalid tier value.
    """
    text = _strip_decorations(raw)
    if not text:
        raise ValueError("deliberation parse failure: empty input")

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Try to extract a JSON object from free text
        try:
            data = _extract_json_braces(text)
        except ValueError as exc:
            raise ValueError(f"deliberation parse failure: {exc}") from exc

    tier = data.get("tier")
    if tier is None:
        raise ValueError("deliberation parse failure: missing 'tier' key")
    if tier not in VALID_TIERS:
        raise ValueError(
            f"deliberation parse failure: invalid tier {tier!r}, expected one of {VALID_TIERS}"
        )

    rationale = data.get("rationale", "")
    return tier, rationale


def _extract_json_braces(text: str) -> dict:
    """Extract the first JSON object bounded by braces."""
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
    raise ValueError("no valid JSON found in response")


# ---------------------------------------------------------------------------
# Monotonic clamp
# ---------------------------------------------------------------------------

def clamp_tier(proposed: str, floor: str) -> Tuple[str, bool]:
    """Enforce monotonicity: return tier >= floor, never lower.

    Returns (clamped_tier, was_adjusted).
    *was_adjusted* is True when the proposed tier was below floor and got raised.
    """
    prop_idx = _tier_index(proposed)
    floor_idx = _tier_index(floor)
    if prop_idx < floor_idx:
        return floor, True
    return proposed, False


# ---------------------------------------------------------------------------
# Retry + fallback
# ---------------------------------------------------------------------------

def run_phase_with_retry(
    call_llm_fn: Callable[[str], str],
    prompt: str,
    floor_tier: str,
) -> Tuple[str, str]:
    """Execute an LLM phase with one retry and monotonic clamp.

    Returns (tier, error_string).
    *error_string* is empty on success; non-empty when fallback was used.
    """
    # Attempt 1
    try:
        raw = call_llm_fn(prompt)
        tier, rationale = parse_deliberation_response(raw)
        clamped, adjusted = clamp_tier(tier, floor_tier)
        if adjusted:
            logger.warning(
                "tier lowering clamped: proposed %s, floor %s, clamped to %s",
                tier, floor_tier, clamped,
            )
        return clamped, ""
    except (ValueError, Exception) as exc:
        logger.warning("first attempt failed: %s — retrying", exc)

    # Attempt 2 (retry)
    try:
        raw = call_llm_fn(prompt)
        tier, rationale = parse_deliberation_response(raw)
        clamped, adjusted = clamp_tier(tier, floor_tier)
        if adjusted:
            logger.warning(
                "tier lowering clamped on retry: proposed %s, floor %s, clamped to %s",
                tier, floor_tier, clamped,
            )
        return clamped, ""
    except (ValueError, Exception) as exc:
        logger.error("retry exhausted, falling back to floor tier %s: %s", floor_tier, exc)
        return floor_tier, f"retry exhausted: {exc}"
