"""Custom-prompt runner: execute domain prompt templates, aggregate results."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from string import Template
from typing import Any, Callable, Dict, List, Optional

from rubric import (
    TIMING_VOCABULARY,
    Tier,
    TIER_UNKNOWN,
    AggregatedDomainRun,
    DomainParseError,
    ParsedDomainResponse,
    aggregate_domain_runs,
    parse_domain_response,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CUSTOM_PROMPT_MAX_TOKENS = int(os.environ.get("CUSTOM_PROMPT_MAX_TOKENS", "512"))

# Timing acuity ordering — higher index = more urgent
_TIMING_ACUTITY = {t: i for i, t in enumerate(TIMING_VOCABULARY)}


# ---------------------------------------------------------------------------
# Caller type
# ---------------------------------------------------------------------------
# The caller is any callable matching the requests.post signature:
#   caller(url, json=body) -> response with .json(), .raise_for_status()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DomainPromptResult:
    domain: str
    tier: str  # tier name string or "unknown"
    timing: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    evidence: str = ""
    errors: List[str] = field(default_factory=list)
    runs: List[ParsedDomainResponse] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_with_retry(
    prompt_text: str,
    caller: Callable,
    url: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call the LLM, retry once on empty/invalid content."""
    for attempt in range(2):
        body = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        body["chat_template_kwargs"] = {"enable_thinking": False}

        resp = caller(url, json=body)
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        if content:
            return content

        # Empty content — retry once
    raise ValueError("empty response after retry")


def _aggregate_with_tiebreak(
    runs: List[ParsedDomainResponse],
) -> AggregatedDomainRun:
    """Aggregate runs with deterministic tie-break toward more acute timing."""
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
        elif run.tier is not None and run.tier == best_tier and run.timing is not None and best_timing is not None:
            # Tie-break: prefer more acute timing
            if _timing_more_acute(run.timing, best_timing):
                best_timing = run.timing

    return AggregatedDomainRun(
        domain=domain,
        tier=best_tier,
        timing=best_timing,
        flags=sorted(all_flags),
        run_errors=all_errors,
    )


def _timing_more_acute(a: str, b: str) -> bool:
    """Return True if timing `a` is more acute (urgent) than `b`."""
    return _TIMING_ACUTITY.get(a, -1) > _TIMING_ACUTITY.get(b, -1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_domain_prompts(
    text: str,
    age: str,
    entry_date: str,
    templates: Dict[str, str],
    caller: Callable,
    url: str = "http://localhost:12434/engines/v1/chat/completions",
    model_id: str = "",
    runs: int = 3,
    temperature: float = 0.3,
) -> Dict[str, DomainPromptResult]:
    """Run custom domain prompts, returning per-domain aggregated results.

    Args:
        text: The user entry text.
        age: Patient age string.
        entry_date: Entry date string.
        templates: Dict mapping domain slug -> prompt template with
                   ``$text``, ``$age``, ``$entry_date`` placeholders.
        caller: Injected callable for network I/O
                (``caller(url, json=body) -> response``).
        url: Endpoint URL (defaulted for convenience).
        model_id: Model identifier (passed in request body).
        runs: Number of independent runs per prompt.
        temperature: Sampling temperature.

    Returns:
        Dict mapping domain -> DomainPromptResult.
    """
    results: Dict[str, DomainPromptResult] = {}

    for domain, template in templates.items():
        prompt_text = Template(template).substitute(
            text=text, age=age, entry_date=entry_date, scores_block=""
        )

        parsed_runs: List[ParsedDomainResponse] = []
        error_msgs: List[str] = []

        for _run_idx in range(runs):
            try:
                raw = _call_with_retry(
                    prompt_text, caller, url, temperature, CUSTOM_PROMPT_MAX_TOKENS
                )
                parsed = parse_domain_response(raw)
                parsed.domain = domain  # ensure domain is set
                parsed_runs.append(parsed)
            except Exception as exc:
                error_msgs.append(f"run_{_run_idx}: {exc}")
                parsed_runs.append(
                    ParsedDomainResponse(
                        domain=domain,
                        tier=None,
                        timing=None,
                        errors=[DomainParseError("call_error", str(exc))],
                    )
                )

        agg = _aggregate_with_tiebreak(parsed_runs)
        error_msgs.extend(agg.run_errors)

        tier_str = TIER_UNKNOWN if agg.tier is None else agg.tier.name

        results[domain] = DomainPromptResult(
            domain=domain,
            tier=tier_str,
            timing=agg.timing,
            flags=agg.flags,
            evidence=_evidence_from_highest_tier(parsed_runs),
            errors=error_msgs,
            runs=parsed_runs,
        )

    return results


def _evidence_from_highest_tier(runs: List[ParsedDomainResponse]) -> str:
    """Return evidence string from the run with the highest tier (tie-break: more acute)."""
    best: Optional[ParsedDomainResponse] = None
    for run in runs:
        if run.tier is not None:
            if best is None or best.tier is None or run.tier > best.tier:
                best = run
            elif best.tier is not None and run.tier == best.tier and best.timing is not None:
                if _timing_more_acute(run.timing or "", best.timing):
                    best = run
    return best.evidence if best else ""
