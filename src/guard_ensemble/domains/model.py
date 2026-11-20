import json
import math
import re
import time
from typing import Any, TypedDict

import polars as pl

from guard_ensemble.domains.config import (
    ESCALATION_FLAGS,
    GATE_THRESHOLD,
    LOGPROBS,
    MIN_TIER_MASS,
    MINIMUM_FLAGS,
    MODEL,
    RATING_MAX_TOKENS,
    TEMPERATURE,
    TIMING_VOCABULARY,
)
from guard_ensemble.domains.post_process import with_entry_tier
from guard_ensemble.domains.prompts import DomainPrompt, domain_prompts, render
from guard_ensemble.domains.types import TIER_UNKNOWN, TIERS, DomainResult
from guard_ensemble.llm import runner
from guard_ensemble.llm.lib import create_messages

## Model calls


def chat(stage: str, prompts: list[str], **sampling) -> list[dict]:
    """One completion per prompt ({"text", "logprobs"}), timed and logged."""
    if not prompts:
        return []
    started = time.monotonic()
    completions = runner.chat(
        MODEL,
        create_messages(prompts),
        chat_template_kwargs={"enable_thinking": False},
        **sampling,
    )
    print(f"{stage}: {len(prompts)} prompts ({time.monotonic() - started:.0f}s)")
    return [choices[0] for choices in completions]


def rating_schema(criterion_tiers: dict[str, str]) -> dict:
    """Decoding is constrained to this schema (vLLM structured outputs): valid
    JSON, these fields in this order, enum values only. The prompt's output
    schema (prompts/_shared/output_schema.txt) describes the same fields."""
    return {
        "type": "object",
        "properties": {
            "domain": {"type": "string", "maxLength": 60},
            "criterion": {"enum": list(criterion_tiers)},
            "quotes": {
                "type": "array",
                "items": {"type": "string", "maxLength": 200},
                "maxItems": 2,
            },
            "tier": {"enum": list(TIERS)},
            "timing": {"enum": list(TIMING_VOCABULARY)},
            "flags": {
                "type": "array",
                "items": {"enum": [*ESCALATION_FLAGS, *MINIMUM_FLAGS]},
                "maxItems": len(ESCALATION_FLAGS) + len(MINIMUM_FLAGS),
            },
        },
        "required": ["domain", "criterion", "quotes", "tier", "timing", "flags"],
        "additionalProperties": False,
    }


## Tier probabilities
#
# A completion's "logprobs" is one dict per generated token: the top LOGPROBS
# candidate tokens and their log probabilities at that step.


def tier_distribution(step: dict[str, float]) -> tuple[dict[str, float], float]:
    """At one step: the probability of each tier word, renormalised over the
    three, and the total probability the step put on tier words at all."""
    mass_per_tier = dict.fromkeys(TIERS, 0.0)
    for token, logprob in step.items():
        word = token.strip().strip(':"').strip().lower()
        if word in mass_per_tier:
            mass_per_tier[word] += math.exp(logprob)
    mass = sum(mass_per_tier.values())
    probs = {tier: p / mass for tier, p in mass_per_tier.items()} if mass else {}
    return probs, mass


def first_tier_distribution(steps: list[dict[str, float]]) -> dict[str, float]:
    for step in steps:
        probs, mass = tier_distribution(step)
        if mass >= MIN_TIER_MASS:
            return probs
    return {}


def gate_distribution(completion: dict) -> tuple[dict[str, float], float]:
    """The gate answers with one token, so its distribution is the first step's."""
    steps = completion.get("logprobs") or [{}]
    return tier_distribution(steps[0])


_TIER_KEY_RE = re.compile(r'"tier"\s*:\s*"?\s*$')


def rating_distribution(completion: dict) -> dict[str, float]:
    """The tier distribution where the model writes the tier value: the step
    right after `"tier": "`, found by replaying the chosen (highest probability)
    token of each step. Under the grammar that step allows only tier words.
    Falls back to the first step dominated by tier words; {} if there is none."""
    steps = completion.get("logprobs") or []
    written = ""
    for step in steps:
        if step and _TIER_KEY_RE.search(written):
            probs, mass = tier_distribution(step)
            if mass >= MIN_TIER_MASS:
                return probs
            break
        written += max(step, key=step.get) if step else ""
    return first_tier_distribution(steps)


## Gate stage


def gate_passes(probs: dict[str, float], mass: float) -> bool:
    """Rate the domain unless the gate confidently answered green."""
    return mass < MIN_TIER_MASS or probs["yellow"] + probs["red"] >= GATE_THRESHOLD


def chunked(items: list, size: int) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


class DomainScore(TypedDict):
    tier_probs: dict[str, float]
    tier_mass: float


type DomainScores = dict[str, DomainScore]


def run_gate(
    rows: list[dict[str, Any]], prompts: dict[str, DomainPrompt]
) -> tuple[list[DomainScores], dict[str, list[int]]]:
    """Gate every (entry, domain) in one domain-major batch. Returns the gate
    scores per row and, per domain, the indices of the rows to rate."""
    keys = list(prompts)
    if not rows:
        return [], {key: [] for key in keys}
    completions = chat(
        "gate",
        [render(prompts[key].gate, row) for key in keys for row in rows],
        temperature=0,
        max_tokens=1,
        logprobs=LOGPROBS,
    )
    scores: list[DomainScores] = [{} for _ in rows]
    to_rate: dict[str, list[int]] = {key: [] for key in keys}
    for key, domain_completions in zip(keys, chunked(completions, len(rows))):
        for index, completion in enumerate(domain_completions):
            probs, mass = gate_distribution(completion)
            scores[index][key] = {"tier_probs": probs, "tier_mass": mass}
            if gate_passes(probs, mass):
                to_rate[key].append(index)
    return scores, to_rate


## Rating stage


def evidence_string(criterion: str, quotes: list[str]) -> str:
    """'<criterion>: "quote", "quote"', the format yellow_concerns and the
    inspect scripts read."""
    if not quotes:
        return criterion

    quoted = ", ".join(f'"{quote}"' for quote in quotes)
    return f"{criterion}: {quoted}"


def parse_rating(
    text: str, criterion_tiers: dict[str, str]
) -> tuple[DomainResult, list[str]]:
    """The rated fields from the model's JSON, plus errors as "<kind>: <detail>".
    The grammar guarantees valid JSON and enum values unless the output was cut
    off at RATING_MAX_TOKENS, which leaves the tier unknown."""
    try:
        rating = json.loads(text)
    except ValueError as exc:
        unknown = {"tier": TIER_UNKNOWN, "timing": None, "flags": [], "evidence": ""}
        return unknown, [f"json: {exc}"]

    errors: list[str] = []
    model_tier = rating.get("tier")
    criterion = rating.get("criterion", "")
    criterion_tier = criterion_tiers.get(criterion, TIER_UNKNOWN)
    if criterion_tier == TIER_UNKNOWN:
        errors.append(f"unknown_criterion: {criterion!r}")
    elif criterion_tier != model_tier:
        errors.append(
            f"tier_mismatch: model tier {model_tier!r}, "
            f"but criterion {criterion!r} is {criterion_tier}"
        )
    quotes = [quote for quote in rating.get("quotes", []) if quote]
    fields = {
        "tier": model_tier,
        "timing": rating.get("timing"),
        "flags": rating.get("flags", []),
        "evidence": evidence_string(criterion, quotes),
    }
    return fields, errors


def rated_result(
    completion: dict, criterion_tiers: dict[str, str]
) -> tuple[DomainResult, list[str]]:
    """A rated domain: the parsed fields plus the tier-token distribution and
    confidence = P(chosen tier)."""
    fields, errors = parse_rating(completion["text"], criterion_tiers)
    probs = rating_distribution(completion)
    result = {
        **fields,
        "tier_probs": probs,
        "confidence": probs.get(fields["tier"]),
        "scored": True,
    }
    return result, errors


def unscored_result(gate_score: DomainScore) -> DomainResult:
    """A gated-out domain: green, carrying the gate's distribution so that
    every domain has a weight."""
    probs = gate_score["tier_probs"]
    return {
        "tier": "green",
        "tier_probs": probs,
        "confidence": probs.get("green"),
        "timing": None,
        "flags": [],
        "evidence": "",
        "scored": False,
    }


def run_rating(
    rows: list[dict[str, Any]],
    prompts: dict[str, DomainPrompt],
    to_rate: dict[str, list[int]],
) -> tuple[list[dict[str, DomainResult]], list[dict], list[list[str]]]:
    """Rate the pairs that passed the gate, one batch per domain since each has
    its own grammar. Returns per-row results, raw model texts and errors, with
    only the rated domains filled in."""
    rated: list[dict] = [{} for _ in rows]
    raw: list[dict] = [{} for _ in rows]
    errors: list[list[str]] = [[] for _ in rows]
    for key, prompt in prompts.items():
        indices = to_rate[key]
        completions = chat(
            f"rating {key}",
            [render(prompt.rating, rows[index]) for index in indices],
            temperature=TEMPERATURE,
            max_tokens=RATING_MAX_TOKENS,
            logprobs=LOGPROBS,
            json_schema=rating_schema(prompt.criterion_tiers),
        )
        for index, completion in zip(indices, completions):
            result, domain_errors = rated_result(completion, prompt.criterion_tiers)
            rated[index][key] = result
            raw[index][key] = completion["text"]
            errors[index] += [f"{key}: {error}" for error in domain_errors]
    return rated, raw, errors


def run(df: pl.DataFrame, validate=False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Gate every (entry, domain), rate the pairs that pass, and fill the rest
    in as unscored greens."""

    if validate:
        print("WARN: Domain model input validation is TBD")

    rows = df.to_dicts()
    prompts = domain_prompts()
    scores, to_rate = run_gate(rows, prompts)
    rated, raw, errors = run_rating(rows, prompts, to_rate)
    results = [
        {
            key: rated[index].get(key) or unscored_result(scores[index][key])
            for key in prompts
        }
        for index in range(len(rows))
    ]
    incomplete = [
        any(result["tier"] == TIER_UNKNOWN for result in row.values())
        for row in results
    ]
    cp_errors = [";".join(row) for row in errors]

    raw_keys = sorted({k for d in raw for k in d})
    raw_dtype = pl.Struct({k: pl.String for k in raw_keys})

    out = df.with_columns(
        pl.Series(scores).alias("gate_scores"),
        pl.Series(results).alias("domain_results"),
        pl.Series(incomplete).alias("domain_incomplete"),
        pl.Series(cp_errors).alias("custom_prompt_errors"),
        pl.Series(raw, dtype=raw_dtype).alias("domain_raw"),
    )

    return with_entry_tier(out), df.clone().clear()
