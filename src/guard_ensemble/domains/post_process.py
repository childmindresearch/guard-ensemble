import re

import pandas as pd

from guard_ensemble.domains import tiers
from guard_ensemble.domains.config import RED_RAISE
from guard_ensemble.domains.types import TIER_UNKNOWN, DomainResult, Row, Tier


def normalised_words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split())


def quote_grams(evidence: str, text: str) -> set[str]:
    """Word 4-grams of the evidence's quotes that occur in the entry. Without
    such a quote (none, or an invented one: the model sometimes copies the
    schema's example), the whole entry's: the claim can't show a separate
    concern."""
    entry = normalised_words(text)
    quotes = [
        quote
        for quote in map(normalised_words, re.findall(r'"([^"]+)"', evidence))
        if quote and quote in entry
    ]
    words = (" ".join(quotes) if quotes else entry).split()
    if len(words) < 4:
        return {" ".join(words)} if words else set()
    return {" ".join(words[start : start + 4]) for start in range(len(words) - 3)}


def yellow_concerns(evidence_by_domain: dict[str, str], text: str) -> list[list[str]]:
    """Group the yellow domains whose quotes share a 4-gram: one passage claimed
    by several domains is one concern, not several."""
    grams = {
        domain: quote_grams(evidence, text)
        for domain, evidence in evidence_by_domain.items()
    }
    groups: list[list[str]] = []
    for domain in evidence_by_domain:
        overlapping = [
            group
            for group in groups
            if any(grams[domain] & grams[other] for other in group)
        ]
        merged = [other for group in overlapping for other in group] + [domain]
        groups = [group for group in groups if group not in overlapping] + [merged]
    return groups


def effective_tier(result: DomainResult) -> str:
    """The domain's tier, raised to red when a scored green or yellow has
    P(red) >= RED_RAISE."""
    probability_red = (result.get("tier_probs") or {}).get("red", 0)
    raisable = result["tier"] in ("green", "yellow") and result.get("scored")
    return "red" if raisable and probability_red >= RED_RAISE else result["tier"]


def participant_age(row: Row) -> int | None:
    age = row.get("age")
    return None if age is None or pd.isna(age) else int(age)


def entry_tier(row: Row) -> dict:
    """The entry's final tier, the reasons for it, and the yellow concerns."""
    results: dict[str, DomainResult] = row["domain_results"]
    eff_tiers = {domain: effective_tier(result) for domain, result in results.items()}
    raised = [
        domain for domain, tier in eff_tiers.items() if tier != results[domain]["tier"]
    ]
    known_tiers = {
        domain: None if tier == TIER_UNKNOWN else Tier[tier]
        for domain, tier in eff_tiers.items()
    }
    flags = frozenset(
        flag for result in results.values() for flag in result.get("flags") or []
    )
    yellow_evidence = {
        domain: results[domain].get("evidence") or ""
        for domain, tier in eff_tiers.items()
        if tier == "yellow"
    }
    concerns = yellow_concerns(yellow_evidence, str(row["text"]))

    highest = tiers.highest_tier(known_tiers)
    escalation = tiers._escalate(highest.tier, flags, len(concerns))
    minimum = tiers._minimums(participant_age(row), flags)
    incomplete = bool(row.get("domain_incomplete")) or highest.incomplete
    unknown = tiers._unknown_floor(incomplete)
    final = tiers.floor_tier(
        highest.tier,
        escalation.tier,
        minimum.tier,
        unknown.tier,
        escalation.reasons,
        minimum.reasons,
        unknown.reasons,
    )
    return {
        "entry_tier": final.tier.name,
        "entry_reasons": [f"red_probability:{domain}" for domain in raised]
        + final.reasons,
        "yellow_concerns": concerns,
    }


def with_entry_tier(results: pd.DataFrame) -> pd.DataFrame:
    """Add entry_tier, entry_reasons and yellow_concerns. Works on a saved run."""
    out = results.copy()
    tiers = [entry_tier(row) for row in results.to_dict("records")]
    for column in ("entry_tier", "entry_reasons", "yellow_concerns"):
        out[column] = [tier[column] for tier in tiers]
    return out
