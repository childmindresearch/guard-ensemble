"""Deterministic rule engine for risk-tier resolution."""

from dataclasses import dataclass, field
from typing import AbstractSet, Dict, List, Optional, Set, Tuple

from rubric import (
    CANONICAL_DOMAINS,
    FLAG_VOCABULARY,
    Tier,
)

ESCALATION_FLAGS = tuple(FLAG_VOCABULARY["escalation"])
MINIMUM_FLAGS = tuple(FLAG_VOCABULARY["minimum"])

LEGACY_MAP = {
    Tier.green: "no_low_risk",
    Tier.yellow: "concerning",
    Tier.red: "high_risk",
}


# ── highest_tier ─────────────────────────────────────────────────────────────

@dataclass
class HighestTierResult:
    tier: Optional[Tier]
    incomplete: bool  # True if any domain returned unknown


def highest_tier(domain_tiers: Dict[str, Optional[Tier]]) -> HighestTierResult:
    """Return the highest tier among known domains; ignore ``unknown``.

    ``incomplete`` is True when at least one domain has tier ``None``.
    """
    known = [t for t in domain_tiers.values() if t is not None]
    incomplete = any(t is None for t in domain_tiers.values())
    return HighestTierResult(
        tier=max(known) if known else None,
        incomplete=incomplete,
    )


# ── Escalation ───────────────────────────────────────────────────────────────

@dataclass
class EscalationResult:
    tier: Optional[Tier]
    reasons: List[str] = field(default_factory=list)


def _escalate(
    current: Optional[Tier],
    all_flags: AbstractSet[str],
    yellow_count: int,
) -> EscalationResult:
    """Raise Yellow to Red only. Never escalates Green. At most one tier."""
    if current != Tier.yellow:
        return EscalationResult(tier=current, reasons=[])

    reasons: List[str] = []
    for flag in ESCALATION_FLAGS:
        if flag in all_flags:
            reasons.append(flag)

    if yellow_count >= 2:
        reasons.append("multi_domain_yellow")

    if reasons:
        return EscalationResult(tier=Tier.red, reasons=reasons)
    return EscalationResult(tier=current, reasons=[])


# ── Minimums ─────────────────────────────────────────────────────────────────

@dataclass
class MinimumResult:
    tier: Optional[Tier]
    reasons: List[str] = field(default_factory=list)


def _minimums(
    age: Optional[int],
    all_flags: AbstractSet[str],
) -> MinimumResult:
    """Floor at Yellow for minors with abuse disclosure or any lifetime-history flag."""
    reasons: List[str] = []

    # Minor abuse disclosure: strictly under 18
    if age is not None and age < 18 and "minor_abuse_disclosure" in all_flags:
        reasons.append("minor_abuse_disclosure")

    # Any lifetime-history flag
    for flag in MINIMUM_FLAGS:
        if flag != "minor_abuse_disclosure" and flag in all_flags:
            reasons.append(flag)

    if reasons:
        return MinimumResult(tier=Tier.yellow, reasons=reasons)
    return MinimumResult(tier=None, reasons=[])


# ── Unknown-floor ────────────────────────────────────────────────────────────

@dataclass
class UnknownFloorResult:
    tier: Optional[Tier]
    reasons: List[str] = field(default_factory=list)


def _unknown_floor(incomplete: bool) -> UnknownFloorResult:
    """Floor at Yellow when any domain is unknown/unavailable."""
    if incomplete:
        return UnknownFloorResult(tier=Tier.yellow, reasons=["domain_unavailable"])
    return UnknownFloorResult(tier=None, reasons=[])


# ── apply_floor ──────────────────────────────────────────────────────────────

def _apply_floor(base: Optional[Tier], floor: Optional[Tier]) -> Optional[Tier]:
    """Return the higher of base and floor; never lowers."""
    if floor is None:
        return base
    if base is None:
        return floor
    return max(base, floor)


# ── floor_tier ───────────────────────────────────────────────────────────────

@dataclass
class FloorTierResult:
    tier: Tier
    reasons: List[str] = field(default_factory=list)


def floor_tier(
    highest: Optional[Tier],
    escalated: Optional[Tier],
    minimum: Optional[Tier],
    unknown_floor: Optional[Tier],
    escalation_reasons: List[str],
    minimum_reasons: List[str],
    unknown_reasons: List[str],
) -> FloorTierResult:
    """Compute final tier as max(highest, escalated, minimum, unknown_floor).

    Collects all contributing reasons.
    """
    reasons: List[str] = []

    t = _apply_floor(highest, escalated)
    reasons.extend(escalation_reasons)
    t = _apply_floor(t, minimum)
    reasons.extend(minimum_reasons)
    t = _apply_floor(t, unknown_floor)
    reasons.extend(unknown_reasons)

    # t must never be None here because unknown_floor catches incomplete coverage
    return FloorTierResult(tier=t if t is not None else Tier.green, reasons=reasons)


# ── to_legacy_risk ───────────────────────────────────────────────────────────

def to_legacy_risk(tier: Tier) -> str:
    """Map tier to legacy risk label."""
    return LEGACY_MAP[tier]


# ── Public convenience ───────────────────────────────────────────────────────

@dataclass
class RiskDecision:
    tier: Tier
    reasons: List[str] = field(default_factory=list)
    legacy_label: str = ""

    def __post_init__(self):
        if not self.legacy_label:
            self.legacy_label = to_legacy_risk(self.tier)


def resolve(
    domain_tiers: Dict[str, Optional[Tier]],
    all_flags: AbstractSet[str] = frozenset(),
    age: Optional[int] = None,
) -> RiskDecision:
    """End-to-end deterministic tier resolution."""
    ht = highest_tier(domain_tiers)

    yellow_count = sum(1 for t in domain_tiers.values() if t is Tier.yellow)
    esc = _escalate(ht.tier, all_flags, yellow_count)
    min_ = _minimums(age, all_flags)
    unk = _unknown_floor(ht.incomplete)

    ft = floor_tier(
        ht.tier, esc.tier, min_.tier, unk.tier,
        esc.reasons, min_.reasons, unk.reasons,
    )
    return RiskDecision(tier=ft.tier, reasons=ft.reasons)
