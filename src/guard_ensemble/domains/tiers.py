from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

from guard_ensemble.domains.config import ESCALATION_FLAGS, MINIMUM_FLAGS
from guard_ensemble.domains.types import Tier

# Highest Tier


@dataclass
class HighestTierResult:
    tier: Tier | None
    incomplete: bool


def highest_tier(domain_tiers: dict[str, Tier | None]) -> HighestTierResult:
    """
    Return the highest tier among known domains; ignore `unknown`.

    `incomplete` is True when no domains are present or any domain has tier ``None``.
    """
    if not domain_tiers:
        return HighestTierResult(tier=None, incomplete=True)
    known = [t for t in domain_tiers.values() if t is not None]
    incomplete = any(t is None for t in domain_tiers.values())
    return HighestTierResult(
        tier=max(known) if known else None,
        incomplete=incomplete,
    )


# Escalation


@dataclass
class EscalationResult:
    tier: Tier | None
    reasons: list[str] = field(default_factory=list)


def _escalate(
    current: Tier | None,
    all_flags: AbstractSet[str],
    yellow_count: int,
) -> EscalationResult:
    """Raise Yellow to Red only. Never escalates Green. At most one tier."""
    if current != Tier.yellow:
        return EscalationResult(tier=current, reasons=[])

    reasons: list[str] = []
    for flag in ESCALATION_FLAGS:
        if flag in all_flags:
            reasons.append(flag)

    if yellow_count >= 2:
        reasons.append("multi_domain_yellow")

    if reasons:
        return EscalationResult(tier=Tier.red, reasons=reasons)
    return EscalationResult(tier=current, reasons=[])


# Minimum


@dataclass
class MinimumResult:
    tier: Tier | None
    reasons: list[str] = field(default_factory=list)


def _minimums(
    age: int | None,
    all_flags: AbstractSet[str],
) -> MinimumResult:
    """Floor at Yellow for minors with abuse disclosure or any lifetime-history flag."""
    reasons: list[str] = []

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
    tier: Tier | None
    reasons: list[str] = field(default_factory=list)


def _unknown_floor(incomplete: bool) -> UnknownFloorResult:
    """Floor at Yellow when any domain is unknown/unavailable."""
    if incomplete:
        return UnknownFloorResult(tier=Tier.yellow, reasons=["domain_unavailable"])
    return UnknownFloorResult(tier=None, reasons=[])


# ── apply_floor ──────────────────────────────────────────────────────────────


# ── floor_tier ───────────────────────────────────────────────────────────────


def _apply_floor(base: Tier | None, floor: Tier | None) -> Tier | None:
    """Return the higher of base and floor; never lowers."""
    if floor is None:
        return base
    if base is None:
        return floor
    return max(base, floor)


@dataclass
class FloorTierResult:
    tier: Tier
    reasons: list[str] = field(default_factory=list)


def floor_tier(
    highest: Tier | None,
    escalated: Tier | None,
    minimum: Tier | None,
    unknown_floor: Tier | None,
    escalation_reasons: list[str],
    minimum_reasons: list[str],
    unknown_reasons: list[str],
) -> FloorTierResult:
    """Compute final tier as max(highest, escalated, minimum, unknown_floor).

    Collects all contributing reasons.
    """
    reasons: list[str] = []

    t = _apply_floor(highest, escalated)
    reasons.extend(escalation_reasons)
    t = _apply_floor(t, minimum)
    reasons.extend(minimum_reasons)
    t = _apply_floor(t, unknown_floor)
    reasons.extend(unknown_reasons)

    # t must never be None here because unknown_floor catches incomplete coverage
    return FloorTierResult(tier=t if t is not None else Tier.green, reasons=reasons)
