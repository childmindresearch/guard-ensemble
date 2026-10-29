"""Comprehensive tests for the deterministic rule engine."""

import pytest

from rules import (
    floor_tier,
    highest_tier,
    resolve,
    to_legacy_risk,
)
from rubric import Tier


# ──────────────────────────────────────────────────────────────────────────────
# highest_tier
# ──────────────────────────────────────────────────────────────────────────────

class TestHighestTier:

    def test_all_green(self):
        result = highest_tier({"a": Tier.green, "b": Tier.green})
        assert result.tier is Tier.green
        assert result.incomplete is False

    def test_yellow_wins_over_green(self):
        result = highest_tier({"a": Tier.green, "b": Tier.yellow})
        assert result.tier is Tier.yellow
        assert result.incomplete is False

    def test_red_wins_over_all(self):
        result = highest_tier({"a": Tier.red, "b": Tier.yellow, "c": Tier.green})
        assert result.tier is Tier.red
        assert result.incomplete is False

    def test_ignores_unknown(self):
        result = highest_tier({"a": Tier.green, "b": None})
        assert result.tier is Tier.green
        assert result.incomplete is True

    def test_all_unknown(self):
        result = highest_tier({"a": None, "b": None})
        assert result.tier is None
        assert result.incomplete is True

    def test_empty(self):
        result = highest_tier({})
        assert result.tier is None
        assert result.incomplete is False

    def test_unknown_does_not_dilute_red(self):
        result = highest_tier({"a": Tier.red, "b": None, "c": None})
        assert result.tier is Tier.red
        assert result.incomplete is True


# ──────────────────────────────────────────────────────────────────────────────
# Escalation
# ──────────────────────────────────────────────────────────────────────────────

class TestEscalation:

    def test_green_never_escalates(self):
        d = resolve(
            {"a": Tier.green, "b": Tier.green},
            all_flags={"lethal_means_access", "current_intoxication", "alone_no_support"},
        )
        assert d.tier is Tier.green

    def test_lethal_means_escalates_yellow_to_red(self):
        d = resolve({"a": Tier.yellow}, all_flags={"lethal_means_access"})
        assert d.tier is Tier.red
        assert "lethal_means_access" in d.reasons

    def test_current_intoxication_escalates_yellow_to_red(self):
        d = resolve({"a": Tier.yellow}, all_flags={"current_intoxication"})
        assert d.tier is Tier.red
        assert "current_intoxication" in d.reasons

    def test_alone_no_support_escalates_yellow_to_red(self):
        d = resolve({"a": Tier.yellow}, all_flags={"alone_no_support"})
        assert d.tier is Tier.red
        assert "alone_no_support" in d.reasons

    def test_multi_domain_yellow_escalates(self):
        d = resolve({"a": Tier.yellow, "b": Tier.yellow})
        assert d.tier is Tier.red
        assert "multi_domain_yellow" in d.reasons

    def test_yellow_single_domain_no_flags_stays_yellow(self):
        d = resolve({"a": Tier.yellow})
        assert d.tier is Tier.yellow

    def test_escalation_at_most_one_tier(self):
        # Green + escalation flags → stays Green (never escalates)
        d = resolve(
            {"a": Tier.green},
            all_flags={"lethal_means_access", "alone_no_support"},
        )
        assert d.tier is Tier.green

    def test_multiple_escalation_reasons(self):
        d = resolve(
            {"a": Tier.yellow},
            all_flags={"lethal_means_access", "alone_no_support", "current_intoxication"},
        )
        assert d.tier is Tier.red
        assert len(d.reasons) == 3

    def test_red_is_idempotent(self):
        d = resolve(
            {"a": Tier.red},
            all_flags={"lethal_means_access"},
        )
        assert d.tier is Tier.red
        # No escalation reasons added since it's already red
        assert not d.reasons

    def test_two_yellow_domains_triggers_even_with_no_flags(self):
        d = resolve({"a": Tier.yellow, "b": Tier.yellow, "c": Tier.green})
        assert d.tier is Tier.red
        assert "multi_domain_yellow" in d.reasons


# ──────────────────────────────────────────────────────────────────────────────
# Minimums
# ──────────────────────────────────────────────────────────────────────────────

class TestMinimums:

    def test_minor_plus_abuse_floors_to_yellow(self):
        d = resolve({"a": Tier.green}, all_flags={"minor_abuse_disclosure"}, age=16)
        assert d.tier is Tier.yellow
        assert "minor_abuse_disclosure" in d.reasons

    def test_age_18_does_not_trigger_minor_abuse(self):
        d = resolve({"a": Tier.green}, all_flags={"minor_abuse_disclosure"}, age=18)
        assert d.tier is Tier.green

    def test_adult_abuse_does_not_floor(self):
        d = resolve({"a": Tier.green}, all_flags={"minor_abuse_disclosure"}, age=25)
        assert d.tier is Tier.green

    def test_lifetime_flag_floors(self):
        for flag in ["past_suicide_attempt", "past_self_harm", "past_sexual_assault",
                      "past_psychosis", "past_eating_disorder"]:
            d = resolve({"a": Tier.green}, all_flags={flag})
            assert d.tier is Tier.yellow, f"Failed for flag {flag}"
            assert flag in d.reasons

    def test_minor_abuse_no_age_does_not_floor(self):
        d = resolve({"a": Tier.green}, all_flags={"minor_abuse_disclosure"}, age=None)
        assert d.tier is Tier.green

    def test_lifetime_flag_on_yellow_is_idempotent(self):
        d = resolve({"a": Tier.yellow}, all_flags={"past_self_harm"})
        assert d.tier is Tier.yellow
        assert "past_self_harm" in d.reasons

    def test_lifetime_flag_on_red_does_not_lower(self):
        d = resolve({"a": Tier.red}, all_flags={"past_suicide_attempt"})
        assert d.tier is Tier.red

    def test_minor_abuse_with_no_green_domains(self):
        d = resolve({"a": Tier.yellow}, all_flags={"minor_abuse_disclosure"}, age=17)
        assert d.tier is Tier.yellow

    def test_resolved_adult_historical_abuse_alone_does_not_floor(self):
        """An adult with historical_resolved abuse should not be floored."""
        d = resolve({"a": Tier.green}, all_flags={"minor_abuse_disclosure"}, age=30)
        assert d.tier is Tier.green


# ──────────────────────────────────────────────────────────────────────────────
# Unknown-floor
# ──────────────────────────────────────────────────────────────────────────────

class TestUnknownFloor:

    def test_unknown_domain_floors_to_yellow(self):
        d = resolve({"a": Tier.green, "b": None})
        assert d.tier is Tier.yellow
        assert "domain_unavailable" in d.reasons

    def test_all_known_no_floor(self):
        d = resolve({"a": Tier.green, "b": Tier.green})
        assert d.tier is Tier.green
        assert "domain_unavailable" not in d.reasons


# ──────────────────────────────────────────────────────────────────────────────
# floor_tier / apply_floor
# ──────────────────────────────────────────────────────────────────────────────

class TestFloorTier:

    def test_max_of_components(self):
        ft = floor_tier(
            highest=Tier.green,
            escalated=Tier.red,
            minimum=Tier.yellow,
            unknown_floor=None,
            escalation_reasons=["lethal_means_access"],
            minimum_reasons=[],
            unknown_reasons=[],
        )
        assert ft.tier is Tier.red

    def test_apply_floor_never_lowers(self):
        ft = floor_tier(
            highest=Tier.red,
            escalated=Tier.yellow,
            minimum=None,
            unknown_floor=None,
            escalation_reasons=[],
            minimum_reasons=[],
            unknown_reasons=[],
        )
        assert ft.tier is Tier.red

    def test_fallback_to_green_when_all_none(self):
        ft = floor_tier(
            highest=None,
            escalated=None,
            minimum=None,
            unknown_floor=None,
            escalation_reasons=[],
            minimum_reasons=[],
            unknown_reasons=[],
        )
        assert ft.tier is Tier.green

    def test_multiple_reasons_collected(self):
        ft = floor_tier(
            highest=Tier.green,
            escalated=Tier.red,
            minimum=Tier.yellow,
            unknown_floor=Tier.yellow,
            escalation_reasons=["lethal_means_access"],
            minimum_reasons=["past_self_harm"],
            unknown_reasons=["domain_unavailable"],
        )
        assert ft.tier is Tier.red
        assert len(ft.reasons) == 3


# ──────────────────────────────────────────────────────────────────────────────
# to_legacy_risk
# ──────────────────────────────────────────────────────────────────────────────

class TestToLegacyRisk:

    def test_green(self):
        assert to_legacy_risk(Tier.green) == "no_low_risk"

    def test_yellow(self):
        assert to_legacy_risk(Tier.yellow) == "concerning"

    def test_red(self):
        assert to_legacy_risk(Tier.red) == "high_risk"


# ──────────────────────────────────────────────────────────────────────────────
# Integration (resolve)
# ──────────────────────────────────────────────────────────────────────────────

class TestResolve:

    def test_legacy_label_set(self):
        d = resolve({"a": Tier.red})
        assert d.legacy_label == "high_risk"

    def test_green_domains_no_flags_no_age(self):
        d = resolve({"a": Tier.green, "b": Tier.green})
        assert d.tier is Tier.green

    def test_minor_abuse_and_lifetime_flag_and_unknown(self):
        """Multiple independent floors should all be tracked."""
        d = resolve(
            {"a": Tier.green, "b": None},
            all_flags={"minor_abuse_disclosure", "past_self_harm"},
            age=15,
        )
        assert d.tier is Tier.yellow
        assert "minor_abuse_disclosure" in d.reasons
        assert "past_self_harm" in d.reasons
        assert "domain_unavailable" in d.reasons

    def test_escalation_plus_minimum(self):
        d = resolve(
            {"a": Tier.yellow},
            all_flags={"lethal_means_access", "past_self_harm"},
        )
        assert d.tier is Tier.red
        assert "lethal_means_access" in d.reasons
        assert "past_self_harm" in d.reasons
