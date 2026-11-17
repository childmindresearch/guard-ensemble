"""Tests for rubric.py — tier constants, domains, flags, prompts, parsing, aggregation."""

import json
import os
import tempfile
from textwrap import dedent

import pytest

from rubric import (
    CANONICAL_DOMAINS,
    ESCALATION_FLAGS,
    FLAG_VOCABULARY,
    MINIMUM_FLAGS,
    TIMING_VOCABULARY,
    Tier,
    TIER_UNKNOWN,
    AggregatedDomainRun,
    ParsedDomainResponse,
    aggregate_domain_runs,
    load_domain_prompt_templates,
    parse_domain_response,
)


# ── Tier ordering ────────────────────────────────────────────────────────────

class TestTierOrdering:
    def test_green_lt_yellow(self):
        assert Tier.green < Tier.yellow

    def test_yellow_lt_red(self):
        assert Tier.yellow < Tier.red

    def test_green_lt_red(self):
        assert Tier.green < Tier.red

    def test_unknown_is_separate_constant(self):
        assert TIER_UNKNOWN == "unknown"
        assert TIER_UNKNOWN not in (Tier.green, Tier.yellow, Tier.red)


# ── Canonical domains ────────────────────────────────────────────────────────

EXPECTED_DOMAINS = [
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
]


class TestCanonicalDomains:
    def test_count_is_21(self):
        assert len(CANONICAL_DOMAINS) == 21

    def test_contains_expected_slugs(self):
        assert list(CANONICAL_DOMAINS) == EXPECTED_DOMAINS


# ── Flag vocabulary ─────────────────────────────────────────────────────────

class TestFlagVocabulary:
    def test_escalation_flags(self):
        assert "lethal_means_access" in ESCALATION_FLAGS
        assert "current_intoxication" in ESCALATION_FLAGS
        assert "alone_no_support" in ESCALATION_FLAGS

    def test_minimum_flags(self):
        assert "minor_abuse_disclosure" in MINIMUM_FLAGS
        assert "past_suicide_attempt" in MINIMUM_FLAGS
        assert "past_self_harm" in MINIMUM_FLAGS
        assert "past_sexual_assault" in MINIMUM_FLAGS
        assert "past_psychosis" in MINIMUM_FLAGS
        assert "past_eating_disorder" in MINIMUM_FLAGS

    def test_timing_vocabulary(self):
        expected = {
            "historical_resolved",
            "historical_unresolved",
            "recent",
            "current",
            "imminent",
            "unclear",
        }
        assert set(TIMING_VOCABULARY) == expected

    def test_flag_vocabulary_structure(self):
        assert "escalation" in FLAG_VOCABULARY
        assert "minimum" in FLAG_VOCABULARY
        assert "timing" in FLAG_VOCABULARY


# ── Prompt assembly ─────────────────────────────────────────────────────────

class TestLoadDomainPromptTemplates:
    def _write_files(self, tmp_dir, files):
        for name, content in files.items():
            path = os.path.join(tmp_dir, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)

    def test_basic_assembly(self, tmp_path):
        self._write_files(str(tmp_path), {
            "prelude.txt": "Prelude $text",
            "schema.txt": "Schema $age",
            "foo.txt": "Body $entry_date",
        })
        result = load_domain_prompt_templates(
            str(tmp_path),
            os.path.join(str(tmp_path), "prelude.txt"),
            os.path.join(str(tmp_path), "schema.txt"),
        )
        assert "foo" in result
        assert "Prelude" in result["foo"]
        assert "Body" in result["foo"]
        assert "Schema" in result["foo"]

    def test_skips_shared_files(self, tmp_path):
        self._write_files(str(tmp_path), {
            "prelude.txt": "P",
            "schema.txt": "S",
            "foo.txt": "$text",
            "foo_shared.txt": "SKIP ME $text",
        })
        result = load_domain_prompt_templates(
            str(tmp_path),
            os.path.join(str(tmp_path), "prelude.txt"),
            os.path.join(str(tmp_path), "schema.txt"),
        )
        assert "foo" in result
        assert "foo_shared" not in result

    def test_validates_required_placeholders(self, tmp_path):
        # $text, $age, $entry_date are required in domain body — no, they are
        # just placeholders that must be recognized (not unknown). The validation
        # is that *unknown* placeholders are rejected.
        self._write_files(str(tmp_path), {
            "prelude.txt": "P",
            "schema.txt": "S",
            "foo.txt": "$text $age $entry_date",
        })
        result = load_domain_prompt_templates(
            str(tmp_path),
            os.path.join(str(tmp_path), "prelude.txt"),
            os.path.join(str(tmp_path), "schema.txt"),
        )
        assert "foo" in result

    def test_optional_scores_block_accepted(self, tmp_path):
        self._write_files(str(tmp_path), {
            "prelude.txt": "P",
            "schema.txt": "S",
            "foo.txt": "$text $scores_block",
        })
        result = load_domain_prompt_templates(
            str(tmp_path),
            os.path.join(str(tmp_path), "prelude.txt"),
            os.path.join(str(tmp_path), "schema.txt"),
        )
        assert "foo" in result

    def test_unknown_placeholder_rejected(self, tmp_path):
        self._write_files(str(tmp_path), {
            "prelude.txt": "P",
            "schema.txt": "S",
            "bad.txt": "$text $mystery_placeholder",
        })
        with pytest.raises(ValueError, match="unknown placeholder"):
            load_domain_prompt_templates(
                str(tmp_path),
                os.path.join(str(tmp_path), "prelude.txt"),
                os.path.join(str(tmp_path), "schema.txt"),
            )


# ── Response parsing ────────────────────────────────────────────────────────

class TestParseDomainResponse:
    def _json_payload(self, **overrides):
        base = {
            "domain": "suicidality",
            "tier": "green",
            "timing": "current",
            "flags": [],
            "evidence": "test",
        }
        base.update(overrides)
        return json.dumps(base)

    def test_valid_json(self):
        resp = parse_domain_response(self._json_payload(tier="red"))
        assert resp.tier is Tier.red
        assert resp.domain == "suicidality"
        assert not resp.errors

    def test_invalid_json(self):
        resp = parse_domain_response("not json at all")
        assert resp.tier is None
        assert len(resp.errors) > 0

    def test_think_block_stripped(self):
        raw = "<think>thinking here</think>\n" + self._json_payload()
        resp = parse_domain_response(raw)
        assert resp.tier is Tier.green
        assert not resp.errors

    def test_code_fence_stripped(self):
        raw = "```\n" + self._json_payload() + "\n```"
        resp = parse_domain_response(raw)
        assert resp.tier is Tier.green
        assert not resp.errors

    def test_unknown_tier_recorded_as_error(self):
        resp = parse_domain_response(self._json_payload(tier="purple"))
        assert resp.tier is None
        assert any(e.kind == "unknown_tier" for e in resp.errors)

    def test_unknown_flag_recorded_not_applied(self):
        resp = parse_domain_response(
            self._json_payload(flags=["lethal_means_access", "fake_flag"])
        )
        assert "lethal_means_access" in resp.flags
        assert "fake_flag" not in resp.flags
        assert any(e.kind == "unknown_flag" for e in resp.errors)

    def test_all_flags_valid(self):
        resp = parse_domain_response(
            self._json_payload(flags=["lethal_means_access", "past_self_harm"])
        )
        assert resp.flags == ["lethal_means_access", "past_self_harm"]
        assert not resp.errors


# ── Aggregation ──────────────────────────────────────────────────────────────

class TestAggregateDomainRuns:
    def _run(self, tier=None, timing=None, flags=None, errors=None):
        return ParsedDomainResponse(
            domain="suicidality",
            tier=tier,
            timing=timing,
            flags=flags or [],
            errors=errors or [],
        )

    def test_severity_max_tier(self):
        runs = [
            self._run(tier=Tier.green, timing="recent"),
            self._run(tier=Tier.red, timing="current"),
        ]
        agg = aggregate_domain_runs(runs)
        assert agg.tier is Tier.red

    def test_flags_union(self):
        runs = [
            self._run(tier=Tier.green, flags=["lethal_means_access"]),
            self._run(tier=Tier.green, flags=["alone_no_support"]),
        ]
        agg = aggregate_domain_runs(runs)
        assert set(agg.flags) == {"lethal_means_access", "alone_no_support"}

    def test_timing_from_highest_tier(self):
        runs = [
            self._run(tier=Tier.green, timing="recent"),
            self._run(tier=Tier.red, timing="current"),
        ]
        agg = aggregate_domain_runs(runs)
        assert agg.timing == "current"

    def test_all_failed_yields_unknown(self):
        from rubric import DomainParseError

        runs = [
            self._run(
                tier=None,
                errors=[DomainParseError("json", "broken")],
            ),
            self._run(
                tier=None,
                errors=[DomainParseError("json", "also broken")],
            ),
        ]
        agg = aggregate_domain_runs(runs)
        assert agg.tier is None
        assert len(agg.run_errors) == 2

    def test_run_errors_recorded(self):
        from rubric import DomainParseError

        runs = [
            self._run(
                tier=Tier.green,
                errors=[DomainParseError("unknown_flag", "flag=x")],
            ),
        ]
        agg = aggregate_domain_runs(runs)
        assert len(agg.run_errors) == 1
        assert "unknown_flag" in agg.run_errors[0]

    def test_empty_runs(self):
        agg = aggregate_domain_runs([])
        assert agg.tier is None
        assert agg.timing is None
