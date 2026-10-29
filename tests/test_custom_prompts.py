"""Tests for custom_prompts.py — per-prompt runner, retry, aggregation, tie-break."""

import json
from unittest import mock

import pytest

from custom_prompts import (
    CUSTOM_PROMPT_MAX_TOKENS,
    DomainPromptResult,
    _timing_more_acute,
    run_domain_prompts,
)
from rubric import Tier, TIER_UNKNOWN, parse_domain_response


# ---------------------------------------------------------------------------
# Helper: build a mock JSON response for a given tier
# ---------------------------------------------------------------------------

def _json_payload(tier="green", timing="current", flags=None, evidence="ok"):
    return {
        "domain": "suicidality",
        "tier": tier,
        "timing": timing,
        "flags": flags or [],
        "evidence": evidence,
    }


def _mock_response(tier="green", timing="current", flags=None, evidence="ok"):
    """Return a mock response object yielding a valid JSON payload."""
    r = mock.Mock()
    r.json.return_value = {
        "choices": [
            {"message": {"content": json.dumps(_json_payload(tier, timing, flags, evidence))}}
        ]
    }
    r.raise_for_status = mock.Mock()
    return r


def _mock_empty_response():
    """Return a mock response yielding empty content."""
    r = mock.Mock()
    r.json.return_value = {"choices": [{"message": {"content": ""}}]}
    r.raise_for_status = mock.Mock()
    return r


# ---------------------------------------------------------------------------
# Basic run
# ---------------------------------------------------------------------------

class TestRunDomainPromptsBasic:
    def test_returns_result_per_domain(self):
        """Each domain in templates yields a result."""
        templates = {
            "suicidality": "$text $age $entry_date",
            "violence_harm_to_others": "$text $age $entry_date",
        }
        caller = mock.Mock(return_value=_mock_response(tier="green", timing="recent"))

        results = run_domain_prompts(
            text="sample text", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller,
        )
        assert "suicidality" in results
        assert "violence_harm_to_others" in results
        assert isinstance(results["suicidality"], DomainPromptResult)

    def test_default_runs_is_three(self):
        """Default runs=3 means 3 calls per domain."""
        templates = {"suicidality": "$text $age $entry_date"}
        caller = mock.Mock(return_value=_mock_response())

        run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller,
        )
        assert caller.call_count == 3

    def test_custom_runs_count(self):
        """runs=1 means exactly one call."""
        templates = {"suicidality": "$text $age $entry_date"}
        caller = mock.Mock(return_value=_mock_response())

        run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=1,
        )
        assert caller.call_count == 1

    def test_two_domains_total_calls(self):
        """Two domains with runs=2 → 4 calls."""
        templates = {
            "a": "$text",
            "b": "$text",
        }
        caller = mock.Mock(return_value=_mock_response())

        run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=2,
        )
        assert caller.call_count == 4


# ---------------------------------------------------------------------------
# Request body contents
# ---------------------------------------------------------------------------

class TestRequestBody:
    def test_temperature_in_body(self):
        """Request body must carry the supplied temperature."""
        templates = {"suicidality": "$text"}
        caller = mock.Mock(return_value=_mock_response())

        run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=1, temperature=0.7,
        )
        body = caller.call_args.kwargs["json"]
        assert body["temperature"] == 0.7

    def test_max_tokens_is_custom_prompt_max_tokens(self):
        """Request body max_tokens equals CUSTOM_PROMPT_MAX_TOKENS."""
        templates = {"suicidality": "$text"}
        caller = mock.Mock(return_value=_mock_response())

        run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=1,
        )
        body = caller.call_args.kwargs["json"]
        assert body["max_tokens"] == CUSTOM_PROMPT_MAX_TOKENS

    def test_thinking_disabled(self):
        """Request body must disable thinking."""
        templates = {"suicidality": "$text"}
        caller = mock.Mock(return_value=_mock_response())

        run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=1,
        )
        body = caller.call_args.kwargs["json"]
        assert body["chat_template_kwargs"]["enable_thinking"] is False

    def test_default_max_tokens_is_512(self):
        assert CUSTOM_PROMPT_MAX_TOKENS == 512


# ---------------------------------------------------------------------------
# Retry on empty/invalid content
# ---------------------------------------------------------------------------

class TestRetry:
    def test_retry_on_empty_first_call(self):
        """If first call returns empty, retry once."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_empty_response()
            return _mock_response()

        templates = {"suicidality": "$text"}
        caller = mock.Mock(side_effect=side_effect)

        results = run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=1,
        )
        assert caller.call_count == 2
        assert results["suicidality"].tier == "green"

    def test_both_attempts_fail_records_error(self):
        """If both attempts return empty, run recorded as error, tier unknown."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return _mock_empty_response()

        templates = {"suicidality": "$text"}
        caller = mock.Mock(side_effect=side_effect)

        results = run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=1,
        )
        assert caller.call_count == 2  # initial + retry
        assert results["suicidality"].tier == TIER_UNKNOWN
        assert len(results["suicidality"].errors) > 0

    def test_retry_count_per_run(self):
        """Each run retries independently; runs=2, both empty → 4 calls."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return _mock_empty_response()

        templates = {"suicidality": "$text"}
        caller = mock.Mock(side_effect=side_effect)

        run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=2,
        )
        assert call_count[0] == 4  # 2 runs * 2 attempts each


# ---------------------------------------------------------------------------
# Failure isolation: partial failures do not block aggregation
# ---------------------------------------------------------------------------

class TestFailureIsolation:
    def test_partial_failure_still_aggregates(self):
        """2 of 3 runs succeed → aggregation still works."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("network error")
            return _mock_response(tier="red", timing="imminent")

        templates = {"suicidality": "$text"}
        caller = mock.Mock(side_effect=side_effect)

        results = run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=3,
        )
        assert results["suicidality"].tier == "red"
        assert results["suicidality"].timing == "imminent"

    def test_all_runs_fail_yields_unknown(self):
        """All runs raise → tier is unknown."""
        def side_effect(*args, **kwargs):
            raise RuntimeError("connection refused")

        templates = {"suicidality": "$text"}
        caller = mock.Mock(side_effect=side_effect)

        results = run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=3,
        )
        assert results["suicidality"].tier == TIER_UNKNOWN
        assert len(results["suicidality"].errors) > 0


# ---------------------------------------------------------------------------
# Aggregation: severity-max tier
# ---------------------------------------------------------------------------

class TestSeverityMax:
    def test_red_wins_over_green(self):
        """Highest tier across runs wins."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_response(tier="green", timing="recent")
            if call_count[0] == 2:
                return _mock_response(tier="red", timing="imminent")
            return _mock_response(tier="yellow", timing="current")

        templates = {"suicidality": "$text"}
        caller = mock.Mock(side_effect=side_effect)

        results = run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=3,
        )
        assert results["suicidality"].tier == "red"

    def test_yellow_wins_over_green(self):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_response(tier="green")
            return _mock_response(tier="yellow")

        templates = {"suicidality": "$text"}
        caller = mock.Mock(side_effect=side_effect)

        results = run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=2,
        )
        assert results["suicidality"].tier == "yellow"


# ---------------------------------------------------------------------------
# Flag union
# ---------------------------------------------------------------------------

class TestFlagUnion:
    def test_flags_are_unioned(self):
        """Flags from different runs are OR-ed."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_response(
                    tier="green", flags=["lethal_means_access"]
                )
            return _mock_response(
                tier="green", flags=["alone_no_support"]
            )

        templates = {"suicidality": "$text"}
        caller = mock.Mock(side_effect=side_effect)

        results = run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=2,
        )
        flags = set(results["suicidality"].flags)
        assert "lethal_means_access" in flags
        assert "alone_no_support" in flags


# ---------------------------------------------------------------------------
# Timing choice: from highest tier, tie-break toward more acute
# ---------------------------------------------------------------------------

class TestTimingChoice:
    def test_timing_from_highest_tier(self):
        """Timing comes from the run that produced the winning tier."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_response(tier="green", timing="recent")
            if call_count[0] == 2:
                return _mock_response(tier="red", timing="imminent")
            return _mock_response(tier="yellow", timing="current")

        templates = {"suicidality": "$text"}
        caller = mock.Mock(side_effect=side_effect)

        results = run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=3,
        )
        assert results["suicidality"].timing == "imminent"

    def test_tiebreak_more_acute_timing(self):
        """When tiers tie, more acute timing wins."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_response(tier="red", timing="current")
            return _mock_response(tier="red", timing="imminent")

        templates = {"suicidality": "$text"}
        caller = mock.Mock(side_effect=side_effect)

        results = run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=2,
        )
        assert results["suicidality"].timing == "imminent"

    def test_tiebreak_acutity_order(self):
        """imminent > current > recent > historical_unresolved > historical_resolved."""
        assert _timing_more_acute("imminent", "current")
        assert _timing_more_acute("current", "recent")
        assert _timing_more_acute("recent", "historical_unresolved")
        assert _timing_more_acute("historical_unresolved", "historical_resolved")
        assert not _timing_more_acute("historical_resolved", "imminent")


# ---------------------------------------------------------------------------
# Evidence from highest tier
# ---------------------------------------------------------------------------

class TestEvidence:
    def test_evidence_from_highest_tier(self):
        """Evidence string comes from the highest-tier run."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_response(tier="green", evidence="green_evidence")
            return _mock_response(tier="red", evidence="red_evidence")

        templates = {"suicidality": "$text"}
        caller = mock.Mock(side_effect=side_effect)

        results = run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=2,
        )
        assert results["suicidality"].evidence == "red_evidence"


# ---------------------------------------------------------------------------
# Network I/O exclusively through caller
# ---------------------------------------------------------------------------

class TestCallerInjection:
    def test_caller_is_only_network_path(self):
        """The caller is invoked exactly `runs` times per domain (no hidden I/O)."""
        templates = {"suicidality": "$text"}
        caller = mock.Mock(return_value=_mock_response())

        run_domain_prompts(
            text="t", age="25", entry_date="2024-01-01",
            templates=templates, caller=caller, runs=3,
        )
        assert caller.call_count == 3

    def test_caller_receives_proper_url_and_body(self):
        """Caller receives URL as first arg, body as json kwarg."""
        templates = {"suicidality": "$text"}
        caller = mock.Mock(return_value=_mock_response())

        run_domain_prompts(
            text="hello", age="30", entry_date="2025-01-01",
            templates=templates, caller=caller, runs=1,
        )
        call_args = caller.call_args
        assert call_args.args[0] == "http://localhost:12434/engines/v1/chat/completions"
        body = call_args.kwargs["json"]
        assert body["messages"][0]["content"] == "hello"
