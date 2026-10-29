"""Tests for deliberation module: builders, parser, clamp, raise-only, retry, template validation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deliberation import (
    VALID_TIERS,
    _CONSOLIDATION_REQUIRED,
    _FINAL_UPDATE_REQUIRED,
    _find_placeholder,
    _tier_index,
    clamp_tier,
    consolidation_build,
    final_update_build,
    load_consolidation_prompt,
    load_final_update_prompt,
    parse_deliberation_response,
    run_phase_with_retry,
)

# Clear lru_cache so templates are reloaded each test session
load_consolidation_prompt.cache_clear()
load_final_update_prompt.cache_clear()


# ---------------------------------------------------------------------------
# Template validation
# ---------------------------------------------------------------------------

class TestTemplateValidation:
    def test_consolidation_loads_default(self):
        prompt = load_consolidation_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_consolidation_has_all_placeholders(self):
        prompt = load_consolidation_prompt()
        for ph in _CONSOLIDATION_REQUIRED:
            assert ph in prompt

    def test_final_update_loads_default(self):
        prompt = load_final_update_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_final_update_has_all_placeholders(self):
        prompt = load_final_update_prompt()
        for ph in _FINAL_UPDATE_REQUIRED:
            assert ph in prompt

    def test_consolidation_missing_placeholder(self, tmp_path):
        bad = tmp_path / "bad.txt"
        bad.write_text("$text only")
        with pytest.raises(ValueError, match="missing placeholder"):
            load_consolidation_prompt(str(bad))

    def test_consolidation_unknown_placeholder(self, tmp_path):
        bad = tmp_path / "bad.txt"
        good_placeholders = " ".join(f"${p}" for p in _CONSOLIDATION_REQUIRED)
        bad.write_text(f"{good_placeholders} $unknown_ph")
        with pytest.raises(ValueError, match="unknown placeholder"):
            load_consolidation_prompt(str(bad))

    def test_final_update_missing_consolidation_block(self, tmp_path):
        bad = tmp_path / "bad.txt"
        bad.write_text(" ".join(_CONSOLIDATION_REQUIRED))
        with pytest.raises(ValueError, match="missing placeholder"):
            load_final_update_prompt(str(bad))

    def test_final_update_unknown_placeholder(self, tmp_path):
        bad = tmp_path / "bad.txt"
        good_placeholders = " ".join(f"${p}" for p in _FINAL_UPDATE_REQUIRED)
        bad.write_text(f"{good_placeholders} $bogus")
        with pytest.raises(ValueError, match="unknown placeholder"):
            load_final_update_prompt(str(bad))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

class TestConsolidationBuild:
    def test_substitutes_all_placeholders(self):
        result = consolidation_build(
            text="sample text",
            context_block="ctx",
            guard_block="guard",
            domain_block="domain",
            rule_block="rules",
        )
        assert "sample text" in result
        assert "ctx" in result
        assert "guard" in result
        assert "domain" in result
        assert "rules" in result

    def test_no_dollar_residue(self):
        result = consolidation_build(
            text="hello",
            context_block="ctx",
            guard_block="guard",
            domain_block="domain",
            rule_block="rules",
        )
        import re
        assert not re.search(r"\$[A-Za-z_]", result)

    def test_custom_prompt(self):
        template = "TEXT: $text GUARD: $guard_block"
        result = consolidation_build(
            text="t",
            context_block="c",
            guard_block="g",
            domain_block="d",
            rule_block="r",
            prompt=template,
        )
        assert "$" not in result
        assert "t" in result
        assert "g" in result

    def test_dollar_in_block_value_is_ok(self):
        # $ can appear in block values — only unreplaced placeholders are bad
        result = consolidation_build(
            text="x",
            context_block="cost $5",
            guard_block="g",
            domain_block="d",
            rule_block="r",
        )
        # The $5 is in the substituted content, not a placeholder
        assert "$5" in result


class TestFinalUpdateBuild:
    def test_substitutes_all_placeholders(self):
        result = final_update_build(
            text="sample text",
            context_block="ctx",
            guard_block="guard",
            domain_block="domain",
            rule_block="rules",
            consolidation_block="consol",
        )
        assert "sample text" in result
        assert "ctx" in result
        assert "guard" in result
        assert "domain" in result
        assert "rules" in result
        assert "consol" in result

    def test_no_dollar_residue(self):
        result = final_update_build(
            text="hello",
            context_block="ctx",
            guard_block="guard",
            domain_block="domain",
            rule_block="rules",
            consolidation_block="consol",
        )
        import re
        assert not re.search(r"\$[A-Za-z_]", result)

    def test_custom_prompt(self):
        template = "TEXT: $text CONSOL: $consolidation_block"
        result = final_update_build(
            text="t",
            context_block="c",
            guard_block="g",
            domain_block="d",
            rule_block="r",
            consolidation_block="x",
            prompt=template,
        )
        assert "$" not in result
        assert "t" in result
        assert "x" in result


# ---------------------------------------------------------------------------
# Strict JSON parser
# ---------------------------------------------------------------------------

class TestParseDeliberationResponse:
    def test_valid_green(self):
        raw = json.dumps({"tier": "green", "rationale": "safe"})
        tier, rationale = parse_deliberation_response(raw)
        assert tier == "green"
        assert rationale == "safe"

    def test_valid_yellow(self):
        raw = json.dumps({"tier": "yellow", "rationale": "concern"})
        tier, rationale = parse_deliberation_response(raw)
        assert tier == "yellow"

    def test_valid_red(self):
        raw = json.dumps({"tier": "red", "rationale": "acute risk"})
        tier, rationale = parse_deliberation_response(raw)
        assert tier == "red"

    def test_rationale_defaults_to_empty(self):
        raw = json.dumps({"tier": "green"})
        tier, rationale = parse_deliberation_response(raw)
        assert rationale == ""

    def test_strips_think_blocks(self):
        raw = '<antThinking>thinking here</antThinking>{"tier": "red", "rationale": "risk"}'
        tier, rationale = parse_deliberation_response(raw)
        assert tier == "red"

    def test_strips_unterminated_think(self):
        raw = '<antThinking>thinking without close\n{"tier": "yellow", "rationale": "x"}'
        tier, rationale = parse_deliberation_response(raw)
        assert tier == "yellow"

    def test_strips_code_fences(self):
        raw = '```json\n{"tier": "green", "rationale": "ok"}\n```'
        tier, rationale = parse_deliberation_response(raw)
        assert tier == "green"

    def test_json_embedded_in_prose(self):
        raw = 'Here is my assessment: {"tier": "yellow", "rationale": "concerning"}'
        tier, rationale = parse_deliberation_response(raw)
        assert tier == "yellow"

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty input"):
            parse_deliberation_response("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty input"):
            parse_deliberation_response("   \n  ")

    def test_invalid_tier_value_raises(self):
        raw = json.dumps({"tier": "orange", "rationale": "x"})
        with pytest.raises(ValueError, match="invalid tier"):
            parse_deliberation_response(raw)

    def test_missing_tier_key_raises(self):
        raw = json.dumps({"rationale": "no tier here"})
        with pytest.raises(ValueError, match="missing 'tier'"):
            parse_deliberation_response(raw)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="no valid JSON"):
            parse_deliberation_response("not json at all")

    def test_think_block_with_no_json_inside(self):
        raw = "<antThinking>thinking\n</antThinking>"
        with pytest.raises(ValueError, match="empty input"):
            parse_deliberation_response(raw)


# ---------------------------------------------------------------------------
# Monotonic clamp
# ---------------------------------------------------------------------------

class TestClampTier:
    def test_same_tier_unchanged(self):
        for tier in VALID_TIERS:
            result, adjusted = clamp_tier(tier, tier)
            assert result == tier
            assert adjusted is False

    def test_raise_allowed(self):
        result, adjusted = clamp_tier("yellow", "green")
        assert result == "yellow"
        assert adjusted is False

    def test_raise_to_red_allowed(self):
        result, adjusted = clamp_tier("red", "green")
        assert result == "red"
        assert adjusted is False

    def test_lowering_green_to_green_clamped(self):
        # proposed green, floor yellow → clamped to yellow
        result, adjusted = clamp_tier("green", "yellow")
        assert result == "yellow"
        assert adjusted is True

    def test_lowering_red_to_yellow_clamped(self):
        result, adjusted = clamp_tier("yellow", "red")
        assert result == "red"
        assert adjusted is True

    def test_lowering_green_to_red_clamped(self):
        result, adjusted = clamp_tier("green", "red")
        assert result == "red"
        assert adjusted is True

    def test_lowering_yellow_to_red_clamped(self):
        result, adjusted = clamp_tier("yellow", "red")
        assert result == "red"
        assert adjusted is True

    def test_step_down_green_from_yellow_clamped(self):
        result, adjusted = clamp_tier("green", "yellow")
        assert result == "yellow"
        assert adjusted is True


# ---------------------------------------------------------------------------
# Raise-only semantics
# ---------------------------------------------------------------------------

class TestRaiseOnly:
    """Clamp + run_phase must never return a tier below the floor."""

    def test_consolidation_never_below_floor(self):
        # Every valid proposed tier vs floor combination
        for floor in VALID_TIERS:
            for proposed in VALID_TIERS:
                result, adjusted = clamp_tier(proposed, floor)
                assert _tier_index(result) >= _tier_index(floor), (
                    f"clamp({proposed}, {floor}) = {result} is below floor {floor}"
                )

    def test_run_phase_respects_floor_on_success(self):
        fn = MagicMock(return_value=json.dumps({"tier": "green", "rationale": "safe"}))
        tier, err = run_phase_with_retry(fn, "prompt", floor_tier="red")
        assert tier == "red"
        assert err == ""

    def test_run_phase_allows_raise(self):
        fn = MagicMock(return_value=json.dumps({"tier": "red", "rationale": "risk"}))
        tier, err = run_phase_with_retry(fn, "prompt", floor_tier="green")
        assert tier == "red"
        assert err == ""


# ---------------------------------------------------------------------------
# Retry + fallback
# ---------------------------------------------------------------------------

class TestRetry:
    def test_first_attempt_succeeds(self):
        fn = MagicMock(return_value=json.dumps({"tier": "yellow", "rationale": "x"}))
        tier, err = run_phase_with_retry(fn, "prompt", floor_tier="green")
        assert tier == "yellow"
        assert err == ""
        assert fn.call_count == 1

    def test_second_attempt_succeeds(self):
        fn = MagicMock(side_effect=[ValueError("bad"), json.dumps({"tier": "green", "rationale": "ok"})])
        tier, err = run_phase_with_retry(fn, "prompt", floor_tier="green")
        assert tier == "green"
        assert err == ""
        assert fn.call_count == 2

    def test_both_fail_fallback_to_floor(self):
        fn = MagicMock(side_effect=[ValueError("err1"), ValueError("err2")])
        tier, err = run_phase_with_retry(fn, "prompt", floor_tier="yellow")
        assert tier == "yellow"
        assert "retry exhausted" in err
        assert fn.call_count == 2

    def test_empty_response_retried(self):
        fn = MagicMock(side_effect=["", json.dumps({"tier": "green", "rationale": "ok"})])
        tier, err = run_phase_with_retry(fn, "prompt", floor_tier="green")
        assert tier == "green"
        assert err == ""
        assert fn.call_count == 2

    def test_invalid_tier_on_first_retried(self):
        fn = MagicMock(
            side_effect=[
                json.dumps({"tier": "invalid", "rationale": "x"}),
                json.dumps({"tier": "yellow", "rationale": "ok"}),
            ]
        )
        tier, err = run_phase_with_retry(fn, "prompt", floor_tier="green")
        assert tier == "yellow"
        assert err == ""
        assert fn.call_count == 2

    def test_fallback_error_string_is_non_empty(self):
        fn = MagicMock(side_effect=[ValueError("x"), ValueError("y")])
        _, err = run_phase_with_retry(fn, "prompt", floor_tier="red")
        assert len(err) > 0
