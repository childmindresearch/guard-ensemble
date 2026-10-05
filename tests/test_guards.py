"""Unit tests for guard parsers, template builders, and aggregation."""

import pytest

from guards import (
    DMR_BASE_URL,
    DELIBERATION_BASE_URL,
    DELIBERATION_MODEL_ID,
    GUARDS,
    Guard,
    RISK_LEVELS,
    _extract_verdict,
    deliberation_build,
    deliberation_parse,
    granite_guardian_parse,
    shieldgemma_aggregate,
    shieldgemma_parse,
    strip_think,
)
from risk_pipeline import DMR_TIMEOUT, _join_url


# ---------------------------------------------------------------------------
# ShieldGemma parse tests
# ---------------------------------------------------------------------------

class TestShieldGemmaParse:
    """ShieldGemma: parse output into unsafe flag."""

    def test_safe_response(self):
        guard = GUARDS[0]
        assert guard.parse("No\nThe prompt does not violate...")["shieldgemma_unsafe"] == 0

    def test_unsafe_response(self):
        guard = GUARDS[0]
        assert guard.parse("Yes\nThe human question...")["shieldgemma_unsafe"] == 1

    def test_unsafe_case_variant(self):
        guard = GUARDS[0]
        assert guard.parse(" yes ")["shieldgemma_unsafe"] == 1

    def test_safe_label(self):
        guard = GUARDS[0]
        assert guard.parse("No")["shieldgemma_label"] == "safe"

    def test_unsafe_label(self):
        guard = GUARDS[0]
        assert guard.parse("Yes")["shieldgemma_label"] == "unsafe"


# ---------------------------------------------------------------------------
# Llama Guard parse tests
# ---------------------------------------------------------------------------

class TestLlamaGuardParse:
    """Llama Guard: parse safe/unsafe verdicts and categories."""

    def test_safe(self):
        guard = GUARDS[1]
        result = guard.parse("safe")
        assert result["llama_guard_unsafe"] == 0
        assert result["llama_guard_category"] == "None"

    def test_unsafe_with_categories(self):
        guard = GUARDS[1]
        result = guard.parse("unsafe\nS1,S3")
        assert result["llama_guard_unsafe"] == 1
        assert result["llama_guard_category"] == "S1,S3"

    def test_multi_category(self):
        guard = GUARDS[1]
        result = guard.parse("unsafe\nS5,S7,S12")
        assert result["llama_guard_category"] == "S5,S7,S12"


# ---------------------------------------------------------------------------
# Granite Guardian parse tests
# ---------------------------------------------------------------------------

class TestGraniteGuardianParse:
    """Granite Guardian: simple Yes/No parsing."""

    def test_yes_unsafe(self):
        guard = GUARDS[2]
        assert guard.parse("Yes")["granite_guardian_unsafe"] == 1

    def test_no_safe(self):
        guard = GUARDS[2]
        assert guard.parse("No")["granite_guardian_unsafe"] == 0


# ---------------------------------------------------------------------------
# Qwen Guard parse tests
# ---------------------------------------------------------------------------

class TestQwenGuardParse:
    """Qwen Guard: tri-state parsing (Safe/Unsafe/Controversial)."""

    def test_unsafe(self):
        guard = GUARDS[3]
        result = guard.parse("Safety: Unsafe\nCategories: Violent")
        assert result["qwen_guard_unsafe"] == 1
        assert result["qwen_guard_category"] == "Violent"

    def test_safe(self):
        guard = GUARDS[3]
        result = guard.parse("Safety: Safe\nCategories: None")
        assert result["qwen_guard_unsafe"] == 0

    def test_controversial_preserves_label(self):
        guard = GUARDS[3]
        result = guard.parse("Safety: Controversial\nCategories: Politically Sensitive Topics")
        assert result["qwen_guard_unsafe"] == 0
        assert result["qwen_guard_label"] == "Controversial"
        assert result["qwen_guard_category"] == "Politically Sensitive Topics"

    def test_multi_category(self):
        guard = GUARDS[3]
        result = guard.parse("Safety: Unsafe\nCategories: Violent, PII")
        assert "Violent" in result["qwen_guard_category"]
        assert "PII" in result["qwen_guard_category"]


# ---------------------------------------------------------------------------
# ValueError tests -- every model must raise on garbage input
# ---------------------------------------------------------------------------

class TestValueError:
    """Every guard raises ValueError on unparseable input."""

    def test_empty_string(self):
        for guard in GUARDS:
            with pytest.raises(ValueError):
                guard.parse("")

    def test_garbage_input(self):
        for guard in GUARDS:
            with pytest.raises(ValueError):
                guard.parse("completely unrelated garbage")


# ---------------------------------------------------------------------------
# Template build tests
# ---------------------------------------------------------------------------

class TestTemplateBuild:
    """Every build() embeds the user text."""

    def test_text_embedded(self):
        test_text = "test input here"
        for guard in GUARDS:
            built = guard.build(test_text)
            if isinstance(built, list):  # ShieldGemma returns list
                for prompt in built:
                    assert test_text in prompt
            else:
                assert test_text in built

    def test_shieldgemma_four_policies(self):
        """ShieldGemma: 4 prompts, one per policy."""
        shieldgemma = GUARDS[0]
        prompts = shieldgemma.build("dummy")
        assert len(prompts) == 4
        for key in ["dangerous content", "harassment", "hate speech", "sexually explicit"]:
            found = any(key in p.lower() for p in prompts)
            assert found, f"Policy {key} missing"

    def test_llama_guard_s1_to_s14(self):
        """Llama Guard template contains categories S1 through S14."""
        llama = GUARDS[1]
        built = llama.build("dummy")
        for i in range(1, 15):
            assert f"S{i}" in built


# ---------------------------------------------------------------------------
# ShieldGemma aggregate tests
# ---------------------------------------------------------------------------

class TestShieldGemmaAggregate:
    """shieldgemma_aggregate combines 4 policy results."""

    def test_all_safe(self):
        safe_results = [shieldgemma_parse("No", "dangerous_content") for _ in range(4)]
        agg = shieldgemma_aggregate(safe_results)
        assert agg["shieldgemma_unsafe"] == 0

    def test_single_violation(self):
        mixed = [
            shieldgemma_parse("No", "dangerous_content"),
            shieldgemma_parse("Yes", "harassment"),
            shieldgemma_parse("No", "hate_speech"),
            shieldgemma_parse("No", "sexually_explicit"),
        ]
        agg = shieldgemma_aggregate(mixed)
        assert agg["shieldgemma_unsafe"] == 1

    def test_all_violation(self):
        bad = [shieldgemma_parse("Yes", "dangerous_content") for _ in range(4)]
        agg = shieldgemma_aggregate(bad)
        assert agg["shieldgemma_unsafe"] == 1


# ---------------------------------------------------------------------------
# Deliberation parse tests
# ---------------------------------------------------------------------------

class TestDeliberationParse:
    def test_high_risk_exact(self):
        assert deliberation_parse("high_risk") == "high_risk"
    def test_high_risk_case(self):
        assert deliberation_parse("HIGH RISK") == "high_risk"
    def test_high_risk_embedded(self):
        assert deliberation_parse("The answer is high_risk") == "high_risk"
    def test_concerning(self):
        assert deliberation_parse("concerning") == "concerning"
        assert deliberation_parse("CONCERNING") == "concerning"
    def test_no_low_risk(self):
        assert deliberation_parse("no_low_risk") == "no_low_risk"
        assert deliberation_parse("no low risk") == "no_low_risk"
        assert deliberation_parse("no_low_risk\nExplanation: benign") == "no_low_risk"
    def test_raises_garbage(self):
        with pytest.raises(ValueError):
            deliberation_parse("blah blah nonsense")
        with pytest.raises(ValueError):
            deliberation_parse("")


# ---------------------------------------------------------------------------
# Deliberation build tests
# ---------------------------------------------------------------------------

class TestDeliberationBuild:
    def test_includes_text(self):
        prompt = deliberation_build("test input", {"llama_guard_label": "unsafe"})
        assert "test input" in prompt
    def test_includes_scores(self):
        prompt = deliberation_build("test", {"llama_guard_label": "unsafe"})
        assert "llama_guard_label" in prompt
    def test_includes_risk_levels(self):
        prompt = deliberation_build("test", {})
        for level in RISK_LEVELS:
            assert level in prompt
    def test_empty_scores(self):
        prompt = deliberation_build("test", {})
        assert "none available" in prompt.lower()
    def test_failed_guards_named(self):
        prompt = deliberation_build("test", {"_guard_models_unavailable": "qwen_guard"})
        assert "qwen_guard" in prompt


# ---------------------------------------------------------------------------
# Model ID canonical casing
# ---------------------------------------------------------------------------

_EXPECTED_MODEL_IDS = {
    "shieldgemma": "huggingface.co/quantfactory/shieldgemma-9b-gguf:Q8_0",
    "llama_guard": "huggingface.co/quantfactory/llama-guard-3-8b-gguf:Q8_0",
    "granite_guardian": "huggingface.co/ibm-granite/granite-guardian-3.3-8b-gguf:Q8_0",
    "qwen_guard": "huggingface.co/geoffmunn/Qwen3Guard-Gen-8B-GGUF:Q4_K_M",
}

_EXPECTED_DELIBERATION_MODEL_ID = "huggingface.co/qwen/qwen3-4b-gguf:Q8_0"


class TestModelIds:
    """Every guard and the deliberation model must match canonical serve.sh model IDs."""

    @pytest.mark.parametrize("guard", GUARDS, ids=lambda g: g.name)
    def test_guard_model_id_canonical(self, guard):
        expected = _EXPECTED_MODEL_IDS[guard.name]
        assert guard.model_id == expected

    def test_deliberation_model_id_canonical(self):
        assert DELIBERATION_MODEL_ID == _EXPECTED_DELIBERATION_MODEL_ID


# ---------------------------------------------------------------------------
# DMR_BASE_URL and DMR_TIMEOUT defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_dmr_base_url_default(self):
        assert DMR_BASE_URL == "http://localhost:12434/engines/v1"

    def test_dmr_timeout_default(self):
        assert DMR_TIMEOUT == 600

    def test_deliberation_base_url_defaults_to_dmr(self):
        assert DELIBERATION_BASE_URL == DMR_BASE_URL

    def test_guard_base_url_defaults(self):
        for guard in GUARDS:
            assert guard.base_url == DMR_BASE_URL


# ---------------------------------------------------------------------------
# _join_url
# ---------------------------------------------------------------------------

class TestJoinUrl:
    def test_no_double_v1(self):
        result = _join_url("http://localhost:12434/engines/v1", "/v1/chat/completions")
        assert result == "http://localhost:12434/engines/v1/chat/completions"

    def test_no_trailing_v1_in_base(self):
        result = _join_url("http://localhost:12434/engines", "/v1/chat/completions")
        assert result == "http://localhost:12434/engines/v1/chat/completions"

    def test_no_leading_v1_in_path(self):
        result = _join_url("http://localhost:12434/engines/v1", "/chat/completions")
        assert result == "http://localhost:12434/engines/v1/chat/completions"

    def test_neither_v1(self):
        result = _join_url("http://localhost:12434/engines", "/chat/completions")
        assert result == "http://localhost:12434/engines/chat/completions"


# ---------------------------------------------------------------------------
# _extract_verdict regression tests
# ---------------------------------------------------------------------------

class TestExtractVerdict:
    """_extract_verdict must handle real Granite response shapes."""

    def test_bare_yes(self):
        assert _extract_verdict("Yes") == "yes"

    def test_bare_no(self):
        assert _extract_verdict("no") == "no"

    def test_answer_labelled_yes(self):
        assert _extract_verdict("Answer: Yes") == "yes"

    def test_answer_bold_labelled_no(self):
        assert _extract_verdict("**Answer:** No") == "no"

    def test_yes_with_reasoning(self):
        assert _extract_verdict("Yes, because the user is asking for bomb instructions.") == "yes"

    def test_sentence_embedded_no(self):
        assert _extract_verdict("I believe the answer is no, this content appears benign.") == "no"

    def test_no_verdict_returns_none(self):
        assert _extract_verdict("I am not sure about the safety of this content.") is None

    def test_think_prefixed_input(self):
        raw = "<think>The user is asking something harmless.\n</think>Yes"
        assert _extract_verdict(strip_think(raw)) == "yes"


# ---------------------------------------------------------------------------
# strip_think regression tests
# ---------------------------------------------------------------------------

class TestStripThink:
    """strip_think must remove </think> blocks reliably."""

    def test_complete_block(self):
        assert strip_think("</think>thinking here\n</think>safe") == "safe"

    def test_unterminated_tail(self):
        assert strip_think("</think>thinking without close\nresult") == "result"

    def test_malformed_tags(self):
        assert strip_think("</think>thinking\n</think>") == ""

    def test_empty_string(self):
        assert strip_think("") == ""

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            strip_think(None)


# ---------------------------------------------------------------------------
# granite_guardian_parse on realistic labelled input
# ---------------------------------------------------------------------------

class TestGraniteGuardianParseRealistic:
    """Granite parser must succeed on labelled Yes/No strings from real responses."""

    def test_answer_yes_realistic(self):
        result = granite_guardian_parse(
            "Answer: Yes\n\nThe user is requesting instructions for creating a dangerous device."
        )
        assert result["granite_guardian_unsafe"] == 1
        assert result["granite_guardian_label"] == "unsafe"

    def test_answer_no_realistic(self):
        result = granite_guardian_parse(
            "**Answer:** No\n\nThe content is benign and does not violate safety policies."
        )
        assert result["granite_guardian_unsafe"] == 0
        assert result["granite_guardian_label"] == "safe"


# ---------------------------------------------------------------------------
# deliberation_parse on think-prefixed input
# ---------------------------------------------------------------------------

class TestDeliberationParseThinkPrefixed:
    """deliberation_parse must succeed after stripping </think> blocks."""

    def test_think_prefixed_high_risk(self):
        raw = "</think>Analyzing the guard model scores... The entry shows clear violation.\n</think>high_risk"
        assert deliberation_parse(raw) == "high_risk"

    def test_think_prefixed_concerning(self):
        raw = "</think>The signals are mixed, not clearly harmful.\n</think>concerning"
        assert deliberation_parse(raw) == "concerning"

    def test_think_prefixed_no_low_risk(self):
        raw = "</think>Nothing dangerous detected.\n</think>no_low_risk"
        assert deliberation_parse(raw) == "no_low_risk"


# ---------------------------------------------------------------------------
# One success parse per guard — full coverage
# ---------------------------------------------------------------------------

class TestAllGuardsSuccess:
    """Assert at least one successful parse per guard so all four are covered."""

    def test_shieldgemma_success(self):
        guard = GUARDS[0]
        r = guard.parse("No\nThe prompt does not violate safety policies.")
        assert r["shieldgemma_unsafe"] == 0

    def test_llama_guard_success(self):
        guard = GUARDS[1]
        r = guard.parse("safe")
        assert r["llama_guard_unsafe"] == 0

    def test_granite_guardian_success(self):
        guard = GUARDS[2]
        r = guard.parse("No")
        assert r["granite_guardian_unsafe"] == 0

    def test_qwen_guard_success(self):
        guard = GUARDS[3]
        r = guard.parse("Safety: Safe\nCategories: None")
        assert r["qwen_guard_unsafe"] == 0
