"""Tests for the Kev client: loader, builder, HTTP client, and response parser."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from kev_client import (
    KEV_DIR,
    KEV_FLAG_THRESHOLD,
    KEV_TIMEOUT,
    KEV_MODEL,
    KEV_BASE_URL,
    KEV_API_KEY,
    TIER_NAMES,
    _decode_choice,
    _decode_noul,
    _decode_score,
    _post_kev,
    build_kev_questions,
    load_kev_questions,
    run_kev_domains,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_TEXT = "I am feeling ok today."
SAMPLE_AGE = 16
SAMPLE_DATE = "2025-01-15"


@pytest.fixture
def keq():
    """Load the real Kev YAML config."""
    return load_kev_questions()


# ── Loader tests ──────────────────────────────────────────────────────────────


class TestLoadKevQuestions:

    def test_returns_expected_keys(self, keq: dict) -> None:
        for key in ("reading_rules", "timing", "flags", "domains"):
            assert key in keq

    def test_21_domains(self, keq: dict) -> None:
        assert len(keq["domains"]) == 21

    def test_9_flags(self, keq: dict) -> None:
        assert len(keq["flags"]) == 9

    def test_reading_rules_is_string(self, keq: dict) -> None:
        assert isinstance(keq["reading_rules"], str)
        assert len(keq["reading_rules"]) > 0

    def test_domain_has_slug(self, keq: dict) -> None:
        for d in keq["domains"]:
            assert isinstance(d["domain"], str)
            assert len(d["domain"]) > 0

    def test_domain_tier_criteria_count(self, keq: dict) -> None:
        for d in keq["domains"]:
            assert len(d["tier"]["criteria"]) == 3

    def test_shared_timing_is_choice(self, keq: dict) -> None:
        assert keq["timing"]["type"] == "choice"

    def test_flags_are_noul(self, keq: dict) -> None:
        for fdef in keq["flags"].values():
            assert fdef["type"] == "noul"

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with patch.object(KEV_DIR, "__class__", Path):
            with patch("kev_client.KEV_DIR", tmp_path):
                with pytest.raises(ValueError, match="No domain YAML"):
                    load_kev_questions()

    def test_malformed_shared_raises(self, tmp_path: Path) -> None:
        shared = tmp_path / "_shared.yaml"
        shared.write_text("not: a: valid: yaml: mapping")
        with patch("kev_client.KEV_DIR", tmp_path):
            with pytest.raises(ValueError):
                load_kev_questions()


# ── Question builder tests ───────────────────────────────────────────────────


class TestBuildKevQuestions:

    def test_question_count(self, keq: dict) -> None:
        # 21 domains * 2 (tier + timing) + 9 flags = 51
        questions = build_kev_questions(keq)
        assert len(questions) == 21 * 2 + 9

    def test_tier_questions_exist(self, keq: dict) -> None:
        questions = build_kev_questions(keq)
        tier_ids = [q["id"] for q in questions if q["type"] == "score"]
        assert len(tier_ids) == 21
        for slug in (d["domain"] for d in keq["domains"]):
            assert f"{slug}__tier" in tier_ids

    def test_timing_questions_exist(self, keq: dict) -> None:
        questions = build_kev_questions(keq)
        timing_ids = [q["id"] for q in questions if q["type"] == "choice"]
        assert len(timing_ids) == 21

    def test_flag_questions_exist(self, keq: dict) -> None:
        questions = build_kev_questions(keq)
        flag_ids = [q["id"] for q in questions if q["type"] == "noul"]
        assert len(flag_ids) == 9

    def test_reading_rules_injected(self, keq: dict) -> None:
        questions = build_kev_questions(keq)
        tier_qs = [q for q in questions if q["type"] == "score"]
        for q in tier_qs:
            assert "<reading_rules>" not in q["instructions"]
            assert keq["reading_rules"] in q["instructions"]


# ── Score decoder tests ──────────────────────────────────────────────────────


class TestDecodeScore:

    def test_clear_green(self) -> None:
        answer = {"score": 0.5, "probabilities": {"0": 0.9, "1": 0.08, "2": 0.02}, "confidence": 0.95}
        result = _decode_score(answer)
        assert result["tier"] == "green"
        assert result["mean_score"] == 0.5
        assert result["confidence"] == 0.95
        assert result["mode_probability"] == 0.9

    def test_clear_red(self) -> None:
        answer = {"score": 2.5, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "confidence": 0.8}
        result = _decode_score(answer)
        assert result["tier"] == "red"
        assert result["mode_probability"] == 0.7

    def test_tie_higher_tier_wins(self) -> None:
        answer = {"score": 1.0, "probabilities": {"0": 0.5, "1": 0.5, "2": 0.0}, "confidence": 0.4}
        result = _decode_score(answer)
        assert result["tier"] == "yellow"  # tie between 0 and 1 -> higher index (1)

    def test_three_way_tie(self) -> None:
        answer = {"score": 1.0, "probabilities": {"0": 0.33, "1": 0.33, "2": 0.34}, "confidence": 0.3}
        result = _decode_score(answer)
        assert result["tier"] == "red"

    def test_empty_probabilities(self) -> None:
        answer = {"score": 1.0, "probabilities": {}, "confidence": None}
        result = _decode_score(answer)
        assert result["tier"] == "unknown"
        assert result["mean_score"] is None

    def test_no_probabilities_key(self) -> None:
        answer = {"score": 1.0}
        result = _decode_score(answer)
        assert result["tier"] == "unknown"


# ── Choice decoder tests ─────────────────────────────────────────────────────


class TestDecodeChoice:

    def test_normal_choice(self) -> None:
        answer = {"choice": "recent", "probabilities": {"recent": 0.8}, "confidence": 0.9}
        result = _decode_choice(answer)
        assert result["timing"] == "recent"
        assert result["confidence"] == 0.9

    def test_minimal_choice(self) -> None:
        answer = {}
        result = _decode_choice(answer)
        assert result["timing"] is None
        assert result["probabilities"] == {}


# ── Noul decoder tests ────────────────────────────────────────────────────────


class TestDecodeNoul:

    def test_above_threshold(self) -> None:
        result = _decode_noul({"noul": 0.8})
        assert result["flagged"] is True

    def test_below_threshold(self) -> None:
        result = _decode_noul({"noul": 0.3})
        assert result["flagged"] is False

    def test_at_threshold(self) -> None:
        result = _decode_noul({"noul": KEV_FLAG_THRESHOLD})
        assert result["flagged"] is True

    def test_none_probability(self) -> None:
        result = _decode_noul({"noul": None})
        assert result["flagged"] is False


# ── HTTP client tests ────────────────────────────────────────────────────────


class TestPostKev:

    def test_success(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"answers": {}}
        with patch("kev_client.requests.post", return_value=mock_resp):
            result = _post_kev({"model": "x", "state": {}, "questions": []})
        assert result == {"answers": {}}

    def test_5xx_retries(self) -> None:
        calls = []
        def side_effect(*a, **kw):
            calls.append(1)
            r = MagicMock()
            r.status_code = 500
            r.text = "Internal"
            return r
        with patch("kev_client.requests.post", side_effect=side_effect):
            with pytest.raises(Exception, match="Server error"):
                _post_kev({"model": "x", "state": {}, "questions": []}, max_retries=1)
        assert len(calls) == 2  # initial + 1 retry

    def test_4xx_no_retry(self) -> None:
        calls = []
        def side_effect(*a, **kw):
            calls.append(1)
            r = MagicMock()
            r.status_code = 400
            r.text = "Bad"
            return r
        with patch("kev_client.requests.post", side_effect=side_effect):
            with pytest.raises(Exception, match="400"):
                _post_kev({"model": "x", "state": {}, "questions": []}, max_retries=2)
        assert len(calls) == 1  # no retry for 4xx

    def test_connection_error_retries(self) -> None:
        calls = []
        def side_effect(*a, **kw):
            calls.append(1)
            raise requests.ConnectionError("timeout")
        with patch("kev_client.requests.post", side_effect=side_effect):
            with pytest.raises(Exception):
                _post_kev({"model": "x", "state": {}, "questions": []}, max_retries=1)
        assert len(calls) == 2


# ── run_kev_domains integration tests ─────────────────────────────────────────


class TestRunKevDomains:

    def _mock_resp(self, domains: list[str], flags: dict[str, float]):
        answers = {}
        for slug in domains:
            answers[f"{slug}__tier"] = {
                "score": 0.5,
                "probabilities": {"0": 0.9, "1": 0.08, "2": 0.02},
                "confidence": 0.95,
            }
            answers[f"{slug}__timing"] = {
                "choice": "recent",
                "probabilities": {"recent": 0.8},
                "confidence": 0.9,
            }
        for fname, prob in flags.items():
            answers[f"flag__{fname}"] = {"noul": prob}
        return {"answers": answers}

    def test_full_success(self, keq: dict) -> None:
        domain_slugs = [d["domain"] for d in keq["domains"]]
        flag_probs = {name: 0.1 for name in keq["flags"]}
        mock_resp = self._mock_resp(domain_slugs, flag_probs)

        with patch("kev_client._post_kev", return_value=mock_resp):
            result = run_kev_domains(SAMPLE_TEXT, SAMPLE_AGE, SAMPLE_DATE, keq)

        assert len(result["errors"]) == 0
        assert len(result["domains"]) == 21
        for d in result["domains"].values():
            assert d["tier"] == "green"
            assert d["timing"] == "recent"
        assert len(result["global_flags"]) == 9
        assert len(result["flag_probabilities"]) == 9

    def test_request_payload_shape(self, keq: dict) -> None:
        domain_slugs = [d["domain"] for d in keq["domains"]]
        flag_probs = {name: 0.0 for name in keq["flags"]}
        mock_resp = self._mock_resp(domain_slugs, flag_probs)

        captured = {}
        def capture(payload):
            captured.update(payload)
            return mock_resp

        with patch("kev_client._post_kev", side_effect=capture):
            run_kev_domains(SAMPLE_TEXT, SAMPLE_AGE, SAMPLE_DATE, keq)

        assert captured["state"]["entry_text"] == SAMPLE_TEXT
        assert captured["state"]["participant_age"] == SAMPLE_AGE
        assert captured["state"]["entry_date"] == SAMPLE_DATE
        assert captured["model"] == KEV_MODEL
        assert len(captured["questions"]) == 51

    def test_missing_answer_yields_unknown(self, keq: dict) -> None:
        with patch("kev_client._post_kev", return_value={"answers": {}}):
            result = run_kev_domains(SAMPLE_TEXT, SAMPLE_AGE, SAMPLE_DATE, keq)

        for d in result["domains"].values():
            assert d["tier"] == "unknown"
        assert len(result["errors"]) > 0

    def test_no_evidence_field(self, keq: dict) -> None:
        domain_slugs = [d["domain"] for d in keq["domains"]]
        flag_probs = {name: 0.0 for name in keq["flags"]}
        mock_resp = self._mock_resp(domain_slugs, flag_probs)

        with patch("kev_client._post_kev", return_value=mock_resp):
            result = run_kev_domains(SAMPLE_TEXT, SAMPLE_AGE, SAMPLE_DATE, keq)

        for d in result["domains"].values():
            assert "evidence" not in d

    def test_request_failure_all_unknown(self, keq: dict) -> None:
        with patch("kev_client._post_kev", side_effect=Exception("network down")):
            result = run_kev_domains(SAMPLE_TEXT, SAMPLE_AGE, SAMPLE_DATE, keq)

        for d in result["domains"].values():
            assert d["tier"] == "unknown"
        assert len(result["errors"]) == 1

    def test_flag_threshold(self, keq: dict) -> None:
        domain_slugs = [d["domain"] for d in keq["domains"]]
        flag_probs = {
            "lethal_means_access": 0.7,
            "current_intoxication": 0.3,
            "alone_no_support": 0.5,
            "minor_abuse_disclosure": 0.1,
            "past_suicide_attempt": 0.0,
            "past_self_harm": 0.9,
            "past_sexual_assault": 0.2,
            "past_psychosis": 0.4,
            "past_eating_disorder": 0.6,
        }
        mock_resp = self._mock_resp(domain_slugs, flag_probs)

        with patch("kev_client._post_kev", return_value=mock_resp):
            result = run_kev_domains(SAMPLE_TEXT, SAMPLE_AGE, SAMPLE_DATE, keq)

        flagged_names = [f["name"] for f in result["global_flags"] if f["flagged"]]
        assert "lethal_means_access" in flagged_names
        assert "alone_no_support" in flagged_names  # at threshold
        assert "past_self_harm" in flagged_names
        assert "past_eating_disorder" in flagged_names
        assert "current_intoxication" not in flagged_names
        assert "minor_abuse_disclosure" not in flagged_names
