"""Tests for risk_pipeline: CLI flags, prompt plumbing, mocked end-to-end pipeline."""

import json
from unittest import mock

import pandas as pd
import pytest

from risk_pipeline import (
    DEMO_ROWS,
    dry_run,
    run,
    validate_input,
    _summary,
    run_deterministic_phases,
    _build_rule_block,
    run_kev_domain_stage,
    run_consolidation_and_final_update,
    _join_url,
)
from guards import DELIBERATION_BASE_URL, DELIBERATION_MODEL_ID, DELIBERATION_PATH, DELIBERATION_MAX_TOKENS
from rubric import Tier
from rules import to_legacy_risk


class TestInputValidation:
    """Required input columns and data validation."""

    def test_valid_input_passes(self):
        df = pd.DataFrame([
            {"id": 0, "text": "hello", "age": 25, "entry_date": "2025-01-01"},
        ])
        result = validate_input(df)
        assert result is not None
        assert result["age"].dtype in ("int64", "int32", "int")

    def test_missing_required_column(self):
        df = pd.DataFrame([{"id": 0, "text": "hello", "age": 25}])
        with pytest.raises(SystemExit):
            validate_input(df)

    def test_invalid_age(self):
        df = pd.DataFrame([{"id": 0, "text": "hello", "age": "abc", "entry_date": "2025-01-01"}])
        with pytest.raises(SystemExit):
            validate_input(df)

    def test_negative_age(self):
        df = pd.DataFrame([{"id": 0, "text": "hello", "age": -5, "entry_date": "2025-01-01"}])
        with pytest.raises(SystemExit):
            validate_input(df)

    def test_missing_date(self):
        df = pd.DataFrame([{"id": 0, "text": "hello", "age": 25, "entry_date": None}])
        with pytest.raises(SystemExit):
            validate_input(df)

    def test_invalid_date(self):
        df = pd.DataFrame([{"id": 0, "text": "hello", "age": 25, "entry_date": "not-a-date"}])
        with pytest.raises(SystemExit):
            validate_input(df)

    def test_row_indices_in_error(self, capsys):
        df = pd.DataFrame([
            {"id": 0, "text": "hello", "age": 25, "entry_date": "2025-01-01"},
            {"id": 1, "text": "world", "age": "bad", "entry_date": "2025-01-02"},
        ])
        with pytest.raises(SystemExit):
            validate_input(df)
        captured = capsys.readouterr()
        assert "row 1" in captured.err


class TestDryRun:
    """dry_run prints all prompts with no unresolved placeholders."""

    def test_dry_run_prints_guard_prompts(self, capsys):
        df = pd.DataFrame(DEMO_ROWS)
        dry_run(df)
        captured = capsys.readouterr()
        for guard_name in ("shieldgemma", "llama_guard", "granite_guardian", "qwen_guard"):
            assert f"[{guard_name}]" in captured.out

    def test_dry_run_prints_kev_questions(self, capsys):
        df = pd.DataFrame(DEMO_ROWS)
        dry_run(df)
        captured = capsys.readouterr()
        assert "[kev]" in captured.out

    def test_dry_run_prints_consolidation(self, capsys):
        df = pd.DataFrame(DEMO_ROWS)
        dry_run(df)
        captured = capsys.readouterr()
        assert "[consolidation]" in captured.out

    def test_dry_run_prints_final_update(self, capsys):
        df = pd.DataFrame(DEMO_ROWS)
        dry_run(df)
        captured = capsys.readouterr()
        assert "[final_update]" in captured.out


class TestMockedEndToEnd:
    """Mocked end-to-end pipeline: 10 calls/entry, columns, legacy mapping, error handling."""

    @pytest.fixture
    def input_df(self):
        return pd.DataFrame([
            {"id": 0, "text": "test entry", "age": 25, "entry_date": "2025-01-01", "participant_id": "P001"},
        ])

    def _patch_requests(self, call_counter):
        """Patch both risk_pipeline and kev_client requests.post, incrementing shared counter once."""
        import threading
        lock = threading.Lock()
        in_call = [False]
        def wrapped(url, **kwargs):
            with lock:
                if in_call[0]:
                    return self._response_for_call(url, call_counter[0])
                in_call[0] = True
                call_counter[0] += 1
                result = self._response_for_call(url, call_counter[0])
                in_call[0] = False
                return result
        return (
            mock.patch("risk_pipeline.requests.post", side_effect=wrapped),
            mock.patch("kev_client.requests.post", side_effect=wrapped),
        )

    def _response_for_call(self, url, call_num):
        """Return the mock response for a given call number."""
        r = mock.Mock()
        r.status_code = 200
        r.raise_for_status = mock.Mock()

        # Calls 1-4: ShieldGemma (completions endpoint)
        if url.endswith("/v1/completions"):
            r.json.return_value = {"choices": [{"text": "No"}]}
            return r

        # Call 5-7: llama_guard, granite_guardian, qwen_guard
        if 5 <= call_num <= 7:
            if call_num == 5:
                content = "safe"
            elif call_num == 6:
                content = "No"
            else:
                content = "Safety: Safe\nCategories: None"
            r.json.return_value = {"choices": [{"message": {"content": content}}]}
            return r

        # Call 8: Kev (/v1/systemone)
        if "/v1/systemone" in url:
            r.json.return_value = self._make_kev_response()
            return r

        # Calls 9-10: consolidation + final_update
        r.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"tier": "green", "rationale": "low risk"}'
                }
            }]
        }
        return r

    def _fake_keq(self):
        """Return a minimal Kev question config for testing."""
        return {
            "reading_rules": "test rules",
            "timing": {"type": "choice", "instructions": "test", "criteria": {"unclear": "unclear"}},
            "flags": {"lethal_means_access": {"type": "noul", "instructions": "test"}},
            "domains": [
                {"domain": "suicidality", "tier": {"type": "score", "instructions": "test", "criteria": ["g", "y", "r"]}},
            ],
        }

    def _make_kev_response(self):
        """Build a Kev response body with all domains green."""
        return {
            "answers": {
                "suicidality__tier": {
                    "score": 0.1,
                    "probabilities": {"0": 0.9, "1": 0.09, "2": 0.01},
                    "confidence": 0.95,
                },
                "suicidality__timing": {
                    "choice": "unclear",
                    "probabilities": {"unclear": 0.9},
                    "confidence": 0.9,
                },
                "flag__lethal_means_access": {"noul": 0.1},
            }
        }

    def _make_side_effect(self, call_counter):
        """Build a side_effect for requests.post that returns safe/green responses.

        Call pattern per entry:
        - 4 ShieldGemma completions calls
        - 3 chat guard calls (llama, granite, qwen)
        - 1 Kev call (/v1/systemone)
        - 1 consolidation call
        - 1 final update call
        Total: 10 calls
        """
        def side_effect(url, **kwargs):
            call = call_counter[0]

            # Calls 1-4: ShieldGemma (completions endpoint)
            if url.endswith("/v1/completions"):
                r = mock.Mock()
                r.json.return_value = {"choices": [{"text": "No"}]}
                r.raise_for_status = mock.Mock()
                return r

            # Call 5-7: llama_guard, granite_guardian, qwen_guard
            if 5 <= call <= 7:
                r = mock.Mock()
                if call == 5:
                    content = "safe"
                elif call == 6:
                    content = "No"
                else:
                    content = "Safety: Safe\nCategories: None"
                r.json.return_value = {"choices": [{"message": {"content": content}}]}
                r.raise_for_status = mock.Mock()
                return r

            # Call 8: Kev (/v1/systemone)
            if "/v1/systemone" in url:
                r = mock.Mock()
                r.json.return_value = self._make_kev_response()
                r.raise_for_status = mock.Mock()
                return r

            # Calls 9-10: consolidation + final_update
            r = mock.Mock()
            r.json.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"tier": "green", "rationale": "low risk"}'
                    }
                }]
            }
            r.raise_for_status = mock.Mock()
            return r

        return side_effect

    def test_10_calls_per_entry(self, input_df, tmp_path):
        call_counter = [0]
        patches = self._patch_requests(call_counter)

        out = tmp_path / "out.csv"
        with patches[0], patches[1]:
            df = run(input_df, str(out), keq=self._fake_keq())

        assert call_counter[0] == 10, f"Expected 10 calls, got {call_counter[0]}"

    def test_expected_columns(self, input_df, tmp_path):
        call_counter = [0]
        patches = self._patch_requests(call_counter)

        out = tmp_path / "out.csv"
        with patches[0], patches[1]:
            df = run(input_df, str(out), keq=self._fake_keq())

        expected = [
            "id", "text", "participant_id", "age", "entry_date",
            "domain_results", "global_flags", "global_flag_probabilities",
            "highest_domain_tier", "escalated_tier",
            "escalation_reasons", "minimum_tier", "minimum_reasons",
            "floor_tier", "floor_reasons", "consolidated_tier", "consolidation_rationale",
            "final_tier", "final_rationale", "final_adjusted", "final_risk",
            "domain_incomplete", "custom_prompt_errors",
            "consolidation_error", "final_update_error", "errors",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

        # Guard columns should exist
        assert "shieldgemma_unsafe" in df.columns
        assert "llama_guard_unsafe" in df.columns
        assert "granite_guardian_unsafe" in df.columns
        assert "qwen_guard_unsafe" in df.columns

    def test_legacy_mapping(self, input_df, tmp_path):
        call_counter = [0]
        patches = self._patch_requests(call_counter)

        out = tmp_path / "out.csv"
        with patches[0], patches[1]:
            df = run(input_df, str(out), keq=self._fake_keq())

        # All green -> no_low_risk
        assert df["final_risk"].iloc[0] == "no_low_risk"

    def test_guard_error_preserved(self, input_df, tmp_path):
        """If a guard fails, the error is recorded and no row is dropped."""
        def side_effect(url, **kwargs):
            r = mock.Mock()
            if url.endswith("/v1/completions"):
                r.json.return_value = {"choices": [{"text": "No"}]}
            else:
                raise RuntimeError("connection refused")
            r.raise_for_status = mock.Mock()
            return r

        out = tmp_path / "out.csv"
        with mock.patch("risk_pipeline.requests.post", side_effect=side_effect):
            df = run(input_df, str(out), keq=self._fake_keq(), guards_only=True)

        assert len(df) == 1, "Row should not be dropped on guard failure"
        assert "errors" in df.columns
        errors_val = str(df["errors"].iloc[0])
        assert "llama_guard" in errors_val
        assert "connection refused" in errors_val

    def test_csv_written(self, input_df, tmp_path):
        call_counter = [0]
        patches = self._patch_requests(call_counter)

        out = tmp_path / "out.csv"
        with patches[0], patches[1]:
            df = run(input_df, str(out), keq=self._fake_keq())

        assert out.exists()
        saved_df = pd.read_csv(out)
        assert len(saved_df) == 1

    def test_domain_results_has_confidence_fields(self, input_df, tmp_path):
        """domain_results contains tier, timing, tier_score, tier_probabilities, etc."""
        call_counter = [0]
        patches = self._patch_requests(call_counter)

        out = tmp_path / "out.csv"
        with patches[0], patches[1]:
            df = run(input_df, str(out), keq=self._fake_keq())

        dr = df["domain_results"].iloc[0]
        if isinstance(dr, str):
            dr = json.loads(dr)
        domain_data = dr.get("suicidality", {})
        assert "tier" in domain_data
        assert "timing" in domain_data
        assert "tier_score" in domain_data
        assert "tier_probabilities" in domain_data
        assert "tier_confidence" in domain_data
        assert "tier_mode_probability" in domain_data
        assert "timing_probabilities" in domain_data
        assert "timing_confidence" in domain_data
        # No evidence or per-domain flags
        assert "evidence" not in domain_data
        assert "flags" not in domain_data

    def test_global_flags_column_present(self, input_df, tmp_path):
        """global_flags and global_flag_probabilities columns exist and are populated."""
        call_counter = [0]
        patches = self._patch_requests(call_counter)

        out = tmp_path / "out.csv"
        with patches[0], patches[1]:
            df = run(input_df, str(out), keq=self._fake_keq())

        gf = df["global_flags"].iloc[0]
        if isinstance(gf, str):
            gf = json.loads(gf)
        assert isinstance(gf, list)
        # Should have at least one flag entry (lethal_means_access)
        assert len(gf) >= 1

        gfp = df["global_flag_probabilities"].iloc[0]
        if isinstance(gfp, str):
            gfp = json.loads(gfp)
        assert "lethal_means_access" in gfp


class TestGuardsOnly:
    """Guards-only mode skips remaining stages."""

    def test_guards_only(self, tmp_path):
        input_df = pd.DataFrame([
            {"id": 0, "text": "test", "age": 25, "entry_date": "2025-01-01"},
        ])

        call_counter = [0]

        def side_effect(url, **kwargs):
            call_counter[0] += 1
            r = mock.Mock()
            if url.endswith("/v1/completions"):
                r.json.return_value = {"choices": [{"text": "No"}]}
            else:
                c = call_counter[0]
                if c == 5:
                    content = "safe"
                elif c == 6:
                    content = "No"
                else:
                    content = "Safety: Safe\nCategories: None"
                r.json.return_value = {"choices": [{"message": {"content": content}}]}
            r.raise_for_status = mock.Mock()
            return r

        out = tmp_path / "out.csv"
        with mock.patch("risk_pipeline.requests.post", side_effect=side_effect):
            df = run(input_df, str(out), keq={}, guards_only=True)

        assert call_counter[0] == 7  # 4 SG + 3 others
        assert len(df) == 1
        # New columns should be present
        assert "global_flags" in df.columns
        assert "global_flag_probabilities" in df.columns


class TestDemoData:
    """Demo records include age/date/participant_id."""

    def test_demo_rows_have_required_fields(self):
        for row in DEMO_ROWS:
            assert "age" in row
            assert "entry_date" in row
            assert "participant_id" in row
            assert "id" in row
            assert "text" in row

    def test_demo_age_is_integer(self):
        for row in DEMO_ROWS:
            assert isinstance(row["age"], int)


class TestSummary:
    """Summary reports guard flags/errors, domain tier distribution, etc."""

    def test_summary_prints_guard_info(self, capsys):
        df = pd.DataFrame([
            {
                "shieldgemma_unsafe": 0, "shieldgemma_error": None,
                "llama_guard_unsafe": 1, "llama_guard_error": None,
                "granite_guardian_unsafe": 0, "granite_guardian_error": None,
                "qwen_guard_unsafe": 0, "qwen_guard_error": None,
            }
        ])
        _summary(df)
        captured = capsys.readouterr()
        assert "shieldgemma" in captured.out
        assert "llama_guard" in captured.out


class TestMainCLI:
    """CLI integration tests."""

    def test_bad_consolidation_prompt_exits(self):
        with mock.patch("sys.argv", [
            "risk_pipeline.py", "--demo",
            "--consolidation-prompt", "/no/such/file.txt"
        ]):
            from risk_pipeline import main
            with pytest.raises(SystemExit):
                main()

    def test_main_guards_only(self, tmp_path):
        out = tmp_path / "out.csv"
        call_counter = [0]

        def side_effect(url, **kwargs):
            call_counter[0] += 1
            r = mock.Mock()
            if url.endswith("/v1/completions"):
                r.json.return_value = {"choices": [{"text": "No"}]}
            else:
                c = call_counter[0]
                if c <= 4:
                    content = "safe"
                elif c == 5:
                    content = "No"
                else:
                    content = "Safety: Safe\nCategories: None"
                r.json.return_value = {"choices": [{"message": {"content": content}}]}
            r.raise_for_status = mock.Mock()
            return r

        with mock.patch("sys.argv", [
            "risk_pipeline.py", "--demo", "--guards-only",
            f"--output={out}"
        ]), mock.patch("risk_pipeline.requests.post", side_effect=side_effect):
            from risk_pipeline import main
            main()

        assert out.exists()
        # 4 demo rows x 7 guards = 28 calls
        assert call_counter[0] == 28


class TestDeterministicPhases:
    """Deterministic phases: tier ordering, incompleteness, floor_reasons, global_flags."""

    def _make_df(self, domain_results, domain_incomplete=False, global_flags=None):
        """Helper to build a minimal DataFrame for run_deterministic_phases."""
        row = {
            "id": 0, "text": "test", "age": 25, "entry_date": "2025-01-01",
            "domain_results": domain_results,
            "domain_incomplete": domain_incomplete,
        }
        if global_flags is not None:
            row["global_flags"] = global_flags
        return pd.DataFrame([row])

    def test_empty_domain_results(self):
        """Empty domain_results yields incomplete=True, floor=yellow, domain_unavailable reason."""
        df = self._make_df({})
        result = run_deterministic_phases(df)
        assert result["domain_incomplete"].iloc[0] == True
        assert result["highest_domain_tier"].iloc[0] == "unknown"
        assert result["floor_tier"].iloc[0] == "yellow"
        assert "domain_unavailable" in result["floor_reasons"].iloc[0]

    def test_custom_stage_exception_sets_domain_incomplete(self):
        """When custom stage sets domain_incomplete=True, deterministic phase preserves it."""
        df = self._make_df({}, domain_incomplete=True)
        result = run_deterministic_phases(df)
        assert result["domain_incomplete"].iloc[0] == True
        assert "domain_unavailable" in result["floor_reasons"].iloc[0]

    def test_one_unknown_plus_all_green(self):
        """One unknown + all green -> incomplete, floor yellow."""
        dr = {
            "a": {"tier": "green", "flags": []},
            "b": {"tier": "green", "flags": []},
            "c": {"tier": "unknown", "flags": []},
        }
        df = self._make_df(dr)
        result = run_deterministic_phases(df)
        assert result["domain_incomplete"].iloc[0] == True
        assert result["highest_domain_tier"].iloc[0] == "green"
        assert result["floor_tier"].iloc[0] == "yellow"
        assert "domain_unavailable" in result["floor_reasons"].iloc[0]

    def test_red_plus_yellow_highest_is_red(self):
        """red + yellow -> highest_domain_tier == red."""
        dr = {
            "a": {"tier": "red", "flags": []},
            "b": {"tier": "yellow", "flags": []},
        }
        df = self._make_df(dr)
        result = run_deterministic_phases(df)
        assert result["highest_domain_tier"].iloc[0] == "red"

    def test_green_yellow_red_highest_is_red(self):
        """green + yellow + red -> highest_domain_tier == red (tier ordering, not lexicographic)."""
        dr = {
            "a": {"tier": "green", "flags": []},
            "b": {"tier": "yellow", "flags": []},
            "c": {"tier": "red", "flags": []},
        }
        df = self._make_df(dr)
        result = run_deterministic_phases(df)
        assert result["highest_domain_tier"].iloc[0] == "red"

    def test_floor_reasons_persisted(self):
        """floor_reasons column is written by run_deterministic_phases."""
        dr = {"a": {"tier": "green", "flags": ["minor_abuse_disclosure"]}}
        df = self._make_df(dr)
        result = run_deterministic_phases(df)
        assert "floor_reasons" in result.columns

    def test_rule_block_includes_new_fields(self):
        """_build_rule_block includes highest_domain_tier, domain_incomplete, floor_reasons."""
        row = {
            "highest_domain_tier": "red",
            "domain_incomplete": True,
            "escalated_tier": "yellow",
            "escalation_reasons": "",
            "minimum_tier": "",
            "minimum_reasons": "",
            "floor_tier": "red",
            "floor_reasons": "domain_unavailable",
        }
        block = _build_rule_block(row)
        assert "Highest domain tier: red" in block
        assert "Domain coverage incomplete: True" in block
        assert "Floor reasons: domain_unavailable" in block

    def test_global_flags_unioned_with_per_domain_flags(self):
        """global_flags are unioned with per-domain flags for escalation/minimums."""
        dr = {
            "a": {"tier": "yellow", "flags": []},
            "b": {"tier": "yellow", "flags": []},
        }
        global_flags = [
            {"name": "lethal_means_access", "flagged": True, "probability": 0.8},
            {"name": "current_intoxication", "flagged": False, "probability": 0.1},
        ]
        df = self._make_df(dr, global_flags=global_flags)
        result = run_deterministic_phases(df)
        # lethal_means_access is an escalation flag; yellow + escalation -> red
        assert result["highest_domain_tier"].iloc[0] == "yellow"
        # Should see escalation to red
        assert result["escalated_tier"].iloc[0] == "red"

    def test_kev_failure_sets_domain_incomplete_and_yellow_floor(self):
        """When Kev fails and domain_incomplete=True, floor is yellow (unknown-domain floor)."""
        dr = {}
        df = self._make_df(dr, domain_incomplete=True, global_flags=[])
        result = run_deterministic_phases(df)
        assert result["domain_incomplete"].iloc[0] == True
        assert result["floor_tier"].iloc[0] == "yellow"
        assert "domain_unavailable" in result["floor_reasons"].iloc[0]


class TestGuardsOnlyFloorReasons:
    """Guards-only mode includes floor_reasons column."""

    def test_guards_only_includes_floor_reasons(self, tmp_path):
        input_df = pd.DataFrame([
            {"id": 0, "text": "test", "age": 25, "entry_date": "2025-01-01"},
        ])

        def side_effect(url, **kwargs):
            r = mock.Mock()
            if url.endswith("/v1/completions"):
                r.json.return_value = {"choices": [{"text": "No"}]}
            else:
                r.json.return_value = {"choices": [{"message": {"content": "safe"}}]}
            r.raise_for_status = mock.Mock()
            return r

        out = tmp_path / "out.csv"
        with mock.patch("risk_pipeline.requests.post", side_effect=side_effect):
            df = run(input_df, str(out), keq={}, guards_only=True)

        assert "floor_reasons" in df.columns


class TestDeliberationModelRouting:
    """DELIBERATION_MODEL_ID and resolved URL flow through all LLM stages."""

    @pytest.fixture
    def input_df(self):
        return pd.DataFrame([
            {"id": 0, "text": "test entry", "age": 25, "entry_date": "2025-01-01", "participant_id": "P001"},
        ])

    def _patch_requests(self, call_counter, request_bodies):
        """Patch both risk_pipeline and kev_client requests.post, incrementing shared counter once."""
        patched = [False]
        def wrapped(url, **kwargs):
            if not patched[0]:
                patched[0] = True
                call_counter[0] += 1
                if "json" in kwargs:
                    request_bodies.append(kwargs["json"])
                patched[0] = False
                return self._response_for_call(url, call_counter[0], request_bodies)
            return self._response_for_call(url, call_counter[0], request_bodies)
        return (
            mock.patch("risk_pipeline.requests.post", side_effect=wrapped),
            mock.patch("kev_client.requests.post", side_effect=wrapped),
        )

    def _response_for_call(self, url, call_num, request_bodies=None):
        """Return the mock response for a given call number."""
        r = mock.Mock()
        r.status_code = 200
        r.raise_for_status = mock.Mock()
        if url.endswith("/v1/completions"):
            r.json.return_value = {"choices": [{"text": "No"}]}
            return r
        if 5 <= call_num <= 7:
            if call_num == 5:
                content = "safe"
            elif call_num == 6:
                content = "No"
            else:
                content = "Safety: Safe\nCategories: None"
            r.json.return_value = {"choices": [{"message": {"content": content}}]}
            return r
        if "/v1/systemone" in url:
            r.json.return_value = self._make_kev_response()
            return r
        r.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"tier": "green", "rationale": "ok"}'
                }
            }]
        }
        return r

    def _fake_keq(self):
        """Return a minimal Kev question config for testing."""
        return {
            "reading_rules": "test rules",
            "timing": {"type": "choice", "instructions": "test", "criteria": {"unclear": "unclear"}},
            "flags": {"lethal_means_access": {"type": "noul", "instructions": "test"}},
            "domains": [
                {"domain": "suicidality", "tier": {"type": "score", "instructions": "test", "criteria": ["g", "y", "r"]}},
            ],
        }

    def _make_kev_response(self):
        """Build a Kev response body with all domains green."""
        return {
            "answers": {
                "suicidality__tier": {
                    "score": 0.1,
                    "probabilities": {"0": 0.9, "1": 0.09, "2": 0.01},
                    "confidence": 0.95,
                },
                "suicidality__timing": {
                    "choice": "unclear",
                    "probabilities": {"unclear": 0.9},
                    "confidence": 0.9,
                },
                "flag__lethal_means_access": {"noul": 0.1},
            }
        }

    def _make_side_effect(self, call_counter, request_bodies):
        """Build side_effect that captures all request bodies."""
        def side_effect(url, **kwargs):
            call_counter[0] += 1
            if "json" in kwargs:
                request_bodies.append(kwargs["json"])
            r = mock.Mock()
            if url.endswith("/v1/completions"):
                r.json.return_value = {"choices": [{"text": "No"}]}
            elif "/v1/systemone" in url:
                # Kev call
                r.json.return_value = self._make_kev_response()
            else:
                call = call_counter[0]
                if 5 <= call <= 7:
                    # Guard calls
                    if call == 5:
                        content = "safe"
                    elif call == 6:
                        content = "No"
                    else:
                        content = "Safety: Safe\nCategories: None"
                    r.json.return_value = {"choices": [{"message": {"content": content}}]}
                else:
                    # Deliberation calls (consolidation + final update)
                    r.json.return_value = {
                        "choices": [{
                            "message": {
                                "content": '{"tier": "green", "rationale": "ok"}'
                            }
                        }]
                    }
            r.raise_for_status = mock.Mock()
            return r
        return side_effect

    def test_full_run_no_errors(self, input_df, tmp_path):
        """Full run() completes without custom/consolidation/final errors."""
        call_counter = [0]
        request_bodies = []
        patches = self._patch_requests(call_counter, request_bodies)
        out = tmp_path / "out.csv"
        with patches[0], patches[1]:
            df = run(input_df, str(out), keq=self._fake_keq())
        assert df["custom_prompt_errors"].iloc[0] == ""
        assert df["consolidation_error"].iloc[0] == ""
        assert df["final_update_error"].iloc[0] == ""

    def test_all_deliberation_bodies_contain_model_id(self, input_df, tmp_path):
        """Every deliberation request body contains DELIBERATION_MODEL_ID."""
        call_counter = [0]
        request_bodies = []
        patches = self._patch_requests(call_counter, request_bodies)
        out = tmp_path / "out.csv"
        with patches[0], patches[1]:
            run(input_df, str(out), keq=self._fake_keq())
        # Calls 1-7 are guard calls (4 SG + 3 chat), call 8 is Kev (no model_id)
        # Calls 9-10 are deliberation calls (consolidation + final)
        deliberation_bodies = request_bodies[8:]  # skip first 8 calls (7 guards + 1 Kev)
        for body in deliberation_bodies:
            assert body["model"] == DELIBERATION_MODEL_ID

    def test_monkeypatched_url_used(self, input_df, tmp_path):
        """Monkeypatch DELIBERATION_BASE_URL and DELIBERATION_PATH, assert resolved URL."""
        custom_url = "http://custom-host:9999"
        custom_path = "/v1/custom/chat/completions"
        call_counter = [0]
        urls_seen = []

        patched = [False]
        def side_effect(url, **kwargs):
            if not patched[0]:
                patched[0] = True
                call_counter[0] += 1
                urls_seen.append(url)
                patched[0] = False
            else:
                urls_seen.append(url)
            r = mock.Mock()
            r.status_code = 200
            if url.endswith("/v1/completions"):
                r.json.return_value = {"choices": [{"text": "No"}]}
            elif "/v1/systemone" in url:
                r.json.return_value = self._make_kev_response()
            else:
                call = call_counter[0]
                if 5 <= call <= 7:
                    if call == 5:
                        content = "safe"
                    elif call == 6:
                        content = "No"
                    else:
                        content = "Safety: Safe\nCategories: None"
                    r.json.return_value = {"choices": [{"message": {"content": content}}]}
                else:
                    r.json.return_value = {
                        "choices": [{
                            "message": {
                                "content": '{"tier": "green", "rationale": "ok"}'
                            }
                        }]
                    }
            r.raise_for_status = mock.Mock()
            return r

        expected = _join_url(custom_url, custom_path)
        out = tmp_path / "out.csv"
        with mock.patch("risk_pipeline.requests.post", side_effect=side_effect), \
             mock.patch("kev_client.requests.post", side_effect=side_effect), \
             mock.patch("risk_pipeline.DELIBERATION_BASE_URL", custom_url), \
             mock.patch("risk_pipeline.DELIBERATION_PATH", custom_path):
            run(input_df, str(out), keq=self._fake_keq())
        # All deliberation calls (calls 9-10) should use the monkeypatched URL
        deliberation_urls = urls_seen[8:]
        for url in deliberation_urls:
            assert url == expected

    def test_consolidation_defaults(self):
        """run_consolidation_and_final_update defaults resolve to deliberation model/URL/max_tokens."""
        df = pd.DataFrame([{
            "id": 0, "text": "hi", "age": 25, "entry_date": "2025-01-01",
            "shieldgemma_unsafe": 0, "shieldgemma_label": "safe", "shieldgemma_category": "none",
            "llama_guard_unsafe": 0, "llama_guard_label": "safe", "llama_guard_category": "none",
            "granite_guardian_unsafe": 0, "granite_guardian_label": "No", "granite_guardian_category": "none",
            "qwen_guard_unsafe": 0, "qwen_guard_label": "safe", "qwen_guard_category": "none",
            "domain_results": {}, "domain_incomplete": True,
            "highest_domain_tier": "unknown", "escalated_tier": "unknown",
            "escalation_reasons": "", "minimum_tier": "", "minimum_reasons": "",
            "floor_tier": "yellow", "floor_reasons": "domain_unavailable",
        }])
        captured = {}

        def fake_call(url, **kwargs):
            captured["url"] = url
            captured["model"] = kwargs.get("json", {}).get("model", "")
            captured["max_tokens"] = kwargs.get("json", {}).get("max_tokens", 0)
            r = mock.Mock()
            r.json.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"tier": "green", "rationale": "ok"}'
                    }
                }]
            }
            r.raise_for_status = mock.Mock()
            return r

        with mock.patch("risk_pipeline.requests.post", side_effect=fake_call):
            run_consolidation_and_final_update(df)

        assert captured["model"] == DELIBERATION_MODEL_ID
        assert captured["url"] == _join_url(DELIBERATION_BASE_URL, DELIBERATION_PATH)
        assert captured["max_tokens"] == DELIBERATION_MAX_TOKENS


class TestKeDomainStage:
    """Kev domain stage: run_kev_domain_stage produces correct columns."""

    def _fake_keq(self):
        return {
            "reading_rules": "test rules",
            "timing": {"type": "choice", "instructions": "test", "criteria": {"unclear": "unclear"}},
            "flags": {"lethal_means_access": {"type": "noul", "instructions": "test"}},
            "domains": [
                {"domain": "suicidality", "tier": {"type": "score", "instructions": "test", "criteria": ["g", "y", "r"]}},
            ],
        }

    def test_kev_stage_columns(self):
        """run_kev_domain_stage adds expected columns."""
        df = pd.DataFrame([{"id": 0, "text": "hi", "age": 25, "entry_date": "2025-01-01"}])

        call_counter = [0]
        def fake_post(url, **kwargs):
            call_counter[0] += 1
            r = mock.Mock()
            r.status_code = 200
            r.json.return_value = {
                "answers": {
                    "suicidality__tier": {
                        "score": 0.1,
                        "probabilities": {"0": 0.9, "1": 0.09, "2": 0.01},
                        "confidence": 0.95,
                    },
                    "suicidality__timing": {
                        "choice": "unclear",
                        "probabilities": {"unclear": 0.9},
                        "confidence": 0.9,
                    },
                    "flag__lethal_means_access": {"noul": 0.1},
                }
            }
            r.raise_for_status = mock.Mock()
            return r

        with mock.patch("kev_client.requests.post", side_effect=fake_post):
            result = run_kev_domain_stage(df, keq=self._fake_keq())

        assert "domain_results" in result.columns
        assert "domain_incomplete" in result.columns
        assert "custom_prompt_errors" in result.columns
        assert "global_flags" in result.columns
        assert "global_flag_probabilities" in result.columns
        assert call_counter[0] == 1  # only one Kev call per row

    def test_kev_failure_sets_domain_incomplete(self):
        """When Kev call fails, domain_incomplete=True and error is recorded."""
        df = pd.DataFrame([{"id": 0, "text": "hi", "age": 25, "entry_date": "2025-01-01"}])

        def fake_post(url, **kwargs):
            raise RuntimeError("connection refused")

        with mock.patch("kev_client.requests.post", side_effect=fake_post):
            result = run_kev_domain_stage(df, keq=self._fake_keq())

        assert result["domain_incomplete"].iloc[0] == True
        assert "connection refused" in result["custom_prompt_errors"].iloc[0]

    def test_threshold_override_via_run_kev_domain_stage(self):
        """run_kev_domain_stage forwards threshold to run_kev_domains."""
        df = pd.DataFrame([{"id": 0, "text": "hi", "age": 25, "entry_date": "2025-01-01"}])

        captured = {}

        def fake_run_kev_domains(text, age, entry_date, keq, threshold):
            captured["threshold"] = threshold
            return {
                "domains": {
                    "suicidality": {
                        "tier": "green", "timing": "unclear", "mean_score": 0.1,
                        "probabilities": {"0": 0.9, "1": 0.09, "2": 0.01},
                        "confidence": 0.95, "mode_probability": 0.9,
                        "timing_probabilities": {"unclear": 0.9},
                        "timing_confidence": 0.9,
                    }
                },
                "errors": [],
                "global_flags": [{"name": "lethal_means_access", "flagged": False, "probability": 0.1}],
                "flag_probabilities": {"lethal_means_access": 0.1},
            }

        with mock.patch("risk_pipeline.run_kev_domains", side_effect=fake_run_kev_domains):
            run_kev_domain_stage(df, keq=self._fake_keq(), threshold=0.75)

        assert captured["threshold"] == 0.75

    def test_threshold_zero_is_not_treated_as_unset(self):
        """threshold=0.0 must not fall back to KEV_FLAG_THRESHOLD."""
        df = pd.DataFrame([{"id": 0, "text": "hi", "age": 25, "entry_date": "2025-01-01"}])

        captured = {}

        def fake_run_kev_domains(text, age, entry_date, keq, threshold):
            captured["threshold"] = threshold
            return {
                "domains": {
                    "suicidality": {
                        "tier": "green", "timing": "unclear", "mean_score": 0.1,
                        "probabilities": {"0": 0.9, "1": 0.09, "2": 0.01},
                        "confidence": 0.95, "mode_probability": 0.9,
                        "timing_probabilities": {"unclear": 0.9},
                        "timing_confidence": 0.9,
                    }
                },
                "errors": [],
                "global_flags": [{"name": "lethal_means_access", "flagged": False, "probability": 0.1}],
                "flag_probabilities": {"lethal_means_access": 0.1},
            }

        with mock.patch("risk_pipeline.run_kev_domains", side_effect=fake_run_kev_domains):
            run_kev_domain_stage(df, keq=self._fake_keq(), threshold=0.0)

        assert captured["threshold"] == 0.0
