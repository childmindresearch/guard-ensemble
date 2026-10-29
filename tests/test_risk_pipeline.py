"""Tests for risk_pipeline: CLI flags, prompt plumbing, mocked end-to-end pipeline."""

from unittest import mock

import pandas as pd
import pytest

from risk_pipeline import (
    DEMO_ROWS,
    dry_run,
    run,
    validate_input,
    _summary,
)
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
    """dry_run prints all prompts with no unresolved $."""

    def test_dry_run_no_unresolved_placeholders(self, capsys):
        df = pd.DataFrame(DEMO_ROWS)
        dry_run(df)
        captured = capsys.readouterr()
        import re
        assert not re.search(r"\$[A-Za-z_]", captured.out), "Unresolved $ placeholder found"

    def test_dry_run_prints_guard_prompts(self, capsys):
        df = pd.DataFrame(DEMO_ROWS)
        dry_run(df)
        captured = capsys.readouterr()
        for guard_name in ("shieldgemma", "llama_guard", "granite_guardian", "qwen_guard"):
            assert f"[{guard_name}]" in captured.out

    def test_dry_run_prints_domain_prompts(self, capsys):
        df = pd.DataFrame(DEMO_ROWS)
        dry_run(df)
        captured = capsys.readouterr()
        assert "[domain:" in captured.out

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
    """Mocked end-to-end pipeline: 72 calls/entry, columns, legacy mapping, error handling."""

    @pytest.fixture
    def input_df(self):
        return pd.DataFrame([
            {"id": 0, "text": "test entry", "age": 25, "entry_date": "2025-01-01", "participant_id": "P001"},
        ])

    def _make_side_effect(self, call_counter):
        """Build a side_effect for requests.post that returns safe/green responses.

        Call pattern per entry:
        - 4 ShieldGemma completions calls
        - 3 chat guard calls (llama, granite, qwen)
        - 63 custom prompt calls (21 domains x 3 runs)
        - 1 consolidation call
        - 1 final update call
        Total: 72 calls
        """
        def side_effect(url, **kwargs):
            call_counter[0] += 1
            call = call_counter[0]

            # Calls 1-4: ShieldGemma (completions endpoint)
            if url.endswith("/v1/completions"):
                r = mock.Mock()
                r.json.return_value = {"choices": [{"text": "No"}]}
                r.raise_for_status = mock.Mock()
                return r

            # All other calls are chat completions
            # Calls 5-7: llama_guard, granite_guardian, qwen_guard
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

            # Calls 8-70: custom prompt calls (21 domains x 3 runs = 63)
            if 8 <= call <= 70:
                r = mock.Mock()
                r.json.return_value = {
                    "choices": [{
                        "message": {
                            "content": '{"domain": "test_domain", "tier": "green", "timing": "unclear", "flags": [], "evidence": "no risk"}'
                        }
                    }]
                }
                r.raise_for_status = mock.Mock()
                return r

            # Calls 71-72: consolidation + final_update
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

    def test_72_calls_per_entry(self, input_df, tmp_path):
        call_counter = [0]
        side_effect = self._make_side_effect(call_counter)

        out = tmp_path / "out.csv"
        with mock.patch("risk_pipeline.requests.post", side_effect=side_effect):
            df = run(input_df, str(out))

        assert call_counter[0] == 72, f"Expected 72 calls, got {call_counter[0]}"

    def test_expected_columns(self, input_df, tmp_path):
        call_counter = [0]
        side_effect = self._make_side_effect(call_counter)

        out = tmp_path / "out.csv"
        with mock.patch("risk_pipeline.requests.post", side_effect=side_effect):
            df = run(input_df, str(out))

        expected = [
            "id", "text", "participant_id", "age", "entry_date",
            "domain_results", "highest_domain_tier", "escalated_tier",
            "escalation_reasons", "minimum_tier", "minimum_reasons",
            "floor_tier", "consolidated_tier", "consolidation_rationale",
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
        side_effect = self._make_side_effect(call_counter)

        out = tmp_path / "out.csv"
        with mock.patch("risk_pipeline.requests.post", side_effect=side_effect):
            df = run(input_df, str(out))

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
            df = run(input_df, str(out), guards_only=True)

        assert len(df) == 1, "Row should not be dropped on guard failure"
        assert "errors" in df.columns
        errors_val = str(df["errors"].iloc[0])
        assert "llama_guard" in errors_val
        assert "connection refused" in errors_val

    def test_csv_written(self, input_df, tmp_path):
        call_counter = [0]
        side_effect = self._make_side_effect(call_counter)

        out = tmp_path / "out.csv"
        with mock.patch("risk_pipeline.requests.post", side_effect=side_effect):
            df = run(input_df, str(out))

        assert out.exists()
        saved_df = pd.read_csv(out)
        assert len(saved_df) == 1


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
            df = run(input_df, str(out), guards_only=True)

        assert call_counter[0] == 7  # 4 SG + 3 others
        assert len(df) == 1


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
