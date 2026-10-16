"""Tests for risk_pipeline CLI flag and prompt plumbing."""
import sys
from unittest import mock

import pandas as pd
import pytest
from risk_pipeline import call_deliberator, dry_run, run, run_deliberation


class TestDeliberationPromptFlag:
    def test_dry_run_uses_custom_prompt(self, capsys):
        """dry_run should output content from a custom template string."""
        custom = "CUSTOM_MARKER: $text scores: $scores_block"
        dry_run(["hello"], deliberation_prompt=custom)
        captured = capsys.readouterr()
        assert "CUSTOM_MARKER" in captured.out
        assert "$" not in captured.out

    def test_dry_run_default_prompt(self, capsys):
        """dry_run without custom prompt should produce default deliberator output."""
        dry_run(["test"], deliberation_prompt=None)
        captured = capsys.readouterr()
        assert "[deliberator]" in captured.out
        assert "ENTRY TEXT:" in captured.out
        assert "$" not in captured.out

    def test_call_deliberator_custom_prompt_in_body(self):
        """call_deliberator should send custom prompt in request body."""
        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "high_risk"}}]
        }
        mock_response.raise_for_status = mock.Mock()
        custom = "MY_PROMPT: $text"
        with mock.patch("risk_pipeline.requests.post", return_value=mock_response) as mock_post:
            result = call_deliberator(
                "sample text",
                {"guard_x": 1},
                deliberation_prompt=custom,
            )
            assert result == "high_risk"
            body = mock_post.call_args.kwargs["json"]
            content = body["messages"][0]["content"]
            assert "MY_PROMPT" in content
            assert "sample text" in content
            assert "$" not in content

    def test_run_deliberation_plumbs_prompt(self):
        """run_deliberation should pass prompt to call_deliberator."""
        import pandas as pd
        df = pd.DataFrame([{"id": 0, "text": "safe text", "llama_guard_label": "safe"}])
        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "no_low_risk"}}]
        }
        mock_response.raise_for_status = mock.Mock()
        custom = "PLUMB_CHECK: $text $scores_block"
        with mock.patch("risk_pipeline.requests.post", return_value=mock_response) as mock_post:
            result_df = run_deliberation(df, deliberation_prompt=custom)
            call_body = mock_post.call_args.kwargs["json"]
            assert "PLUMB_CHECK" in call_body["messages"][0]["content"]
            assert result_df["deliberation_risk"].iloc[0] == "no_low_risk"

    def test_main_bad_path_exits_before_run(self):
        """main() with nonexistent --deliberation-prompt should exit before calling run()."""
        with mock.patch("risk_pipeline.run") as mock_run, \
             mock.patch("sys.argv", ["risk_pipeline.py", "--demo",
                                    "--deliberation-prompt", "/no/such/file.txt"]):
            with pytest.raises(SystemExit):
                from risk_pipeline import main
                main()
            assert not mock_run.called


class TestErrorsNoNaN:
    """Regression: errors column must not contain ':nan' when a guard fails."""

    def test_errors_no_nan_on_guard_failure(self, tmp_path):
        """One guard raises, others succeed — errors column has real message, no ':nan'."""
        call_count = [0]

        def side_effect(url, **kwargs):
            call_count[0] += 1
            # ShieldGemma: 4 policy calls (call_count 1-4)
            if call_count[0] <= 4:
                r = mock.Mock()
                r.json.return_value = {"choices": [{"text": "No"}]}
                r.raise_for_status = mock.Mock()
                return r
            # llama_guard (call_count 5) — raise
            if call_count[0] == 5:
                raise RuntimeError("connection refused")
            # granite_guardian (call_count 6) and qwen_guard (call_count 7) — succeed
            r = mock.Mock()
            r.json.return_value = {"choices": [{"message": {"content": "safe"}}]}
            r.raise_for_status = mock.Mock()
            return r

        out = tmp_path / "out.csv"
        with mock.patch("risk_pipeline.requests.post", side_effect=side_effect):
            df = run(["test text"], str(out), skip_deliberation=True)

        errors_val = df["errors"].iloc[0]
        assert ":nan" not in errors_val
        assert "llama_guard" in errors_val
        assert "connection refused" in errors_val


class TestShieldGemmaPerPolicy:
    """ShieldGemma mixed policy verdicts produce correct per-policy columns."""

    def test_shieldgemma_only_harassment_flagged(self, tmp_path):
        """harassment=Yes, others=No → shieldgemma_category=='harassment', p_harassment==1."""
        sg_call = [0]

        def side_effect(url, **kwargs):
            sg_call[0] += 1
            # ShieldGemma policy order: dangerous_content, harassment, hate_speech, sexually_explicit
            if sg_call[0] == 1:  # dangerous_content
                text = "No"
            elif sg_call[0] == 2:  # harassment
                text = "Yes"
            elif sg_call[0] == 3:  # hate_speech
                text = "No"
            elif sg_call[0] == 4:  # sexually_explicit
                text = "No"
            else:
                text = "safe"  # other guards

            r = mock.Mock()
            r.json.return_value = {
                "choices": [{"text": text} if sg_call[0] <= 4 else {"message": {"content": text}}]
            }
            r.raise_for_status = mock.Mock()
            return r

        out = tmp_path / "out.csv"
        with mock.patch("risk_pipeline.requests.post", side_effect=side_effect):
            df = run(["test text"], str(out), skip_deliberation=True)

        row = df.iloc[0]
        assert row["shieldgemma_unsafe"] == 1
        assert row["shieldgemma_category"] == "harassment"
        assert row["shieldgemma_p_dangerous_content"] == 0
        assert row["shieldgemma_p_harassment"] == 1
        assert row["shieldgemma_p_hate_speech"] == 0
        assert row["shieldgemma_p_sexually_explicit"] == 0


class TestDeliberationRetry:
    """run_deliberation should retry once on empty content."""

    def test_retry_on_empty_content(self):
        """First call returns empty content, second returns valid risk level."""
        call_count = [0]

        def side_effect(url, **kwargs):
            call_count[0] += 1
            r = mock.Mock()
            if call_count[0] == 1:
                r.json.return_value = {
                    "choices": [{"message": {"content": ""}, "finish_reason": "length"}]
                }
            else:
                r.json.return_value = {
                    "choices": [{"message": {"content": "high_risk"}}]
                }
            r.raise_for_status = mock.Mock()
            return r

        df = pd.DataFrame([{"id": 0, "text": "test", "llama_guard_label": "safe"}])
        with mock.patch("risk_pipeline.requests.post", side_effect=side_effect):
            result_df = run_deliberation(df)
        assert result_df["deliberation_risk"].iloc[0] == "high_risk"
        assert not str(result_df["deliberation_risk"].iloc[0]).startswith("error:")

    def test_retry_count_is_two(self):
        """Should make exactly two calls on retry."""
        call_count = [0]

        def side_effect(url, **kwargs):
            call_count[0] += 1
            r = mock.Mock()
            if call_count[0] == 1:
                r.json.return_value = {
                    "choices": [{"message": {"content": ""}}]
                }
            else:
                r.json.return_value = {
                    "choices": [{"message": {"content": "concerning"}}]
                }
            r.raise_for_status = mock.Mock()
            return r

        df = pd.DataFrame([{"id": 0, "text": "test", "llama_guard_label": "safe"}])
        with mock.patch("risk_pipeline.requests.post", side_effect=side_effect):
            run_deliberation(df)
        assert call_count[0] == 2


class TestDisableThinkingRequestBody:
    """When DELIBERATION_DISABLE_THINKING is on, request body includes chat_template_kwargs."""

    def test_chat_template_kwargs_included(self):
        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "high_risk"}}]
        }
        mock_response.raise_for_status = mock.Mock()
        with mock.patch("risk_pipeline.DELIBERATION_DISABLE_THINKING", True), \
             mock.patch("risk_pipeline.requests.post", return_value=mock_response) as mock_post:
            call_deliberator("sample", {"guard_x": 1})
            body = mock_post.call_args.kwargs["json"]
            assert "chat_template_kwargs" in body
            assert body["chat_template_kwargs"]["enable_thinking"] is False

    def test_chat_template_kwargs_omitted_when_disabled(self):
        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "high_risk"}}]
        }
        mock_response.raise_for_status = mock.Mock()
        with mock.patch("risk_pipeline.DELIBERATION_DISABLE_THINKING", False), \
             mock.patch("risk_pipeline.requests.post", return_value=mock_response) as mock_post:
            call_deliberator("sample", {"guard_x": 1})
            body = mock_post.call_args.kwargs["json"]
            assert "chat_template_kwargs" not in body
