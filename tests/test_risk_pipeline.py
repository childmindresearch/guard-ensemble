"""Tests for risk_pipeline CLI flag and prompt plumbing."""
import sys
from unittest import mock

import pytest
from risk_pipeline import call_deliberator, dry_run, run_deliberation


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
