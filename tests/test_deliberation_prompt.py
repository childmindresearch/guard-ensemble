"""Tests for deliberation prompt loading and template rendering."""
import re
import sys
from pathlib import Path
from unittest import mock

import pytest
from guards import load_deliberation_prompt, deliberation_build, RISK_LEVELS

# ---------------------------------------------------------------------------
# Load tests
# ---------------------------------------------------------------------------

class TestLoadDeliberationPrompt:
    def test_default_returns_file_content(self):
        content = load_deliberation_prompt()
        assert isinstance(content, str)
        assert len(content) > 100

    def test_default_contains_placeholders(self):
        content = load_deliberation_prompt()
        assert "$text" in content
        assert "$scores_block" in content

    def test_custom_path(self, tmp_path):
        custom = tmp_path / "custom.txt"
        custom.write_text("$text is here")
        result = load_deliberation_prompt(str(custom))
        assert "is here" in result

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_deliberation_prompt("/nonexistent/path/prompt.txt")

    def test_missing_placeholder(self, tmp_path):
        bad = tmp_path / "bad.txt"
        bad.write_text("no placeholders at all")
        with pytest.raises(ValueError, match="missing placeholder"):
            load_deliberation_prompt(str(bad))

    def test_unknown_placeholder(self, tmp_path):
        bad = tmp_path / "bad.txt"
        bad.write_text("$text and $mystery")
        with pytest.raises(ValueError, match="unknown placeholder"):
            load_deliberation_prompt(str(bad))

# ---------------------------------------------------------------------------
# Build tests
# ---------------------------------------------------------------------------

class TestDeliberationBuildWithTemplate:
    def test_custom_prompt_renders(self):
        template = "Custom: $text with $scores_block"
        result = deliberation_build("hello", {"a": "1"}, prompt=template)
        assert "hello" in result
        assert "Custom:" in result

    def test_default_prompt_no_dollar_residue(self):
        result = deliberation_build("test", {})
        assert "$" not in result

    def test_empty_scores_uses_fallback(self):
        result = deliberation_build("test", {})
        assert "none available" in result.lower()
