"""Validate all built-in prompt files: 21 domain prompts, shared files, and both LLM templates."""

import os
import re
from pathlib import Path
from string import Template

import pytest

from rubric import (
    CANONICAL_DOMAINS,
    load_domain_prompt_templates,
)
from deliberation import (
    load_consolidation_prompt,
    load_final_update_prompt,
    consolidation_build,
    final_update_build,
)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DOMAINS_DIR = PROMPTS_DIR / "domains"
SHARED_DIR = PROMPTS_DIR / "_shared"

# Domain files have numeric prefixes (01_, 02_, etc.) — strip them for comparison.
def _strip_prefix(name: str) -> str:
    m = re.match(r"^\d+_(.*)", name)
    return m.group(1) if m else name


class TestDomainPromptFiles:
    """Validate all 21 built-in domain prompts."""

    @pytest.fixture
    def templates(self):
        return load_domain_prompt_templates(
            str(DOMAINS_DIR),
            str(SHARED_DIR / "rubric_prelude.txt"),
            str(SHARED_DIR / "output_schema.txt"),
        )

    def test_count_is_21(self, templates):
        assert len(templates) == 21, f"Expected 21 domain templates, got {len(templates)}"

    def test_all_canonical_domains_present(self, templates):
        slugs = {_strip_prefix(k) for k in templates}
        for domain in CANONICAL_DOMAINS:
            assert domain in slugs, f"Missing domain template: {domain}"

    def test_no_extra_domains(self, templates):
        for key in templates:
            slug = _strip_prefix(key)
            assert slug in CANONICAL_DOMAINS, f"Unexpected domain: {slug}"

    def test_each_prompt_has_required_placeholders(self, templates):
        """Each domain template must contain $text, $age, $entry_date."""
        for domain, content in templates.items():
            assert "$text" in content, f"{domain} missing $text"
            assert "$age" in content, f"{domain} missing $age"
            assert "$entry_date" in content, f"{domain} missing $entry_date"

    def test_no_unknown_placeholders(self, templates):
        """No placeholder outside the known set."""
        known = {"$text", "$age", "$entry_date", "$scores_block", "$$$"}
        for domain, content in templates.items():
            found = set(re.findall(r"\$\w+", content))
            unknown = found - known
            assert not unknown, f"{domain} has unknown placeholders: {unknown}"

    def test_all_prompts_render_cleanly(self, templates):
        """Every template renders without unresolved $ variables."""
        sample = {
            "text": "Sample entry text.",
            "age": "18",
            "entry_date": "2025-01-01",
            "scores_block": "",
        }
        for domain, content in templates.items():
            rendered = Template(content).substitute(**sample)
            assert not re.search(r"\$[A-Za-z_]", rendered), (
                f"{domain} has unresolved placeholder after render"
            )

    def test_file_count_on_disk(self):
        """21 domain .txt files exist on disk."""
        txt_files = sorted(DOMAINS_DIR.glob("*.txt"))
        assert len(txt_files) == 21, f"Expected 21 domain files, found {len(txt_files)}"

    def test_shared_files_skipped(self, templates):
        """Files with '_shared' in the name should not appear in templates."""
        for domain in templates:
            assert "_shared" not in domain, f"_shared file leaked as domain: {domain}"


class TestSharedFiles:
    """Validate shared prompt files."""

    def test_prelude_exists(self):
        prelude = SHARED_DIR / "rubric_prelude.txt"
        assert prelude.exists(), "rubric_prelude.txt not found"
        content = prelude.read_text()
        assert "Tier definitions" in content

    def test_schema_exists(self):
        schema = SHARED_DIR / "output_schema.txt"
        assert schema.exists(), "output_schema.txt not found"
        content = schema.read_text()
        assert "domain" in content
        assert "tier" in content

    def test_prelude_no_unknown_placeholders(self):
        content = (SHARED_DIR / "rubric_prelude.txt").read_text()
        found = set(re.findall(r"\$\w+", content))
        assert not found, f"rubric_prelude.txt has unexpected placeholders: {found}"

    def test_schema_no_unknown_placeholders(self):
        content = (SHARED_DIR / "output_schema.txt").read_text()
        found = set(re.findall(r"\$\w+", content))
        assert not found, f"output_schema.txt has unexpected placeholders: {found}"


class TestLLMTemplates:
    """Validate consolidation and final_update LLM templates."""

    def test_consolidation_loads_default(self):
        prompt = load_consolidation_prompt()
        assert prompt

    def test_consolidation_has_expected_placeholders(self):
        prompt = load_consolidation_prompt()
        for placeholder in ("text", "context_block", "guard_block", "domain_block", "rule_block"):
            assert f"${placeholder}" in prompt, f"Consolidation prompt missing ${placeholder}"

    def test_final_update_loads_default(self):
        prompt = load_final_update_prompt()
        assert prompt

    def test_final_update_has_expected_placeholders(self):
        prompt = load_final_update_prompt()
        for placeholder in ("text", "context_block", "guard_block", "domain_block", "rule_block", "consolidation_block"):
            assert f"${placeholder}" in prompt, f"Final update prompt missing ${placeholder}"

    def test_consolidation_builds_clean(self):
        rendered = consolidation_build(
            text="Sample text",
            context_block="Age: 25",
            guard_block="Guard results",
            domain_block="Domain results",
            rule_block="Rules applied",
        )
        assert "Sample text" in rendered
        assert not re.search(r"\$[A-Za-z_]", rendered), "Unresolved $ in consolidation prompt"

    def test_final_update_builds_clean(self):
        rendered = final_update_build(
            text="Sample text",
            context_block="Age: 25",
            guard_block="Guard results",
            domain_block="Domain results",
            rule_block="Rules applied",
            consolidation_block="Consolidated tier: yellow",
        )
        assert "Sample text" in rendered
        assert "yellow" in rendered
        assert not re.search(r"\$[A-Za-z_]", rendered), "Unresolved $ in final update prompt"
