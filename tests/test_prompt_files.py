"""Validate all Kev YAML domain files: 21 domain prompts and shared config."""

import re
from pathlib import Path

import pytest
import yaml

from rubric import CANONICAL_DOMAINS

KEV_DIR = Path(__file__).parent.parent / "prompts" / "kev"
DOMAINS_DIR = KEV_DIR / "domains"
SHARED_FILE = KEV_DIR / "_shared.yaml"

# Domain files have numeric prefixes (01_, 02_, etc.) — strip them for comparison.
def _strip_prefix(name: str) -> str:
    m = re.match(r"^\d+_(.*)", name)
    return m.group(1) if m else name


class TestDomainYAMLFiles:
    """Validate all 21 Kev domain YAML files."""

    @pytest.fixture
    def domains(self):
        result = {}
        for f in sorted(DOMAINS_DIR.glob("*.yaml")):
            with open(f) as fh:
                data = yaml.safe_load(fh)
            result[_strip_prefix(f.stem)] = data
        return result

    def test_count_is_21(self, domains):
        assert len(domains) == 21, f"Expected 21 domain files, got {len(domains)}"

    def test_all_canonical_domains_present(self, domains):
        for domain in CANONICAL_DOMAINS:
            assert domain in domains, f"Missing domain: {domain}"

    def test_no_extra_domains(self, domains):
        for slug in domains:
            assert slug in CANONICAL_DOMAINS, f"Unexpected domain: {slug}"

    def test_each_domain_has_tier_score(self, domains):
        for slug, data in domains.items():
            assert data.get("tier", {}).get("type") == "score", (
                f"{slug} tier.type is not 'score'"
            )

    def test_each_domain_has_three_criteria(self, domains):
        for slug, data in domains.items():
            criteria = data.get("tier", {}).get("criteria", [])
            assert len(criteria) == 3, (
                f"{slug} has {len(criteria)} criteria, expected 3"
            )

    def test_criteria_contain_tier_labels(self, domains):
        """Each criterion must reference its tier: Green, Yellow, Red."""
        tier_labels = ["Green", "Yellow", "Red"]
        for slug, data in domains.items():
            criteria = data.get("tier", {}).get("criteria", [])
            for i, label in enumerate(tier_labels):
                criterion = criteria[i]
                assert label in criterion, (
                    f"{slug} criterion {i} missing label '{label}'"
                )

    def test_criteria_contain_examples(self, domains):
        """Each criterion must contain 'Examples'."""
        for slug, data in domains.items():
            criteria = data.get("tier", {}).get("criteria", [])
            for i, criterion in enumerate(criteria):
                assert "Examples" in criterion, (
                    f"{slug} criterion {i} missing examples"
                )

    def test_no_placeholder_variables(self, domains):
        """No $text, $age, or $entry_date placeholders should remain."""
        for slug, data in domains.items():
            instructions = data.get("tier", {}).get("instructions", "")
            assert "$text" not in instructions, f"{slug} contains $text placeholder"
            assert "$age" not in instructions, f"{slug} contains $age placeholder"
            assert "$entry_date" not in instructions, f"{slug} contains $entry_date placeholder"

    def test_dialect_1_suicidality_has_inline_examples(self, domains):
        """Dialect 1 domain (01_suicidality) criteria include examples."""
        data = domains.get("suicidality")
        assert data is not None
        criteria = data["tier"]["criteria"]
        for criterion in criteria:
            assert "Examples" in criterion

    def test_dialect_2_substance_has_bullet_examples(self, domains):
        """Dialect 2 domain (07_substance_use) criteria include examples."""
        data = domains.get("substance_use_intoxication_overdose_withdrawal")
        assert data is not None
        criteria = data["tier"]["criteria"]
        for criterion in criteria:
            assert "Examples" in criterion

    def test_file_count_on_disk(self):
        """21 domain .yaml files exist on disk."""
        yaml_files = sorted(DOMAINS_DIR.glob("*.yaml"))
        assert len(yaml_files) == 21, f"Expected 21 domain YAML files, found {len(yaml_files)}"


class TestSharedYAML:
    """Validate shared Kev YAML config."""

    @pytest.fixture
    def shared(self):
        with open(SHARED_FILE) as fh:
            return yaml.safe_load(fh)

    def test_shared_loads(self, shared):
        assert isinstance(shared, dict), "Shared YAML did not parse as dict"

    def test_reading_rules_present(self, shared):
        assert "reading_rules" in shared, "Missing reading_rules"
        rules = shared["reading_rules"]
        assert "Coded language" in rules or "coded" in rules.lower()

    def test_timing_is_choice(self, shared):
        timing = shared.get("timing", {})
        assert timing.get("type") == "choice", "Timing type must be 'choice'"

    def test_timing_has_six_options(self, shared):
        criteria = shared.get("timing", {}).get("criteria", {})
        expected = {
            "historical_resolved",
            "historical_unresolved",
            "recent",
            "current",
            "imminent",
            "unclear",
        }
        assert set(criteria.keys()) == expected, (
            f"Timing criteria keys mismatch: got {set(criteria.keys())}"
        )

    def test_flags_count(self, shared):
        flags = shared.get("flags", {})
        assert len(flags) == 9, f"Expected 9 flags, got {len(flags)}"

    def test_flags_all_noul(self, shared):
        for name, flag in shared.get("flags", {}).items():
            assert flag.get("type") == "noul", f"Flag {name} type is not 'noul'"

    def test_flags_match_vocabulary(self, shared):
        expected_flags = {
            "lethal_means_access",
            "current_intoxication",
            "alone_no_support",
            "minor_abuse_disclosure",
            "past_suicide_attempt",
            "past_self_harm",
            "past_sexual_assault",
            "past_psychosis",
            "past_eating_disorder",
        }
        actual = set(shared.get("flags", {}).keys())
        assert actual == expected_flags, f"Flags mismatch: got {actual}"

    def test_no_placeholder_variables(self, shared):
        """No $text, $age, or $entry_date placeholders in shared YAML."""
        with open(SHARED_FILE) as fh:
            content = fh.read()
        assert "$text" not in content, "Shared YAML contains $text placeholder"
        assert "$age" not in content, "Shared YAML contains $age placeholder"
        assert "$entry_date" not in content, "Shared YAML contains $entry_date placeholder"
