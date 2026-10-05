"""Guard prompt templates, parsers, and guard definitions for risk detection."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Union

# ---------------------------------------------------------------------------
# DMR endpoint configuration
# ---------------------------------------------------------------------------

DMR_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/engines/v1")

# ---------------------------------------------------------------------------
# ShieldGemma policies (verbatim from model card)
# ---------------------------------------------------------------------------
SHIELDGEMMA_POLICIES: dict[str, str] = {
    "dangerous_content": (
        '"No Dangerous Content": The prompt shall not contain or seek generation of content '
        'that harming oneself and/or others (e.g., accessing or building firearms and explosive devices, '
        'promotion of terrorism, instructions for suicide).'
    ),
    "harassment": (
        '"No Harassment": The prompt shall not contain or seek generation of content '
        'that is malicious, intimidating, bullying, or abusive content targeting another individual '
        '(e.g., physical threats, denial of tragic events, disparaging victims of violence).'
    ),
    "hate_speech": (
        '"No Hate Speech": The prompt shall not contain or seek generation of content '
        'that is targeting identity and/or protected attributes (e.g., racial slurs, promotion of '
        'discrimination, calls to violence against protected groups) or dehumanizing/belittling/'
        'vilifying on the bases of race, ethnicity, religion, nationality, veteran status, age, '
        'sexual orientation, gender, gender identity, caste, or any other protected status. '
        'We do allow content that includes a reference to a protected individual or group '
        'in a positive manner.'
    ),
    "sexually_explicit": (
        '"No Sexually Explicit Information": The prompt shall not contain or seek generation of '
        'content that contains references to sexual acts or other lewd content '
        '(e.g., sexually graphic descriptions, content aimed at causing arousal). '
        'Medical or scientific terms related to human anatomy and sex education are permitted.'
    ),
}

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def shieldgemma_build(text: str, policy_key: str) -> str:
    """Build a single ShieldGemma prompt for one policy."""
    policy = SHIELDGEMMA_POLICIES[policy_key]
    return (
        f"You are a policy expert trying to help determine whether a user prompt is in violation of the defined safety policies.\n"
        f"<start_of_turn>\n"
        f"Human Question: {text.strip()}\n"
        f"<end_of_turn>\n"
        f"Our safety principle is defined in the below:\n"
        f"{policy}\n"
        f"Does the human question violate the above principle? Your answer must start with 'Yes' or 'No'. "
        f"And then walk through step by step to be sure we answer correctly."
    )


def _shieldgemma_build_all(text: str) -> list[str]:
    """Return a list of 4 ShieldGemma prompts, one per policy."""
    return [shieldgemma_build(text, k) for k in SHIELDGEMMA_POLICIES]


def llama_guard_build(text: str) -> str:
    task_block = (
        "Task: Check if there is unsafe content in 'User' messages in conversations "
        "according to our safety policy with the below categories.\n\n"
        "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
        "S1: Violent Crimes.\n"
        "S2: Non-Violent Crimes.\n"
        "S3: Sex Crimes.\n"
        "S4: Child Sexual Exploitation.\n"
        "S5: Defamation.\n"
        "S6: Specialized Advice.\n"
        "S7: Privacy.\n"
        "S8: Intellectual Property.\n"
        "S9: Indecent Content.\n"
        "S10: Hate.\n"
        "S11: Suicide & Self-Harm.\n"
        "S12: Sexual Content.\n"
        "S13: Elections.\n"
        "S14: Code Interpreter Abuse.\n"
        "<END UNSAFE CONTENT CATEGORIES>\n\n"
        "<BEGIN CONVERSATION>\n\n"
        f"User: {text}\n\n"
        "<END CONVERSATION>\n\n"
        "Provide your safety assessment for User in the above conversation:\n"
        "- If unsafe, a second line must include a comma-separated list of violated categories."
    )
    return task_block


def granite_guardian_build(text: str) -> str:
    return (
        "Is the user message harmful based on the risk definition? "
        "Your answer must be either 'Yes' or 'No'.\n\n"
        "Conversation:\n"
        f"User: {text}"
    )


def qwen_guard_build(text: str) -> str:
    return text


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def strip_think(raw: str) -> str:
    """Remove <think> blocks and unterminated trailing fragments."""
    raw = re.sub(r"<think>.*?", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
    return raw.strip()

_SHIELDGEMMA_YES_RE = re.compile(r"^\s*([Yy][Ee]?[Ss]|[Nn][Oo]?)")

_GRANITE_LABELLED_RE = re.compile(
    r"(?:answer|verdict|harmful)\s*[:*\s]*\s*(yes|no)", re.IGNORECASE
)
_GRANITE_BARE_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _extract_verdict(raw: str) -> str | None:
    """Extract yes/no verdict from Granite Guardian output.

    Tries labelled form first (answer/verdict/harmful : Yes),
    then bare yes/no anywhere in the text.
    """
    m = _GRANITE_LABELLED_RE.search(raw)
    if m:
        return m.group(1).lower()
    m = _GRANITE_BARE_RE.search(raw)
    if m:
        return m.group(1).lower()
    return None


def shieldgemma_parse(raw: str, policy_name: str, guard_name: str = "shieldgemma") -> dict:
    raw = strip_think(raw)
    m = _SHIELDGEMMA_YES_RE.match(raw)
    if not m:
        raise ValueError(f"ShieldGemma parse failure: no Yes/No found in: {raw!r}")
    answer = m.group(1).upper()
    if answer in ("Y", "YE", "YES"):
        return {
            f"{guard_name}_label": "unsafe",
            f"{guard_name}_unsafe": 1,
            f"{guard_name}_category": policy_name,
        }
    if answer in ("N", "NO"):
        return {
            f"{guard_name}_label": "safe",
            f"{guard_name}_unsafe": 0,
            f"{guard_name}_category": "None",
        }
    raise ValueError(f"ShieldGemma parse failure: unexpected match {answer!r}")


def _shieldgemma_parse_all(results_raw: list[str], guard_name: str = "shieldgemma") -> list[dict]:
    """Parse 4 raw outputs in policy order."""
    return [shieldgemma_parse(raw, policy, guard_name) for raw, policy in zip(results_raw, SHIELDGEMMA_POLICIES)]


_LLAMA_FIRST_LINE_RE = re.compile(r"^\s*(safe|unsafe)", re.IGNORECASE)


def llama_guard_parse(raw: str, guard_name: str = "llama_guard") -> dict:
    raw = strip_think(raw)
    m = _LLAMA_FIRST_LINE_RE.match(raw)
    if not m:
        raise ValueError(f"Llama Guard parse failure: no safe/unsafe found in: {raw!r}")
    verdict = m.group(1).lower()
    lines = raw.strip().splitlines()
    if verdict == "safe":
        return {
            f"{guard_name}_label": "safe",
            f"{guard_name}_unsafe": 0,
            f"{guard_name}_category": "None",
        }
    # unsafe — second line may contain S-codes
    category = "None"
    if len(lines) >= 2:
        second = lines[1].strip()
        if second:
            category = second
    return {
        f"{guard_name}_label": "unsafe",
        f"{guard_name}_unsafe": 1,
        f"{guard_name}_category": category,
    }


def granite_guardian_parse(raw: str, guard_name: str = "granite_guardian") -> dict:
    raw = strip_think(raw)
    verdict = _extract_verdict(raw)
    if verdict is None:
        raise ValueError(f"Granite Guardian parse failure: no Yes/No found in: {raw!r}")
    if verdict == "yes":
        return {
            f"{guard_name}_label": "unsafe",
            f"{guard_name}_unsafe": 1,
            f"{guard_name}_category": "harm",
        }
    return {
        f"{guard_name}_label": "safe",
        f"{guard_name}_unsafe": 0,
        f"{guard_name}_category": "None",
    }


_QWEN_SAFETY_RE = re.compile(
    r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE
)
_QWEN_CATEGORIES_RE = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|None)",
    re.IGNORECASE,
)


def qwen_guard_parse(raw: str, guard_name: str = "qwen_guard") -> dict:
    raw = strip_think(raw)
    sm = _QWEN_SAFETY_RE.search(raw)
    if not sm:
        raise ValueError(f"Qwen Guard parse failure: no Safety line in: {raw!r}")
    verdict = sm.group(1)
    unsafe = 1 if verdict.lower() == "unsafe" else 0
    categories = _QWEN_CATEGORIES_RE.findall(raw)
    # Exclude "None" from category string
    categories = [c for c in categories if c.lower() != "none"]
    category_str = ",".join(categories) if categories else "None"
    return {
        f"{guard_name}_label": verdict,
        f"{guard_name}_unsafe": unsafe,
        f"{guard_name}_category": category_str,
    }


# ---------------------------------------------------------------------------
# Deliberation stage
# ---------------------------------------------------------------------------

DELIBERATION_MODEL_ID = "huggingface.co/qwen/qwen3-4b-gguf:Q8_0"
DELIBERATION_BASE_URL = os.environ.get("DELIBERATION_BASE_URL", DMR_BASE_URL)
DELIBERATION_PATH = "/v1/chat/completions"
DELIBERATION_MAX_TOKENS = 1024

RISK_LEVELS = ["high_risk", "concerning", "no_low_risk"]

_RISK_LEVEL_GUIDE = """- high_risk: the text is clearly dangerous, illegal, hateful, violent, \
or sexual in a way that violates safety policy. Explicit intent, threats, or targeted abuse.
- concerning: the text shows ambiguous or borderline signals. Mild policy concerns, edgy or \
ambiguous phrasing, topics that are sensitive but not clearly harmful.
- no_low_risk: the text appears safe, benign, or ordinary. No meaningful safety concerns."""


def deliberation_build(text: str, guard_scores: dict) -> str:
    """Build the deliberation prompt from the raw text plus guard model ratings."""
    if guard_scores:
        score_lines = "\n".join(f"- {k}: {v}" for k, v in guard_scores.items())
        scores_block = f"Guard model ratings for this entry:\n{score_lines}"
    else:
        scores_block = (
            "Guard model ratings for this entry: none available "
            "(all guard models failed or were skipped). Judge from the text alone."
        )

    return (
        "You are a risk analyst reviewing content that has already been screened by "
        "specialised guard language models.\n\n"
        f"ENTRY TEXT:\n{text}\n\n"
        f"{scores_block}\n\n"
        "Weigh the entry text together with the guard model ratings above and decide a "
        "single overall risk level. Trust the guard models on their specific categories, but "
        "use your own judgement on the text where guard models disagree or report nothing.\n\n"
        f"Risk levels:\n{_RISK_LEVEL_GUIDE}\n\n"
        "Respond with exactly one of: high_risk, concerning, no_low_risk. "
        "Output that single word and nothing else."
    )


_DELIBERATION_RISK_RE = re.compile(
    r"\b(high[_\s-]?risk|concerning|no[_\s-]?low[_\s-]?risk|no[_\s-]?risk|low[_\s-]?risk|safe)\b",
    re.IGNORECASE,
)


def deliberation_parse(raw: str) -> str:
    raw = strip_think(raw)
    m = _DELIBERATION_RISK_RE.search(raw)
    if not m:
        raise ValueError(f"deliberation parse failure: no risk level found in: {raw!r}")
    token = m.group(1).lower().replace(" ", "_").replace("-", "_")
    if token.startswith("high"):
        return "high_risk"
    if token == "concerning":
        return "concerning"
    return "no_low_risk"


# ---------------------------------------------------------------------------
# Guard dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Guard:
    name: str
    model_id: str
    base_url: str = field(default="")
    path: str = "/v1/chat/completions"
    max_tokens: int = 0
    build: Callable[[str], Union[str, list[str]]] = field(default=None)  # type: ignore[arg-type]
    parse: Callable[[str], dict] = field(default=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Guard instances
# ---------------------------------------------------------------------------

GUARDS: list[Guard] = [
    Guard(
        name="shieldgemma",
        model_id="huggingface.co/quantfactory/shieldgemma-9b-gguf:Q8_0",
        base_url=os.environ.get("SHIELDGEMMA_BASE_URL", DMR_BASE_URL),
        path="/v1/completions",
        max_tokens=5,
        build=_shieldgemma_build_all,
        parse=lambda raw: shieldgemma_parse(raw, policy_name="dangerous_content"),
    ),
    Guard(
        name="llama_guard",
        model_id="huggingface.co/quantfactory/llama-guard-3-8b-gguf:Q8_0",
        base_url=os.environ.get("LLAMA_GUARD_BASE_URL", DMR_BASE_URL),
        path="/v1/chat/completions",
        max_tokens=32,
        build=llama_guard_build,
        parse=llama_guard_parse,
    ),
    Guard(
        name="granite_guardian",
        model_id="huggingface.co/ibm-granite/granite-guardian-3.3-8b-gguf:Q8_0",
        base_url=os.environ.get("GRANITE_GUARDIAN_BASE_URL", DMR_BASE_URL),
        path="/v1/chat/completions",
        max_tokens=256,
        build=granite_guardian_build,
        parse=granite_guardian_parse,
    ),
    Guard(
        name="qwen_guard",
        model_id="huggingface.co/geoffmunn/Qwen3Guard-Gen-8B-GGUF:Q4_K_M",
        base_url=os.environ.get("QWEN_GUARD_BASE_URL", DMR_BASE_URL),
        path="/v1/chat/completions",
        max_tokens=64,
        build=qwen_guard_build,
        parse=qwen_guard_parse,
    ),
]


# ---------------------------------------------------------------------------
# ShieldGemma aggregator
# ---------------------------------------------------------------------------

def shieldgemma_aggregate(policy_results: list[dict]) -> dict:
    """Aggregate 4 ShieldGemma policy parse dicts into a single result dict."""
    any_unsafe = any(d.get("shieldgemma_unsafe", 0) for d in policy_results)
    violated = [d["shieldgemma_category"] for d in policy_results if d.get("shieldgemma_unsafe", 0) == 1]

    return {
        "shieldgemma_unsafe": 1 if any_unsafe else 0,
        "shieldgemma_label": "unsafe" if any_unsafe else "safe",
        "shieldgemma_category": "|".join(violated) if violated else "None",
        "shieldgemma_p_dangerous_content": 1 if any(
            d.get("shieldgemma_category") == "dangerous_content" and d.get("shieldgemma_unsafe", 0) == 1
            for d in policy_results
        ) else 0,
        "shieldgemma_p_harassment": 1 if any(
            d.get("shieldgemma_category") == "harassment" and d.get("shieldgemma_unsafe", 0) == 1
            for d in policy_results
        ) else 0,
        "shieldgemma_p_hate_speech": 1 if any(
            d.get("shieldgemma_category") == "hate_speech" and d.get("shieldgemma_unsafe", 0) == 1
            for d in policy_results
        ) else 0,
        "shieldgemma_p_sexually_explicit": 1 if any(
            d.get("shieldgemma_category") == "sexually_explicit" and d.get("shieldgemma_unsafe", 0) == 1
            for d in policy_results
        ) else 0,
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(len(GUARDS))
