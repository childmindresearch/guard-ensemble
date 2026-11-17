import re
from collections.abc import Hashable, Iterable

from guard_ensemble.guards import shieldgemma
from guard_ensemble.llm import runner
from guard_ensemble.llm.lib import create_messages
from guard_ensemble.util import PACKAGE_DIR

type Outs = list[list[str]]

OUT = PACKAGE_DIR / "out"


## Common


def first_out(o: list[str]):
    return o[0].strip() if len(o) > 0 else None


def eq_no(s: str | None):
    return s.strip().lower().startswith("no") if s else False


def dedupe[T: Hashable](vs: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(vs))


## Llama Guard


def llama_guard_categories(verdict: str) -> list[str]:
    sp = verdict.split("\n")
    catstr = sp[1] if len(sp) > 1 else None
    if not catstr:
        return []
    cats = {r for r in (c.strip() for c in catstr.split(",")) if r}
    return list(set(cats))


def run_llama_guard(texts: list[str]):
    model = "llama_guard"
    runner.load(model)

    messages = create_messages(texts)
    outs = runner.chat(model, messages, temperature=0, max_tokens=32)
    out_texts = [first_out(o) for o in outs]
    text_cats = [
        llama_guard_categories(o) if o and o != "safe" else [] for o in out_texts
    ]

    return [
        {
            "text": text,
            "llama_guard_out": out,
            "llama_guard_unsafe_categories": cats,
            "llama_guard_flagged": len(cats) > 0 or out is None,
        }
        for text, out, cats in zip(texts, out_texts, text_cats)
    ]


## Shield Gemma


type ShieldGemmaResult = dict[str, str | bool | None]


def run_shieldgemma(texts: list[str]):
    model = "shieldgemma"
    runner.load(model)

    messages = create_messages(texts)

    results: list[ShieldGemmaResult] = [{"text": text} for text in texts]

    for key, policy in shieldgemma.POLICIES.items():
        outs = runner.chat(
            model,
            messages,
            chat_template_kwargs={"guideline": policy},
            temperature=0,
            max_tokens=1,
        )
        out_texts = [first_out(o) for o in outs]
        out_key = f"shieldgemma_{key}_out"

        for i in range(len(results)):
            results[i][out_key] = out_texts[i]
            if not eq_no(out_texts[i]):
                results[i]["shieldgemma_flagged"] = True

    return results


## Granite Guardian


GRANITE_GUARDIAN_SCORE_RE = r"<score>(.*?)</score>"


def granite_guardian_score(out_text: str) -> str | None:
    matches: list[str] = re.findall(GRANITE_GUARDIAN_SCORE_RE, out_text, re.DOTALL)

    if len(matches) == 0:
        return None

    return matches[-1].strip().lower()


def run_granite_guardian(texts: list[str]):
    model = "granite_guardian"
    runner.load(model)

    messages = create_messages(texts)
    outs = runner.chat(model, messages, temperature=0, max_tokens=32)
    out_texts = [first_out(o) for o in outs]
    text_scores = [granite_guardian_score(o) if o else None for o in out_texts]
    return [
        {
            "text": text,
            "granite_guardian_out": out,
            "granite_guardian_flagged": not eq_no(score),
        }
        for text, out, score in zip(texts, out_texts, text_scores)
    ]


## Qwen Guard


def qwen_guard_labels(out_text: str) -> tuple[str | None, str | None]:
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
    safe_label_match = re.search(safe_pattern, out_text)
    label: str | None = safe_label_match.group(1) if safe_label_match else None
    categories: list[str] = re.findall(category_pattern, out_text)
    return label, ",".join(categories) or None


def eq_unsafe(s: str | None) -> bool:
    return s is None or s.lower() == "unsafe"


def run_qwen_guard(texts: list[str]):
    model = "qwen_guard"
    runner.load(model)

    messages = create_messages(texts)
    outs = runner.chat(model, messages, temperature=0, max_tokens=32)
    out_texts = [first_out(o) for o in outs]
    text_labels = [qwen_guard_labels(o) if o else (None, None) for o in out_texts]
    return [
        {
            "text": text,
            "qwen_guard_out": out,
            "qwen_guard_label": label,
            "qwen_guard_categories": cats,
            "qwen_guard_flagged": eq_unsafe(label),
        }
        for text, out, (label, cats) in zip(texts, out_texts, text_labels)
    ]
