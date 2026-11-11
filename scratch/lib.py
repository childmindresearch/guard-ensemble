import re
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from vllm.renderers.chat_utils import ChatCompletionMessageParam


type Ins = list[str]
type Outs = list[list[str]]


## Runner Helpers


def create_message(text: str) -> list["ChatCompletionMessageParam"]:
    param: ChatCompletionMessageParam = {"role": "user", "content": text}
    return [param]


def create_messages(texts: list[str]):
    return [create_message(t) for t in texts]


## Common


def eq_no(s: str | None):
    return s.strip().lower().startswith("no") if s else False


def dedupe[T: Hashable](vs: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(vs))


## Llama Guard


class LlamaGuardPartition(TypedDict):
    safe: list[tuple[str, str]]
    unsafe: list[tuple[str, frozenset[str]]]


def llama_guard_categories(verdict: str) -> frozenset[str]:
    sp = verdict.split("\n")
    catstr = sp[1] if len(sp) > 1 else None
    if not catstr:
        return frozenset()
    cats = {r for r in (c.strip() for c in catstr.split(",")) if r}
    return frozenset(cats)


def llama_guard_partition(ins: Ins, outs: Outs) -> LlamaGuardPartition:
    out_texts = [o[0].strip() for o in outs]
    results = list(zip(ins, out_texts))
    return {
        "safe": dedupe([(i, o) for i, o in results if o == "safe"]),
        "unsafe": dedupe(
            [(i, llama_guard_categories(o)) for i, o in results if o != "safe"]
        ),
    }


## Shield Gemma


class ShieldGemmaPartition(TypedDict):
    safe: list[tuple[str, str]]
    unsafe: list[tuple[str, str | None]]


def shieldgemma_partition(ins: Ins, outs: Outs) -> ShieldGemmaPartition:
    results = list(zip(ins, [next(iter(o), None) for o in outs]))

    return {
        "safe": dedupe([(t, r) for t, r in results if r is not None and eq_no(r)]),
        "unsafe": dedupe([(t, r) for t, r in results if not eq_no(r)]),
    }


## Granite Guardian


class GraniteGuardianPartition(TypedDict):
    safe: list[tuple[str, str]]
    unsafe: list[tuple[str, str | None]]


GRANITE_GUARDIAN_SCORE_RE = r"<score>(.*?)</score>"


def granite_guardian_score(out: list[str]) -> str | None:
    out_text = next(iter(out), None)

    if out_text is None:
        return None

    matches: list[str] = re.findall(GRANITE_GUARDIAN_SCORE_RE, out_text, re.DOTALL)

    if len(matches) == 0:
        return None

    return matches[-1].strip().lower()


def granite_guardian_partition(ins: Ins, outs: Outs) -> GraniteGuardianPartition:
    results = list(zip(ins, [granite_guardian_score(o) for o in outs]))
    return {
        "safe": dedupe([(t, r) for t, r in results if r is not None and eq_no(r)]),
        "unsafe": dedupe([(t, r) for t, r in results if not eq_no(r)]),
    }


## Qwen Guard


class QwenGuardPartition(TypedDict):
    safe: list[tuple[str, tuple[str, str]]]
    unsafe: list[tuple[str, tuple[str, str]]]


def qwen_guard_labels(out: list[str]) -> tuple[str, str]:
    out_text = next(iter(out), None)

    if out_text is None:
        return "None", "None"

    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
    safe_label_match = re.search(safe_pattern, out_text)
    label: str = safe_label_match.group(1) if safe_label_match else "None"
    categories: list[str] = re.findall(category_pattern, out_text)
    return label, ",".join(categories) or "None"


def eq_unsafe(s: str | None) -> bool:
    return s is None or s.lower() == "unsafe"


def qwen_guard_partition(ins: Ins, outs: Outs) -> QwenGuardPartition:
    results = list(zip(ins, [qwen_guard_labels(o) for o in outs]))
    return {
        "safe": dedupe([(t, (l, c)) for t, (l, c) in results if not eq_unsafe(l)]),
        "unsafe": dedupe([(t, (l, c)) for t, (l, c) in results if eq_unsafe(l)]),
    }
