from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vllm.renderers.chat_utils import ChatCompletionMessageParam

## Runner Helpers


def create_message(text: str) -> list["ChatCompletionMessageParam"]:
    param: ChatCompletionMessageParam = {"role": "user", "content": text}
    return [param]


def create_messages(texts: list[str]):
    return [create_message(t) for t in texts]


## Common


def dedupe[T: Hashable](vs: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(vs))


## Llama Guard


def llama_guard_categories(verdict: str):
    sp = verdict.split("\n")
    catstr = sp[1] if len(sp) > 1 else None
    if not catstr:
        return []
    cats = {r for r in (c.strip() for c in catstr.split(",")) if r}
    return frozenset(cats)


def llama_guard_partition(ins, outs):
    out_texts = [o[0].strip() for o in outs]
    results = list(zip(ins, out_texts))
    return {
        "safe": dedupe([(i, o) for i, o in results if o == "safe"]),
        "unsafe": dedupe(
            [(i, llama_guard_categories(o)) for i, o in results if o != "safe"]
        ),
    }


## Shield Gemma


def shield_gemma_partition(
    results: list[tuple[str, str | None]],
) -> dict[Literal["safe", "unsafe"], list[tuple[str, str | None]]]:

    is_safe = lambda s: s and s.strip().startswith("No")

    return {
        "safe": dedupe([(t, r) for t, r in results if is_safe(r)]),
        "unsafe": dedupe([(t, r) for t, r in results if not is_safe(r)]),
    }
