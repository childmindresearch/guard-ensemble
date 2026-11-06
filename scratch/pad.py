import random
from collections.abc import Iterable

import pandas as pd
from vllm.renderers.chat_utils import ChatCompletionMessageParam

## -- Loads meta-llama/Llama-Guard-3-8B model into memory
from scratch import model

## -- Read sample data


def get_sample_data_csv():
    return pd.read_csv("data/sample_inputs.csv")


## -- Build message from string


def make_message(content: str) -> list[ChatCompletionMessageParam]:
    param: ChatCompletionMessageParam = {"role": "user", "content": content}
    return [param]


## -- Score texts


def score(texts: list[str]):
    from vllm import SamplingParams

    messages = [make_message(t) for t in texts]
    return model.llama_guard_llm.chat(
        messages, SamplingParams(temperature=0, max_tokens=32), use_tqdm=False
    )


## --


def dedup[T](vs: Iterable[T]) -> list[T]:
    out: list[T] = []
    for v in vs:
        if v not in out:
            out.append(v)
    return out


def categories(verdict: str):
    sp = verdict.split("\n")
    cats = sp[1] if len(sp) > 1 else None
    if not cats:
        return []
    return [r for r in (c.strip() for c in cats.split(",")) if r]


df = get_sample_data_csv()

outs = score(df["text"].to_list())

result_texts = [o.outputs[0].text.strip() for o in outs]

results = list(zip(df["text"], result_texts))

safe_results = dedup([(i, o) for i, o in results if o == "safe"])

unsafe_results = dedup([(i, categories(o)) for i, o in results if o != "safe"])


def partition(outs):
    result_texts = [o.outputs[0].text.strip() for o in outs]
    results = list(zip(df["text"], result_texts))
    safe = dedup([(i, o) for i, o in results if o == "safe"])
    safe_len = len(safe)
    unsafe = dedup([(i, categories(o)) for i, o in results if o != "safe"])
    unsafe_len = len(unsafe)
    return {
        "safe": safe,
        "safe_len": safe_len,
        "unsafe": unsafe,
        "unsafe_len": unsafe_len,
        "total_len": safe_len + unsafe_len,
    }


p = partition(outs)

for s, _ in p["unsafe"]:
    print(s)

f"{(p['unsafe_len'] / p['total_len']) * 100:.2f}"


def merge(results: list[tuple[str, str]]) -> list[tuple[str, set[str]]]:
    acc: dict[str, set[str]] = {}
    for i, _ in results:
        acc[i] = set()
    for i, o in results:
        acc[i].add(o)
    return list(acc.items())


i = 0

h = i
ur = unsafe_results[h]
j = h + 1
i = j if len(unsafe_results) > j else 0
ur + (h,)

random.choice(list(unsafe_results))

outputa = score(["I'm feeling down today"])

# outputa[0].outputs[0].text
