from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from vllm.renderers.chat_utils import ChatCompletionMessageParam

from scratch import runner
from scratch.shieldgemma import SHIELDGEMMA_POLICIES, shieldgemma_build

# import importlib
#
# importlib.reload(runner)

# runner.load("shieldgemma")
runner.load("llama_guard")

## -- Read sample data


def read_sample_csv():
    return pd.read_csv("data/sample_inputs.csv")


## -- Build message from string


def make_message(content: str) -> list["ChatCompletionMessageParam"]:
    param: ChatCompletionMessageParam = {"role": "user", "content": content}
    return [param]


## -- Score texts


def score(texts: list[str]):
    messages = [make_message(t) for t in texts]
    return runner.chat("llama_guard", messages, temperature=0, max_tokens=32)


## -- Deduplicate results (test data only)


def dedupe[T: Hashable](vs: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(vs))


## -- Extract categories from score


def categories(verdict: str):
    sp = verdict.split("\n")
    catstr = sp[1] if len(sp) > 1 else None
    if not catstr:
        return []
    cats = {r for r in (c.strip() for c in catstr.split(",")) if r}
    return frozenset(cats)


def partition(ins, outs):
    out_texts = [o.outputs[0].text.strip() for o in outs]
    results = list(zip(ins, out_texts))
    safe = dedupe([(i, o) for i, o in results if o == "safe"])
    unsafe = dedupe([(i, categories(o)) for i, o in results if o != "safe"])
    safe_len = len(safe)
    unsafe_len = len(unsafe)
    return {
        "safe": safe,
        "safe_len": safe_len,
        "unsafe": unsafe,
        "unsafe_len": unsafe_len,
        "total_len": safe_len + unsafe_len,
    }


## -- Execute program

df = read_sample_csv()

texts: list[str] = df["text"].to_list()

#### Llamma Guard

outs = score(texts)

p = partition(df["text"], outs)

for s, _ in p["unsafe"]:
    print(s)

f"{(p['unsafe_len'] / p['total_len']) * 100:.2f}"


outputa = score(["I'm feeling down today"])

outputa[0][0]

#### Shield Gemma

# policy_responses = {
#     key: runner.generate(
#         "shieldgemma",
#         [shieldgemma_build(t, key) for t in texts],
#         temperature=0,
#         max_tokens=5,
#     )
#     for key in SHIELDGEMMA_POLICIES
# }
#
#
# def partition_sg(
#     results: list[tuple[str, str | None]],
# ) -> dict[Literal["safe", "unsafe"], list[tuple[str, str | None]]]:
#     return {
#         "safe": dedupe([(t, r) for t, r in results if r and r.strip() == "No"]),
#         "unsafe": dedupe([(t, r) for t, r in results if not r or r.strip() != "No"]),
#     }
#
#
# policy_results = {
#     key: partition_sg(
#         [(t, next(iter(r), None)) for t, r in zip(texts, policy_responses[key])]
#     )
#     for key in policy_responses
# }
#

#### Compare defined policies with default

# # pyrefly: ignore[missing-import]
# from huggingface_hub import snapshot_download
#
# # pyrefly: ignore[missing-import]
# from transformers import AutoTokenizer
#
# tok = AutoTokenizer.from_pretrained(
#     snapshot_download(
#         "google/shieldgemma-9b", allow_patterns=["*.json", "tokenizer.model"]
#     )
# )
# t, k = texts[0], "dangerous_content"
#
# for policy in dict.keys(SHIELDGEMMA_POLICIES):
#     print(
#         tok.apply_chat_template(
#             [{"role": "user", "content": texts[0]}],
#             guideline=SHIELDGEMMA_POLICIES[policy],
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#     )
#     print(shieldgemma_build(t, policy))
