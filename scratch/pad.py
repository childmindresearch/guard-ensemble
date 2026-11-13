import json

import pandas as pd

from scratch import domains, lib, models, runner, shieldgemma


def reload_modules():
    import importlib

    importlib.reload(domains)
    importlib.reload(lib)
    importlib.reload(models)
    importlib.reload(runner)
    importlib.reload(shieldgemma)


# reload_modules()  # <- Uncomment to reload scratchpad

# runner.stop()

## Read sample data


def read_sample_csv():
    return pd.read_csv("data/sample_inputs.csv")


def read_labelled_csv():
    return pd.read_csv("labelled_data/clinician_inputs.csv")


## Programs --------------------------------------------------------------------


# Llamma Guard


def run_llama_guard():
    model = "llama_guard"

    df = read_sample_csv()
    texts: list[str] = df["text"].to_list()

    runner.load(model)

    messages = lib.create_messages(texts)

    outs = runner.chat(model, messages, temperature=0, max_tokens=32)

    return lib.llama_guard_partition(texts, outs)


llama_results = run_llama_guard()

llama_results["unsafe"]


#### Shield Gemma


def run_shieldgemma():
    runner.load("shieldgemma")

    df = read_sample_csv()
    texts: list[str] = df["text"].to_list()

    policy_responses = {
        key: runner.chat(
            "shieldgemma",
            lib.create_messages(texts),
            chat_template_kwargs={"guideline": policy},
            temperature=0,
            max_tokens=1,
        )
        for key, policy in shieldgemma.POLICIES.items()
    }

    return {
        key: lib.shieldgemma_partition(texts, policy_responses[key])
        for key in policy_responses
    }


gemma_results = run_shieldgemma()
#
# [(p, t) for p, rs in gemma_results.items() for t, _ in rs["unsafe"]]


#### Granite Guardian


def run_granite_guardian():
    runner.load("granite_guardian")

    df = read_sample_csv()
    texts: list[str] = df["text"].to_list()

    outs = runner.chat(
        "granite_guardian",
        lib.create_messages(texts),
        temperature=0,
        max_tokens=32,
    )

    return lib.granite_guardian_partition(texts, outs)


# granite_results = run_granite_guardian()


#### Qwen Guard


def run_qwen_guard():
    runner.load("qwen_guard")

    df = read_sample_csv()
    texts: list[str] = df["text"].to_list()

    outs = runner.chat(
        "qwen_guard",
        lib.create_messages(texts),
        temperature=0,
        max_tokens=32,
    )

    return lib.qwen_guard_partition(texts, outs)


# qwen_results = run_qwen_guard()


#### Domains: gate -> scoped JSON -> adjudicate (scratch/domains.py)

from collections.abc import Hashable
from typing import Any, cast

reload_modules()

runner.stop()

data = domains.read_sample_csv(path="labelled_data/clinician_inputs.csv")

domains.MODEL = "gemma3_4b"

res = domains.run(data)

domains.save(res, "key_domains_gemma3_4b_labelled_0001")

# ---


def as_df(v: Any) -> pd.DataFrame:
    return cast(pd.DataFrame, v)


def select_results(domains_out: pd.DataFrame) -> list[dict[Hashable, Any]]:
    selected = as_df(domains_out[["text", "domain_results"]])
    return selected.to_dict("records")


select_results(res)


false_ai_flags_ = {
    (text, risks, json.dumps(drs))
    for text, risks, drs in [
        (
            r["text"],
            frozenset(
                [
                    (k, v["tier"])
                    for k, v in dict.items(r["domain_results"])
                    if v["tier"] != "green"
                ]
            ),
            {**r["domain_results"]},
        )
        for r in res.to_dict("records")
    ]
    if "04_ai_related_safety_risk" in {k for k, v in risks}
    and "My chatbot is the only one" not in text
}

false_ai_flags = [
    {"text": text, "risks": list(risks), "domain_results": json.loads(drs)}
    for text, risks, drs in false_ai_flags_
]

print(json.dumps({(f["text"], f["risks"]) for f in false_ai_flags}))


def serialize_sets(obj):
    if isinstance(obj, set):
        return [*obj]
    return obj


with open("out/false_ai_flags.json", "w") as f:
    json.dump(false_ai_flags, f, indent=2, default=tuple, ensure_ascii=False)


# ---

res.to_dict("records")

rs = pd.DataFrame(
    [
        {
            "text": "If I'm not here next week, it's not anyone's fault.",
            "age": "17",
            "entry_date": "2025-01-27",
        }
    ]
)
res2 = domains.run(rs, runs=1)

[
    {
        "text": r["text"],
        "concerns": [
            (k, v["tier"], v)
            for k, v in dict.items(r["domain_results"])
            if v["tier"] != "green"
        ],
    }
    for r in cast(pd.DataFrame, res2[["text", "domain_results"]]).to_dict("records")
]

# Re-adjudicate a saved JSON run without regenerating:
# res = domains.adjudicate(domains.load("gated_qwen3_4b_head100_scoped"))


#### Kev Experimentation

import importlib
from concurrent.futures import ThreadPoolExecutor

import kev_client

importlib.reload(kev_client)

keq = kev_client.load_kev_questions()

result = kev_client.run_kev_domains(
    "I don't want to be here anymore", 19, "2025-01-11", keq
)

dict.keys(result["response"])

{key: result["response"][key] for key in ["usage", "latency_ms"]}


def run_kev_domains_many(rows, keq, concurrency=16):
    def one(row):
        return kev_client.run_kev_domains(
            row["text"], int(row["age"]), str(row["entry_date"]), keq
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, rows))


data = read_sample_csv()
rows = data.head(100).to_dict("records")

out = run_kev_domains_many(rows, keq)
