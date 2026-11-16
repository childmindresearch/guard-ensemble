import pandas as pd

from scratch import domains, lib, models, runner, shieldgemma


def reload_modules():
    import importlib

    importlib.reload(domains)
    importlib.reload(lib)
    importlib.reload(models)
    importlib.reload(runner)
    importlib.reload(shieldgemma)


reload_modules()  # <- Uncomment to reload scratchpad

# runner.stop()

## Read sample data


def read_sample_csv():
    return pd.read_csv("data/sample_inputs.csv")


def read_labelled_csv():
    return pd.read_csv("data/labelled_clinician_inputs.csv")


# ------------------------------------------------------------------------------
# Guard Runs
# ------------------------------------------------------------------------------


### Llamma Guard


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


# ------------------------------------------------------------------------------
# Domain Runs
# ------------------------------------------------------------------------------


#### Domains: gate -> scoped JSON (scratch/domains.py)

runner.stop()

reload_modules()

labelled_data = lib.load_input_csv("data/labelled_clinician_inputs.csv")
llm_mock_data = lib.load_input_csv("data/sample_inputs.csv")

res = domains.run(labelled_data)
# res = domains.run(llm_mock_data)

lib.save_out_jsonl(res, "key_domains_qwen3_4b_2507_labelled_0005")


# ------------------------------------------------------------------------------
# Other Experiments
# ------------------------------------------------------------------------------


#### Kev Experimentation

import importlib
from concurrent.futures import ThreadPoolExecutor

import kev_client

importlib.reload(kev_client)

keq = kev_client.load_kev_questions()

result = kev_client.run_kev_domains(
    "I don't want to be here anymore", 19, "2025-01-11", keq
)

dict.keys(result["domains"])


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
