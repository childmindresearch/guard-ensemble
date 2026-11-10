import pandas as pd

from scratch import lib, models, runner, shieldgemma


def reload_modules():
    import importlib

    importlib.reload(lib)

    importlib.reload(models)
    importlib.reload(runner)
    importlib.reload(shieldgemma)


# reload_modules()  # <- Uncomment to reload scratchpad

## Read sample data


def read_sample_csv():
    return pd.read_csv("data/sample_inputs.csv")


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


# llama_results = run_llama_guard()
#
# llama_results["unsafe"]


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


# gemma_results = run_shieldgemma()
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
