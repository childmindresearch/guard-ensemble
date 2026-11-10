import pandas as pd

from scratch import lib, models, runner, shieldgemma


def reload_modules():
    import importlib

    importlib.reload(lib)
    importlib.reload(models)
    importlib.reload(runner)
    importlib.reload(shieldgemma)


# reload_modules() # <- Uncomment to reload scratchpad

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


score_results = run_llama_guard()


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
        key: lib.shield_gemma_partition(
            [(t, next(iter(r), None)) for t, r in zip(texts, policy_responses[key])]
        )
        for key in policy_responses
    }


policy_results = run_shieldgemma()

[
    (p, t)
    for p, rs in policy_results.items()
    for t, _ in rs["unsafe"]
]
