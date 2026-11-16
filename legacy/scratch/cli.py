import argparse
import json

from scratch import lib, models, runner, shieldgemma

## Llama Guard


def run_llama_guard(texts: list[str]):
    model = "llama_guard"

    runner.load(model)

    messages = lib.create_messages(texts)

    outs = runner.chat(model, messages, temperature=0, max_tokens=32)

    return lib.llama_guard_partition(texts, outs)


## Shield Gemma


def run_shieldgemma(texts: list[str]):
    runner.load("shieldgemma")

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


## Granite Guardian


def run_granite_guardian(texts: list[str]):
    runner.load("granite_guardian")

    outs = runner.chat(
        "granite_guardian",
        lib.create_messages(texts),
        temperature=0,
        max_tokens=32,
    )

    return lib.granite_guardian_partition(texts, outs)


## Qwen Guard


def run_qwen_guard(texts: list[str]):
    runner.load("qwen_guard")

    outs = runner.chat(
        "qwen_guard",
        lib.create_messages(texts),
        temperature=0,
        max_tokens=32,
    )

    return lib.qwen_guard_partition(texts, outs)


def main():
    parser = argparse.ArgumentParser(description="Guard Ensemble")
    parser.add_argument("model", choices=models.MODEL_KEYS, help="Guard model")
    parser.add_argument("text")

    args = parser.parse_args()

    model: models.MODEL_KEY = args.model
    texts = [args.text]

    result = None

    match model:
        case "llama_guard":
            result = run_llama_guard(texts)
        case "shieldgemma":
            result = run_shieldgemma(texts)
        case "granite_guardian":
            result = run_granite_guardian(texts)
        case "qwen_guard":
            result = run_qwen_guard(texts)

    out = json.dumps({"result": result})
    print(out)


if __name__ == "__main__":
    main()
