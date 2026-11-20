import argparse
import json
from typing import Literal

import polars as pl

import guard_ensemble.guards.model as guard

GUARD_KEY = Literal["llama_guard", "shieldgemma", "granite_guardian", "qwen_guard"]

GUARD_KEYS: frozenset[GUARD_KEY] = frozenset({
    "llama_guard",
    "shieldgemma",
    "granite_guardian",
    "qwen_guard",
})


def main():
    parser = argparse.ArgumentParser(description="Guard Ensemble")
    parser.add_argument("model", choices=GUARD_KEYS, help="Guard model")
    parser.add_argument("csv", help="CSV file containing 'text' column")

    args = parser.parse_args()

    model: GUARD_KEY = args.model
    csv_file: str = args.csv

    df = pl.read_csv(csv_file, schema={"text": pl.String})

    if "text" not in df.columns or df["text"].dtype != "string":
        raise ValueError("Invalid CSV file: column 'text' is not of type 'string")

    texts = df["text"].to_list()

    result = None

    match model:
        case "llama_guard":
            result = guard.run_llama_guard(texts)
        case "shieldgemma":
            result = guard.run_shieldgemma(texts)
        case "granite_guardian":
            result = guard.run_granite_guardian(texts)
        case "qwen_guard":
            result = guard.run_qwen_guard(texts)

    for row in result:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
