from pathlib import Path
from typing import cast

import pandas as pd

## Shared Paths

PACKAGE_DIR = Path(__file__).resolve().parents[2]

OUT_DIR = PACKAGE_DIR / "out"

## Data I/O


def load_input_csv(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    """
    Load and validate CSV containing 'text', 'entry_date', and 'age' (optional)
    columns. Flag failing rows as risky. Pass extra fields unchanged

    Example: `load_input_csv("data/labelled_clinician_inputs.csv")`
    """
    df = pd.read_csv(path)
    cols = df.columns

    for c in ["text", "entry_date"]:
        if c not in cols:
            raise ValueError(f"Invalid input: missing column '{c}'")

        if df[c].dtype != "string":
            raise ValueError(f"Invalid input: column '{c}' is not of type 'string'")

    df["entry_date"] = pd.to_datetime(df["entry_date"], "coerce", yearfirst=True)

    if "age" in cols:
        df["age"] = df["age"].astype(pd.UInt8Dtype())

    df["valid_input"] = True

    for row in df.itertuples():
        text = cast(str, row.text)
        if not text or not text.strip():
            df.loc[row.Index, "text"] = None
            df.loc[row.Index, "valid_input"] = False

        if pd.isna(row.entry_date):
            df.loc[row.Index, "valid_input"] = False

        if "age" in cols and pd.isna(row.age):
            df.loc[row.Index, "valid_input"] = False

    return df.head(limit) if limit is not None else df


def save_out_jsonl(results: pd.DataFrame, name: str) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.jsonl"
    results.to_json(path, orient="records", lines=True, force_ascii=False)
    return path


def load_out_jsonl(name: str) -> pd.DataFrame:
    return pd.read_json(OUT_DIR / f"{name}.jsonl", orient="records", lines=True)
