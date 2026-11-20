import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, assert_never
from uuid import UUID, uuid4

import polars as pl

from guard_ensemble.guards import shieldgemma
from guard_ensemble.llm import runner
from guard_ensemble.llm.lib import create_messages
from guard_ensemble.util import PACKAGE_DIR, save_out_jsonl

type Outs = list[list[str]]

OUT = PACKAGE_DIR / "out"


## Validation


def validate_df(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # TODO: Throw error and/or log warning on schema mismatch
    if df.schema.get("text") != pl.String:
        return df.clone().clear(), df
    cond = (pl.col("text").is_not_null()) & (pl.col("text").str.strip_chars() != "")
    valid_df = df.filter(cond)
    invalid_df = df.filter(~cond)
    return valid_df, invalid_df


## Common


def first_out(o: list[str]):
    return o[0].strip() if len(o) > 0 else None


def eq_no(s: str | None):
    return s.strip().lower().startswith("no") if s else False


## Llama Guard


def llama_guard_categories(verdict: str) -> list[str]:
    sp = verdict.split("\n")
    catstr = sp[1] if len(sp) > 1 else None
    if not catstr:
        return []
    cats = {r for r in (c.strip() for c in catstr.split(",")) if r}
    return list(set(cats))


def run_llama_guard(df: pl.DataFrame, validate: bool = True):
    valid_df, invalid_df = validate_df(df) if validate else (df, df.clone().clear())

    if valid_df.is_empty():
        return valid_df, invalid_df

    model = "llama_guard"
    runner.load(model)

    texts = valid_df["text"].to_list()
    messages = create_messages(texts)
    outs = runner.chat(model, messages, temperature=0, max_tokens=32)
    out_texts = [first_out(o) for o in outs]
    text_cats = [
        llama_guard_categories(o) if o and o != "safe" else [] for o in out_texts
    ]
    flagged = [len(cats) > 0 or out is None for out, cats in zip(out_texts, text_cats)]

    out_df = valid_df.with_columns(
        pl.Series(out_texts).alias("llama_guard_out"),
        pl.Series(text_cats).alias("llama_guard_unsafe_categories"),
        pl.Series(flagged).alias("llama_guard_flagged"),
        pl.Series([True for _ in texts]).alias("llama_guard_success"),
    )

    return out_df, invalid_df


## Shield Gemma


SHIELDGEMMA_RESULT_SCHEMA = {
    f"shieldgemma_{key}_out": pl.String for key in list(shieldgemma.POLICIES)
} | {f"shieldgemma_{key}_flagged": pl.Boolean for key in list(shieldgemma.POLICIES)}


def run_shieldgemma(df: pl.DataFrame, validate: bool = True):
    valid_df, invalid_df = validate_df(df) if validate else (df, df.clone().clear())

    if valid_df.is_empty():
        return valid_df, invalid_df

    model = "shieldgemma"
    runner.load(model)

    texts = valid_df["text"].to_list()

    messages = create_messages(texts)

    results: list[dict[str, str | bool | None]] = [{} for text in texts]

    for key, policy in shieldgemma.POLICIES.items():
        outs = runner.chat(
            model,
            messages,
            chat_template_kwargs={"guideline": policy},
            temperature=0,
            max_tokens=1,
        )
        out_texts = [first_out(o) for o in outs]
        out_key = f"shieldgemma_{key}_out"
        flag_key = f"shieldgemma_{key}_flagged"

        for i in range(len(results)):
            results[i][out_key] = out_texts[i]
            results[i][flag_key] = not eq_no(out_texts[i])

    res_df = pl.from_dicts(results, schema=SHIELDGEMMA_RESULT_SCHEMA)
    out_df = valid_df.hstack(res_df).with_columns(
        pl.Series([True for _ in texts]).alias("shieldgemma_success")
    )

    return out_df, invalid_df


## Granite Guardian


GRANITE_GUARDIAN_SCORE_RE = r"<score>(.*?)</score>"


def granite_guardian_score(out_text: str) -> str | None:
    matches: list[str] = re.findall(GRANITE_GUARDIAN_SCORE_RE, out_text, re.DOTALL)

    if len(matches) == 0:
        return None

    return matches[-1].strip().lower()


def run_granite_guardian(df: pl.DataFrame, validate: bool = True):
    valid_df, invalid_df = validate_df(df) if validate else (df, df.clone().clear())

    if valid_df.is_empty():
        return valid_df, invalid_df

    model = "granite_guardian"
    runner.load(model)

    texts = valid_df["text"].to_list()
    messages = create_messages(texts)
    outs = runner.chat(model, messages, temperature=0, max_tokens=32)
    out_texts = [first_out(o) for o in outs]
    scores = [granite_guardian_score(o) if o else None for o in out_texts]
    flagged = [not eq_no(score) for score in scores]

    out_df = valid_df.with_columns(
        pl.Series(out_texts).alias("granite_guardian_out"),
        pl.Series(flagged).alias("granite_guardian_flagged"),
        pl.Series([True for _ in texts]).alias("granite_guardian_success"),
    )

    return out_df, invalid_df


## Qwen Guard


def qwen_guard_labels(out_text: str) -> tuple[str | None, str | None]:
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
    safe_label_match = re.search(safe_pattern, out_text)
    label: str | None = safe_label_match.group(1) if safe_label_match else None
    categories: list[str] = re.findall(category_pattern, out_text)
    return label, ",".join(categories) or None


def eq_unsafe(s: str | None) -> bool:
    return s is None or s.lower() == "unsafe"


def run_qwen_guard(df: pl.DataFrame, validate: bool = True):
    valid_df, invalid_df = validate_df(df) if validate else (df, df.clone().clear())

    if valid_df.is_empty():
        return valid_df, invalid_df

    model = "qwen_guard"
    runner.load(model)

    texts = valid_df["text"].to_list()
    messages = create_messages(texts)
    outs = runner.chat(model, messages, temperature=0, max_tokens=32)
    out_texts = [first_out(o) for o in outs]
    text_labels = [qwen_guard_labels(o) if o else (None, None) for o in out_texts]
    labels, cats = map(list, zip(*text_labels))
    flagged = [eq_unsafe(label) for label in labels]
    out_df = valid_df.with_columns(
        pl.Series(labels).alias("qwen_guard_label"),
        pl.Series(cats).alias("qwen_guard_categories"),
        pl.Series(flagged).alias("qwen_guard_flagged"),
        pl.Series([True for _ in texts]).alias("qwen_guard_success"),
    )

    return out_df, invalid_df


## Run Safe

type GuardModel = Literal[
    "llama_guard", "shieldgemma", "granite_guardian", "qwen_guard"
]


class ModelDispatch(Protocol):
    def __call__(
        self, df: pl.DataFrame, *, validate: bool = True
    ) -> tuple[pl.DataFrame, pl.DataFrame]: ...


@dataclass(frozen=True)
class ModelSuccess:
    success = True
    tagged_df: pl.DataFrame
    invalid_df: pl.DataFrame


@dataclass(frozen=True)
class ModelFailure:
    success = False
    tagged_df: pl.DataFrame


type ModelResult = ModelSuccess | ModelFailure


def guard_runner(model: GuardModel) -> ModelDispatch:
    match model:
        case "llama_guard":
            return run_llama_guard
        case "shieldgemma":
            return run_shieldgemma
        case "granite_guardian":
            return run_granite_guardian
        case "qwen_guard":
            return run_qwen_guard
        case _:
            assert_never(model)


def run_safe(guard: GuardModel, df: pl.DataFrame, validate: bool = True) -> ModelResult:
    run = guard_runner(guard)
    try:
        tagged_df, invalid_df = run(df, validate=validate)
        return ModelSuccess(tagged_df, invalid_df)
    except Exception:  # noqa: BLE001
        tagged_df = df.with_columns(pl.lit(False).alias(f"{guard}_success"))
        return ModelFailure(tagged_df)


## Run Suite


def run_safe_out(
    guard: GuardModel,
    df: pl.DataFrame,
    id: UUID,
    timestamp: str,
    out_dir: Path | None = None,
    validate: bool = True,
) -> ModelResult:
    guard_out = run_safe(guard, df, validate=validate)
    out_status = "success" if guard_out.success else "failure"
    out_stem = f"{timestamp}_ensemble-{id}_{guard}_{out_status}_out"
    save_out_jsonl(guard_out.tagged_df, out_stem, out_dir)
    return guard_out


def run_all(df: pl.DataFrame, out_dir: Path | None = None) -> ModelResult:
    valid_df, invalid_df = validate_df(df)

    id = uuid4()
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    invalid_stem = f"{ts}_ensemble-{id}_pipeline_invalid.json"

    if valid_df.is_empty() and not invalid_df.is_empty():
        tagged_df = invalid_df.with_columns(data_valid=pl.lit(False))
        save_out_jsonl(tagged_df, invalid_stem, out_dir)

        if valid_df.is_empty():
            return ModelFailure(tagged_df)

    def _run(guard: GuardModel):
        return run_safe_out(guard, valid_df, id, ts, out_dir, validate=False)

    llama_guard_result = _run("llama_guard")
    shieldgemma_result = _run("shieldgemma")
    granite_guardian_result = _run("granite_guardian")
    qwen_guard_result = _run("qwen_guard")

    def drop_cols(res: ModelResult):
        return res.tagged_df.drop(valid_df.columns, strict=False)

    tagged_df = (
        valid_df
        .hstack(drop_cols(llama_guard_result))
        .hstack(drop_cols(shieldgemma_result))
        .hstack(drop_cols(granite_guardian_result))
        .hstack(drop_cols(qwen_guard_result))
    )

    return ModelSuccess(tagged_df, invalid_df)
