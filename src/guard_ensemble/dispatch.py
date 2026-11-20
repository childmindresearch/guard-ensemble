import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, assert_never
from uuid import UUID, uuid4

import polars as pl

from guard_ensemble.domains import model as domain
from guard_ensemble.guards import model as guard
from guard_ensemble.llm.catalog import Model
from guard_ensemble.util import save_out_jsonl


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


def model_dispatch(model: Model) -> ModelDispatch:
    match model:
        case "llama_guard":
            return guard.run_llama_guard
        case "shieldgemma":
            return guard.run_shieldgemma
        case "granite_guardian":
            return guard.run_granite_guardian
        case "qwen_guard":
            return guard.run_qwen_guard
        case "qwen3_4b_2507":
            return domain.run
        case _:
            assert_never(model)


def run_safe(model: Model, df: pl.DataFrame, validate: bool = True) -> ModelResult:
    run = model_dispatch(model)
    try:
        tagged_df, invalid_df = run(df, validate=validate)
        return ModelSuccess(tagged_df, invalid_df)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        tagged_df = df.with_columns(pl.lit(False).alias(f"{model}_success"))
        return ModelFailure(tagged_df)


## Run Suite


def run_safe_out(
    guard: Model,
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
    valid_df, invalid_df = guard.validate_df(df)

    id = uuid4()
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    invalid_stem = f"{ts}_ensemble-{id}_pipeline_invalid.json"

    if not invalid_df.is_empty():
        save_out_jsonl(invalid_df.with_columns(data_valid=pl.lit(False)), invalid_stem, out_dir)
    if valid_df.is_empty():
        return ModelFailure(invalid_df)

    def _run(model: Model):
        return run_safe_out(model, valid_df, id, ts, out_dir, validate=False)

    llama_guard_result = _run("llama_guard")
    shieldgemma_result = _run("shieldgemma")
    granite_guardian_result = _run("granite_guardian")
    qwen_guard_result = _run("qwen_guard")
    domain_result = _run("qwen3_4b_2507")

    def drop_cols(res: ModelResult):
        return res.tagged_df.drop(valid_df.columns, strict=False)

    combined_df = (
        valid_df
        .hstack(drop_cols(llama_guard_result))
        .hstack(drop_cols(shieldgemma_result))
        .hstack(drop_cols(granite_guardian_result))
        .hstack(drop_cols(qwen_guard_result))
        .hstack(drop_cols(domain_result))
    )

    combined_stem = f"{ts}_ensemble-{id}_pipeline_success.json"
    save_out_jsonl(combined_df, combined_stem, out_dir)

    return ModelSuccess(combined_df, invalid_df)
