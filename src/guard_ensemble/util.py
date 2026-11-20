import types
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union, get_args, get_origin

import polars as pl
from pydantic import BaseModel, ConfigDict, ValidationError

## Shared Paths

PACKAGE_DIR = Path(__file__).resolve().parents[2]

OUT_DIR = PACKAGE_DIR / "out"

## DataFrame Schemas


def to_dtype(tp: Any) -> Any:
    origin, args = get_origin(tp), get_args(tp)

    if origin is Annotated:
        return to_dtype(args[0])
    if origin in (Union, types.UnionType):
        inner = [a for a in args if a is not type(None)]
        if len(inner) != 1:
            raise TypeError(f"unsupported union: {tp}")
        return to_dtype(inner[0])
    if origin in (list, set, frozenset):
        return pl.List(to_dtype(args[0]))
    if origin is Literal:
        return to_dtype(type(args[0]))
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return pl.Struct({
            n: to_dtype(f.annotation) for n, f in tp.model_fields.items()
        })

    return pl.DataType.from_python(tp)


# def polars_schema(model: type[BaseModel]) -> dict[str, Any]:
#     out: dict[str, Any] = {}
#     for name, field in model.model_fields.items():
#         try:
#             out[name] = to_dtype(field.annotation)
#         except TypeError as e:
#             raise TypeError(
#                 f"{model.__name__}.{name} ({field.annotation!r}): {e}"
#             ) from e
#     return out


def polars_schema(model: type[BaseModel]) -> dict[str, Any]:
    return {n: to_dtype(f.annotation) for n, f in model.model_fields.items()}


## Data I/O


class CsvEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    entry_date: datetime
    age: Optional[int] = None  # noqa: UP045


type RowResult[T] = (
    tuple[Literal["valid"], T] | tuple[Literal["invalid"], dict[str, Any]]
)


def row_valid[T: BaseModel](model: type[T], row: dict[str, Any]) -> RowResult[T]:
    try:
        return ("valid", model.model_validate(row))
    except ValidationError:
        return ("invalid", row)


def validate_frame[T: BaseModel](
    model: type[T], df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    valid_entries: list[T] = []
    invalid_entries: list[dict[str, Any]] = []

    for row in df.iter_rows(named=True):
        match row_valid(model, row):
            case ("valid", entry):
                valid_entries.append(entry)
            case ("invalid", raw):
                invalid_entries.append(raw)

    model_schema = polars_schema(model)
    schema = {k: model_schema.get(k, dtype) for k, dtype in df.schema.items()}
    schema |= {k: t for k, t in model_schema.items() if k not in schema}

    valid_df = pl.DataFrame([e.model_dump() for e in valid_entries], schema=schema)
    invalid_df = pl.from_dicts(invalid_entries, schema=df.schema)
    return (valid_df, invalid_df)


def load_input_csv(
    path: str | Path, limit: int | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load and validate CSV containing 'text', 'entry_date', and 'age' (optional)
    columns. Flag failing rows as risky. Pass extra fields unchanged

    Example: `load_input_csv("data/labelled_clinician_inputs.csv")`
    """
    csv_df = pl.read_csv(path, infer_schema_length=None)
    _limit = limit if limit is not None and limit > 0 else None
    return validate_frame(CsvEntry, csv_df.head(_limit) if _limit else csv_df)


def save_out_jsonl(results: pl.DataFrame, name: str) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.jsonl"
    results.write_ndjson(path)
    return path


def load_out_jsonl(name: str) -> pl.DataFrame:
    return pl.read_ndjson(OUT_DIR / f"{name}.jsonl")
