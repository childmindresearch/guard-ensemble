# ruff: noqa: UP045
import json
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import boto3
import polars as pl
from pydantic import BaseModel, ConfigDict, ValidationError

from guard_ensemble import util

if TYPE_CHECKING:
    from types_boto3_s3.type_defs import ObjectTypeDef

MSM_RESEARCH_ID = 5

MIRROR_DATA_ACCESS_SECRET_ARN = os.getenv("MIRROR_DATA_ACCESS_SECRET_ARN")
MIRROR_DATA_BUCKET = os.getenv("MIRROR_DATA_BUCKET")

DEBUG = False


def config_debug(debug=False):
    global DEBUG
    DEBUG = debug


class MirrorEntryResearch(BaseModel):
    model_config = ConfigDict(extra="allow")

    research_id: int
    research_uid: str
    research_name: str
    created_on: datetime
    external_id: Optional[str] = None
    platform_external_id: Optional[str] = None


class MirrorEntryUser(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    is_opted_in: Optional[bool] = None
    pronouns: Optional[str] = None
    region: Optional[str] = None
    zipcode: Optional[str] = None
    user_research: Optional[list[MirrorEntryResearch]] = []


class MirrorEntryMoodCheck(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    created_on: datetime
    mood_type: Optional[str] = None
    pleasant_scale: Optional[float] = None
    energy_scale: Optional[float] = None
    description: Optional[str] = None


class MirrorEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    created_on: datetime
    text: str
    entry_type: str
    user: MirrorEntryUser
    mood_check: Optional[MirrorEntryMoodCheck] = None
    remix: Optional[list] = None
    transcription: Optional[str] = None
    summary: Optional[str] = None
    summary_en: Optional[str] = None
    prompt_en: Optional[str] = None
    prompt_lang: Optional[str] = None
    hidden: Optional[bool] = None
    summary_feedback: Optional[str] = None
    video_entry_type: Optional[str] = None
    presigned_url: Optional[str] = None
    summary_status: Optional[str] = None
    transcription_status: Optional[str] = None
    num_words: Optional[int] = None
    num_images_attached: Optional[int] = None
    num_drawings_attached: Optional[int] = None
    length_audio_or_video: Optional[float] = None
    vibe_check_after: Optional[float] = None
    was_reviewed: Optional[bool] = None
    remix_status: Optional[str] = None
    high_risk: Optional[bool] = None
    concerning: Optional[bool] = None
    recorded_on: Optional[datetime] = None
    media_location: Optional[str] = None


KEY_RE = re.compile(
    r"sub_(?P<user_id>\d+)"
    r"-entry_(?P<entry_id>\d+)"
    r"-remix_(?P<remix>\w+)"
    r"-timestamp_(?P<pulled_at>\d+)"
    r"-(?P<entry_type>[A-Za-z]+)"
    r"\.(?P<extension>\w+)$",
    re.ASCII,
)


class MirrorKeyLabels(BaseModel):
    key: str
    user_id: int
    entry_id: int
    remix: str
    pulled_at: datetime
    entry_type: str
    extension: str


class MirrorRecord(BaseModel):
    key_labels: MirrorKeyLabels
    entry: MirrorEntry


def get_ro_access_secret():
    access_key_id_env = os.getenv("MIRROR_DATA_ACCESS_KEY_ID")
    secret_access_key_env = os.getenv("MIRROR_DATA_SECRET_ACCESS_KEY")

    if access_key_id_env and secret_access_key_env:
        return (access_key_id_env, secret_access_key_env)

    if not MIRROR_DATA_ACCESS_SECRET_ARN:
        raise RuntimeError(
            "Failed to access Mirror data: MIRROR_DATA_ACCESS_SECRET_ARN "
            "environment variable is unset"
        )

    secrets_client = boto3.client("secretsmanager")
    secret_resp = secrets_client.get_secret_value(
        SecretId=MIRROR_DATA_ACCESS_SECRET_ARN
    )
    creds = json.loads(secret_resp["SecretString"])
    access_key_id = creds["msm-mirror-data-access-key"]
    secret_access_key = creds["msm-mirror-data-secret-access-key"]
    return (access_key_id, secret_access_key)


def create_s3_client():
    access_key_id, secret_access_key = get_ro_access_secret()
    return boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )


def datetime_from_str_ts(ts: str):
    return datetime.fromtimestamp(int(ts) / 1000)


def convert_pulled_at(key_labels: list[dict]):
    return [
        {**label, "pulled_at": datetime_from_str_ts(label["pulled_at"])}
        for label in key_labels
    ]


def re_dict(pattern: re.Pattern[str], string: str):
    result = pattern.search(string)
    return result.groupdict() if result else None


def parse_key_labels(obj_list: list[ObjectTypeDef]) -> list[MirrorKeyLabels]:
    return [
        MirrorKeyLabels.model_validate({**d, "key": o["Key"]})
        for o in obj_list
        if (d := re_dict(KEY_RE, o["Key"]))
    ]


def entry_in_study(entry: MirrorEntry):
    return MSM_RESEARCH_ID in [r.research_id for r in entry.user.user_research or []]


def fetch_records(
    key_labels: list[MirrorKeyLabels],
    bucket: str,
    *,
    curr_len: int = 0,
    limit: int | None = None,
) -> list[MirrorRecord]:
    s3_client = create_s3_client()

    records = []
    for labels in key_labels:
        if labels.entry_type not in {"Written", "Guided"}:
            if DEBUG:
                print("x", end="", flush=True)
            continue

        object_key = labels.key
        resp = s3_client.get_object(Bucket=bucket, Key=object_key)

        body = resp["Body"].read().decode("utf-8")
        data = json.loads(body)

        entry: MirrorEntry

        try:
            entry = MirrorEntry(**data)
        except ValidationError as e:
            if DEBUG:
                print(f"Validation failed: {e.errors()}")
            continue

        in_study = entry_in_study(entry)
        has_text = data.get("entry_type") in {"Written", "Guided"}

        if in_study and has_text:
            if DEBUG:
                print(".", end="", flush=True)
            records.append(MirrorRecord(key_labels=labels, entry=entry))

            if limit and (curr_len + len(records)) >= limit:
                break
        else:
            if DEBUG:
                print("x", end="", flush=True)

    return records


def pull(*, limit: int | None = 500, since: datetime | None = None):
    if not MIRROR_DATA_BUCKET:
        raise ValueError(
            "Failed to pull Mirror data: MIRROR_DATA_BUCKET environment "
            "variable is unset"
        )
    s3_client = create_s3_client()

    cont_token = None
    records = []

    while True:
        kwargs = {"Bucket": MIRROR_DATA_BUCKET}
        if cont_token:
            kwargs["ContinuationToken"] = cont_token

        list_resp = s3_client.list_objects_v2(**kwargs)

        obj_list = list_resp["Contents"]

        if not obj_list:
            break

        key_labels = parse_key_labels(list_resp["Contents"])
        key_labels_since = (
            [t for t in key_labels if t.pulled_at > since] if since else key_labels
        )

        records.extend(
            fetch_records(
                key_labels_since, MIRROR_DATA_BUCKET, curr_len=len(records), limit=limit
            )
        )

        if limit and len(records) > limit:
            break

        if list_resp.get("IsTruncated"):
            cont_token = list_resp["NextContinuationToken"]
        else:
            break

    if DEBUG and len(records) > 0:
        print("pull complete")

    return records


def inner_struct_fields(col: str, df: pl.DataFrame):
    dtype = df.schema[col]
    assert isinstance(dtype, pl.List)
    inner = dtype.inner
    assert isinstance(inner, pl.Struct)
    return inner.fields


def research_column(name: str):
    return (
        pl
        .col("research")
        .list.eval(pl.element().struct.field(name))
        .alias(f"research_{name}")
    )


def frame_records(records: list[MirrorRecord]):
    input = [rec.model_dump() for rec in records]
    research = pl.col("user").struct.field("user_research").fill_null([])

    df = (
        pl
        .DataFrame(
            input, schema=util.polars_schema(MirrorRecord), infer_schema_length=None
        )
        .unnest("entry")
        .with_columns(research=research)
    )

    fields = inner_struct_fields("research", df)

    return (
        df
        .with_columns(research_column(f.name) for f in fields)
        .drop("research")
        .with_columns(
            pl.col("created_on").alias("entry_date"),
            pl.col("user").name.prefix_fields("user_"),
            pl.col("user").alias("_user"),
        )
        .unnest("user")
        .drop("user_user_research")
        .rename({
            "research_research_id": "research_id",
            "research_research_uid": "research_uid",
        })
        .select(pl.exclude("key_labels"), "key_labels")
        .rename({"key_labels": "_key_labels"})
    )


def select_msm_frame(records_frame: pl.DataFrame):
    return records_frame.with_columns(
        pl
        .col("research_platform_external_id")
        .list.filter(pl.element().fill_null("").str.contains(r"^z\d+$"))
        .list.first()
    ).filter(
        pl.col("research_platform_external_id").str.strip_prefix("z").cast(pl.Int64)
        > 100010
    )
