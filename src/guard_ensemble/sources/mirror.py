import datetime
import json
import os
import re

import boto3
from types_boto3_s3.type_defs import ObjectTypeDef

MIRROR_DATA_ACCESS_SECRET_ARN = os.getenv("MIRROR_DATA_ACCESS_SECRET_ARN")
MIRROR_DATA_BUCKET = os.getenv("MIRROR_DATA_BUCKET")

KEY_RE = re.compile(
    r"sub_(?P<user_id>\d+)"
    r"-entry_(?P<entry_id>\d+)"
    r"-remix_(?P<remix>\w+)"
    r"-timestamp_(?P<pulled_at>\d+)"
    r"-(?P<entry_type>[A-Za-z]+)"
    r"\.(?P<extension>\w+)$",
    re.ASCII,
)


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


s3_client = create_s3_client()


def datetime_from_str_ts(ts: str):
    return datetime.datetime.fromtimestamp(int(ts) / 1000)


def convert_pulled_at(parsed_keys: list[dict]):
    return [
        {**g, "pulled_at": datetime_from_str_ts(g["pulled_at"])} for g in parsed_keys
    ]


def parse_object_keys(objs: list[ObjectTypeDef]):
    parsed = [
        {**m.groupdict(), "key": o["Key"]}
        for o in objs
        if (m := KEY_RE.search(o["Key"]))
    ]
    return convert_pulled_at(parsed)


def get_objects(parsed_keys: list[dict], bucket: str):
    parsed = []
    for p in parsed_keys:
        resp = s3_client.get_object(Bucket=bucket, Key=p["key"])
        data = json.loads(resp["Body"].read().decode("utf-8"))
        parsed.append(data)
    return parsed


def pull(limit: int | None = 500):
    if not MIRROR_DATA_BUCKET:
        raise ValueError(
            "Failed to pull Mirror data: MIRROR_DATA_BUCKET environment "
            "variable is unset"
        )
    s3_client = create_s3_client()

    cont_token = None
    entries = []

    while True:
        kwargs = {"Bucket": MIRROR_DATA_BUCKET}
        if cont_token:
            kwargs["ContinuationToken"] = cont_token

        list_resp = s3_client.list_objects_v2(**kwargs)
        parsed_keys = parse_object_keys(list_resp["Contents"])
        entries.extend(get_objects(parsed_keys, MIRROR_DATA_BUCKET))

        if limit and len(entries) > limit:
            return entries[:limit]

        if not list_resp["IsTruncated"]:
            cont_token = list_resp["NextContinuationToken"]
        else:
            return entries
