import datetime
import json
import os
import re

import boto3

RO_ACCESS_SECRET_ARN = "arn:aws:secretsmanager:us-east-1:103625405479:secret:msm-controller-mirror-data-access-E2tK8E"
MIRROR_BUCKET = "cmi-mirror-research-data-production-5895463cf3cfa3086595eb0b89"

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

    secrets_client = boto3.client("secretsmanager")
    secret_resp = secrets_client.get_secret_value(SecretId=RO_ACCESS_SECRET_ARN)
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


def list_objects():
    return s3_client.list_objects_v2(Bucket=MIRROR_BUCKET)


def datetime_from_str_ts(ts: str):
    return datetime.datetime.fromtimestamp(int(ts) / 1000)


def convert_pulled_at(parsed_keys: list[dict]):
    return [
        {**g, "pulled_at": datetime_from_str_ts(g["pulled_at"])} for g in parsed_keys
    ]


def parse_keys(objs: list[dict]):
    parsed = [
        {**m.groupdict(), "key": o["Key"]}
        for o in objs
        if (m := KEY_RE.search(o["Key"]))
    ]
    return convert_pulled_at(parsed)


def pull_json_objects(parsed_keys: list[dict], length: int = 100):
    parsed = []
    for _, p in zip(range(length), parsed_keys):
        obj = s3_client.get_object(Bucket=MIRROR_BUCKET, Key=p["key"])
        data = json.loads(obj["Body"].read().decode("utf-8"))
        parsed.append(data)
    return parsed
