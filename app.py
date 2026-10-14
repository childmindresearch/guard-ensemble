"""Thin orchestrator: fans texts out to vLLM model servers, runs the deliberator.

No GPU code lives here. Inference happens in the vLLM containers; this service
only builds prompts, calls them concurrently (bounded by MAX_INFLIGHT), parses
results and manages batch jobs.

Wire in your existing guards.py: GUARDS (name, build, parse), deliberation_build,
deliberation_parse, shieldgemma_aggregate.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import uuid

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from guards import GUARDS, deliberation_build, deliberation_parse

MODEL_URLS: dict[str, str] = json.loads(os.environ["MODEL_URLS"])
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "4"))
GUARD_MAX_TOKENS = 32
DELIBERATOR_MAX_TOKENS = 256

app = FastAPI(title="risk-detection")
client = httpx.AsyncClient(timeout=120)
sem = asyncio.Semaphore(MAX_INFLIGHT)       # caps concurrent model calls across ALL requests
jobs: dict[str, dict] = {}                  # in-memory; move to SQS + S3 before relying on it


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

async def complete(model: str, prompt: str, max_tokens: int) -> str:
    """One completion against the vLLM server for `model`."""
    body = {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": max_tokens}
    async with sem:
        resp = await client.post(f"{MODEL_URLS[model]}/completions", json=body)
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


async def chat(model: str, prompt: str, max_tokens: int, **extra) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        **extra,
    }
    async with sem:
        resp = await client.post(f"{MODEL_URLS[model]}/chat/completions", json=body)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def run_guard(guard, text: str) -> dict:
    prompt = guard.build(text)
    call = chat if guard.path == "/v1/chat/completions" else complete
    raw = await call(guard.name, prompt, GUARD_MAX_TOKENS)
    return guard.parse(raw)

async def score_text(text: str) -> dict:
    """All configured guards concurrently, then the deliberator."""
    active = [g for g in GUARDS if g.name in MODEL_URLS]
    results = await asyncio.gather(*(run_guard(g, text) for g in active), return_exceptions=True)

    row: dict = {"text": text}
    unavailable = []
    for guard, res in zip(active, results):
        if isinstance(res, Exception):
            row[f"{guard.name}_error"] = f"{type(res).__name__}: {res}"
            unavailable.append(guard.name)
        else:
            row.update(res)

    scores = {k: v for k, v in row.items() if k != "text" and not k.endswith("_error")}
    if unavailable:
        scores["_guard_models_unavailable"] = ", ".join(unavailable)
    try:
        raw = await chat("deliberator", deliberation_build(text, scores), DELIBERATOR_MAX_TOKENS,
                 chat_template_kwargs={"enable_thinking": False})
        row["deliberation_risk"] = deliberation_parse(raw)
    except Exception as e:  # noqa: BLE001 - keep the row, surface the failure
        row["deliberation_risk"] = f"error: {type(e).__name__}: {e}"
    return row


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    csv_url: str                # pre-signed or public URL; swap for bucket/key once in AWS
    text_col: str = "text"


@app.get("/health")
async def health():
    return {"ok": True, "models": list(MODEL_URLS)}


@app.post("/detect")
async def detect(req: DetectRequest):
    return await score_text(req.text)


@app.post("/batches", status_code=202)
async def create_batch(req: BatchRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "done": 0, "total": None, "rows": []}
    bg.add_task(run_batch, job_id, req)
    return {"job_id": job_id}


@app.get("/batches/{job_id}")
async def get_batch(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in job.items() if k != "rows"}


@app.get("/batches/{job_id}/results")
async def get_results(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    return job["rows"]


async def run_batch(job_id: str, req: BatchRequest) -> None:
    job = jobs[job_id]
    try:
        resp = await client.get(req.csv_url)
        resp.raise_for_status()
        texts = [r[req.text_col] for r in csv.DictReader(io.StringIO(resp.text))]
        job.update(status="running", total=len(texts))

        async def one(i: int, text: str) -> dict:
            row = await score_text(text)
            job["done"] += 1
            return {"id": i, **row}

        # Rows run concurrently; the semaphore in complete()/chat() keeps the servers from flooding.
        job["rows"] = await asyncio.gather(*(one(i, t) for i, t in enumerate(texts)))
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        job.update(status="failed", error=f"{type(e).__name__}: {e}")
