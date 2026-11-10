# syntax=docker/dockerfile:1
# vLLM runtime for Llama Guard 3 8B (bf16). Needs a >=24 GB GPU (e.g. A10G).
# Weights are NOT baked in: they download into the mounted volume on first run.
#
#   docker build -t guard-ensemble .
#
#   docker run --gpus all \
#     -e HOME=/tmp -e HF_TOKEN="$HF_TOKEN" \
#     -v hf-cache:/models -v vllm-cache:/tmp/.cache/vllm -v ./scratch/logs:/app/scratch/logs \
#     guard-ensemble <llama_guard|shieldgemma|granite_guardian|qwen_guard> "<text>"

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir vllm huggingface_hub

RUN pip install --no-cache-dir pandas jupyter_server ipykernel

ENV HF_HOME=/models
ENV VLLM_USE_FLASHINFER_SAMPLER=0

WORKDIR /app

COPY *.py /app

COPY prompts/ /app/prompts/

COPY scratch/ /app/scratch/

ENTRYPOINT ["python", "-m", "scratch.cli"]
