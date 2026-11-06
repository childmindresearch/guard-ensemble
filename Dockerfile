# syntax=docker/dockerfile:1
# vLLM runtime for Llama Guard 3 8B (bf16). Needs a >=24 GB GPU (e.g. A10G).
# Weights are NOT baked in: they download into the mounted volume on first run.
#
#   docker build -t guard-llama .
#   docker run --gpus all -v hf-cache:/models -e HF_TOKEN guard-llama "first text" "second text"
#
# Dev (Jupyter server for molten-nvim; reach it via `ssh -L 8888:localhost:8888 <host>`):
#   docker run --gpus all -p 127.0.0.1:8888:8888 -e JUPYTER_TOKEN=dev -e HF_TOKEN \
#     -v hf-cache:/models -v vllm-cache:/root/.cache/vllm -v "$PWD:/app" \
#     --entrypoint jupyter guard-llama server --ip 0.0.0.0 --allow-root --no-browser

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir vllm huggingface_hub

RUN pip install --no-cache-dir pandas jupyter_server ipykernel

ENV HF_HOME=/models
ENV VLLM_USE_FLASHINFER_SAMPLER=0

WORKDIR /app

COPY scratch_vllm.py /app
ENTRYPOINT ["python", "scratch_vllm.py"]
