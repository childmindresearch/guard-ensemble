# vLLM runtime for the guard/domain models. Expects a 24GB GPU (e.g. A10G).
# Weights are downloaded into the mounted volume on first run.
#
#   docker build -t guard-ensemble .
#
#   docker run --gpus all \
#     -e HOME=/tmp -e HF_TOKEN="$HF_TOKEN" \
#     -v hf-cache:/models -v vllm-cache:/tmp/.cache/vllm \
#     guard-ensemble
#
# For the long-lived Jupyter kernel see scripts/run-kernel.sh.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group gpu --group kernel --no-install-project

COPY README.md ./
COPY src/ ./src/
COPY prompts/ ./prompts/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group gpu --group kernel

ENV HF_HOME=/models
ENV VLLM_USE_FLASHINFER_SAMPLER=0

ENTRYPOINT ["guard-ensemble"]
