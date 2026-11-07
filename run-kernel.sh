#!/usr/bin/env bash

# Run Jupyter Kernel on devbox, opening sockets that will be forwarded over SSH.
#
#   ./run-kernel.sh          start the kernel, or reuse it if already running
#   ./run-kernel.sh stop     kill the kernel container (frees all GPU memory)
#
# After an nvim session dies, the kernel (and any loaded model) is still alive:
# reattach with :MoltenInit <connection file> instead of starting a new one.

set -eou pipefail

NAME=guard-kernel
IMAGE=guard-llama
RUNTIME_DIR=/tmp/svim-k

if [[ "${1:-}" == "stop" ]]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

if [[ -n "$(docker ps -q -f "name=^${NAME}$")" ]]; then
  echo "$NAME already running; reattach with:"
  ls -t "$RUNTIME_DIR"/kernel-*.json | head -1
  exit 0
fi

HF_TOKEN="$(cat hf_token)"

docker run \
  --name "$NAME" \
  --rm \
  --entrypoint jupyter \
  --user "$(id -u):$(id -g)" \
  --gpus all \
  -e HOME=/tmp \
  -e HF_TOKEN="$HF_TOKEN" \
  -e JUPYTER_RUNTIME_DIR=/run/kernel \
  -v hf-cache:/models \
  -v vllm-cache:/tmp/.cache/vllm \
  -v "$RUNTIME_DIR":/run/kernel \
  -v "$PWD:/app" \
  "$IMAGE" \
  kernel --kernel=python3 \
         --KernelManager.transport=ipc \
         --KernelManager.ip=/run/kernel/k