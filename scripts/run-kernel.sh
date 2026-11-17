#!/usr/bin/env bash

# Run Jupyter Kernel on devbox, opening sockets that will be forwarded over SSH.
#
# Examples:
#
#   scripts/run-kernel.sh            # start the kernel, no-op if already running
#   scripts/run-kernel.sh --detach   # start the kernel in the background
#   scripts/run-kernel.sh stop       # stop the kernel
#

set -eou pipefail

NAME=guard-kernel
IMAGE=guard-ensemble
RUNTIME_DIR=/tmp/svim-k

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == "stop" ]]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

if [[ -n "$(docker ps -q -f "name=^${NAME}$")" ]]; then
  echo "$NAME already running"
  exit 0
fi

HF_TOKEN="$(cat hf_token)"

DETACH=""

if [[ "${1:-}" == "--detach" ]]; then
  DETACH="1"
fi

kernel_args=(
  --kernel=python3
  --KernelManager.transport=ipc
  --KernelManager.ip=/run/kernel/k
)

docker run \
  --name "$NAME" \
  --rm \
  ${DETACH:+--detach} \
  --entrypoint jupyter \
  --user "$(id -u):$(id -g)" \
  --gpus all \
  --network host \
  -e HOME=/tmp \
  -e HF_TOKEN="$HF_TOKEN" \
  -e JUPYTER_RUNTIME_DIR=/run/kernel \
  -v /opt/dlami/nvme/hf-cache:/models \
  -v vllm-cache:/tmp/.cache/vllm \
  -v "$RUNTIME_DIR":/run/kernel \
  -v "$PWD:/app" \
  "$IMAGE" \
  kernel "${kernel_args[@]}"
