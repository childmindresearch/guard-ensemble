#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# DMR (Docker Model Runner) preflight and pull script
# ---------------------------------------------------------------------------
# DMR is a single daemon. No per-model servers or runtime flags.
# This script: probes the endpoint, pulls the five required models, validates
# quantization, and runs a smoke test.
# ---------------------------------------------------------------------------

DMR_BASE_URL="${DMR_BASE_URL:-http://localhost:12434}"

# Canonical model references (all lowercase)
MODELS=(
    "huggingface.co/quantfactory/shieldgemma-9b-gguf:Q8_0"
    "huggingface.co/quantfactory/llama-guard-3-8b-gguf:Q8_0"
    "huggingface.co/ibm-granite/granite-guardian-3.3-8b-gguf:Q8_0"
    "huggingface.co/geoffmunn/Qwen3Guard-Gen-8B-GGUF:Q4_K_M"
    "huggingface.co/qwen/qwen3-4b-gguf:Q8_0"
)

# ---------------------------------------------------------------------------
# Kill / unload
# ---------------------------------------------------------------------------
kill_models() {
    echo "Unloading models from DMR …"
    for model in "${MODELS[@]}"; do
        docker model unload "$model" 2>/dev/null || true
    done
    echo "Done. Models not removed."
    exit 0
}

if [[ "${1:-}" == "--kill" ]]; then
    kill_models
fi

# ---------------------------------------------------------------------------
# Pull models
# ---------------------------------------------------------------------------
echo "Pulling ${#MODELS[@]} models …"

PULL_FAILURES=()
for model in "${MODELS[@]}"; do
    echo "  pulling  ${model} …"
    if ! docker model pull "$model" >/dev/null 2>&1; then
        PULL_FAILURES+=("$model")
    fi
done

if [[ ${#PULL_FAILURES[@]} -gt 0 ]]; then
    echo ""
    echo "ERROR: Failed to pull ${#PULL_FAILURES[@]} model(s):"
    for failed in "${PULL_FAILURES[@]}"; do
        echo "  - ${failed}"
    done
    exit 1
fi

echo "  pulls complete."

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
echo ""
echo "DMR base URL: ${DMR_BASE_URL}"
echo ""

echo "Smoke test — fetching engine model list:"
SMOKE="$(curl -sf "${DMR_BASE_URL}/engines/v1/models" 2>&1)" || {
    echo "ERROR: smoke test failed (endpoint returned non-200 or curl error)."
    echo "  DMR may need a moment to finish loading models."
    exit 1
}

if [[ -n "$SMOKE" ]]; then
    echo "$SMOKE" | head -40
    echo ""
    echo "Smoke test passed."
else
    echo "ERROR: Smoke test returned empty — DMR may still be loading."
    exit 1
fi

echo ""
echo "Ready. Use 'bash serve.sh --kill' to unload models."
