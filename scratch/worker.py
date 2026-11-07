"""Child-process side: load one model, then serve batches until stdin closes.

Started by scratch.runner as:

    python -m scratch.worker <model>

Protocol: one JSON object per line.
  worker -> parent, once:   {"ok": true, "ready": true}  or  {"ok": false, "error": "..."}
  parent -> worker:         {"mode": "chat"|"generate", "inputs": [...], "sampling": {...}}
  worker -> parent:         {"ok": true, "result": [[text, ...], ...]}  or  {"ok": false, "error": "..."}

The original stdout is reserved for the protocol; fd 1 is redirected to stderr
(the log file) so vLLM's prints and child processes can't corrupt it.
When stdin closes (parent stopped us or died), the worker exits.

Importing this module is cheap (vllm is only imported in serve), so the parent
can read MODELS without touching the GPU.
"""

import json
import os
import sys
import traceback
from typing import Literal

DEFAULT_ALLOW_PATTERNS = ["*.json", "*.safetensors"]

MODEL_KEY = Literal["llama_guard", "shieldgemma"]

MODELS: dict[MODEL_KEY, dict] = {
    "llama_guard": {
        "repo": "meta-llama/Llama-Guard-3-8B",
        "llm_kwargs": {"max_model_len": 2048, "quantization": "fp8_per_tensor"},
    },
    "shieldgemma": {
        "repo": "google/shieldgemma-9b",
        "llm_kwargs": {"max_model_len": 2048, "quantization": "fp8_per_tensor"},
    },
}


def serve(model: MODEL_KEY) -> None:
    proto = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)

    def send(msg: dict) -> None:
        proto.write(json.dumps(msg) + "\n")
        proto.flush()

    try:
        from huggingface_hub import snapshot_download
        from vllm import LLM, SamplingParams

        if model not in MODELS:
            raise ValueError(f"Unsupported model: {model}")

        spec = MODELS[model]
        path = snapshot_download(
            spec["repo"],
            allow_patterns=spec.get("allow_patterns", DEFAULT_ALLOW_PATTERNS),
        )
        llm = LLM(model=path, **spec.get("llm_kwargs", {}))
    except Exception:  # noqa: BLE001 -- any load failure is reported to the parent
        send({"ok": False, "error": traceback.format_exc()})
        return

    send({"ok": True, "ready": True})

    for line in sys.stdin:
        try:
            req = json.loads(line)
            params = SamplingParams(**req["sampling"])
            if req["mode"] == "chat":
                outs = llm.chat(
                    req["inputs"],
                    params,
                    use_tqdm=False,
                    chat_template_kwargs=req.get("chat_template_kwargs"),
                )
            elif req["mode"] == "generate":
                outs = llm.generate(req["inputs"], params, use_tqdm=False)
            else:
                raise ValueError(f"unknown mode {req['mode']!r}")
            send({"ok": True, "result": [[c.text for c in o.outputs] for o in outs]})
        except Exception:  # noqa: BLE001 -- report and keep serving; model stays loaded
            send({"ok": False, "error": traceback.format_exc()})


if __name__ == "__main__":
    model = sys.argv[1]
    if model not in MODELS:
        raise ValueError(f"Unsupported model: {model}")
    serve(model)
