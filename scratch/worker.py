"""Child-process side: load one model, then serve batches until stdin closes.

Started by scratch.runner as:

    python -m scratch.worker <model>

Protocol: one JSON object per line.
  worker -> parent, once:   {"ok": true, "ready": true}  or  {"ok": false, "error": "..."}
  parent -> worker:         {"mode": "chat"|"generate", "inputs": [...], "sampling": {...}}
  worker -> parent:         {"ok": true, "result": [[completion, ...], ...]}  or  {"ok": false, "error": "..."}

A completion is the generated text, or, when sampling includes "logprobs",
{"text": str, "logprobs": [{token: logprob, ...}, ...]} with one dict per
generated token (the top-`logprobs` candidates plus the sampled token).

vLLM's progress bar goes to the log: `tail -f scratch/logs/<model>.log`.

The original stdout is reserved for the protocol; fd 1 is redirected to stderr
(the log file) so vLLM's prints and child processes can't corrupt it.
When stdin closes (parent stopped us or died), the worker exits.

Model configs live in scratch/models.py. Importing this module is cheap (vllm
is only imported in serve), so the parent can reload it without touching the GPU.
"""

import json
import math
import os
import sys
import traceback

from scratch.models import MODEL_KEY, MODELS

DEFAULT_ALLOW_PATTERNS = ["*.json", "*.safetensors"]


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
                    use_tqdm=True,  # progress bar goes to the log
                    chat_template_kwargs=req.get("chat_template_kwargs"),
                )
            elif req["mode"] == "generate":
                outs = llm.generate(req["inputs"], params, use_tqdm=True)
            else:
                raise ValueError(f"unknown mode {req['mode']!r}")
            with_lp = req["sampling"].get("logprobs") is not None
            result = [[_completion(c, with_lp) for c in o.outputs] for o in outs]
            send({"ok": True, "result": result})
        except Exception:  # noqa: BLE001 -- report and keep serving; model stays loaded
            send({"ok": False, "error": traceback.format_exc()})


def _completion(c, with_logprobs: bool):
    """Text only, or text plus per-token {decoded_token: logprob} dicts."""
    if not with_logprobs:
        return c.text
    steps = []
    for step in c.logprobs or []:
        # step: {token_id: Logprob(logprob, rank, decoded_token)}
        d: dict[str, float] = {}
        for tok_id, lp in step.items():
            key = lp.decoded_token if lp.decoded_token is not None else str(tok_id)
            # Different ids can decode to the same text: add their probabilities.
            prev = d.get(key)
            if prev is None:
                d[key] = lp.logprob
            else:
                hi, lo = max(prev, lp.logprob), min(prev, lp.logprob)
                d[key] = hi + math.log1p(math.exp(lo - hi))
        steps.append(d)
    return {"text": c.text, "logprobs": steps}


if __name__ == "__main__":
    model = sys.argv[1]
    if model not in MODELS:
        raise ValueError(f"Unsupported model: {model}")
    serve(model)
