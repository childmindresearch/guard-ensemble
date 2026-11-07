"""Keep one model loaded in a persistent child process.

The worker stays up between calls and is only replaced when a different model
is requested (or on stop()). The kernel never imports vllm; stopping the worker
frees all of its GPU memory.

    from scratch import runner
    outs = runner.chat("llama_guard", conversations, temperature=0, max_tokens=32)
    outs[i][0]              # first sample for conversation i (len(outs[i]) == n)
    runner.generate("some_model", ["raw prompt", ...], max_tokens=5)  # swaps models
    runner.current()        # "some_model"
    runner.stop()           # free the GPU

Keyword args are passed straight to vllm.SamplingParams.
Worker output (vLLM logs, tracebacks) goes to scratch/logs/<model>.log.

If a call is interrupted (Ctrl-C) the worker is killed, since its reply would
otherwise arrive out of sync; the next call reloads the model.

`importlib.reload(runner)` keeps the running worker. scratch/worker.py is
re-read every time a worker starts, so new MODELS entries need no reload.
If the kernel restarts, the old worker is found via a pidfile and killed on
the next load().
"""

from __future__ import annotations

import atexit
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING

from scratch import worker

if TYPE_CHECKING:
    from scratch.worker import MODEL_KEY

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / "logs"
PIDFILE = Path("/tmp/scratch-worker.pid")

# globals().get(...) so importlib.reload(runner) keeps the running worker.
_proc: subprocess.Popen | None = globals().get("_proc")
_model: str | None = globals().get("_model")
_log: IO | None = globals().get("_log")


class WorkerError(RuntimeError):
    """The worker raised while handling a request (it is still running)."""


def current() -> str | None:
    """Name of the loaded model, or None."""
    if _proc is not None and _proc.poll() is None:
        return _model
    return None


def load(model: MODEL_KEY) -> None:
    """Make sure `model` is loaded, replacing any other model."""
    global _proc, _model, _log
    if current() == model:
        return
    importlib.reload(worker)  # pick up edits to MODELS
    if model not in worker.MODELS:
        raise KeyError(f"unknown model {model!r}; known: {sorted(worker.MODELS)}")
    stop()
    _kill_stale()

    LOG_DIR.mkdir(exist_ok=True)
    _log = open(LOG_DIR / f"{model}.log", "w")
    _proc = subprocess.Popen(
        [sys.executable, "-m", "scratch.worker", model],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=_log,
        text=True,
        start_new_session=True,  # own process group, so we can kill vLLM's children too
    )
    _model = model
    PIDFILE.write_text(str(_proc.pid))
    try:
        _recv()
    except BaseException:
        stop()
        raise


def chat(
    model: MODEL_KEY,
    conversations: list,
    chat_template_kwargs: dict | None = None,
    **sampling,
) -> list[list[str]]:
    """Run llm.chat over a list of conversations (each a list of message dicts).

    chat_template_kwargs are extra variables for the model's chat template and
    apply to the whole batch, e.g. {"guideline": policy} for ShieldGemma.
    """
    extra = {"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else {}
    return _request(model, "chat", conversations, sampling, extra)


def generate(model: MODEL_KEY, prompts: list[str], **sampling) -> list[list[str]]:
    """Run llm.generate over raw prompt strings (no chat template applied)."""
    return _request(model, "generate", prompts, sampling)


def stop(grace: float = 15.0) -> None:
    """Shut down the worker (if any) and free its GPU memory."""
    global _proc, _model, _log
    proc, log = _proc, _log
    _proc = _model = _log = None
    if proc is not None:
        if proc.stdin is not None:
            try:
                proc.stdin.close()  # worker exits its loop and lets vLLM shut down
            except OSError:
                pass
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        _kill_group(proc.pid, reap=proc)
    if log is not None:
        log.close()
    PIDFILE.unlink(missing_ok=True)


atexit.register(stop)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _request(
    model: MODEL_KEY,
    mode: str,
    inputs: list,
    sampling: dict,
    extra: dict | None = None,
) -> list[list[str]]:
    load(model)
    _, stdin, _ = _pipes()
    try:
        req = {"mode": mode, "inputs": list(inputs), "sampling": sampling, **(extra or {})}
        stdin.write(json.dumps(req) + "\n")
        stdin.flush()
        return _recv()["result"]
    except WorkerError:
        raise
    except BaseException:
        stop()
        raise


def _pipes() -> tuple[subprocess.Popen, IO[str], IO[str]]:
    """The running worker and its stdin/stdout; raises if there is none."""
    proc = _proc
    if proc is None or proc.stdin is None or proc.stdout is None:
        raise RuntimeError("no worker running; call load() first")
    return proc, proc.stdin, proc.stdout


def _recv() -> dict:
    proc, _, stdout = _pipes()
    line = stdout.readline()
    if not line:
        rc = proc.wait()
        raise RuntimeError(f"{_model} worker died (exit {rc}):\n{_tail()}")
    msg = json.loads(line)
    if not msg["ok"]:
        raise WorkerError(msg["error"])
    return msg


def _kill_stale() -> None:
    """Kill a worker left behind by a previous kernel / module reload."""
    try:
        pid = int(PIDFILE.read_text())
    except (OSError, ValueError):
        return
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists() and b"scratch.worker" not in cmdline.read_bytes():
        PIDFILE.unlink(missing_ok=True)  # pid reused by something else
        return
    _kill_group(pid)
    PIDFILE.unlink(missing_ok=True)


def _kill_group(
    pgid: int, reap: subprocess.Popen | None = None, grace: float = 10.0
) -> None:
    """SIGTERM the process group (worker + vLLM children), SIGKILL after `grace`."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if reap is not None:
            reap.poll()
        try:
            os.killpg(pgid, 0)  # anything left in the group?
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if reap is not None:
        reap.wait()


def _tail(lines: int = 30) -> str:
    if _model is None:
        return ""
    try:
        return "\n".join((LOG_DIR / f"{_model}.log").read_text().splitlines()[-lines:])
    except OSError:
        return ""
