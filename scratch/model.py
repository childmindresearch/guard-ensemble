import os

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

from huggingface_hub import snapshot_download
from vllm import LLM

LLAMA_GUARD_REPO = "meta-llama/Llama-Guard-3-8B"

llama_guard_dir = snapshot_download(
    LLAMA_GUARD_REPO, allow_patterns=["*.json", "*.safetensors"]
)

_LLAMA_GUARD_LLM: LLM | None = None


def get_llama_guard_llm() -> LLM:
    global _LLAMA_GUARD_LLM
    if _LLAMA_GUARD_LLM is not None:
        return _LLAMA_GUARD_LLM
    llm = LLM(model=llama_guard_dir, max_model_len=4096, quantization="fp8")
    _LLAMA_GUARD_LLM = llm
    return llm


llama_guard_llm = get_llama_guard_llm()
