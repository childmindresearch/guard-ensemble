from typing import Literal

MODEL_KEY = Literal[
    "llama_guard",
    "shieldgemma",
    "granite_guardian",
    "qwen_guard",
    "qwen3_4b",
    "qwen3_4b_2507",
    "gemma3_4b",
    "qwen3_8b",
]

MODELS: dict[MODEL_KEY, dict] = {
    "llama_guard": {
        "repo": "meta-llama/Llama-Guard-3-8B",
        "llm_kwargs": {
            "max_model_len": 2048,
            "quantization": "fp8_per_tensor",
            "enforce_eager": True,
        },
    },
    "shieldgemma": {
        "repo": "google/shieldgemma-9b",
        "llm_kwargs": {
            "max_model_len": 2048,
            "quantization": "fp8_per_tensor",
            "gpu_memory_utilization": 0.85,
            "max_num_seqs": 64,
            "enforce_eager": True,
        },
    },
    "granite_guardian": {
        "repo": "ibm-granite/granite-guardian-3.3-8b",
        "llm_kwargs": {
            "max_model_len": 2048,
            "quantization": "fp8_per_tensor",
            "enforce_eager": True,
        },
    },
    "qwen_guard": {
        "repo": "Qwen/Qwen3Guard-Gen-8B",
        "llm_kwargs": {
            "max_model_len": 2048,
            "quantization": "fp8_per_tensor",
            "enforce_eager": True,
        },
    },
    "qwen3_4b": {
        "repo": "Qwen/Qwen3-4B",
        "llm_kwargs": {
            "max_model_len": 8192,
            "disable_log_stats": False,
            "max_num_seqs": 512,
        },
    },
    "qwen3_4b_2507": {
        "repo": "Qwen/Qwen3-4B-Instruct-2507",
        "llm_kwargs": {
            "max_model_len": 8192,
            "disable_log_stats": False,
            "max_num_seqs": 512,
        },
    },
}

MODEL_KEYS = frozenset(dict.keys(MODELS))
