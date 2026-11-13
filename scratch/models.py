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
    # Domain-prompt / deliberation model. bf16 (~8 GB) fits with room for KV
    # cache. A domain prompt (scratch/domains.py) is ~2-3k tokens. 16k was sized
    # for a removed all-domains triage prompt (~10k); 4096 would now do.
    "qwen3_4b": {
        "repo": "Qwen/Qwen3-4B",
        # disable_log_stats=False: periodic scheduler stats in the log (running /
        # waiting requests, KV cache usage, prefix cache hit rate).
        "llm_kwargs": {
            "max_model_len": 8192,
            "disable_log_stats": False,
            "max_num_seqs": 512,
        },
    },
    # Qwen3-4B retrained for non-thinking use (July 2025): better instruction
    # following than qwen3_4b with enable_thinking=False. Drop-in replacement.
    "qwen3_4b_2507": {
        "repo": "Qwen/Qwen3-4B-Instruct-2507",
        "llm_kwargs": {
            "max_model_len": 8192,
            "disable_log_stats": False,
            "max_num_seqs": 512,
        },
    },
    # A different model family at the same size, for uncorrelated errors.
    # Multimodal checkpoint used text-only; no system role in its chat template.
    "gemma3_4b": {
        "repo": "google/gemma-3-4b-it",
        "llm_kwargs": {
            "max_model_len": 8192,
            "limit_mm_per_prompt": {"image": 0},
            "disable_log_stats": False,
            "max_num_seqs": 256,
        },
    },
    # Same role as qwen3_4b, to test whether a larger model follows "rate only
    # this domain". FP8 weight-only (Marlin on Ampere), like the guards: ~9 GB.
    "qwen3_8b": {
        "repo": "Qwen/Qwen3-8B",
        "llm_kwargs": {
            "max_model_len": 4096,
            "quantization": "fp8_per_tensor",
            "disable_log_stats": False,
            "max_num_seqs": 256,
        },
    },
}

MODEL_KEYS = frozenset(dict.keys(MODELS))
