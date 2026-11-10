from typing import Literal

MODEL_KEY = Literal["llama_guard", "shieldgemma"]

MODELS: dict[MODEL_KEY, dict] = {
    "llama_guard": {
        "repo": "meta-llama/Llama-Guard-3-8B",
        "llm_kwargs": {"max_model_len": 2048, "quantization": "fp8_per_tensor"},
    },
    "shieldgemma": {
        "repo": "google/shieldgemma-9b",
        "llm_kwargs": {
            "max_model_len": 2048,
            "quantization": "fp8_per_tensor",
            "gpu_memory_utilization": 0.85,
            "max_num_seqs": 64,
        },
    },
}
