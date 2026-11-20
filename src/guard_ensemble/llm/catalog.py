from typing import Any, Literal, TypedDict, get_args

Model = Literal[
    "llama_guard",
    "shieldgemma",
    "granite_guardian",
    "qwen_guard",
    "qwen3_4b_2507",
]

# *.jinja: newer HF repos ship the chat template as chat_template.jinja
# instead of inside tokenizer_config.json.
DEFAULT_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.jinja"]


class ModelConfig(TypedDict):
    repo: str
    llm_kwargs: dict[str, Any]
    allow_patterns: list[str] | None


def model_config(model: Model) -> ModelConfig:
    match model:
        case "llama_guard":
            return {
                "repo": "meta-llama/Llama-Guard-3-8B",
                "llm_kwargs": {
                    "max_model_len": 2048,
                    "quantization": "fp8_per_tensor",
                    "enforce_eager": True,
                },
                "allow_patterns": DEFAULT_ALLOW_PATTERNS,
            }
        case "shieldgemma":
            return {
                "repo": "google/shieldgemma-9b",
                "llm_kwargs": {
                    "max_model_len": 2048,
                    "quantization": "fp8_per_tensor",
                    "gpu_memory_utilization": 0.85,
                    "max_num_seqs": 64,
                    "enforce_eager": True,
                },
                "allow_patterns": DEFAULT_ALLOW_PATTERNS,
            }
        case "granite_guardian":
            return {
                "repo": "ibm-granite/granite-guardian-3.3-8b",
                "llm_kwargs": {
                    "max_model_len": 2048,
                    "quantization": "fp8_per_tensor",
                    "enforce_eager": True,
                },
                "allow_patterns": DEFAULT_ALLOW_PATTERNS,
            }
        case "qwen_guard":
            return {
                "repo": "Qwen/Qwen3Guard-Gen-8B",
                "llm_kwargs": {
                    "max_model_len": 2048,
                    "quantization": "fp8_per_tensor",
                    "enforce_eager": True,
                },
                "allow_patterns": DEFAULT_ALLOW_PATTERNS,
            }
        case "qwen3_4b_2507":
            return {
                "repo": "Qwen/Qwen3-4B-Instruct-2507",
                "llm_kwargs": {
                    "max_model_len": 8192,
                    "disable_log_stats": False,
                    "max_num_seqs": 512,
                },
                "allow_patterns": DEFAULT_ALLOW_PATTERNS,
            }


MODEL_KEYS: frozenset[Model] = frozenset(get_args(Model))
