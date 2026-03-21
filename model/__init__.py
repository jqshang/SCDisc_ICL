from model.base import BaseLLM
from model.gemma import GemmaModel
from model.llama import LlamaModel
from model.qwen import QwenModel
from model.gpt4 import GPT4Model
from model.deepseek import DeepSeekR1Model

MODEL_REGISTRY = {
    "gemma3": GemmaModel,
    "llama3": LlamaModel,
    "qwen": QwenModel,
    "gpt4": GPT4Model,
    "deepseek-r1": DeepSeekR1Model,
}


def get_model(model_key: str, **kwargs) -> BaseLLM:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_key](**kwargs)
