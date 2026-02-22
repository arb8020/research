"""Model implementations for inference.

Custom model implementations that work with our attention backend
and KV cache. Weights are loaded from HuggingFace format.

Supported models:
- LlamaForCausalLM: Llama 2, Llama 3, and similar
- Qwen2ForCausalLM: TODO
- Qwen3ForCausalLM: TODO
- Qwen3MoeForCausalLM: TODO (MoE variant)
"""

from .base import BaseModel
from .config import ModelConfig, load_model_config
from .llama import LlamaForCausalLM
from .qwen import Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM
from .weight import load_weights

__all__ = [
    "BaseModel",
    "ModelConfig",
    "load_model_config",
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "load_weights",
]
