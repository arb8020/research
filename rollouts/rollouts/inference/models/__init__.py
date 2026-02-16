"""Model implementations for inference.

Custom model implementations that work with our attention backend
and KV cache. Weights are loaded from HuggingFace format.
"""

from .base import BaseModel
from .config import ModelConfig, load_model_config
from .llama import LlamaForCausalLM
from .weight import load_weights

__all__ = [
    "BaseModel",
    "ModelConfig",
    "load_model_config",
    "LlamaForCausalLM",
    "load_weights",
]
