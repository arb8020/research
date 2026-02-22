"""Megatron-Core training backend.

Provides distributed training with tensor/pipeline/expert parallelism
for large models like GLM-5 (700B-A40B).

Uses Megatron-Core as a dependency (like we use SGLang for inference).
Orchestration via miniray.

Components:
- initialize: Megatron process group setup
- model: Model/optimizer factory using mbridge/AutoBridge
- mbridge: Custom model bridges for GLM-4, GLM-4.7-Flash
- weight_conversion: Megatron → HuggingFace state dict mapping

Dependencies:
- megatron-core: NVIDIA's distributed training library
- mbridge: HF <-> Megatron weight conversion (https://github.com/ISEEKYAN/mbridge)
"""

from .initialize import init_megatron
from .model import MegatronModelConfig, setup_megatron_model

__all__ = ["init_megatron", "setup_megatron_model", "MegatronModelConfig"]
