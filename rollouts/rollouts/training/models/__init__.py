"""Training models for rollouts."""

from __future__ import annotations

import importlib
from typing import Any

from .adapters import ModelConstructionAdapter
from .backend_lowering import (
    MegatronModelLowering,
    TorchTitanModelLowering,
    lower_model_to_megatron,
    lower_model_to_torchtitan,
)
from .denotation import (
    AttentionKind,
    CheckpointFormat,
    CheckpointSemantics,
    HFModelSource,
    ModelArchitectureSemantics,
    ModelDenotation,
    ModelFamily,
    MoESemantics,
    NormKind,
    RotaryKind,
    RotarySemantics,
    normalize_hf_model_denotation,
)

__all__ = [
    "glm",
    "AttentionKind",
    "CheckpointFormat",
    "CheckpointSemantics",
    "HFModelSource",
    "MegatronModelLowering",
    "ModelArchitectureSemantics",
    "ModelConstructionAdapter",
    "ModelDenotation",
    "ModelFamily",
    "MoESemantics",
    "NormKind",
    "RotaryKind",
    "RotarySemantics",
    "TorchTitanModelLowering",
    "lower_model_to_megatron",
    "lower_model_to_torchtitan",
    "normalize_hf_model_denotation",
]


def __getattr__(name: str) -> Any:
    if name == "glm":
        return importlib.import_module(f"{__name__}.glm")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
