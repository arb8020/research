"""Backend model lowerings derived from backend-neutral model denotation.

These are the model-semantic analogue of the training `RealizationPlan`
lowerings. They derive honest backend-native construction intent; they do not
pretend to be executable backend IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .denotation import ModelDenotation

MegatronAdapterKind = Literal["provider", "raw_gpt", "custom_spec"]
MegatronLoaderKind = Literal["bridge_hf", "megatron_dist"]
NmoeCheckpointKind = Literal["nmoe_native", "hf_import_only"]
NmoeLoaderKind = Literal["nmoe_native", "hf_bridge_provisional"]
TorchTitanLoaderKind = Literal["hf_state_dict_adapter", "direct_state_dict"]


@dataclass(frozen=True)
class MegatronModelLowering:
    """Backend-native Megatron construction intent."""

    denotation: ModelDenotation
    adapter_kind: MegatronAdapterKind
    loader_kind: MegatronLoaderKind
    architecture_args: dict[str, Any] = field(default_factory=dict)
    validation_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TorchTitanModelLowering:
    """Backend-native TorchTitan construction intent."""

    denotation: ModelDenotation
    train_spec_name: str
    model_size: str
    loader_kind: TorchTitanLoaderKind
    validation_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class NmoeModelLowering:
    """Reserved backend-native `nmoe` construction intent."""

    denotation: ModelDenotation
    loader_kind: NmoeLoaderKind
    checkpoint_kind: NmoeCheckpointKind
    runtime_family: str
    validation_notes: tuple[str, ...] = ()


def lower_model_to_megatron(
    denotation: ModelDenotation,
    *,
    bridge_supports_provider: bool,
    architecture_args: dict[str, Any] | None = None,
) -> MegatronModelLowering:
    """Lower internal model denotation into Megatron-native construction intent."""
    family = denotation.architecture.family
    notes: list[str] = []

    if family in {"qwen3_5", "qwen3_5_moe"}:
        adapter_kind: MegatronAdapterKind = "custom_spec"
        notes.append(
            "qwen3_5 lowers through an explicit Megatron custom spec and custom bridge registration; "
            "this family is not honestly represented by the generic provider/raw_gpt paths"
        )
    elif family == "qwen3_next":
        adapter_kind = "custom_spec"
        notes.append(
            "qwen3_next lowers through an explicit Megatron custom spec and custom bridge registration; "
            "this family includes linear-attention semantics beyond the generic provider/raw_gpt paths"
        )
    elif bridge_supports_provider:
        adapter_kind: MegatronAdapterKind = "provider"
    elif family == "qwen3":
        adapter_kind = "custom_spec"
        notes.append(
            "qwen3 lowers through explicit Megatron model args derived from model denotation; "
            "this path mirrors the slime/miles plain-Qwen3 story more closely than raw_gpt fallback"
        )
    else:
        adapter_kind = "raw_gpt"
        if family in {"qwen3", "qwen3_moe"} and denotation.architecture.norm == "rmsnorm":
            notes.append(
                "raw_gpt fallback for RMSNorm Qwen-family models is currently provisional; "
                "prefer provider or custom_spec lowering when available"
            )

    return MegatronModelLowering(
        denotation=denotation,
        adapter_kind=adapter_kind,
        loader_kind="bridge_hf",
        architecture_args=dict(architecture_args or ()),
        validation_notes=tuple(notes),
    )


def lower_model_to_torchtitan(
    denotation: ModelDenotation,
    *,
    model_override: str | None = None,
    size_override: str | None = None,
) -> TorchTitanModelLowering:
    """Lower internal model denotation into TorchTitan-native construction intent."""
    family = denotation.architecture.family
    variant = denotation.variant

    if family == "glm4":
        derived_spec = "glm"
        derived_size = variant or "4.7-flash"
    elif family == "glm4_moe":
        derived_spec = "glm"
        derived_size = variant or "4.7-flash"
    elif family == "qwen3":
        derived_spec = "qwen3"
        derived_size = variant or "0.6B"
    elif family == "llama":
        derived_spec = "llama3"
        derived_size = variant or "8B"
    else:
        raise ValueError(
            f"TorchTitan lowering does not know how to derive a train spec for family {family!r}"
        )

    train_spec_name = model_override or derived_spec
    model_size = size_override or derived_size
    notes: list[str] = []
    if model_override is not None and model_override != derived_spec:
        notes.append(
            f"explicit torchtitan model override {model_override!r} differs from denotation-derived "
            f"train spec {derived_spec!r}"
        )
    if size_override is not None and size_override != derived_size:
        notes.append(
            f"explicit torchtitan size override {size_override!r} differs from denotation-derived "
            f"model size {derived_size!r}"
        )

    return TorchTitanModelLowering(
        denotation=denotation,
        train_spec_name=train_spec_name,
        model_size=model_size,
        loader_kind="hf_state_dict_adapter",
        validation_notes=tuple(notes),
    )


def lower_model_to_nmoe(
    denotation: ModelDenotation,
) -> NmoeModelLowering:
    """Reserve an honest model-lowering shape for a future native `nmoe` path."""
    family = denotation.architecture.family
    runtime_family_map = {
        "glm4": "glm4",
        "glm4_moe": "glm4_moe",
        "qwen3": "qwen3",
        "qwen3_moe": "qwen3_moe",
        "qwen3_5": "qwen3_5",
        "qwen3_5_moe": "qwen3_5_moe",
        "qwen3_next": "qwen3_next",
    }
    if family not in runtime_family_map:
        raise ValueError(
            f"Nmoe model lowering does not know how to derive native runtime intent for {family!r}"
        )

    notes = [
        "placeholder lowering only; no runnable nmoe backend exists yet",
        "HF import remains provisional until we map native nmoe checkpoints and Transformer(cfg)",
    ]
    return NmoeModelLowering(
        denotation=denotation,
        loader_kind="hf_bridge_provisional",
        checkpoint_kind="nmoe_native",
        runtime_family=runtime_family_map[family],
        validation_notes=tuple(notes),
    )
