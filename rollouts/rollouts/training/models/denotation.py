"""Draft backend-neutral model denotation types.

This is the model-semantic analogue of `RealizationPlan` for layout intent.
The goal is to capture model/checkpoint truth once, then lower it honestly into
backend-native construction for Megatron, TorchTitan, or future backends.

This draft is intentionally narrow:
- architecture semantics
- checkpoint/load semantics
- external HF source identity

It does not yet model forward/loss semantics, optimizer/runtime policy, or
parallel/layout lowering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

ModelFamily: TypeAlias = Literal[
    "llama",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "glm4",
    "glm4_moe",
    "unknown",
]
NormKind: TypeAlias = Literal["layernorm", "rmsnorm"]
AttentionKind: TypeAlias = Literal["mha", "gqa", "mla", "linear_attention"]
RotaryKind: TypeAlias = Literal["rope", "yarn"]
CheckpointFormat: TypeAlias = Literal[
    "hf_pretrained",
    "hf_local",
    "megatron_dist",
    "torchtitan_dist",
    "adapter_delta",
]


@dataclass(frozen=True)
class HFModelSource:
    """Source identity in the HuggingFace/Transformers publishing ecosystem."""

    name_or_path: str
    trust_remote_code: bool = True
    revision: str | None = None

    def __post_init__(self) -> None:
        assert self.name_or_path, "HF model source must be non-empty"


@dataclass(frozen=True)
class RotarySemantics:
    """Rotary-position semantics after normalization."""

    kind: RotaryKind
    theta: float
    head_dim: int | None = None
    scaling: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.theta > 0, "rotary theta must be > 0"


@dataclass(frozen=True)
class MoESemantics:
    """Mixture-of-experts architecture facts."""

    num_experts: int
    experts_per_token: int
    shared_expert: bool = False

    def __post_init__(self) -> None:
        assert self.num_experts > 0, "num_experts must be > 0"
        assert self.experts_per_token > 0, "experts_per_token must be > 0"


@dataclass(frozen=True)
class ModelArchitectureSemantics:
    """Backend-neutral architecture semantics after normalization."""

    family: ModelFamily
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    ffn_hidden_size: int
    vocab_size: int
    max_sequence_length: int
    norm: NormKind
    norm_epsilon: float
    attention: AttentionKind
    num_kv_heads: int | None = None
    rotary: RotarySemantics | None = None
    moe: MoESemantics | None = None
    tie_embeddings: bool = True
    uses_qk_layernorm: bool = False
    uses_bias_linear: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.hidden_size > 0, "hidden_size must be > 0"
        assert self.num_layers > 0, "num_layers must be > 0"
        assert self.num_attention_heads > 0, "num_attention_heads must be > 0"
        assert self.ffn_hidden_size > 0, "ffn_hidden_size must be > 0"
        assert self.vocab_size > 0, "vocab_size must be > 0"
        assert self.max_sequence_length > 0, "max_sequence_length must be > 0"
        assert self.norm_epsilon > 0, "norm_epsilon must be > 0"
        if self.num_kv_heads is not None:
            assert self.num_kv_heads > 0, "num_kv_heads must be > 0"


@dataclass(frozen=True)
class CheckpointSemantics:
    """How weights are stored and expected to be loaded."""

    format: CheckpointFormat
    source: HFModelSource | str
    weight_tying: bool | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelDenotation:
    """Backend-neutral model truth before backend lowering."""

    source: HFModelSource
    architecture: ModelArchitectureSemantics
    checkpoint: CheckpointSemantics
    variant: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def _infer_rope_theta(hf_config: Any) -> float | None:
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is not None:
        return float(rope_theta)

    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and rope_parameters.get("rope_theta") is not None:
        return float(rope_parameters["rope_theta"])

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and rope_scaling.get("rope_theta") is not None:
        return float(rope_scaling["rope_theta"])

    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        return _infer_rope_theta(text_config)

    return None


def _infer_model_family(hf_config: Any, source: HFModelSource) -> ModelFamily:
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    architectures = [str(x).lower() for x in getattr(hf_config, "architectures", []) or ()]
    source_name = source.name_or_path.lower()

    if "qwen3_moe" in model_type or "qwen3moe" in source_name:
        return "qwen3_moe"
    if "qwen3" in model_type or any("qwen3" in arch for arch in architectures):
        return "qwen3"
    if "qwen2" in model_type or any("qwen2" in arch for arch in architectures):
        return "qwen2"
    if "glm" in model_type or any("glm" in arch for arch in architectures):
        if getattr(hf_config, "num_experts", None) or getattr(hf_config, "n_routed_experts", None):
            return "glm4_moe"
        return "glm4"
    if "llama" in model_type or any("llama" in arch for arch in architectures):
        return "llama"
    return "unknown"


def _infer_attention_kind(hf_config: Any, family: ModelFamily) -> AttentionKind:
    if getattr(hf_config, "multi_latent_attention", False):
        return "mla"
    if family in {"glm4", "glm4_moe"}:
        return "mla"
    num_attention_heads = getattr(hf_config, "num_attention_heads", None)
    num_kv_heads = getattr(hf_config, "num_key_value_heads", None)
    if num_attention_heads is not None and num_kv_heads not in (None, num_attention_heads):
        return "gqa"
    return "mha"


def _infer_variant(source: HFModelSource, family: ModelFamily) -> str | None:
    name = source.name_or_path.split("/")[-1]
    lowered = name.lower()
    if family in {"qwen2", "qwen3", "qwen3_moe"}:
        for marker in ("0.6B", "1.7B", "4B", "8B", "14B", "30B", "32B", "235B"):
            if marker.lower() in lowered:
                return marker
    if family in {"glm4", "glm4_moe"}:
        if "4.7-flash" in lowered:
            return "4.7-flash"
        if lowered == "glm-5" or "glm-5" in lowered:
            return "5"
    return None


def normalize_hf_model_denotation(
    source: HFModelSource | str,
    *,
    checkpoint_format: CheckpointFormat = "hf_pretrained",
) -> ModelDenotation:
    """Normalize a HuggingFace-published model into our internal denotation."""
    if isinstance(source, str):
        source = HFModelSource(name_or_path=source)

    try:
        from transformers import AutoConfig
    except ImportError as e:
        raise ImportError(
            "transformers is required to normalize HF model semantics into ModelDenotation"
        ) from e

    hf_config = AutoConfig.from_pretrained(
        source.name_or_path,
        trust_remote_code=source.trust_remote_code,
        revision=source.revision,
    )
    family = _infer_model_family(hf_config, source)
    norm_kind: NormKind = (
        "rmsnorm" if getattr(hf_config, "rms_norm_eps", None) is not None else "layernorm"
    )
    norm_epsilon = float(
        getattr(hf_config, "rms_norm_eps", None)
        or getattr(hf_config, "layer_norm_eps", None)
        or getattr(hf_config, "norm_epsilon", None)
        or 1e-5
    )
    rope_theta = _infer_rope_theta(hf_config)
    rotary = None
    if rope_theta is not None:
        rope_kind: RotaryKind = (
            "yarn"
            if (
                isinstance(getattr(hf_config, "rope_scaling", None), dict)
                and getattr(hf_config, "rope_scaling", {}).get("rope_type") == "yarn"
            )
            else "rope"
        )
        scaling = {}
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            scaling = dict(rope_scaling)
        rotary = RotarySemantics(
            kind=rope_kind,
            theta=rope_theta,
            head_dim=getattr(hf_config, "head_dim", None),
            scaling=scaling,
        )

    moe = None
    num_experts = getattr(hf_config, "num_experts", None) or getattr(
        hf_config, "n_routed_experts", None
    )
    experts_per_token = getattr(hf_config, "num_experts_per_tok", None)
    if num_experts:
        moe = MoESemantics(
            num_experts=int(num_experts),
            experts_per_token=int(experts_per_token or 1),
            shared_expert=bool(
                getattr(hf_config, "n_shared_experts", 0)
                or getattr(hf_config, "num_shared_experts", 0)
            ),
        )

    architecture = ModelArchitectureSemantics(
        family=family,
        hidden_size=int(hf_config.hidden_size),
        num_layers=int(hf_config.num_hidden_layers),
        num_attention_heads=int(hf_config.num_attention_heads),
        num_kv_heads=getattr(hf_config, "num_key_value_heads", None),
        ffn_hidden_size=int(hf_config.intermediate_size),
        vocab_size=int(hf_config.vocab_size),
        max_sequence_length=int(hf_config.max_position_embeddings),
        norm=norm_kind,
        norm_epsilon=norm_epsilon,
        attention=_infer_attention_kind(hf_config, family),
        rotary=rotary,
        moe=moe,
        tie_embeddings=bool(getattr(hf_config, "tie_word_embeddings", True)),
        uses_qk_layernorm=bool(
            getattr(hf_config, "qk_layernorm", False)
            or getattr(hf_config, "use_qk_norm", False)
            or (family in {"qwen3", "qwen3_moe"})
        ),
        uses_bias_linear=bool(getattr(hf_config, "attention_bias", True)),
        metadata={
            "hf_config_class": type(hf_config).__name__,
            "architectures": tuple(getattr(hf_config, "architectures", ()) or ()),
            "model_type": getattr(hf_config, "model_type", None),
            "head_dim": getattr(hf_config, "head_dim", None),
        },
    )
    checkpoint = CheckpointSemantics(
        format=checkpoint_format,
        source=source,
        weight_tying=architecture.tie_embeddings,
    )
    return ModelDenotation(
        source=source,
        architecture=architecture,
        checkpoint=checkpoint,
        variant=_infer_variant(source, family),
    )
