"""Model configuration.

Loads config from HuggingFace model and extracts relevant parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..layers.rotary import RotaryConfig


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration extracted from HuggingFace config.

    All parameters needed to construct the model.
    """

    # Model architecture
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int

    # Activation
    hidden_act: str  # "silu" for Llama

    # Rotary embeddings
    rotary_config: RotaryConfig

    # Misc
    tie_word_embeddings: bool
    model_type: str  # "llama", "qwen2", etc.

    # MoE config (optional, only for MoE models like Qwen3-MoE)
    num_experts: int | None = None
    num_experts_per_tok: int | None = None  # top-k
    moe_intermediate_size: int | None = None  # expert intermediate size


def load_model_config(model_path: str) -> ModelConfig:
    """Load model config from HuggingFace model path.

    Args:
        model_path: HuggingFace model name or local path

    Returns:
        ModelConfig with all parameters
    """
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Extract head dimension
    head_dim = getattr(
        hf_config,
        "head_dim",
        hf_config.hidden_size // hf_config.num_attention_heads,
    )

    # Build rotary config
    rotary_dim = getattr(hf_config, "rotary_dim", head_dim)
    rope_base = getattr(hf_config, "rope_theta", 10000.0)
    rope_scaling = getattr(hf_config, "rope_scaling", None)

    scaling_type = None
    scaling_factor = 1.0
    if rope_scaling is not None:
        scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
        scaling_factor = rope_scaling.get("factor", 1.0)

    rotary_config = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=hf_config.max_position_embeddings,
        base=rope_base,
        scaling_type=scaling_type,
        scaling_factor=scaling_factor,
    )

    # MoE config (for models like Qwen3-MoE, Mixtral)
    num_experts = getattr(hf_config, "num_experts", None)
    num_experts_per_tok = getattr(hf_config, "num_experts_per_tok", None)
    moe_intermediate_size = getattr(hf_config, "moe_intermediate_size", None)

    # Some models use different naming
    if num_experts is None:
        num_experts = getattr(hf_config, "num_local_experts", None)
    if num_experts_per_tok is None:
        num_experts_per_tok = getattr(hf_config, "num_selected_experts", None)

    return ModelConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        ),
        head_dim=head_dim,
        rms_norm_eps=hf_config.rms_norm_eps,
        max_position_embeddings=hf_config.max_position_embeddings,
        hidden_act=getattr(hf_config, "hidden_act", "silu"),
        rotary_config=rotary_config,
        tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
        model_type=hf_config.model_type,
        # MoE
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
    )
