"""Megatron to HuggingFace weight conversion.

Converts Megatron state dict keys to HuggingFace format for weight sync
to SGLang inference. Each model architecture has different key mappings.

Ported from SLIME's megatron_to_hf/ with minimal changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .deepseekv3 import convert_deepseekv3_to_hf

# Model name to converter mapping
_CONVERTERS = {
    "glm4moelite": convert_deepseekv3_to_hf,
    "deepseekv3": convert_deepseekv3_to_hf,
    "glm-4.7": convert_deepseekv3_to_hf,  # GLM-4.7-Flash uses deepseekv3 arch
}


def convert_megatron_to_hf(
    model_name: str,
    name: str,
    param: torch.Tensor,
    *,
    vocab_size: int,
    num_layers: int,
    num_attention_heads: int,
    hidden_size: int,
    num_query_groups: int | None = None,
    kv_channels: int | None = None,
    q_lora_rank: int | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Convert a single Megatron parameter to HuggingFace format.

    Args:
        model_name: Model identifier (e.g., "glm4moelite", "deepseekv3")
        name: Megatron parameter name
        param: Parameter tensor
        vocab_size: Vocabulary size (for padding removal)
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        hidden_size: Hidden dimension
        num_query_groups: Number of query groups for GQA
        kv_channels: KV head dimension

    Returns:
        List of (hf_name, tensor) tuples. One Megatron param may map to
        multiple HF params (e.g., fused QKV splits into Q, K, V).

    Raises:
        ValueError: If model architecture is not supported
    """
    # Normalize model name
    model_name_lower = model_name.lower()

    # Find converter
    converter = None
    for key, conv in _CONVERTERS.items():
        if key in model_name_lower:
            converter = conv
            break

    if converter is None:
        raise ValueError(
            f"Unsupported model for weight conversion: {model_name}. "
            f"Supported: {list(_CONVERTERS.keys())}"
        )

    # Build args namespace (SLIME converters expect this format)
    class Args:
        pass

    args = Args()
    args.vocab_size = vocab_size
    args.num_layers = num_layers
    args.num_attention_heads = num_attention_heads
    args.hidden_size = hidden_size
    args.num_query_groups = num_query_groups or num_attention_heads
    args.kv_channels = kv_channels
    args.q_lora_rank = q_lora_rank

    return converter(args, name, param)


def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Remove vocabulary padding from embedding/output layers.

    Megatron pads vocab to be divisible by TP size. This removes the padding.

    Args:
        name: Parameter name
        param: Parameter tensor
        vocab_size: Actual vocabulary size

    Returns:
        Tensor with padding removed (if applicable)
    """
    if "word_embeddings" in name or "output_layer" in name:
        if param.shape[0] > vocab_size:
            return param[:vocab_size]
    return param
