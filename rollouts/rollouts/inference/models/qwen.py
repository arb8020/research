"""Qwen model implementations for inference.

Supports:
- Qwen2ForCausalLM
- Qwen3ForCausalLM
- Qwen3MoeForCausalLM (MoE variant)

TODO: Implement these models. Reference:
- /tmp/mini-sglang/python/minisgl/models/qwen2.py
- /tmp/mini-sglang/python/minisgl/models/qwen3.py
- /tmp/mini-sglang/python/minisgl/models/qwen3_moe.py

Qwen models are very similar to Llama with minor differences:
- Qwen2: Similar to Llama, uses RMSNorm, RoPE, GQA
- Qwen3: Adds QK-Norm (optional), different RoPE scaling
- Qwen3-MoE: Mixture of experts with shared experts

Architecture differences from Llama:
1. Embedding: May have different vocab size
2. Attention: QK-Norm in Qwen3, otherwise identical
3. MLP: Identical structure (SwiGLU)
4. MoE (Qwen3-MoE only): Top-k routing with shared experts
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from ..attention.backend import AttentionBackend, AttentionMetadata
from .config import ModelConfig

if TYPE_CHECKING:
    from ..tp import TPConfig


class Qwen2ForCausalLM(nn.Module):
    """Qwen2 model for causal language modeling.

    TODO: Implement. Structure is nearly identical to LlamaForCausalLM.
    Main difference: config loading and weight names.

    Reference: /tmp/mini-sglang/python/minisgl/models/qwen2.py
    """

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        raise NotImplementedError("TODO: Implement Qwen2ForCausalLM")

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        """Forward pass returning logits for last token of each sequence."""
        raise NotImplementedError("TODO: Implement Qwen2 forward")

    def load_weights(self, weights: dict[str, Tensor]) -> None:
        """Load weights from state dict."""
        raise NotImplementedError("TODO: Implement Qwen2 weight loading")


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 model for causal language modeling.

    TODO: Implement. Differences from Qwen2:
    - Optional QK-Norm (query-key normalization before attention)
    - Different RoPE scaling parameters

    Reference: /tmp/mini-sglang/python/minisgl/models/qwen3.py
    """

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        raise NotImplementedError("TODO: Implement Qwen3ForCausalLM")

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        """Forward pass returning logits for last token of each sequence."""
        raise NotImplementedError("TODO: Implement Qwen3 forward")

    def load_weights(self, weights: dict[str, Tensor]) -> None:
        """Load weights from state dict."""
        raise NotImplementedError("TODO: Implement Qwen3 weight loading")


class Qwen3MoeForCausalLM(nn.Module):
    """Qwen3-MoE model for causal language modeling.

    TODO: Implement. This is the MoE variant with:
    - Multiple expert MLPs per layer
    - Top-k routing (typically top-2)
    - Shared experts that always activate
    - Auxiliary load balancing loss

    For RL training, we need to track routed_experts for routing replay.

    Reference: /tmp/mini-sglang/python/minisgl/models/qwen3_moe.py
    """

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        raise NotImplementedError("TODO: Implement Qwen3MoeForCausalLM")

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
        return_routed_experts: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass returning logits (and optionally routing info).

        Args:
            input_ids: Input token IDs
            positions: Position IDs for RoPE
            attn_backend: Attention backend
            attn_metadata: Attention metadata
            out_loc: KV cache slots
            return_routed_experts: If True, return (logits, routed_experts)

        Returns:
            logits, or (logits, routed_experts) if return_routed_experts=True
        """
        raise NotImplementedError("TODO: Implement Qwen3-MoE forward")

    def load_weights(self, weights: dict[str, Tensor]) -> None:
        """Load weights from state dict."""
        raise NotImplementedError("TODO: Implement Qwen3-MoE weight loading")


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT REMAPPING
# ═══════════════════════════════════════════════════════════════════════════════


def remap_weights_qwen(
    hf_weights: dict[str, Tensor],
    num_layers: int,
    model_type: str = "qwen2",
) -> dict[str, Tensor]:
    """Remap HuggingFace Qwen weights to our naming convention.

    TODO: Implement weight name mapping from HF format.
    Similar to remap_weights_llama but with Qwen-specific names.

    Args:
        hf_weights: Weights from HuggingFace checkpoint
        num_layers: Number of transformer layers
        model_type: "qwen2", "qwen3", or "qwen3_moe"

    Returns:
        Remapped weights dict
    """
    raise NotImplementedError("TODO: Implement Qwen weight remapping")
