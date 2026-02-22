"""Mixture of Experts (MoE) layer implementations.

Supports:
- Standard MoE with top-k routing
- Shared experts (always active)
- Fused MoE kernels for performance

TODO: Implement MoE layers. Reference:
- /tmp/mini-sglang/python/minisgl/layers/moe.py
- /tmp/mini-sglang/python/minisgl/moe/fused.py
- /tmp/mini-sglang/python/minisgl/kernel/triton/fused_moe.py

For RL training, we need to track which experts were routed to
for each token (return_routed_experts). This enables routing replay
during training to ensure consistent expert selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from ..tp import TPConfig


class MoeGate(nn.Module):
    """Router/gating network for MoE layer.

    Computes routing probabilities and selects top-k experts per token.

    TODO: Implement:
    1. Linear layer: hidden_size -> num_experts
    2. Softmax over experts
    3. Top-k selection
    4. Optional load balancing auxiliary loss
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        raise NotImplementedError("TODO: Implement MoeGate")

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute routing for input tokens.

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            topk_weights: [num_tokens, top_k] - routing weights
            topk_indices: [num_tokens, top_k] - selected expert indices
            router_logits: [num_tokens, num_experts] - raw router logits (for aux loss)
        """
        raise NotImplementedError("TODO: Implement MoeGate forward")


class MoeExperts(nn.Module):
    """Collection of expert MLPs.

    Each expert is a standard MLP (gate_proj, up_proj, down_proj with SwiGLU).

    TODO: Implement:
    1. Store num_experts separate MLPs
    2. Efficient batched expert computation
    3. Optional tensor parallelism (experts sharded across GPUs)
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        raise NotImplementedError("TODO: Implement MoeExperts")

    def forward(
        self,
        hidden_states: Tensor,
        topk_weights: Tensor,
        topk_indices: Tensor,
    ) -> Tensor:
        """Route tokens to experts and combine outputs.

        Args:
            hidden_states: [num_tokens, hidden_size]
            topk_weights: [num_tokens, top_k]
            topk_indices: [num_tokens, top_k]

        Returns:
            [num_tokens, hidden_size]
        """
        raise NotImplementedError("TODO: Implement MoeExperts forward")


class MoeLayer(nn.Module):
    """Full MoE layer with routing and experts.

    Combines MoeGate + MoeExperts + optional shared experts.

    TODO: Implement:
    1. Router to select experts
    2. Expert computation
    3. Shared experts (if configured)
    4. Combine routed + shared expert outputs
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        num_shared_experts: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        raise NotImplementedError("TODO: Implement MoeLayer")

    def forward(
        self,
        hidden_states: Tensor,
        return_routed_experts: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass through MoE layer.

        Args:
            hidden_states: [num_tokens, hidden_size]
            return_routed_experts: If True, also return routing decisions

        Returns:
            output: [num_tokens, hidden_size]
            routed_experts: [num_tokens, top_k] expert indices (if return_routed_experts)
        """
        raise NotImplementedError("TODO: Implement MoeLayer forward")


# ═══════════════════════════════════════════════════════════════════════════════
# FUSED MOE KERNEL
# ═══════════════════════════════════════════════════════════════════════════════


def fused_moe(
    hidden_states: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    topk_weights: Tensor,
    topk_indices: Tensor,
) -> Tensor:
    """Fused MoE kernel for efficient expert computation.

    TODO: Implement using Triton kernel.
    Reference: /tmp/mini-sglang/python/minisgl/kernel/triton/fused_moe.py

    This fuses:
    1. Gather tokens for each expert
    2. Expert MLP forward (gate_proj, up_proj, down_proj, SwiGLU)
    3. Scatter-add results weighted by routing

    Much faster than naive for-loop over experts.

    Args:
        hidden_states: [num_tokens, hidden_size]
        gate_weight: [num_experts, intermediate_size, hidden_size]
        up_weight: [num_experts, intermediate_size, hidden_size]
        down_weight: [num_experts, hidden_size, intermediate_size]
        topk_weights: [num_tokens, top_k]
        topk_indices: [num_tokens, top_k]

    Returns:
        [num_tokens, hidden_size]
    """
    raise NotImplementedError("TODO: Implement fused MoE kernel")
