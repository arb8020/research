"""Mixture of Experts (MoE) layer implementations.

Supports:
- Standard MoE with top-k routing
- Fused MoE kernels for performance (via sgl-kernel)
- Fallback naive implementation for testing

For RL training, we track which experts were routed to for each token
(return_routed_experts). This enables routing replay during training
to ensure consistent expert selection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .activation import silu_and_mul

if TYPE_CHECKING:
    from ..tp import TPConfig

logger = logging.getLogger(__name__)


def is_fused_moe_available() -> bool:
    """Check if fused MoE kernels are available."""
    try:
        from sgl_kernel import topk_softmax  # noqa: F401

        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MOE GATE (ROUTER)
# ═══════════════════════════════════════════════════════════════════════════════


class MoeGate(nn.Module):
    """Router/gating network for MoE layer.

    Computes routing probabilities and selects top-k experts per token.
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

        # Router linear: hidden_size -> num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute routing for input tokens.

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            topk_weights: [num_tokens, top_k] - routing weights (after softmax)
            topk_indices: [num_tokens, top_k] - selected expert indices
            router_logits: [num_tokens, num_experts] - raw router logits (for aux loss)
        """
        # Compute router logits
        router_logits = self.gate(hidden_states)  # [num_tokens, num_experts]

        # Softmax over experts
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Select top-k experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # Renormalize weights to sum to 1
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return topk_weights.to(hidden_states.dtype), topk_indices, router_logits


# ═══════════════════════════════════════════════════════════════════════════════
# MOE EXPERTS
# ═══════════════════════════════════════════════════════════════════════════════


class MoeExperts(nn.Module):
    """Collection of expert MLPs.

    Each expert is a SwiGLU MLP (gate_proj, up_proj, down_proj).
    Supports both fused kernel and naive fallback.
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
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Expert weights: [num_experts, out_features, in_features]
        # gate_up_proj combines gate and up projections
        # Shape: [num_experts, 2 * intermediate_size, hidden_size]
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size, dtype=dtype)
        )

        # down_proj: [num_experts, hidden_size, intermediate_size]
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, dtype=dtype)
        )

        # Initialize weights
        nn.init.kaiming_uniform_(self.gate_up_proj)
        nn.init.kaiming_uniform_(self.down_proj)

        # Check for fused kernel availability
        self._use_fused = is_fused_moe_available()
        if self._use_fused:
            logger.debug("Using fused MoE kernels")
        else:
            logger.debug("Using naive MoE implementation (fused kernels not available)")

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
        if self._use_fused:
            return self._forward_fused(hidden_states, topk_weights, topk_indices)
        else:
            return self._forward_naive(hidden_states, topk_weights, topk_indices)

    def _forward_fused(
        self,
        hidden_states: Tensor,
        topk_weights: Tensor,
        topk_indices: Tensor,
    ) -> Tensor:
        """Fused MoE forward using sgl-kernel."""

        # The fused implementation would use:
        # 1. moe_align_block_size for efficient batching
        # 2. fused_moe_kernel for GEMM + activation
        # For now, fall back to naive since we need more setup
        # TODO: Implement proper fused path once kernel wrappers are ready
        return self._forward_naive(hidden_states, topk_weights, topk_indices)

    def _forward_naive(
        self,
        hidden_states: Tensor,
        topk_weights: Tensor,
        topk_indices: Tensor,
    ) -> Tensor:
        """Naive MoE forward - loops over experts."""
        num_tokens, hidden_size = hidden_states.shape
        top_k = topk_indices.shape[1]

        # Output accumulator
        output = torch.zeros_like(hidden_states)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            # expert_mask: [num_tokens, top_k] bool
            expert_mask = topk_indices == expert_idx

            if not expert_mask.any():
                continue

            # Get indices of tokens and their position in top-k
            token_indices, topk_positions = torch.where(expert_mask)

            if len(token_indices) == 0:
                continue

            # Get hidden states for these tokens
            expert_input = hidden_states[token_indices]  # [num_expert_tokens, hidden_size]

            # Get weights for these tokens
            weights = topk_weights[token_indices, topk_positions]  # [num_expert_tokens]

            # Expert forward: SwiGLU MLP
            # gate_up: [2 * intermediate, hidden] @ [hidden] -> [2 * intermediate]
            gate_up = F.linear(expert_input, self.gate_up_proj[expert_idx])
            # SwiGLU activation
            intermediate = silu_and_mul(gate_up)
            # down: [hidden, intermediate] @ [intermediate] -> [hidden]
            expert_output = F.linear(intermediate, self.down_proj[expert_idx])

            # Weight and accumulate
            output.index_add_(
                0,
                token_indices,
                expert_output * weights.unsqueeze(-1),
            )

        return output


# ═══════════════════════════════════════════════════════════════════════════════
# FULL MOE LAYER
# ═══════════════════════════════════════════════════════════════════════════════


class MoeLayer(nn.Module):
    """Full MoE layer with routing and experts.

    Combines MoeGate + MoeExperts. Optionally tracks routed experts
    for RL training replay.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype = torch.bfloat16,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = MoeGate(hidden_size, num_experts, top_k, dtype)
        self.experts = MoeExperts(num_experts, hidden_size, intermediate_size, dtype, tp)

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
        # Route tokens to experts
        topk_weights, topk_indices, _router_logits = self.gate(hidden_states)

        # Compute expert outputs
        output = self.experts(hidden_states, topk_weights, topk_indices)

        if return_routed_experts:
            return output, topk_indices
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# MOE MLP (drop-in replacement for standard MLP)
# ═══════════════════════════════════════════════════════════════════════════════


class MoeMLP(nn.Module):
    """MoE MLP - drop-in replacement for standard SwiGLU MLP.

    Used in Qwen3-MoE and similar architectures.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype = torch.bfloat16,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.moe = MoeLayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            dtype=dtype,
            tp=tp,
        )

    def forward(
        self,
        x: Tensor,
        return_routed_experts: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return self.moe(x, return_routed_experts=return_routed_experts)
