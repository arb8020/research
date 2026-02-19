"""Online statistics and distance metrics for REAP.

Uses Welford's algorithm for numerically stable streaming mean/variance.
"""

from __future__ import annotations

import torch
from torch import Tensor


class OnlineStatsTracker:
    """Track running mean using Welford's algorithm.

    Numerically stable for streaming updates without storing all values.
    Uses Kahan summation to reduce floating-point error accumulation.
    """

    def __init__(self, shape: tuple[int, ...], device: torch.device) -> None:
        self.mean = torch.zeros(shape, device=device)
        self.count = torch.zeros(shape, device=device)
        # Kahan summation compensation terms
        self._comp = torch.zeros(shape, device=device)

    def update(self, values: Tensor, mask: Tensor | None = None) -> None:
        """Update running mean with new values.

        Args:
            values: New values to incorporate, shape matches self.mean
            mask: Boolean mask indicating which elements to update
        """
        if mask is None:
            mask = torch.ones_like(values, dtype=torch.bool)

        # Welford's online update: mean_new = mean + (x - mean) / n
        new_count = self.count + mask.float()
        # Avoid division by zero
        safe_count = torch.where(new_count > 0, new_count, torch.ones_like(new_count))

        delta = values - self.mean
        # Kahan summation for the delta accumulation
        y = delta * mask.float() - self._comp
        t = self.mean + y / safe_count * mask.float()
        self._comp = (t - self.mean) - y / safe_count * mask.float()

        self.mean = torch.where(mask, t, self.mean)
        self.count = new_count

    def get_mean(self) -> Tensor:
        """Return current running mean."""
        return self.mean

    def get_count(self) -> Tensor:
        """Return count of updates per element."""
        return self.count


def angular_distance(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute angular distance between vectors.

    Returns value in [0, 1] where 0 = identical direction, 1 = opposite.

    Args:
        a: First tensor, shape [..., dim]
        b: Second tensor, shape [..., dim]
        eps: Small value for numerical stability

    Returns:
        Angular distance, shape [...]
    """
    # Normalize to unit vectors
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)

    # Cosine similarity, clamped to valid range for acos
    cos_sim = (a_norm * b_norm).sum(dim=-1)
    cos_sim = cos_sim.clamp(-1 + eps, 1 - eps)

    # Angular distance normalized to [0, 1]
    return torch.acos(cos_sim) / torch.pi


def cosine_distance(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute cosine distance between vectors.

    Returns 1 - cosine_similarity, in range [0, 2].

    Args:
        a: First tensor, shape [..., dim]
        b: Second tensor, shape [..., dim]
        eps: Small value for numerical stability

    Returns:
        Cosine distance, shape [...]
    """
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    cos_sim = (a_norm * b_norm).sum(dim=-1)
    return 1 - cos_sim


def compute_reap_scores(
    activation_norms: Tensor,
    routing_weights: Tensor,
    expert_counts: Tensor,
) -> Tensor:
    """Compute REAP scores for each expert.

    REAP score = mean(activation_norm * routing_weight) over routed tokens.
    Higher score = more important expert.

    Args:
        activation_norms: Sum of L2 norms per expert, shape [num_experts]
        routing_weights: Sum of routing weights per expert, shape [num_experts]
        expert_counts: Number of tokens routed to each expert, shape [num_experts]

    Returns:
        REAP scores, shape [num_experts]
    """
    # Avoid division by zero for unused experts
    safe_counts = torch.where(expert_counts > 0, expert_counts, torch.ones_like(expert_counts))

    # REAP = mean(EAN * routing_weight)
    # We tracked sums, so divide by count
    mean_ean = activation_norms / safe_counts
    mean_routing = routing_weights / safe_counts

    scores = mean_ean * mean_routing

    # Zero out scores for unused experts
    scores = torch.where(expert_counts > 0, scores, torch.zeros_like(scores))

    return scores
