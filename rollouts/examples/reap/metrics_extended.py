"""Extended pruning metrics for REAP.

Adds missing pruning criteria from Cerebras REAP:
- ean_ca: Expert Activation Norm with Characteristic Activation
- weighted_frequency_sum: Router-weighted frequency
- reap_l2: REAP with L2 normalization
- weighted_ean_sum_l2: Weighted EAN with L2 normalization
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_ean_ca_scores(
    characteristic_activations: Tensor,
) -> Tensor:
    """Compute EAN-CA (Expert Activation Norm - Characteristic Activation) scores.

    Characteristic activation is the mean activation per expert across all tokens.
    This captures the "signature" activation pattern of each expert.

    Args:
        characteristic_activations: Mean activations per expert, shape [num_experts, hidden_dim]

    Returns:
        EAN-CA scores (L2 norm of characteristic activation), shape [num_experts]
    """
    return torch.linalg.norm(characteristic_activations, dim=-1)


def compute_weighted_frequency_scores(
    weighted_frequency_sum: Tensor,
) -> Tensor:
    """Compute weighted frequency scores.

    Simple sum of routing weights per expert. Experts that are selected
    with higher confidence (weight) score higher.

    Args:
        weighted_frequency_sum: Sum of routing weights per expert, shape [num_experts]

    Returns:
        Weighted frequency scores, shape [num_experts]
    """
    return weighted_frequency_sum


def compute_reap_l2_scores(
    activation_norms: Tensor,
    routing_weights: Tensor,
    expert_counts: Tensor,
) -> Tensor:
    """Compute REAP-L2 scores (REAP with L2 normalization).

    Similar to REAP but with L2 normalization on the activation norms
    before combining with routing weights.

    Args:
        activation_norms: Sum of L2 norms per expert, shape [num_experts]
        routing_weights: Sum of routing weights per expert, shape [num_experts]
        expert_counts: Number of tokens routed to each expert, shape [num_experts]

    Returns:
        REAP-L2 scores, shape [num_experts]
    """
    # Avoid division by zero
    safe_counts = torch.where(expert_counts > 0, expert_counts, torch.ones_like(expert_counts))

    # Mean EAN per expert
    mean_ean = activation_norms / safe_counts

    # L2 normalize the mean EANs across experts
    ean_l2 = mean_ean / (torch.linalg.norm(mean_ean) + 1e-8)

    # Mean routing weight per expert
    mean_routing = routing_weights / safe_counts

    # L2 normalize routing weights
    routing_l2 = mean_routing / (torch.linalg.norm(mean_routing) + 1e-8)

    # Combine (element-wise product, then normalize)
    scores = ean_l2 * routing_l2

    # Zero out unused experts
    scores = torch.where(expert_counts > 0, scores, torch.zeros_like(scores))

    return scores


def compute_weighted_ean_sum_l2_scores(
    weighted_ean_sum: Tensor,
    expert_counts: Tensor,
) -> Tensor:
    """Compute weighted EAN sum with L2 normalization.

    Args:
        weighted_ean_sum: Sum of (EAN * routing_weight) per expert, shape [num_experts]
        expert_counts: Number of tokens routed to each expert, shape [num_experts]

    Returns:
        Weighted EAN-L2 scores, shape [num_experts]
    """
    # Avoid division by zero
    safe_counts = torch.where(expert_counts > 0, expert_counts, torch.ones_like(expert_counts))

    # Mean weighted EAN
    mean_weighted_ean = weighted_ean_sum / safe_counts

    # L2 normalize
    scores = mean_weighted_ean / (torch.linalg.norm(mean_weighted_ean) + 1e-8)

    # Zero out unused experts
    scores = torch.where(expert_counts > 0, scores, torch.zeros_like(scores))

    return scores


def compute_max_activation_scores(
    max_activations: Tensor,
) -> Tensor:
    """Compute scores based on maximum activation values.

    Identifies "super-experts" that produce outlier activations.

    Args:
        max_activations: Maximum activation value per expert, shape [num_experts]

    Returns:
        Max activation scores, shape [num_experts]
    """
    return max_activations


# Registry of all pruning methods
PRUNING_METHODS = {
    "frequency": lambda obs: obs.expert_frequency.float(),
    "ean_sum": lambda obs: compute_ean_mean_scores(obs.ean_sum, obs.expert_frequency),
    "ean_mean": lambda obs: compute_ean_mean_scores(obs.ean_sum, obs.expert_frequency),
    "reap": lambda obs: compute_reap_scores(
        obs.ean_sum, obs.routing_weight_sum, obs.expert_frequency
    ),
    "ean_ca": lambda obs: compute_ean_ca_scores(obs.characteristic_activation),
    "weighted_frequency_sum": lambda obs: compute_weighted_frequency_scores(
        obs.weighted_expert_frequency_sum
    ),
    "weighted_ean_sum": lambda obs: obs.weighted_ean_sum
    / torch.where(obs.expert_counts > 0, obs.expert_counts, torch.ones_like(obs.expert_counts)),
    "reap_l2": lambda obs: compute_reap_l2_scores(
        obs.ean_sum, obs.routing_weight_sum, obs.expert_frequency
    ),
    "weighted_ean_sum_l2": lambda obs: compute_weighted_ean_sum_l2_scores(
        obs.weighted_ean_sum, obs.expert_frequency
    ),
    "max_activations": lambda obs: compute_max_activation_scores(obs.max_activations),
}


def compute_ean_mean_scores(
    ean_sum: Tensor,
    expert_counts: Tensor,
) -> Tensor:
    """Helper: compute mean EAN scores."""
    safe_counts = torch.where(expert_counts > 0, expert_counts, torch.ones_like(expert_counts))
    return ean_sum / safe_counts


def compute_reap_scores(
    activation_norms: Tensor,
    routing_weights: Tensor,
    expert_counts: Tensor,
) -> Tensor:
    """Original REAP score computation (from metrics.py)."""
    safe_counts = torch.where(expert_counts > 0, expert_counts, torch.ones_like(expert_counts))
    mean_ean = activation_norms / safe_counts
    mean_routing = routing_weights / safe_counts
    scores = mean_ean * mean_routing
    scores = torch.where(expert_counts > 0, scores, torch.zeros_like(scores))
    return scores


def get_pruning_scores(observation, method: str) -> Tensor:
    """Get pruning scores for a given method.

    Args:
        observation: LayerObservation dataclass with metric tensors
        method: One of PRUNING_METHODS keys

    Returns:
        Score tensor, shape [num_experts]
    """
    if method not in PRUNING_METHODS:
        raise ValueError(
            f"Unknown pruning method: {method}. Available: {list(PRUNING_METHODS.keys())}"
        )

    return PRUNING_METHODS[method](observation)
