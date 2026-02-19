"""Expert pruning logic for MoE models.

Removes low-scoring experts and updates router accordingly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .config import PruneMethod, ReapConfig
from .metrics import compute_reap_scores
from .metrics_extended import (
    compute_ean_ca_scores,
    compute_reap_l2_scores,
    compute_weighted_ean_sum_l2_scores,
    compute_weighted_frequency_scores,
)
from .observer import LayerObservation, MoEObserver

logger = logging.getLogger(__name__)


@dataclass
class PruneResult:
    """Result of pruning operation."""

    original_num_experts: int
    pruned_num_experts: int
    pruned_expert_indices: dict[int, list[int]]  # layer_idx -> list of pruned indices
    reap_scores: dict[int, Tensor]  # layer_idx -> scores before pruning


def compute_layer_scores(
    obs: LayerObservation,
    method: PruneMethod,
) -> Tensor:
    """Compute expert scores for a single layer.

    Higher score = more important expert.
    """
    if method == PruneMethod.REAP:
        return compute_reap_scores(
            activation_norms=obs.ean_sum,
            routing_weights=obs.routing_weight_sum,
            expert_counts=obs.expert_frequency,
        )
    elif method == PruneMethod.FREQUENCY:
        return obs.expert_frequency.float()
    elif method == PruneMethod.EAN_MEAN:
        safe_counts = torch.where(
            obs.expert_frequency > 0,
            obs.expert_frequency,
            torch.ones_like(obs.expert_frequency),
        )
        return obs.ean_sum / safe_counts
    elif method == PruneMethod.EAN_SUM:
        return obs.ean_sum
    elif method == PruneMethod.EAN_CA:
        return compute_ean_ca_scores(obs.characteristic_activation)
    elif method == PruneMethod.WEIGHTED_FREQUENCY:
        return compute_weighted_frequency_scores(obs.weighted_expert_frequency_sum)
    elif method == PruneMethod.WEIGHTED_EAN_SUM:
        safe_counts = torch.where(
            obs.expert_frequency > 0,
            obs.expert_frequency,
            torch.ones_like(obs.expert_frequency),
        )
        return obs.weighted_ean_sum / safe_counts
    elif method == PruneMethod.REAP_L2:
        return compute_reap_l2_scores(
            activation_norms=obs.ean_sum,
            routing_weights=obs.routing_weight_sum,
            expert_counts=obs.expert_frequency,
        )
    elif method == PruneMethod.WEIGHTED_EAN_SUM_L2:
        return compute_weighted_ean_sum_l2_scores(
            weighted_ean_sum=obs.weighted_ean_sum,
            expert_counts=obs.expert_frequency,
        )
    elif method == PruneMethod.MAX_ACTIVATIONS:
        return obs.max_activations
    else:
        raise ValueError(f"Unknown prune method: {method}")


def get_super_expert_indices(
    observations: dict[int, LayerObservation],
    percentile: float = 99.5,
) -> set[tuple[int, int]]:
    """Identify super-experts with outlier max activations.

    Returns set of (layer_idx, expert_idx) pairs that should be preserved.
    """
    # Collect all max activations across layers
    all_max_acts = []
    for obs in observations.values():
        all_max_acts.append(obs.max_activations)
    all_max_acts = torch.cat(all_max_acts)

    # Compute threshold
    threshold = torch.quantile(all_max_acts, percentile / 100.0)

    # Find experts above threshold
    super_experts = set()
    for layer_idx, obs in observations.items():
        for expert_idx in range(len(obs.max_activations)):
            if obs.max_activations[expert_idx] > threshold:
                super_experts.add((layer_idx, expert_idx))

    logger.info(f"Found {len(super_experts)} super-experts above {percentile}th percentile")
    return super_experts


def select_experts_to_prune(
    observations: dict[int, LayerObservation],
    config: ReapConfig,
) -> dict[int, list[int]]:
    """Select which experts to prune in each layer.

    Returns mapping from layer_idx to list of expert indices to remove.
    """
    # Get super-experts if preservation is enabled
    super_experts = set()
    if config.preserve_super_experts:
        super_experts = get_super_expert_indices(observations)

    pruned_indices: dict[int, list[int]] = {}

    for layer_idx, obs in observations.items():
        num_experts = len(obs.expert_frequency)
        num_to_prune = int(num_experts * config.compression_ratio)

        # Compute scores
        scores = compute_layer_scores(obs, config.prune_method)

        # Sort experts by score (ascending = lowest first)
        sorted_indices = torch.argsort(scores)

        # Select experts to prune, respecting super-expert preservation
        to_prune = []
        for idx in sorted_indices.tolist():
            if len(to_prune) >= num_to_prune:
                break
            if (layer_idx, idx) not in super_experts:
                to_prune.append(idx)

        pruned_indices[layer_idx] = to_prune
        logger.info(f"Layer {layer_idx}: pruning {len(to_prune)}/{num_experts} experts")

    return pruned_indices


def prune_moe_layer(
    moe_block: nn.Module,
    experts_attr: str,
    router_attr: str,
    indices_to_remove: list[int],
) -> None:
    """Remove experts from a single MoE layer.

    Modifies the layer in-place:
    1. Remove expert weight slices (for fused experts) or modules (for ModuleList)
    2. Remove corresponding rows from router weight matrix
    """
    experts = getattr(moe_block, experts_attr)
    router = getattr(moe_block, router_attr)

    # Determine original number of experts
    if hasattr(experts, "num_experts"):
        # Fused experts (e.g., Qwen3MoeExperts)
        original_num = experts.num_experts
    elif hasattr(experts, "__len__"):
        # ModuleList
        original_num = len(experts)
    else:
        raise TypeError(f"Cannot determine number of experts from {type(experts).__name__}")

    indices_to_keep = [i for i in range(original_num) if i not in indices_to_remove]

    # Handle fused expert weights (Qwen3MoeExperts style)
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        # Fused: gate_up_proj [num_experts, intermediate, hidden], down_proj [num_experts, hidden, intermediate/2]
        with torch.no_grad():
            experts.gate_up_proj = nn.Parameter(experts.gate_up_proj[indices_to_keep])
            experts.down_proj = nn.Parameter(experts.down_proj[indices_to_keep])
        experts.num_experts = len(indices_to_keep)
    elif isinstance(experts, nn.ModuleList):
        # Create new ModuleList with remaining experts
        new_experts = nn.ModuleList([experts[i] for i in indices_to_keep])
        setattr(moe_block, experts_attr, new_experts)
    else:
        raise TypeError(f"Don't know how to prune experts of type {type(experts).__name__}")

    # Update router weights
    # Router typically has weight of shape [num_experts, hidden_dim] or [hidden_dim, num_experts]
    if hasattr(router, "weight"):
        weight = router.weight.data
        if weight.shape[0] == original_num:
            # Shape is [num_experts, hidden_dim]
            new_weight = weight[indices_to_keep]
            router.weight = nn.Parameter(new_weight)
            router.out_features = len(indices_to_keep)
        elif weight.shape[1] == original_num:
            # Shape is [hidden_dim, num_experts]
            new_weight = weight[:, indices_to_keep]
            router.weight = nn.Parameter(new_weight)
            router.out_features = len(indices_to_keep)

        if hasattr(router, "bias") and router.bias is not None:
            new_bias = router.bias.data[indices_to_keep]
            router.bias = nn.Parameter(new_bias)


def prune_model(
    model: nn.Module,
    observer: MoEObserver,
    config: ReapConfig,
) -> PruneResult:
    """Prune experts from all MoE layers.

    Modifies model in-place.
    """
    observations = observer.get_observations()
    pruned_indices = select_experts_to_prune(observations, config)

    # Compute scores for result tracking
    reap_scores = {
        layer_idx: compute_layer_scores(obs, config.prune_method)
        for layer_idx, obs in observations.items()
    }

    # Get MoE layers
    moe_layers = observer._get_moe_layers()
    original_num_experts = observer.moe_config.num_experts

    for layer_idx, moe_block in moe_layers:
        if layer_idx in pruned_indices:
            prune_moe_layer(
                moe_block,
                observer.moe_config.experts_attr,
                observer.moe_config.router_attr,
                pruned_indices[layer_idx],
            )

    # Update model config
    new_num_experts = original_num_experts - int(original_num_experts * config.compression_ratio)
    _update_model_config(model, new_num_experts)

    return PruneResult(
        original_num_experts=original_num_experts,
        pruned_num_experts=new_num_experts,
        pruned_expert_indices=pruned_indices,
        reap_scores=reap_scores,
    )


def _update_model_config(model: nn.Module, new_num_experts: int) -> None:
    """Update model config to reflect new expert count."""
    config = model.config

    # Different models use different attribute names
    if hasattr(config, "num_experts"):
        config.num_experts = new_num_experts
    elif hasattr(config, "num_local_experts"):
        config.num_local_experts = new_num_experts
    elif hasattr(config, "n_routed_experts"):
        config.n_routed_experts = new_num_experts
    else:
        logger.warning("Could not find num_experts attribute in model config")
