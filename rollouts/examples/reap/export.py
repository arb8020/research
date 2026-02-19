"""Export pruned model to HuggingFace format."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_pruned_model(
    model: nn.Module,
    tokenizer: Any,
    output_dir: Path,
    config_overrides: dict | None = None,
) -> None:
    """Save pruned model in HuggingFace format.

    Args:
        model: The pruned model
        tokenizer: Associated tokenizer
        output_dir: Directory to save to
        config_overrides: Additional config values to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving pruned model to {output_dir}")

    # Save model weights
    model.save_pretrained(output_dir, safe_serialization=True)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Save additional metadata
    if config_overrides:
        metadata_path = output_dir / "reap_config.json"
        with open(metadata_path, "w") as f:
            json.dump(config_overrides, f, indent=2, default=str)
        logger.info(f"Saved REAP config to {metadata_path}")

    logger.info(f"Model saved to {output_dir}")


def save_pruning_recipe(
    base_model: str,
    experts_to_keep: dict[int, list[int]],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    """Save lightweight pruning recipe instead of full model weights.

    The recipe can be applied to the base model at load time to get the pruned model.
    This saves ~30GB of disk space compared to saving full model weights.

    Args:
        base_model: HuggingFace model ID (e.g., "Qwen/Qwen3-30B-A3B")
        experts_to_keep: Mapping from layer index to list of expert indices to retain
        output_path: Path to save the recipe JSON
        config: Additional metadata (compression_ratio, method, seed, etc.)
    """
    recipe = {
        "base_model": base_model,
        "experts_to_keep": {str(k): v for k, v in experts_to_keep.items()},
        **config,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(recipe, f, indent=2)

    logger.info(f"Saved pruning recipe to {output_path}")


def load_pruning_recipe(recipe_path: Path) -> dict[str, Any]:
    """Load a pruning recipe from disk."""
    with open(recipe_path) as f:
        recipe = json.load(f)

    # Convert string keys back to ints
    recipe["experts_to_keep"] = {int(k): v for k, v in recipe["experts_to_keep"].items()}
    return recipe


def apply_pruning_recipe(
    model: nn.Module,
    experts_to_keep: dict[int, list[int]],
) -> None:
    """Apply a pruning recipe to a model in-place.

    Args:
        model: The loaded model to prune
        experts_to_keep: Mapping from layer index to list of expert indices to retain
    """
    from .observer import detect_moe_config

    moe_config = detect_moe_config(model)

    for layer_idx, keep_indices in experts_to_keep.items():
        # Get the MoE block for this layer
        layer = model.model.layers[layer_idx]
        moe_block = getattr(layer, moe_config.block_attr)
        experts = getattr(moe_block, moe_config.experts_attr)
        router = getattr(moe_block, moe_config.router_attr)

        # Prune fused experts
        if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
            with torch.no_grad():
                experts.gate_up_proj = nn.Parameter(experts.gate_up_proj[keep_indices])
                experts.down_proj = nn.Parameter(experts.down_proj[keep_indices])
            if hasattr(experts, "num_experts"):
                experts.num_experts = len(keep_indices)
        else:
            # ModuleList style
            new_experts = nn.ModuleList([experts[i] for i in keep_indices])
            setattr(moe_block, moe_config.experts_attr, new_experts)

        # Prune router
        if hasattr(router, "weight"):
            original_num = router.weight.shape[0]
            if original_num == len(keep_indices) + (moe_config.num_experts - len(keep_indices)):
                router.weight = nn.Parameter(router.weight[keep_indices])
                router.out_features = len(keep_indices)
            if hasattr(router, "bias") and router.bias is not None:
                router.bias = nn.Parameter(router.bias[keep_indices])

    logger.info(f"Applied pruning recipe: kept {len(keep_indices)} experts per layer")


def get_output_path(
    base_dir: Path,
    model_name: str,
    dataset_name: str,
    prune_method: str,
    compression_ratio: float,
    seed: int,
) -> Path:
    """Generate output path for pruned model.

    Format: {base_dir}/{model_name}/{dataset_name}/{method}-seed_{seed}-{ratio}/
    """
    # Clean names for filesystem
    model_clean = model_name.replace("/", "_")
    dataset_clean = dataset_name.replace("/", "_")

    return (
        base_dir
        / model_clean
        / dataset_clean
        / f"{prune_method}-seed_{seed}-{compression_ratio:.2f}"
    )
