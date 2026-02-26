"""Apply a pruning recipe to create a pruned model.

Script 2 in the pruning pipeline:
1. reap/score_experts.py - Generate recipe from activation analysis
2. reap/apply_recipe.py - Apply recipe to create pruned model (this script)
3. training config - Train the pruned model

Usage:
    # Apply recipe, save to local directory
    python examples/reap/apply_recipe.py \
        --recipe results/reap/glm4_flash_random_50/pruning_recipe.json \
        --output /path/to/pruned_model

    # Apply recipe, push to HuggingFace Hub
    python examples/reap/apply_recipe.py \
        --recipe results/reap/glm4_flash_random_50/pruning_recipe.json \
        --output hf://username/glm4-flash-pruned-50

    # Dry run (just validate recipe)
    python examples/reap/apply_recipe.py \
        --recipe results/reap/glm4_flash_random_50/pruning_recipe.json \
        --dry-run

Requires ~80GB VRAM to load full model. Run on Modal/RunPod with A100-80GB or H100.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_recipe(recipe_path: Path) -> dict:
    """Load and validate a pruning recipe."""
    with open(recipe_path) as f:
        recipe = json.load(f)

    required_keys = ["base_model", "experts_to_keep"]
    for key in required_keys:
        if key not in recipe:
            raise ValueError(f"Recipe missing required key: {key}")

    # Convert string layer indices to int
    recipe["experts_to_keep"] = {int(k): v for k, v in recipe["experts_to_keep"].items()}

    return recipe


def apply_pruning(model: nn.Module, experts_to_keep: dict[int, list[int]]) -> dict:
    """Apply pruning recipe to model in-place.

    Args:
        model: HuggingFace model to prune
        experts_to_keep: Mapping from layer index to list of expert indices to retain

    Returns:
        Dict with pruning statistics
    """
    config = model.config
    original_num_experts = getattr(config, "n_routed_experts", None) or getattr(
        config, "num_experts", None
    )
    if original_num_experts is None:
        raise ValueError("Model config missing n_routed_experts or num_experts")

    stats = {
        "original_num_experts": original_num_experts,
        "layers_pruned": 0,
        "experts_removed": 0,
    }

    for layer_idx, keep_indices in experts_to_keep.items():
        layer = model.model.layers[layer_idx]  # type: ignore[index]

        # Find MoE block - different models use different names
        moe_block = None
        for attr in ["mlp", "block_sparse_moe", "moe"]:
            if hasattr(layer, attr):
                candidate = getattr(layer, attr)
                if hasattr(candidate, "experts") or hasattr(candidate, "gate"):
                    moe_block = candidate
                    break

        if moe_block is None:
            logger.warning(f"Layer {layer_idx}: no MoE block found, skipping")
            continue

        # Get experts container
        experts = None
        experts_attr = None
        for attr in ["experts", "experts_list"]:
            if hasattr(moe_block, attr):
                experts = getattr(moe_block, attr)
                experts_attr = attr
                break

        if experts is None:
            logger.warning(f"Layer {layer_idx}: no experts found in MoE block")
            continue

        num_experts = len(experts) if isinstance(experts, nn.ModuleList) else original_num_experts

        if max(keep_indices) >= num_experts:
            raise ValueError(
                f"Layer {layer_idx}: keep_indices contain index {max(keep_indices)} "
                f"but only {num_experts} experts exist"
            )

        # Prune experts
        if isinstance(experts, nn.ModuleList):
            new_experts = nn.ModuleList([experts[i] for i in keep_indices])
            setattr(moe_block, experts_attr, new_experts)
        else:
            # Fused experts (tensor-based) - e.g., gate_up_proj, down_proj
            if hasattr(experts, "gate_up_proj"):
                with torch.no_grad():
                    experts.gate_up_proj = nn.Parameter(experts.gate_up_proj[keep_indices])
                    experts.down_proj = nn.Parameter(experts.down_proj[keep_indices])
                if hasattr(experts, "num_experts"):
                    experts.num_experts = len(keep_indices)

        # Prune router/gate
        router = None
        for attr in ["gate", "router", "gate_proj"]:
            if hasattr(moe_block, attr):
                router = getattr(moe_block, attr)
                break

        if router is not None and hasattr(router, "weight"):
            weight = router.weight.data
            # Router shape varies: [num_experts, hidden] or [hidden, num_experts]
            if weight.shape[0] == num_experts:
                router.weight = nn.Parameter(weight[keep_indices])
                if hasattr(router, "out_features"):
                    router.out_features = len(keep_indices)
            elif weight.shape[1] == num_experts:
                router.weight = nn.Parameter(weight[:, keep_indices])

            if hasattr(router, "bias") and router.bias is not None:
                if router.bias.shape[0] == num_experts:
                    router.bias = nn.Parameter(router.bias.data[keep_indices])

        stats["layers_pruned"] += 1
        stats["experts_removed"] += num_experts - len(keep_indices)

        logger.info(f"Layer {layer_idx}: pruned {num_experts} -> {len(keep_indices)} experts")

    # Update config
    new_num_experts = len(next(iter(experts_to_keep.values())))
    if hasattr(config, "n_routed_experts"):
        config.n_routed_experts = new_num_experts
    if hasattr(config, "num_experts"):
        config.num_experts = new_num_experts

    stats["new_num_experts"] = new_num_experts

    return stats


def save_model(
    model: nn.Module,
    tokenizer: Any,
    output_path: str,
    recipe: dict,
) -> None:
    """Save pruned model to local path or HuggingFace Hub."""
    if output_path.startswith("hf://"):
        # Push to HuggingFace Hub
        repo_id = output_path[5:]  # Remove "hf://" prefix
        logger.info(f"Pushing to HuggingFace Hub: {repo_id}")

        model.push_to_hub(repo_id, safe_serialization=True)
        tokenizer.push_to_hub(repo_id)

        # Save recipe as a file in the repo
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=json.dumps(recipe, indent=2).encode(),
            path_in_repo="pruning_recipe.json",
            repo_id=repo_id,
        )
        logger.info(f"Model pushed to: https://huggingface.co/{repo_id}")
    else:
        # Save to local directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving to: {output_dir}")
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        # Save recipe alongside model
        recipe_path = output_dir / "pruning_recipe.json"
        with open(recipe_path, "w") as f:
            json.dump(recipe, f, indent=2)

        logger.info(f"Model saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply pruning recipe to create pruned model")
    parser.add_argument(
        "--recipe",
        type=Path,
        required=True,
        help="Path to pruning recipe JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (local dir or hf://repo_id). Required unless --dry-run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate recipe without loading model",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading (default: auto)",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.output:
        parser.error("--output is required unless --dry-run is specified")

    # Load recipe
    logger.info(f"Loading recipe: {args.recipe}")
    recipe = load_recipe(args.recipe)

    base_model = recipe["base_model"]
    experts_to_keep = recipe["experts_to_keep"]

    logger.info(f"Base model: {base_model}")
    logger.info(f"Layers to prune: {len(experts_to_keep)}")
    logger.info(
        f"Experts per layer: {len(next(iter(experts_to_keep.values())))} "
        f"(from {recipe.get('original_num_experts', '?')})"
    )

    if args.dry_run:
        logger.info("Dry run - validating recipe only")

        # Just check that base model config is accessible
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        orig_experts = getattr(config, "n_routed_experts", None) or getattr(
            config, "num_experts", None
        )
        logger.info(f"Model config loaded. Original experts: {orig_experts}")

        # Validate indices
        for layer_idx, keep in experts_to_keep.items():
            if orig_experts and max(keep) >= orig_experts:
                raise ValueError(f"Layer {layer_idx}: index {max(keep)} >= {orig_experts} experts")
        logger.info("Recipe validated successfully")
        return

    # Load model
    logger.info(f"Loading model: {base_model}")
    logger.info("This requires ~80GB VRAM...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Apply pruning
    logger.info("Applying pruning recipe...")
    stats = apply_pruning(model, experts_to_keep)

    logger.info(
        f"Pruning complete: {stats['original_num_experts']} -> {stats['new_num_experts']} experts"
    )
    logger.info(f"Layers pruned: {stats['layers_pruned']}")
    logger.info(f"Total experts removed: {stats['experts_removed']}")

    # Save
    save_model(model, tokenizer, args.output, recipe)
    logger.info("Done!")


if __name__ == "__main__":
    main()
