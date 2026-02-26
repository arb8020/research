"""Run apply_recipe.py on Modal to create a pruned model.

This is a standalone Modal script (not using modal_runner.py) because:
1. Pruning only needs 1x A100-80GB, training may need 8x H100
2. Pruning is a one-time operation, training is iterative
3. Output goes to HF Hub, making it usable by any training config

Usage:
    # Prune GLM-4.7-Flash to 32 experts, push to HF Hub
    modal run examples/reap/apply_recipe_modal.py \
        --recipe results/reap/glm4_flash_random_50/pruning_recipe.json \
        --output hf://your-username/glm4-flash-pruned-50

    # Dry run (validate recipe without GPU)
    modal run examples/reap/apply_recipe_modal.py \
        --recipe results/reap/glm4_flash_random_50/pruning_recipe.json \
        --dry-run

Requires Modal CLI: pip install modal && modal setup
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# App and image setup
app = modal.App("reap-pruning")

# Image with transformers and torch for model loading
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.50.0",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "accelerate",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,  # 1 hour
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def apply_pruning_recipe(
    recipe_json: str,
    output_path: str,
    dry_run: bool = False,
) -> dict:
    """Apply pruning recipe to create pruned model.

    Args:
        recipe_json: JSON string of pruning recipe
        output_path: Output destination (hf://repo_id or local path)
        dry_run: If True, only validate recipe

    Returns:
        Dict with status and details
    """
    import logging

    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse recipe
    recipe = json.loads(recipe_json)
    base_model = recipe["base_model"]
    experts_to_keep = {int(k): v for k, v in recipe["experts_to_keep"].items()}

    logger.info(f"Base model: {base_model}")
    logger.info(f"Layers to prune: {len(experts_to_keep)}")
    logger.info(f"Experts per layer: {len(next(iter(experts_to_keep.values())))}")

    if dry_run:
        # Just validate
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        orig_experts = getattr(config, "n_routed_experts", None) or getattr(
            config, "num_experts", None
        )
        logger.info(f"Original experts: {orig_experts}")
        return {"status": "dry_run", "original_experts": orig_experts}

    # Load model
    logger.info(f"Loading model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Get original config
    config = model.config
    original_num_experts = getattr(config, "n_routed_experts", None) or getattr(
        config, "num_experts", None
    )

    # Apply pruning
    logger.info("Applying pruning...")
    layers_pruned = 0

    for layer_idx, keep_indices in experts_to_keep.items():
        layer = model.model.layers[layer_idx]

        # Find MoE block
        moe_block = None
        for attr in ["mlp", "block_sparse_moe", "moe"]:
            if hasattr(layer, attr):
                candidate = getattr(layer, attr)
                if hasattr(candidate, "experts") or hasattr(candidate, "gate"):
                    moe_block = candidate
                    break

        if moe_block is None:
            logger.warning(f"Layer {layer_idx}: no MoE block found")
            continue

        # Get experts
        experts = getattr(moe_block, "experts", None)
        if experts is None:
            logger.warning(f"Layer {layer_idx}: no experts found")
            continue

        num_experts = len(experts)

        # Prune experts (ModuleList)
        new_experts = nn.ModuleList([experts[i] for i in keep_indices])
        moe_block.experts = new_experts

        # Prune router
        router = getattr(moe_block, "gate", None)
        if router is not None and hasattr(router, "weight"):
            weight = router.weight.data
            if weight.shape[0] == num_experts:
                router.weight = nn.Parameter(weight[keep_indices])
            elif weight.shape[1] == num_experts:
                router.weight = nn.Parameter(weight[:, keep_indices])

            if hasattr(router, "bias") and router.bias is not None:
                if router.bias.shape[0] == num_experts:
                    router.bias = nn.Parameter(router.bias.data[keep_indices])

        layers_pruned += 1
        logger.info(f"Layer {layer_idx}: {num_experts} -> {len(keep_indices)} experts")

    # Update config
    new_num_experts = len(next(iter(experts_to_keep.values())))
    if hasattr(config, "n_routed_experts"):
        config.n_routed_experts = new_num_experts
    if hasattr(config, "num_experts"):
        config.num_experts = new_num_experts

    logger.info(f"Pruning complete: {original_num_experts} -> {new_num_experts} experts")

    # Save
    if output_path.startswith("hf://"):
        repo_id = output_path[5:]
        logger.info(f"Pushing to HuggingFace Hub: {repo_id}")

        model.push_to_hub(repo_id, safe_serialization=True)
        tokenizer.push_to_hub(repo_id)

        # Upload recipe
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=recipe_json.encode(),
            path_in_repo="pruning_recipe.json",
            repo_id=repo_id,
        )

        logger.info(f"Model pushed: https://huggingface.co/{repo_id}")
        return {
            "status": "success",
            "output": f"https://huggingface.co/{repo_id}",
            "original_experts": original_num_experts,
            "new_experts": new_num_experts,
            "layers_pruned": layers_pruned,
        }
    else:
        # Local save (in sandbox - mainly for testing)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        return {
            "status": "success",
            "output": str(output_dir),
            "original_experts": original_num_experts,
            "new_experts": new_num_experts,
            "layers_pruned": layers_pruned,
        }


@app.local_entrypoint()
def main(
    recipe: str,
    output: str = "",
    dry_run: bool = False,
) -> None:
    """CLI entrypoint for Modal.

    Args:
        recipe: Path to pruning recipe JSON
        output: Output path (hf://repo_id or local). Required unless --dry-run
        dry_run: Validate recipe without loading model
    """
    if not dry_run and not output:
        raise ValueError("--output is required unless --dry-run is specified")

    # Read recipe file
    recipe_path = Path(recipe)
    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe not found: {recipe_path}")

    with open(recipe_path) as f:
        recipe_json = f.read()

    # Validate JSON
    recipe_data = json.loads(recipe_json)
    print(f"Recipe: {recipe_path}")
    print(f"  Base model: {recipe_data['base_model']}")
    print(f"  Layers: {len(recipe_data['experts_to_keep'])}")
    print(f"  Experts to keep: {len(next(iter(recipe_data['experts_to_keep'].values())))}")

    if dry_run:
        print("\nDry run - validating recipe on Modal...")
        result = apply_pruning_recipe.remote(recipe_json, "", dry_run=True)
    else:
        print(f"\nApplying recipe on Modal -> {output}")
        print("This will take 10-20 minutes (download + prune + upload)...")
        result = apply_pruning_recipe.remote(recipe_json, output, dry_run=False)

    print(f"\nResult: {json.dumps(result, indent=2)}")
