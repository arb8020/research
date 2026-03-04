"""GLM-4.7-Flash 50% random expert pruning.

Quick test: randomly drop half the experts without REAP scoring.
This is just to get a smaller model for testing Megatron training.

Run:
    python examples/reap/configs/glm4_flash_prune_50_random.py
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "zai-org/GLM-4.7-Flash"
OUTPUT_DIR = Path("results/reap/glm4_flash_pruned_50")
COMPRESSION_RATIO = 0.5
SEED = 42


def prune_experts_random(model: nn.Module, ratio: float, seed: int) -> dict:
    """Randomly prune experts from GLM-4.7-Flash MoE layers."""
    torch.manual_seed(seed)

    config = model.config
    num_experts = config.n_routed_experts  # 64 for GLM-4.7-Flash
    num_to_keep = int(num_experts * (1 - ratio))  # 32

    logger.info(f"Pruning {num_experts} -> {num_to_keep} experts per layer")

    pruned_layers = {}

    # GLM-4.7-Flash structure: model.layers[i].mlp is the MoE block
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp"):
            continue

        mlp = layer.mlp

        # Check if this is an MoE layer (has experts)
        if not hasattr(mlp, "experts"):
            continue

        experts = mlp.experts  # ModuleList of experts
        router = mlp.gate  # Router/gate

        if len(experts) != num_experts:
            logger.warning(f"Layer {layer_idx}: expected {num_experts} experts, got {len(experts)}")
            continue

        # Randomly select experts to keep
        indices_to_keep = torch.randperm(num_experts)[:num_to_keep].sort().values.tolist()
        indices_to_remove = [i for i in range(num_experts) if i not in indices_to_keep]

        logger.info(
            f"Layer {layer_idx}: keeping experts {indices_to_keep[:5]}... (showing first 5)"
        )

        # Prune experts (ModuleList)
        new_experts = nn.ModuleList([experts[i] for i in indices_to_keep])
        mlp.experts = new_experts

        # Prune router weights
        if hasattr(router, "weight"):
            weight = router.weight.data
            # Router shape is typically [hidden_dim, num_experts] or [num_experts, hidden_dim]
            if weight.shape[0] == num_experts:
                new_weight = weight[indices_to_keep]
                router.weight = nn.Parameter(new_weight)
            elif weight.shape[1] == num_experts:
                new_weight = weight[:, indices_to_keep]
                router.weight = nn.Parameter(new_weight)

            if hasattr(router, "bias") and router.bias is not None:
                if router.bias.shape[0] == num_experts:
                    router.bias = nn.Parameter(router.bias.data[indices_to_keep])

        pruned_layers[layer_idx] = indices_to_remove

    # Update config
    model.config.n_routed_experts = num_to_keep

    return {
        "original_num_experts": num_experts,
        "pruned_num_experts": num_to_keep,
        "pruned_layers": pruned_layers,
    }


def main() -> None:
    logger.info(f"Loading {MODEL_NAME}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    logger.info("Pruning experts...")
    result = prune_experts_random(model, COMPRESSION_RATIO, SEED)

    logger.info(f"Experts: {result['original_num_experts']} -> {result['pruned_num_experts']}")
    logger.info(f"Pruned {len(result['pruned_layers'])} MoE layers")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {OUTPUT_DIR}...")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("Done!")
    logger.info(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
