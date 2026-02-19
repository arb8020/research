"""REAP base configuration and pipeline.

Run with:
    python -m rollouts.run --config examples/reap/configs/qwen3_prune_50.py
    python -m rollouts.run --config examples/reap/configs/qwen3_prune_50.py --provision
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .config import ReapConfig
from .data import batch_iterator, load_calibration_data
from .export import get_output_path, save_pruned_model
from .observer import MoEObserver
from .pruner import prune_model

logger = logging.getLogger(__name__)


def run_reap(config: ReapConfig) -> dict[str, Any]:
    """Run REAP expert pruning pipeline.

    Steps:
    1. Load model and tokenizer
    2. Load calibration dataset
    3. Run observation (forward passes to collect activation stats)
    4. Compute REAP scores and select experts to prune
    5. Prune experts from model
    6. Save pruned model

    Args:
        config: REAP configuration

    Returns:
        Dict with pruning results and metadata
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Set seed
    torch.manual_seed(config.seed)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    # Setup observer
    observer = MoEObserver(model, device)

    # Check for cached observations
    output_path = get_output_path(
        config.output_dir,
        config.model_name,
        config.dataset_name,
        config.prune_method.value,
        config.compression_ratio,
        config.seed,
    )
    cache_path = output_path.parent / f"observations_{config.num_samples}.pt"

    if config.cache_observations and cache_path.exists():
        logger.info(f"Loading cached observations from {cache_path}")
        observer.load_observations(str(cache_path))
    else:
        # Load calibration data
        samples = load_calibration_data(
            config.dataset_name,
            tokenizer,
            config.num_samples,
            config.max_seq_len,
            config.seed,
        )

        # Run observation
        logger.info("Running observation phase...")
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(batch_iterator(samples, batch_size=1, device=device)):
                model(**batch)
                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(samples)} samples")

        # Cache observations
        if config.cache_observations:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            observer.save_observations(str(cache_path))

    # Remove hooks before pruning
    observer.remove_hooks()

    # Prune model
    logger.info("Pruning experts...")
    result = prune_model(model, observer, config)

    logger.info(
        f"Pruned {result.original_num_experts} -> {result.pruned_num_experts} experts "
        f"({config.compression_ratio:.0%} compression)"
    )

    # Save pruned model
    save_pruned_model(
        model,
        tokenizer,
        output_path,
        config_overrides={
            "model_name": config.model_name,
            "dataset_name": config.dataset_name,
            "compression_ratio": config.compression_ratio,
            "prune_method": config.prune_method.value,
            "num_samples": config.num_samples,
            "seed": config.seed,
            "original_num_experts": result.original_num_experts,
            "pruned_num_experts": result.pruned_num_experts,
        },
    )

    return {
        "output_path": str(output_path),
        "original_num_experts": result.original_num_experts,
        "pruned_num_experts": result.pruned_num_experts,
        "pruned_indices": {k: v for k, v in result.pruned_expert_indices.items()},
    }


# Alias for consistency with other examples
train = run_reap
