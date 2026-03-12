"""REAP base configuration and pipeline.

Run with:
    python -m argus run --config examples/reap/configs/qwen3_prune_50.py
    python -m argus run --config examples/reap/configs/qwen3_prune_50.py --provision
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import ReapConfig

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
    import sys

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .data import batch_iterator
    from .data_pipeline import load_calibration_data_advanced
    from .export import get_output_path, save_pruned_model, save_pruning_recipe
    from .observer import MoEObserver
    from .pruner import prune_model

    # Configure logging to stdout explicitly (default stderr doesn't appear in pty capture)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[REAP] %(message)s"))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Starting REAP pipeline")

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
    logger.info(f"Model loaded: {model.__class__.__name__}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    logger.info("Tokenizer loaded")

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
        logger.info(
            f"Loading calibration data: {config.dataset_name} ({config.num_samples} samples)"
        )
        if config.split_by_category:
            # Use category-aware data pipeline
            category_samples = load_calibration_data_advanced(
                config.dataset_name,
                tokenizer,
                config.num_samples,
                config.max_seq_len,
                config.seed,
                split_by_category=True,
                samples_per_category=config.samples_per_category,
            )
            # Flatten categories into single list
            samples = []
            for cat, cat_samples in category_samples.items():
                samples.extend(cat_samples)
            logger.info(
                f"Calibration data loaded: {len(samples)} samples from {len(category_samples)} categories"
            )
        else:
            # Use simple data loading
            from .data import load_calibration_data

            samples = load_calibration_data(
                config.dataset_name,
                tokenizer,
                config.num_samples,
                config.max_seq_len,
                config.seed,
            )
            logger.info(f"Calibration data loaded: {len(samples)} samples")

        # Run observation
        logger.info("Running observation phase...")
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(batch_iterator(samples, batch_size=1, device=device)):
                model(**batch)
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(samples)} samples")

        logger.info("Observation phase complete")

        # Cache observations
        if config.cache_observations:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            observer.save_observations(str(cache_path))
            logger.info(f"Saved observations to {cache_path}")

    # Remove hooks before pruning
    observer.remove_hooks()

    # Prune model
    logger.info("Pruning experts...")
    result = prune_model(model, observer, config)

    logger.info(
        f"Pruned {result.original_num_experts} -> {result.pruned_num_experts} experts "
        f"({config.compression_ratio:.0%} compression)"
    )

    # Compute experts to keep (inverse of pruned indices)
    experts_to_keep = {}
    for layer_idx, pruned in result.pruned_expert_indices.items():
        all_experts = list(range(result.original_num_experts))
        experts_to_keep[layer_idx] = [i for i in all_experts if i not in pruned]

    # Save output
    recipe_config = {
        "compression_ratio": config.compression_ratio,
        "prune_method": config.prune_method.value,
        "num_samples": config.num_samples,
        "seed": config.seed,
        "original_num_experts": result.original_num_experts,
        "pruned_num_experts": result.pruned_num_experts,
    }

    if config.save_full_model:
        # Clear HF cache to make room for saving pruned model
        import shutil
        from pathlib import Path as PathLib

        hf_cache = PathLib.home() / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            logger.info(f"Clearing HF cache at {hf_cache} to free disk space...")
            shutil.rmtree(hf_cache, ignore_errors=True)

        logger.info(f"Saving full pruned model to {output_path}")
        save_pruned_model(
            model,
            tokenizer,
            output_path,
            config_overrides={"model_name": config.model_name, **recipe_config},
        )
    else:
        recipe_path = output_path.parent / "pruning_recipe.json"
        logger.info(f"Saving pruning recipe to {recipe_path}")
        save_pruning_recipe(
            base_model=config.model_name,
            experts_to_keep=experts_to_keep,
            output_path=recipe_path,
            config=recipe_config,
        )

    logger.info("REAP pruning complete")

    # Run evaluation if requested
    eval_results = {}
    if config.run_eval and config.save_full_model:
        logger.info("Starting evaluation phase...")
        eval_results = run_eval(output_path, config)

    logger.info("REAP pipeline complete")

    return {
        "output_path": str(output_path),
        "original_num_experts": result.original_num_experts,
        "pruned_num_experts": result.pruned_num_experts,
        "pruned_indices": {k: v for k, v in result.pruned_expert_indices.items()},
        "eval_results": eval_results,
    }


def run_eval(model_path: Path, config: ReapConfig) -> dict[str, Any]:
    """Run lm-eval benchmarks on pruned model.

    Uses new eval_lm_harness module for comprehensive evaluation.

    Args:
        model_path: Path to the pruned model
        config: REAP config with eval settings

    Returns:
        Dict with benchmark results
    """
    from .eval_lm_harness import run_code_eval, run_lm_eval

    logger.info(f"Running evaluation on {model_path}")
    logger.info(f"Tasks: {config.eval_tasks}")

    all_results = {}

    # Run lm-eval
    try:
        lm_results = run_lm_eval(
            model_path=model_path,
            tasks=list(config.eval_tasks),
            use_server=config.use_server,
            port=config.sglang_port,
        )
        all_results["lm_eval"] = lm_results.get("results", {})
    except Exception as e:
        logger.error(f"lm-eval failed: {e}")
        all_results["lm_eval_error"] = str(e)

    # Run code eval if requested
    if config.run_evalplus:
        try:
            code_results = run_code_eval(
                model_path=model_path,
                tasks=list(config.evalplus_tasks),
                port=config.sglang_port + 1,  # Use different port
            )
            all_results["code_eval"] = code_results
        except Exception as e:
            logger.error(f"Code eval failed: {e}")
            all_results["code_eval_error"] = str(e)

    # Save results
    import json

    results_path = model_path.parent / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return all_results


# Alias for consistency with other examples
train = run_reap
