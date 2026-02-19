"""REAP base configuration and pipeline.

Run with:
    python -m rollouts.run --config examples/reap/configs/qwen3_prune_50.py
    python -m rollouts.run --config examples/reap/configs/qwen3_prune_50.py --provision
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

    from .data import batch_iterator, load_calibration_data
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
    """Start SGLang server and run lm-eval benchmarks.

    Uses rollouts.deploy infrastructure for reliable server management.

    Args:
        model_path: Path to the pruned model
        config: REAP config with eval settings

    Returns:
        Dict with benchmark results
    """
    import json
    import subprocess
    import time

    import requests

    from rollouts.training.sglang_launcher import _patch_transformers

    # Apply transformers patch before sglang import
    _patch_transformers()

    logger.info(f"Starting SGLang server on port {config.sglang_port}...")
    logger.info(f"Model path: {model_path}")

    # Start server using the rollouts launcher (handles transformers patch)
    server_cmd = [
        "python",
        "-m",
        "rollouts.training.sglang_launcher",
        "--model-path",
        str(model_path),
        "--port",
        str(config.sglang_port),
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.85",
    ]

    logger.info(f"Server command: {' '.join(server_cmd)}")

    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    base_url = f"http://localhost:{config.sglang_port}"
    logger.info(f"Waiting for server at {base_url}...")

    for attempt in range(180):  # 3 min timeout
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                logger.info(f"SGLang server ready after {attempt}s")
                break
        except requests.RequestException:
            pass

        # Check if process died
        if server_proc.poll() is not None:
            stdout = server_proc.stdout.read().decode() if server_proc.stdout else ""
            logger.error(f"Server process died. Output:\n{stdout[-2000:]}")
            raise RuntimeError("SGLang server process died during startup")

        time.sleep(1)
    else:
        server_proc.terminate()
        raise TimeoutError("SGLang server failed to start within 3 minutes")

    try:
        # Run lm-eval
        logger.info(f"Running lm-eval on tasks: {config.eval_tasks}")

        import lm_eval

        results = lm_eval.simple_evaluate(
            model="local-completions",
            model_args={
                "base_url": f"{base_url}/v1/completions",
                "tokenized_requests": False,
            },
            tasks=list(config.eval_tasks),
            batch_size=8,
        )

        # Extract summary
        summary: dict[str, Any] = {}
        if "results" in results:
            for task, metrics in results["results"].items():
                summary[task] = {
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and not k.startswith("_")
                }

        logger.info("Evaluation results:")
        logger.info(json.dumps(summary, indent=2))

        # Save results
        results_path = model_path.parent / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {results_path}")

        return summary

    finally:
        logger.info("Stopping SGLang server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()


# Alias for consistency with other examples
train = run_reap
