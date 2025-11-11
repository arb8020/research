#!/usr/bin/env python3
"""Generic evaluation runner for rollouts-style agent tasks.

Supports any dataset/environment combination via config.

Usage:
    python run_eval.py --config configs/calc_smoke.py
    python run_eval.py --config configs/screenspot.py
"""

import trio
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from rollouts.dtypes import Endpoint, Actor, AgentState
from rollouts.agents import run_agent, RunConfig, stdout_handler
from rollouts.logging_utils import init_rollout_logging
from rollouts.progress import tqdm

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


def load_config_from_file(config_path: Path, workspace_root: Optional[Path] = None):
    """Load and resolve config from Python file.

    This is a reusable function for loading configs, suitable for
    importing from wrapper scripts like clicker's run_eval.py.

    Tiger Style: Explicit workspace_root parameter, no path inference magic.

    Args:
        config_path: Path to config .py file
        workspace_root: Optional workspace root for resolving relative paths.
                       If provided, relative paths in config are resolved relative to this.
                       If None, uses config file location to infer project root (legacy).

    Returns:
        Loaded config object with resolved paths
    """
    print(f"📝 Loading config from: {config_path}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "config"), "Config file must export 'config' variable"
    config = module.config

    # Determine project root for path resolution
    if workspace_root:
        # Explicit workspace root provided (e.g., from bifrost deployment)
        project_root = workspace_root
        print(f"📁 Using explicit workspace root: {workspace_root}")
    else:
        # Legacy: Infer from config file location
        config_path_abs = Path(config_path).absolute()
        project_root = config_path_abs.parent.parent  # Go up from configs/ to project root
        print(f"📁 Inferred project root: {project_root}")

    # Resolve dataset path
    if hasattr(config, "dataset") and hasattr(config.dataset, "dataset_path"):
        dataset_path = config.dataset.dataset_path
        if not dataset_path.is_absolute():
            absolute_path = (project_root / dataset_path).resolve()
            object.__setattr__(config.dataset, "dataset_path", absolute_path)
            print(f"📂 Resolved dataset path: {absolute_path}")

    # Resolve save_dir path
    if hasattr(config, "output") and hasattr(config.output, "save_dir"):
        save_dir = config.output.save_dir
        if not save_dir.is_absolute():
            absolute_save_dir = (project_root / save_dir).resolve()
            object.__setattr__(config.output, "save_dir", absolute_save_dir)
            print(f"💾 Resolved save directory: {absolute_save_dir}")

    return config


async def run_evaluation(config, result_dir: Path) -> dict:
    """Run evaluation on dataset with environment.

    Generic evaluation loop:
    1. Load dataset
    2. Create endpoint
    3. Create environment
    4. For each sample:
       - Transform to trajectory
       - Run agent
       - Collect results
    5. Save results

    Args:
        config: Config object with dataset, environment, etc.

    Returns:
        Results dict with summary and per-sample results
    """
    logger.info("=" * 60)

    # Load dataset
    assert config.load_dataset is not None, "Config must have load_dataset function"
    assert config.dataset.dataset_path.exists(), f"Dataset not found: {config.dataset.dataset_path}"

    # Load from all annotation files and concatenate
    dataset = []
    for annotation_file in config.dataset.annotation_files:
        logger.info(f"📂 Loading {annotation_file}...")
        file_dataset = config.load_dataset(
            data_path=config.dataset.dataset_path,
            annotation_file=annotation_file,
            limit=None,  # Don't limit per file, apply global limit later
            platforms=config.filters.platforms if config.filters.platforms else None,
            applications=config.filters.applications if config.filters.applications else None,
            ui_types=config.filters.ui_types if config.filters.ui_types else None,
        )
        dataset.extend(file_dataset)
        logger.info(f"   Loaded {len(file_dataset)} samples from {annotation_file}")

    # Apply global limit if specified
    if config.filters.limit is not None:
        dataset = dataset[:config.filters.limit]

    logger.info(f"📊 Total: {len(dataset)} samples from {len(config.dataset.annotation_files)} file(s)")

    # Create endpoint - read API key from environment
    api_key = ""
    # Try config.model.api_key_env_var first (new style), then config.api_key_env_var (old style)
    api_key_env_var = None
    if hasattr(config, "model") and hasattr(config.model, "api_key_env_var"):
        api_key_env_var = config.model.api_key_env_var
    elif hasattr(config, "api_key_env_var"):
        api_key_env_var = config.api_key_env_var

    if api_key_env_var:
        api_key = os.getenv(api_key_env_var, "")
        logger.info(f"🔑 API key from {api_key_env_var}: {'***' + api_key[-4:] if api_key else 'NOT FOUND'}")

    endpoint = Endpoint(
        provider=config.provider,
        model=config.model_name,
        api_base=config.api_base,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
    )
    logger.info(f"🔗 Endpoint: {config.provider} @ {config.api_base}")
    logger.info(f"🤖 Model: {config.model_name}")

    # Create environment instance
    assert config.environment is not None, "Config must have environment"
    environment = config.environment()
    logger.info(f"🌍 Environment: {environment.__class__.__name__}")
    logger.info("")

    # Run on each sample concurrently
    results = [None] * len(dataset)  # Pre-allocate results list
    max_concurrent = config.max_concurrent if hasattr(config, "max_concurrent") else 10
    semaphore = trio.Semaphore(max_concurrent)

    logger.info(f"🚀 Running {len(dataset)} samples with max concurrency: {max_concurrent}")
    logger.info("")

    # Create progress bar
    pbar = tqdm(total=len(dataset), desc="Evaluating", unit="sample")

    # Open JSONL file for incremental writing (if enabled)
    jsonl_file = None
    jsonl_lock = trio.Lock()
    if config.save_jsonl:
        jsonl_path = result_dir / f"{config.experiment_name}_results.jsonl"
        jsonl_file = open(jsonl_path, 'w')
        logger.info(f"📝 Writing incremental results to: {jsonl_path}")
        logger.info("")

    async def run_sample(i: int, row: dict):
        """Run a single sample evaluation."""
        async with semaphore:
            sample_id = f"Sample {i+1}/{len(dataset)}"

            try:
                # Transform to trajectory
                assert config.to_trajectory is not None, "Config must have to_trajectory"
                trajectory = config.to_trajectory(row)

                # Create actor
                actor = Actor(
                    trajectory=trajectory,
                    endpoint=endpoint,
                )

                # Create initial state
                state = AgentState(
                    actor=actor,
                    environment=environment,
                    max_turns=config.max_turns,
                )

                # Run agent
                # Use no-op handler to avoid printing streaming chunks
                async def noop_chunk_handler(chunk):
                    pass

                run_config = RunConfig(on_chunk=noop_chunk_handler)

                states = await run_agent(state, run_config)
                final_state = states[-1]

                # Get final message
                final_message = final_state.actor.trajectory.messages[-1]

                # Compute reward
                reward = 0.0
                if final_state.stop:
                    # Tool-based environments (calculator) use stop.reward
                    # But for ScreenSpot we need to compute reward from response
                    pass

                # Check if environment has compute_reward method (ScreenSpot)
                if hasattr(environment, "compute_reward") and hasattr(trajectory, "metadata"):
                    if "bbox" in trajectory.metadata and "img_size" in trajectory.metadata:
                        reward = environment.compute_reward(
                            final_message.content,
                            trajectory.metadata["bbox"],
                            trajectory.metadata["img_size"]
                        )

                result = {
                    "sample_id": i,
                    "response": final_message.content,
                    "turns": final_state.turn_idx,
                    "reward": reward,
                    "stop_reason": final_state.stop.value if final_state.stop else None,
                    "success": final_state.stop is not None,
                }

                # Include metadata from trajectory if present
                if hasattr(trajectory, "metadata") and trajectory.metadata:
                    result["metadata"] = trajectory.metadata

                results[i] = result

                # Write to JSONL incrementally
                if jsonl_file:
                    async with jsonl_lock:
                        jsonl_file.write(json.dumps(result) + "\n")
                        jsonl_file.flush()  # Ensure it's written to disk

                # Update progress bar with reward in postfix
                pbar.set_postfix({"last_reward": f"{reward:.3f}"})
                pbar.update(1)

            except Exception as e:
                logger.error(f"❌ {sample_id} failed: {e}")
                result = {
                    "sample_id": i,
                    "error": str(e),
                    "success": False,
                }
                results[i] = result

                # Write error to JSONL incrementally
                if jsonl_file:
                    async with jsonl_lock:
                        jsonl_file.write(json.dumps(result) + "\n")
                        jsonl_file.flush()

                pbar.update(1)

    # Run all samples concurrently
    async with trio.open_nursery() as nursery:
        for i, row in enumerate(dataset):
            nursery.start_soon(run_sample, i, row)

    # Close progress bar
    pbar.close()

    # Close JSONL file
    if jsonl_file:
        jsonl_file.close()
        logger.info(f"✅ JSONL results written to: {jsonl_path}")

    # Compute summary
    num_completed = sum(1 for r in results if r and r.get("success", False))
    total_reward = sum(r.get("reward", 0.0) for r in results if r and "reward" in r)
    num_errors = len(dataset) - num_completed

    logger.info(f"\n{'=' * 60}")
    logger.info("📊 SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Completed: {num_completed}")
    logger.info(f"Errors: {num_errors}")
    logger.info(f"Mean reward: {total_reward / num_completed if num_completed > 0 else 0.0:.3f}")

    summary = {
        "experiment_name": config.experiment_name,
        "total_samples": len(dataset),
        "completed": num_completed,
        "errors": num_errors,
        "mean_reward": total_reward / num_completed if num_completed > 0 else 0.0,
        "total_reward": total_reward,
    }

    return {
        "summary": summary,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run generic evaluation")
    parser.add_argument("--config", type=Path, required=True, help="Config file path")
    parser.add_argument("--output", type=Path, help="Output JSON file (optional, overrides default timestamped path)")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        help="Workspace root for resolving relative paths (e.g., from bifrost deployment)"
    )

    args = parser.parse_args()

    # Load config from file using reusable function
    # Pass workspace_root if provided for explicit path resolution
    config = load_config_from_file(args.config, workspace_root=args.workspace_root)

    # Check if we need rollouts-style evaluation
    if config.environment is None or config.to_trajectory is None:
        print("❌ Config missing environment or to_trajectory fields")
        print("   This config appears to be old-style. Please update to rollouts-style.")
        return 1

    # Verify API key is set for the provider
    provider_key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    provider = config.provider.lower()
    if provider in provider_key_map:
        required_key = provider_key_map[provider]
        if not os.getenv(required_key):
            print(f"❌ Error: {required_key} not found in environment")
            print(f"   Provider '{config.provider}' requires {required_key}")
            print(f"   Please add it to your .env file or set it in your environment")
            return 1
        else:
            print(f"✅ Found {required_key} in environment")

    # Initialize logging with timestamped results directory
    result_dir = init_rollout_logging(
        experiment_name=config.experiment_name,
        results_base_dir=config.save_dir if hasattr(config, "save_dir") else Path("results"),
        log_level="INFO",
    )

    logger.info(f"🚀 Running evaluation: {config.experiment_name}")

    # Run evaluation
    results_dict = trio.run(run_evaluation, config, result_dir)

    # Save results to timestamped directory
    if config.save_json:
        output_path = args.output or (result_dir / f"{config.experiment_name}_results.json")

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"💾 Saved results to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
