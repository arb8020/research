#!/usr/bin/env python3
"""Generic evaluation runner for rollouts-style agent tasks.

Supports any dataset/environment combination via config.

Usage:
    python run_eval.py --config configs/calc_smoke.py
    python run_eval.py --config configs/screenspot.py
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from rollouts.dtypes import Endpoint, Actor, AgentState
from rollouts.agents import run_agent, RunConfig, stdout_handler
from rollouts.logging_utils import init_rollout_logging

logger = logging.getLogger(__name__)


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

    # Call dataset loader with appropriate arguments
    # Different loaders need different args, so we pass relevant config fields
    dataset = config.load_dataset(
        data_path=config.dataset.dataset_path,
        annotation_file=config.dataset.annotation_files[0] if config.dataset.annotation_files else None,
        limit=config.filters.limit,
        platforms=config.filters.platforms if config.filters.platforms else None,
        applications=config.filters.applications if config.filters.applications else None,
        ui_types=config.filters.ui_types if config.filters.ui_types else None,
    )
    logger.info(f"📊 Loaded {len(dataset)} samples from {config.dataset.dataset_path}")

    # Create endpoint
    endpoint = Endpoint(
        provider=config.provider,
        model=config.model_name,
        api_base=config.api_base,
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

    # Run on each sample
    results = []
    total_reward = 0.0
    num_completed = 0

    for i, row in enumerate(dataset):
        logger.info(f"\n{'─' * 60}")
        # Log sample identifier (different for each dataset)
        if "question" in row:
            logger.info(f"Sample {i+1}/{len(dataset)}: {row['question'][:80]}")
        elif "instruction" in row:
            logger.info(f"Sample {i+1}/{len(dataset)}: {row['instruction'][:80]}")
        else:
            logger.info(f"Sample {i+1}/{len(dataset)}")
        logger.info(f"{'─' * 60}\n")

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
        # Note: Use None for on_chunk to avoid printing base64 images to stdout
        run_config = RunConfig(on_chunk=None)

        try:
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

            total_reward += reward
            num_completed += 1

            result = {
                "sample_id": i,
                "response": final_message.content,
                "turns": final_state.turn_idx,
                "reward": reward,
                "stop_reason": final_state.stop.reason if final_state.stop else None,
                "success": final_state.stop is not None,
            }

            # Include metadata from trajectory if present
            if hasattr(trajectory, "metadata") and trajectory.metadata:
                result["metadata"] = trajectory.metadata

            results.append(result)

            logger.info(f"\n✅ Completed in {final_state.turn_idx} turns | Reward: {reward}")

        except Exception as e:
            logger.error(f"Sample {i+1} failed: {e}", exc_info=True)

            result = {
                "sample_id": i,
                "error": str(e),
                "success": False,
            }
            results.append(result)

    # Compute summary
    logger.info(f"\n{'=' * 60}")
    logger.info("📊 SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Completed: {num_completed}")
    logger.info(f"Errors: {len(dataset) - num_completed}")
    logger.info(f"Mean reward: {total_reward / num_completed if num_completed > 0 else 0.0:.3f}")

    summary = {
        "experiment_name": config.experiment_name,
        "total_samples": len(dataset),
        "completed": num_completed,
        "errors": len(dataset) - num_completed,
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

    args = parser.parse_args()

    # Load config from file
    print(f"📝 Loading config from: {args.config}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("exp_config", args.config)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "config"), "Config file must export 'config' variable"
    config = module.config

    # Check if we need rollouts-style evaluation
    if config.environment is None or config.to_trajectory is None:
        print("❌ Config missing environment or to_trajectory fields")
        print("   This config appears to be old-style. Please update to rollouts-style.")
        return 1

    # Initialize logging with timestamped results directory
    result_dir = init_rollout_logging(
        experiment_name=config.experiment_name,
        results_base_dir=config.save_dir if hasattr(config, "save_dir") else Path("results"),
        log_level="INFO",
    )

    logger.info(f"🚀 Running evaluation: {config.experiment_name}")

    # Run evaluation
    results_dict = asyncio.run(run_evaluation(config, result_dir))

    # Save results to timestamped directory
    if config.save_json:
        output_path = args.output or (result_dir / f"{config.experiment_name}_results.json")

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"💾 Saved results to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
