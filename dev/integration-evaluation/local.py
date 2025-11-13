#!/usr/bin/env python3
"""Local entrypoint for Prime Intellect integration evaluation.

This script demonstrates Phase 1: Evaluating models with Prime verifiers.

Usage:
    python local.py configs/prime_2048.py

Pattern matches wafer_stuff/clicker/run_eval.py but adapted for Prime integration.
"""

import argparse
import trio
import importlib.util
import sys
import logging
from pathlib import Path
from dataclasses import replace

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add shared and rollouts to path
sys.path.insert(0, str(Path.home() / "research" / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "rollouts"))

from rollouts.evaluation import evaluate
from rollouts.integrations.prime import prime_reward_fn
from rollouts.dtypes import Message
from rollouts.logging_utils import init_rollout_logging

# Import verifiers for loading Prime Hub environments
from verifiers import load_environment

logger = logging.getLogger(__name__)


async def run_evaluation(config_path: Path, result_dir: Path):
    """Run Prime integration evaluation.

    Args:
        config_path: Path to config file
        result_dir: Timestamped results directory for outputs

    Pattern:
        1. Load config
        2. Create Prime environment
        3. Create reward function from Prime
        4. Run evaluation with rollouts framework
    """
    logger.info(f"📝 Loading config from: {config_path}")

    # Load config module
    spec = importlib.util.spec_from_file_location("config", config_path)
    assert spec is not None
    assert spec.loader is not None

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    assert hasattr(config_module, "config"), "Config file must export 'config' variable"
    assert hasattr(config_module, "prepare_messages"), "Config file must export 'prepare_messages' function"
    assert hasattr(config_module, "create_environment"), "Config file must export 'create_environment' function"

    config = config_module.config
    prepare_messages = config_module.prepare_messages
    create_environment = config_module.create_environment

    logger.info(f"🎯 Configuration loaded")
    logger.info(f"   Model: {config.model_name}")
    logger.info(f"   Environment: {config.env_name}")
    logger.info(f"   Samples: {config.num_samples}")
    logger.info(f"   Max concurrent: {config.max_concurrent}")

    # Create Prime environment
    logger.info(f"\n🎮 Loading Prime environment: {config.env_name}")
    prime_env = load_environment(config.env_name)

    logger.info(f"   Dataset size: {len(prime_env.dataset)}")
    logger.info(f"   Rubric: {type(prime_env.rubric).__name__}")
    logger.info(f"   Parser: {type(prime_env.parser).__name__}")
    logger.info(f"   Max turns: {prime_env.max_turns}")

    # Create reward function from Prime environment
    logger.info(f"\n🏆 Creating reward function from Prime rubric")
    reward_fn = prime_reward_fn(prime_env)

    # Use Prime dataset directly (no conversion needed - prepare_messages handles format)
    logger.info(f"\n📊 Using Prime dataset")
    rollouts_dataset = list(prime_env.dataset)
    logger.info(f"   Dataset size: {len(rollouts_dataset)} samples")

    # Setup rollouts evaluation config
    endpoint = config.to_endpoint()
    eval_config = config.to_eval_config(reward_fn)

    # Override output directory to use timestamped result_dir
    eval_config = replace(eval_config, output_dir=result_dir)

    # Create environment factory closure over prime_env
    async def environment_factory(sample_data):
        return await create_environment(prime_env, sample_data)

    logger.info(f"\n🚀 Starting evaluation")
    logger.info(f"   Output dir: {result_dir}")
    logger.info("="*50)

    # Run evaluation within trio_asyncio loop context
    # This allows all Prime API calls to share the same event loop
    import trio_asyncio
    async with trio_asyncio.open_loop():
        try:
            report = await evaluate(
                dataset=iter(rollouts_dataset),
                prepare_messages=prepare_messages,
                environment_factory=environment_factory,
                endpoint=endpoint,
                config=eval_config,
                dataset_path=f"{config.env_name}_dataset",
            )
        finally:
            # Cleanup any remaining sandboxes (in case of errors during evaluation)
            if hasattr(prime_env, 'cleanup_sandboxes'):
                logger.info("\n🧹 Cleaning up sandboxes...")
                prime_env.cleanup_sandboxes()

    # Print detailed results
    logger.info("\n" + "="*50)
    logger.info("📊 EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Total samples: {report.total_samples}")
    logger.info(f"Mean reward: {report.summary_metrics['mean_reward']:.3f}")
    logger.info(f"Min reward: {report.summary_metrics['min_reward']:.3f}")
    logger.info(f"Max reward: {report.summary_metrics['max_reward']:.3f}")
    logger.info(f"Std reward: {report.summary_metrics['std_reward']:.3f}")

    # Show sample-level details
    logger.info(f"\n📝 Sample-level results:")
    for i, sample in enumerate(report.sample_results[:5]):  # Show first 5
        logger.info(f"\n{sample.sample_id}:")
        logger.info(f"  Reward: {sample.metrics['reward']:.3f}")

        # Handle different dataset formats (wiki-search has 'question', acebench doesn't)
        if 'question' in sample.input_data:
            logger.info(f"  Question: {sample.input_data['question'][:80]}...")

        logger.info(f"  Ground truth: {sample.trajectory.metadata.get('prime_ground_truth')}")
        logger.info(f"  Parsed answer: {sample.trajectory.metadata.get('prime_parsed_answer')}")

        # Show model response
        if sample.trajectory.messages:
            last_msg = sample.trajectory.messages[-1]
            if last_msg.role == "assistant":
                response_preview = (last_msg.content or "")[:100]
                logger.info(f"  Model response: {response_preview}...")

    logger.info(f"\n✅ Results saved to: {result_dir}")
    logger.info(f"   Report: {result_dir}/report.json")
    logger.info(f"   Samples: {result_dir}/samples/")
    logger.info(f"   Trajectories: {result_dir}/trajectories/")

    return report


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(
        description="Prime Intellect integration evaluation"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to config file (e.g., configs/prime_2048.py)"
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    # Extract experiment name from config file name
    experiment_name = args.config.stem  # e.g., "prime_wiki" from "prime_wiki.py"

    # Initialize logging and create timestamped results directory
    result_dir = init_rollout_logging(
        experiment_name=experiment_name,
        results_base_dir=Path("eval_results"),
        log_level="INFO",
        logger_levels={
            "httpx": "WARNING",
            "httpcore": "WARNING",
            "openai": "WARNING",
            "anthropic": "WARNING",
            "httpcore.http11": "WARNING",
            "datasets": "WARNING",
            "chromadb": "WARNING",
            "verifiers": "WARNING",
            "verifiers.utils": "WARNING",
            "verifiers.utils.env_utils": "WARNING",
            "verifiers.rubrics": "WARNING",
        }
    )

    logger.info(f"🚀 Running Prime integration evaluation: {experiment_name}")

    # Run evaluation
    trio.run(run_evaluation, args.config, result_dir)


if __name__ == "__main__":
    main()
