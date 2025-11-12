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
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add shared and rollouts to path
sys.path.insert(0, str(Path.home() / "research" / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "rollouts"))

# Use shared logging config
from shared.logging_config import setup_logging

setup_logging(
    level="INFO",  # Show info, warnings, and errors
    logger_levels={
        "httpx": "WARNING",  # Mute HTTP request logs
        "httpcore": "WARNING",  # Mute HTTP core logs
        "openai": "WARNING",  # Mute OpenAI client logs
        "anthropic": "WARNING",  # Mute Anthropic client logs
        "httpcore.http11": "WARNING",  # Mute HTTP/1.1 logs
        "datasets": "WARNING",  # Mute HuggingFace datasets logs
        "chromadb": "WARNING",  # Mute ChromaDB logs
    }
)

# Silence verifiers module loggers specifically
import logging
logging.getLogger("verifiers").setLevel(logging.WARNING)
logging.getLogger("verifiers.utils").setLevel(logging.WARNING)
logging.getLogger("verifiers.utils.env_utils").setLevel(logging.WARNING)
logging.getLogger("verifiers.rubrics").setLevel(logging.WARNING)

from rollouts.evaluation import evaluate
from rollouts.integrations.prime import (
    prime_reward_fn,
    convert_verifiers_dataset_to_rollouts,
)
from rollouts.dtypes import Message

# Import verifiers for loading Prime Hub environments
from verifiers import load_environment


async def run_evaluation(config_path: Path):
    """Run Prime integration evaluation.

    Args:
        config_path: Path to config file

    Pattern:
        1. Load config
        2. Create Prime environment
        3. Create reward function from Prime
        4. Run evaluation with rollouts framework
    """
    print(f"📝 Loading config from: {config_path}")

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

    print(f"🎯 Configuration loaded")
    print(f"   Model: {config.model_name}")
    print(f"   Environment: {config.env_name}")
    print(f"   Samples: {config.num_samples}")
    print(f"   Max concurrent: {config.max_concurrent}")

    # Create Prime environment
    print(f"\n🎮 Loading Prime environment: {config.env_name}")
    prime_env = load_environment(config.env_name)

    print(f"   Dataset size: {len(prime_env.dataset)}")
    print(f"   Rubric: {type(prime_env.rubric).__name__}")
    print(f"   Parser: {type(prime_env.parser).__name__}")
    print(f"   Max turns: {prime_env.max_turns}")

    # Create reward function from Prime environment
    print(f"\n🏆 Creating reward function from Prime rubric")
    reward_fn = prime_reward_fn(prime_env)

    # Convert Prime dataset to rollouts format
    print(f"\n📊 Converting dataset to rollouts format")
    rollouts_dataset = convert_verifiers_dataset_to_rollouts(prime_env)
    print(f"   Converted {len(rollouts_dataset)} samples")

    # Setup rollouts evaluation config
    endpoint = config.to_endpoint()
    eval_config = config.to_eval_config(reward_fn)

    # Create output directory
    eval_config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Starting evaluation")
    print(f"   Output dir: {eval_config.output_dir}")
    print("="*50)

    # Run evaluation
    report = await evaluate(
        dataset=iter(rollouts_dataset),
        prepare_messages=prepare_messages,
        environment_factory=create_environment,
        endpoint=endpoint,
        config=eval_config,
        dataset_path=f"{config.env_name}_dataset",
    )

    # Print detailed results
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {report.total_samples}")
    print(f"Mean reward: {report.summary_metrics['mean_reward']:.3f}")
    print(f"Min reward: {report.summary_metrics['min_reward']:.3f}")
    print(f"Max reward: {report.summary_metrics['max_reward']:.3f}")
    print(f"Std reward: {report.summary_metrics['std_reward']:.3f}")

    # Show sample-level details
    print(f"\n📝 Sample-level results:")
    for i, sample in enumerate(report.sample_results[:5]):  # Show first 5
        print(f"\n{sample.sample_id}:")
        print(f"  Reward: {sample.metrics['reward']:.3f}")
        print(f"  Question: {sample.input_data['question'][:80]}...")
        print(f"  Ground truth: {sample.trajectory.metadata.get('prime_ground_truth')}")
        print(f"  Parsed answer: {sample.trajectory.metadata.get('prime_parsed_answer')}")

        # Show model response
        if sample.trajectory.messages:
            last_msg = sample.trajectory.messages[-1]
            if last_msg.role == "assistant":
                response_preview = (last_msg.content or "")[:100]
                print(f"  Model response: {response_preview}...")

    print(f"\n✅ Results saved to: {eval_config.output_dir}")
    print(f"   Report: {eval_config.output_dir}/report.json")
    print(f"   Samples: {eval_config.output_dir}/samples/")
    print(f"   Trajectories: {eval_config.output_dir}/trajectories/")


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

    # Run evaluation
    trio.run(run_evaluation, args.config)


if __name__ == "__main__":
    main()
