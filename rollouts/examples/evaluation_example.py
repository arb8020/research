"""Example: Using the evaluation framework with composable rewards.

This shows how to:
1. Define reward functions (Trajectory -> Trajectory)
2. Create EvalConfig
3. Run evaluation with agent framework
4. Compose multiple reward signals manually
"""

import asyncio
from dataclasses import replace
from pathlib import Path

from rollouts.dtypes import Endpoint, Message, Trajectory
from rollouts.environments.calculator import CalculatorEnvironment
from rollouts.evaluation import EvalConfig, evaluate, load_jsonl

# ─────────────────────────────────────────────────────────────────────────────
# Define Reward Functions (Trajectory -> Trajectory)
# ─────────────────────────────────────────────────────────────────────────────


def correctness_reward(trajectory: Trajectory) -> Trajectory:
    """Binary correctness: 1.0 if correct, 0.0 otherwise."""
    sample_data = trajectory.metadata.get("sample_data", {})
    ground_truth = sample_data.get("answer")

    # Extract final answer from trajectory
    final_message = trajectory.messages[-1] if trajectory.messages else None
    predicted_answer = extract_answer(final_message.content if final_message else "")

    # Compute score
    correct = (predicted_answer == ground_truth)
    score = 1.0 if correct else 0.0

    # Return trajectory with reward
    return replace(trajectory, rewards=score)


def efficiency_reward(trajectory: Trajectory) -> Trajectory:
    """Reward based on number of turns (fewer is better)."""
    num_turns = len([m for m in trajectory.messages if m.role == "assistant"])

    # Ideal: 3 turns or less (no penalty)
    # Each extra turn: -0.1
    if num_turns <= 3:
        score = 1.0
    else:
        score = max(0.0, 1.0 - (num_turns - 3) * 0.1)

    return replace(trajectory, rewards=score)


def composite_reward(trajectory: Trajectory) -> Trajectory:
    """Manually compose multiple reward signals with weights.

    This shows how users can compose rewards themselves without
    needing RewardComponent/CompositeReward classes.
    """
    # Compute individual signals
    correctness_traj = correctness_reward(trajectory)
    correctness_score = correctness_traj.rewards

    efficiency_traj = efficiency_reward(trajectory)
    efficiency_score = efficiency_traj.rewards

    # Compose with weights
    total_reward = correctness_score * 1.0 + efficiency_score * 0.2

    # Optional: store breakdown for debugging
    metadata = {
        **trajectory.metadata,
        "reward_breakdown": {
            "correctness": correctness_score,
            "efficiency": efficiency_score,
            "correctness_weight": 1.0,
            "efficiency_weight": 0.2,
        }
    }

    return replace(trajectory, rewards=total_reward, metadata=metadata)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> float:
    """Extract numerical answer from text."""
    # Simple extraction - look for "Answer:" or just parse last number
    if "Answer:" in text:
        text = text.split("Answer:")[-1]

    # Extract first number found
    import re
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return float(numbers[0])
    return 0.0


def prepare_messages(sample_data: dict) -> list[Message]:
    """Convert sample data to initial messages."""
    prompt = sample_data.get("question", "")
    return [Message(role="user", content=prompt)]


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """Run evaluation example."""

    # Create eval config
    config = EvalConfig(
        reward_fn=composite_reward,  # Use composite reward function
        max_turns=10,
        max_concurrent=4,  # Parallel evaluation
        max_samples=10,  # Just evaluate 10 samples for demo
        eval_name="calculator_eval",
        output_dir=Path("results/calculator_eval"),
        verbose=True,
    )

    # Setup endpoint
    endpoint = Endpoint(
        provider="sglang",
        model="Qwen/Qwen2.5-7B-Instruct",
        api_base="http://localhost:30000/v1",
    )

    # Load dataset
    dataset_path = Path("data/calculator_dataset.jsonl")
    dataset = load_jsonl(dataset_path)

    # Run evaluation
    report = await evaluate(
        dataset=dataset,
        prepare_messages=prepare_messages,
        environment_factory=lambda: CalculatorEnvironment(),  # Fresh env per sample
        endpoint=endpoint,
        config=config,
        dataset_path=str(dataset_path),
    )

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Total samples: {report.total_samples}")
    print(f"Mean reward: {report.summary_metrics['mean_reward']:.3f}")
    print(f"Mean correctness: {report.summary_metrics.get('mean_correctness', 0):.3f}")
    print(f"Mean efficiency: {report.summary_metrics.get('mean_efficiency', 0):.3f}")
    print(f"\nResults saved to: {config.output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Alternative: Simple Reward (No Composition)
# ─────────────────────────────────────────────────────────────────────────────

async def simple_example():
    """Example with just a single reward function (no composition)."""

    config = EvalConfig(
        reward_fn=correctness_reward,  # Just use correctness
        max_turns=10,
        eval_name="simple_eval",
        verbose=True,
    )

    endpoint = Endpoint(
        provider="sglang",
        model="Qwen/Qwen2.5-7B-Instruct",
        api_base="http://localhost:30000/v1",
    )

    dataset = load_jsonl(Path("data/calculator_dataset.jsonl"))

    report = await evaluate(
        dataset=dataset,
        prepare_messages=prepare_messages,
        environment_factory=lambda: CalculatorEnvironment(),
        endpoint=endpoint,
        config=config,
        dataset_path="data/calculator_dataset.jsonl",
    )

    print(f"Mean accuracy: {report.summary_metrics['mean_reward']:.1%}")


if __name__ == "__main__":
    # Run composite reward example
    asyncio.run(main())

    # Or run simple example
    # asyncio.run(simple_example())
