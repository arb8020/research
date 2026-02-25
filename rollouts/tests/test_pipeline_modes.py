"""Test all pipeline modes on reverse_text task (prime-rl CI style).

This test:
1. Runs GRPO on reverse_text with Prime's SFT model
2. Asserts reward improves over training
3. Tests all three pipeline modes: sync, async, true_pipeline

Run with:
    # Local (requires GPU + SGLang)
    pytest tests/test_pipeline_modes.py -v -s

    # Or directly
    python tests/test_pipeline_modes.py

Based on prime-rl's test pattern:
- Model: PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT
- Task: Reverse text
- Steps: 10 (reduced from 20 for faster CI)
- Assertion: Final reward > initial reward
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def extract_rewards_from_logs(output_dir: Path) -> list[float]:
    """Extract rewards from rollouts.jsonl."""
    rollouts_file = output_dir / "rollouts.jsonl"
    if not rollouts_file.exists():
        return []

    rewards = []
    with open(rollouts_file) as f:
        for line in f:
            try:
                record = json.loads(line)
                if "reward" in record and record["reward"] is not None:
                    rewards.append(float(record["reward"]))
            except (json.JSONDecodeError, ValueError):
                continue

    return rewards


def run_grpo_training(
    pipeline_mode: str,
    output_dir: Path,
    num_steps: int = 10,
    use_modal: bool = True,
) -> tuple[list[float], int]:
    """Run GRPO training and return (rewards, exit_code).

    Args:
        pipeline_mode: One of "sync", "async", "true_pipeline"
        output_dir: Directory for outputs
        num_steps: Number of training steps
        use_modal: Whether to use Modal for GPU

    Returns:
        (list of rewards, exit code)
    """
    # Create a temporary config file
    config_content = f'''
"""Test config for {pipeline_mode} mode."""
from dataclasses import replace
from examples.rl.reverse_text.grpo_01_01 import config as base_config

config = replace(
    base_config,
    output=replace(
        base_config.output,
        output_dir="{output_dir}",
        experiment_name="test_{pipeline_mode}",
    ),
    checkpoint=replace(
        base_config.checkpoint,
        num_steps={num_steps},
        pipeline_mode="{pipeline_mode}",
        weight_sync_mode="{"nccl" if pipeline_mode in ("async", "true_pipeline") else "disk"}",
    ),
)

from examples.rl.reverse_text.base_config import train
'''

    config_file = output_dir / f"config_{pipeline_mode}.py"
    config_file.write_text(config_content)

    # Run training
    cmd = [
        sys.executable,
        "-m",
        "rollouts.run",
        "--config",
        str(config_file),
    ]
    if use_modal:
        cmd.append("--modal")

    print(f"\n{'=' * 60}")
    print(f"Running pipeline_mode={pipeline_mode}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent,
        capture_output=False,  # Show output in real-time
        timeout=600,  # 10 minute timeout
    )

    rewards = extract_rewards_from_logs(output_dir)
    return rewards, result.returncode


def assert_reward_improves(rewards: list[float], mode: str, min_improvement: float = 0.05):
    """Assert that reward improves over training.

    Args:
        rewards: List of rewards from training
        mode: Pipeline mode name (for error messages)
        min_improvement: Minimum improvement required (default 5%)
    """
    assert len(rewards) > 0, f"[{mode}] No rewards recorded"

    # Compare first 10% vs last 10% of rewards
    n = len(rewards)
    early_rewards = rewards[: max(1, n // 10)]
    late_rewards = rewards[-max(1, n // 10) :]

    early_mean = sum(early_rewards) / len(early_rewards)
    late_mean = sum(late_rewards) / len(late_rewards)

    improvement = late_mean - early_mean

    print(f"\n[{mode}] Reward analysis:")
    print(f"  Early mean: {early_mean:.4f}")
    print(f"  Late mean:  {late_mean:.4f}")
    print(f"  Improvement: {improvement:.4f}")

    assert improvement > min_improvement, (
        f"[{mode}] Reward did not improve enough: "
        f"early={early_mean:.4f}, late={late_mean:.4f}, "
        f"improvement={improvement:.4f} < {min_improvement}"
    )


@pytest.mark.gpu
@pytest.mark.slow
class TestPipelineModes:
    """Test all pipeline modes work and improve reward."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory."""
        return tmp_path / "test_output"

    def test_sync_mode(self, output_dir):
        """Test sync pipeline mode (stop-and-go)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        rewards, exit_code = run_grpo_training("sync", output_dir, num_steps=10)
        assert exit_code == 0, "Training failed"
        assert_reward_improves(rewards, "sync")

    def test_async_mode(self, output_dir):
        """Test async pipeline mode (background sampling)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        rewards, exit_code = run_grpo_training("async", output_dir, num_steps=10)
        assert exit_code == 0, "Training failed"
        assert_reward_improves(rewards, "async")

    def test_true_pipeline_mode(self, output_dir):
        """Test true_pipeline mode (PipelineRL-style)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        rewards, exit_code = run_grpo_training("true_pipeline", output_dir, num_steps=10)
        assert exit_code == 0, "Training failed"
        assert_reward_improves(rewards, "true_pipeline")


def main():
    """Run all tests manually."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_base = Path(tmpdir)

        results = {}
        for mode in ["sync", "async", "true_pipeline"]:
            output_dir = output_base / mode
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                rewards, exit_code = run_grpo_training(
                    mode, output_dir, num_steps=10, use_modal=True
                )

                if exit_code != 0:
                    results[mode] = f"FAILED (exit code {exit_code})"
                    continue

                if len(rewards) == 0:
                    results[mode] = "FAILED (no rewards)"
                    continue

                # Check improvement
                n = len(rewards)
                early = rewards[: max(1, n // 10)]
                late = rewards[-max(1, n // 10) :]
                early_mean = sum(early) / len(early)
                late_mean = sum(late) / len(late)
                improvement = late_mean - early_mean

                if improvement > 0.05:
                    results[mode] = f"PASSED (improvement: {improvement:.4f})"
                else:
                    results[mode] = f"FAILED (improvement: {improvement:.4f})"

            except Exception as e:
                results[mode] = f"ERROR: {e}"

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for mode, result in results.items():
            print(f"  {mode}: {result}")

        # Exit with error if any failed
        if any("FAILED" in r or "ERROR" in r for r in results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()
