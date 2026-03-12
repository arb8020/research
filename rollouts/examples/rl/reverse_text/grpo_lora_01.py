"""Reverse Text GRPO with LoRA.

Demonstrates RL fine-tuning with LoRA adapters for parameter-efficient training.
Uses ~10-100x higher learning rate than full fine-tuning as recommended by
the LoRA research (https://thinkingmachines.ai/blog/lora/).

Key insight from the blog: LoRA matches full fine-tuning for RL even at rank-1,
requiring only ~1 bit per episode. This makes it ideal for test-time training.

Run:
    # Local (requires GPU)
    python rollouts/run.py --config examples/rl/reverse_text/grpo_lora_01.py --local

    # Remote (RunPod A100)
    python rollouts/run.py --config examples/rl/reverse_text/grpo_lora_01.py --provision --provider runpod

    # CI mode (asserts reward >= 0.65)
    ROLLOUTS_CHECK_REWARD=1 python rollouts/run.py --config examples/rl/reverse_text/grpo_lora_01.py --local
"""

from examples.rl.reverse_text.base_config import train  # noqa: F401
from rollouts.training.configs import HardwareConfig
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    InferenceConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
)

# =============================================================================
# Hardware Configuration
# =============================================================================

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=1,
    provider="runpod",
    legacy_remote_bootstrap=True,
)

# =============================================================================
# Training Configuration
# =============================================================================

# Use Prime's SFT model (already knows how to reverse text)
DEFAULT_MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

REWARD_THRESHOLD = 0.65

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="reverse_text_grpo_lora_01"),
    model=ModelConfig(
        name=DEFAULT_MODEL,
        # LoRA configuration
        use_lora=True,
        lora_rank=16,  # Rank 16 is plenty for this task
        lora_alpha=32,  # Alpha = 2 * rank is a good default
    ),
    checkpoint=CheckpointConfig(
        num_steps=20,
        checkpoint_every=10,
        sync_weights_every=1,  # On-policy: sync after every step
    ),
    rollout=RolloutConfig(
        batch_size=8,
        n_samples_per_prompt=16,
        temperature=1.0,
        max_seq_len=2048,
        max_tokens=128,
    ),
    trainer=TrainerConfig(
        # KEY: LoRA requires ~10-100x higher LR than full fine-tuning
        # Full FT uses 3e-6, so we use 1e-4 (about 30x higher)
        lr=1e-4,
        num_minibatches=32,
        loss_type="masked",  # Prime-RL ratio masking works well with LoRA
    ),
    inference=InferenceConfig(
        mem_fraction=0.5,
    ),
)


def check_reward_threshold(results: dict, threshold: float = REWARD_THRESHOLD) -> None:
    """Assert final reward meets threshold (for CI)."""
    metrics_history = results.get("metrics_history", [])
    if not metrics_history:
        raise AssertionError("No metrics recorded - training may have failed")

    final_reward = metrics_history[-1].get("mean_reward", 0.0)
    if final_reward < threshold:
        raise AssertionError(f"Final reward {final_reward:.4f} below threshold {threshold:.2f}")
    print(f"Final reward {final_reward:.4f} >= {threshold:.2f} (PASSED)")


if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
