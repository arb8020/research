"""Reverse Text GRPO baseline experiment.

Matches prime-rl nightly CI configuration: examples/reverse_text/rl.toml

Run with:
    # Default: RunPod A100
    python rollouts/run_rl.py --config examples/rl/reverse_text/grpo_01_01.py

    # Modal (fast ~30s cold start)
    python rollouts/run_rl.py --config examples/rl/reverse_text/grpo_01_01.py --provider modal

    # Local (requires GPU)
    python rollouts/run_rl.py --config examples/rl/reverse_text/grpo_01_01.py --local

    # CI mode (asserts reward >= 0.65)
    ROLLOUTS_CHECK_REWARD=1 python rollouts/run_rl.py --config examples/rl/reverse_text/grpo_01_01.py --provider modal

Note:
    Using the base Qwen3-0.6B model without SFT warmup typically achieves
    only ~5% reward because the model doesn't know how to reverse text.

    Prime's SFT model (PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT) was
    trained on willcb/R1-reverse-wikipedia-paragraphs-v1-1000 and starts
    at ~50% reward, which RL can then improve to ~80%.

    For the full SFT → RL pipeline, see sft_then_grpo.py
"""

from examples.rl.reverse_text.base_config import train as _base_train
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
)

# =============================================================================
# Training Configuration
# =============================================================================

# Default: Use Prime's pre-trained SFT model (recommended)
# This model already knows how to reverse text, so RL can refine it
DEFAULT_MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

# Alternative: Base model (will struggle without SFT warmup)
BASE_MODEL = "Qwen/Qwen3-0.6B"

# Matches prime-rl nightly CI: examples/reverse_text/rl.toml
# - batch_size=128, rollouts_per_example=16, max_tokens=128
# - seq_len=2048, max_steps=20, lr=3e-6
# - Tested nightly: reward must reach >= 0.65
REWARD_THRESHOLD = 0.65  # prime-rl nightly CI threshold

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="reverse_text_grpo_01"),
    model=ModelConfig(name=DEFAULT_MODEL),
    checkpoint=CheckpointConfig(
        num_steps=100,
        checkpoint_every=20,
        sync_weights_every=1,  # on-policy
    ),
    rollout=RolloutConfig(
        batch_size=8,  # prompts per step (× 16 rollouts = 128 total)
        n_samples_per_prompt=16,
        temperature=1.0,
        max_seq_len=2048,  # Match prime-rl nightly (reverse_text needs <256 anyway)
        max_tokens=128,
    ),
    trainer=TrainerConfig(
        lr=3e-6,
        num_minibatches=32,  # 128 total / 32 = micro_batch_size=4 (fits 24GB GPU)
        loss_type="masked",  # Importance sampling with ratio masking (GRPO always uses token-level)
    ),
    inference=InferenceConfig(
        mem_fraction=0.5,  # Leave room for training
    ),
)

# For base model variant, create a separate config file or use:
# python -m rollouts.run --config examples/rl/reverse_text/grpo_01_01.py


def check_reward_threshold(results: dict, threshold: float = REWARD_THRESHOLD) -> None:
    """Assert final reward meets threshold (for CI).

    Raises AssertionError if final reward < threshold.
    """
    metrics_history = results.get("metrics_history", [])
    if not metrics_history:
        raise AssertionError("No metrics recorded - training may have failed")

    final_reward = metrics_history[-1].get("mean_reward", 0.0)
    if final_reward < threshold:
        raise AssertionError(f"Final reward {final_reward:.4f} below threshold {threshold:.2f}")
    print(f"✓ Final reward {final_reward:.4f} >= {threshold:.2f} (PASSED)")


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    """Run training, optionally checking reward threshold.

    Set ROLLOUTS_CHECK_REWARD=1 to assert final reward >= REWARD_THRESHOLD (for CI).
    """
    import os

    results = _base_train(config=config, **kwargs)

    if os.environ.get("ROLLOUTS_CHECK_REWARD", "").lower() in ("1", "true"):
        check_reward_threshold(results)

    return results
