"""Reverse Text - matches prime-rl nightly CI exactly.

prime-rl config: examples/reverse_text/rl.toml
prime-rl test: tests/nightly/test_reverse_text.py

Run:
    # Local
    python examples/rl/basic/reverse_text.py

    # Modal (recommended for CI)
    python examples/rl/basic/reverse_text.py --modal

    # RunPod
    python examples/rl/basic/reverse_text.py --provision --provider runpod

    # CI mode (asserts reward >= 0.65)
    ROLLOUTS_CHECK_REWARD=1 python examples/rl/basic/reverse_text.py --modal

Expected: Final reward >= 0.65 after 20 steps
"""

from examples.rl.reverse_text.base_config import (
    load_reverse_text_prompts,
    reverse_text_score_fn,
)
from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    InferenceConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
    grpo_train,
)

# ──────────────────────── prime-rl nightly config ────────────────────────────
# From: examples/reverse_text/rl.toml
#
# max_steps = 20
# seq_len = 2048
# [model] name = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
# [orchestrator] batch_size = 128, rollouts_per_example = 16
# [orchestrator.sampling] max_tokens = 128
# [trainer.optim] lr = 3e-6
#
# Nightly test threshold: reward >= 0.65
# ─────────────────────────────────────────────────────────────────────────────

REWARD_THRESHOLD = 0.65

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="basic_reverse_text"),
    model=ModelConfig(name="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"),
    checkpoint=CheckpointConfig(
        num_steps=20,
        checkpoint_every=5,
        sync_weights_every=1,
    ),
    rollout=RolloutConfig(
        batch_size=8,  # 8 prompts × 16 rollouts = 128 total (matches prime-rl)
        n_samples_per_prompt=16,
        temperature=1.0,
        max_seq_len=2048,
        max_tokens=128,
    ),
    trainer=TrainerConfig(
        lr=3e-6,
        num_minibatches=32,
        loss_type="masked",
    ),
    inference=InferenceConfig(
        mem_fraction=0.5,
    ),
)


def check_reward_threshold(results: dict, threshold: float = REWARD_THRESHOLD) -> None:
    """Assert final reward meets threshold."""
    metrics = results.get("metrics_history", [])
    if not metrics:
        raise AssertionError("No metrics recorded")
    final_reward = metrics[-1].get("mean_reward", 0.0)
    if final_reward < threshold:
        raise AssertionError(f"Final reward {final_reward:.4f} < {threshold}")
    print(f"✓ Final reward {final_reward:.4f} >= {threshold} (PASSED)")


def train(config: GRPOConfig = config, num_samples: int = 1000) -> dict:
    """Run training, optionally checking reward threshold."""
    import os

    prompts = load_reverse_text_prompts(max_samples=num_samples)
    results = grpo_train(
        config=config,
        prompts=prompts,
        score_fn=reverse_text_score_fn,
        environment_cls=BasicEnvironment,
    )

    if os.environ.get("ROLLOUTS_CHECK_REWARD", "").lower() in ("1", "true"):
        check_reward_threshold(results)

    return results


if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
