"""Alphabet Sort - matches prime-rl nightly CI exactly.

prime-rl config: examples/alphabet_sort/rl.toml
prime-rl test: tests/nightly/test_alphabet_sort.py

Run:
    # Local
    python examples/rl/basic/alphabet_sort.py

    # Modal (recommended for CI)
    python examples/rl/basic/alphabet_sort.py --modal

    # RunPod
    python examples/rl/basic/alphabet_sort.py --provision --provider runpod

    # CI mode (checks reward goes up)
    ROLLOUTS_CHECK_REWARD=1 python examples/rl/basic/alphabet_sort.py --modal

Expected: Reward should increase during training (no fixed threshold)
"""

from examples.rl.alphabet_sort.base_config import (
    alphabet_sort_score_fn,
    generate_alphabet_sort_prompts,
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
# From: examples/alphabet_sort/rl.toml
#
# max_steps = 200
# seq_len = 2048
# [model] name = "Qwen/Qwen3-4B-Instruct-2507"
# [orchestrator] batch_size = 512, rollouts_per_example = 8
# [orchestrator.sampling] max_tokens = 768
# [trainer.optim] lr = 1e-5
# [trainer.model.lora] rank = 32, alpha = 64
# [[orchestrator.env]] args = { min_turns = 3, max_turns = 5 }
#
# Nightly test: checks reward_goes_up (no fixed threshold)
# ─────────────────────────────────────────────────────────────────────────────

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="basic_alphabet_sort"),
    model=ModelConfig(
        name="Qwen/Qwen3-4B-Instruct-2507",
        use_lora=True,
        lora_rank=32,
        lora_alpha=64,
    ),
    checkpoint=CheckpointConfig(
        num_steps=200,
        checkpoint_every=20,
        sync_weights_every=1,
    ),
    rollout=RolloutConfig(
        batch_size=64,  # 64 prompts × 8 rollouts = 512 total (matches prime-rl)
        n_samples_per_prompt=8,
        temperature=1.0,
        max_seq_len=2048,
        max_tokens=768,
    ),
    trainer=TrainerConfig(
        lr=1e-5,
        num_minibatches=64,
        loss_type="masked",
    ),
    inference=InferenceConfig(
        mem_fraction=0.5,
    ),
)


def check_reward_goes_up(results: dict) -> None:
    """Assert reward increases during training."""
    metrics = results.get("metrics_history", [])
    if len(metrics) < 2:
        raise AssertionError("Not enough metrics to check trend")

    rewards = [m.get("mean_reward", 0.0) for m in metrics]
    # Compare first quarter avg to last quarter avg
    quarter = max(1, len(rewards) // 4)
    first_avg = sum(rewards[:quarter]) / quarter
    last_avg = sum(rewards[-quarter:]) / quarter

    if last_avg <= first_avg:
        raise AssertionError(
            f"Reward did not improve: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
        )
    print(f"✓ Reward improved: {first_avg:.4f} -> {last_avg:.4f} (PASSED)")


def train(config: GRPOConfig = config, num_episodes: int = 500) -> dict:
    """Run training, optionally checking reward trend."""
    import os

    prompts = generate_alphabet_sort_prompts(
        num_episodes=num_episodes,
        min_turns=3,
        max_turns=5,
    )
    results = grpo_train(
        config=config,
        prompts=prompts,
        score_fn=alphabet_sort_score_fn,
        environment_cls=BasicEnvironment,
    )

    if os.environ.get("ROLLOUTS_CHECK_REWARD", "").lower() in ("1", "true"):
        check_reward_goes_up(results)

    return results


if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
