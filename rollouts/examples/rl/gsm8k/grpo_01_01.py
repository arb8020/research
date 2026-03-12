"""GSM8K GRPO baseline experiment.

Run with:
    python -m argus run --config examples/rl/gsm8k/grpo_01_01.py
    python -m argus run --config examples/rl/gsm8k/grpo_01_01.py --provision
"""

from examples.rl.gsm8k.base_config import train  # noqa: F401
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
)

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="gsm8k_grpo_01"),
    model=ModelConfig(name="Qwen/Qwen3-0.6B"),
    checkpoint=CheckpointConfig(num_steps=100, checkpoint_every=20),
    rollout=RolloutConfig(
        batch_size=8,
        n_samples_per_prompt=8,
        temperature=0.8,
    ),
    trainer=TrainerConfig(lr=1e-6, num_minibatches=8),
)
