"""GSM8K GRPO baseline experiment.

Run with:
    python examples/rl/gsm8k/grpo_01_01.py

Or deploy remotely:
    python -m bifrost deploy examples/rl/gsm8k/grpo_01_01.py --gpu H100
"""

from dataclasses import replace

from examples.rl.gsm8k.base_config import (
    DatasetConfig,
    ModelConfig,
    OrchestratorConfig,
    RLConfig,
    TrainerConfig,
    train,
)

# Base GSM8K config following Miles recipe
config = RLConfig(
    experiment_name="gsm8k_grpo_01",
    num_steps=100,
    checkpoint_every=20,
    dataset=DatasetConfig(
        hf_split="train",
        max_samples=1000,  # Start with subset
    ),
    model=ModelConfig(
        name="Qwen/Qwen3-0.6B",
    ),
    orchestrator=OrchestratorConfig(
        batch_size=8,  # 8 unique prompts
        rollouts_per_example=8,  # 8 samples per prompt = 64 total
        temperature=0.8,
    ),
    trainer=TrainerConfig(
        lr=1e-6,  # Miles uses 1e-6
        num_minibatches=8,  # 64 / 8 = 8 samples per minibatch
    ),
)

if __name__ == "__main__":
    metrics = train(config)
    print(f"Training complete. Final metrics: {len(metrics.get('metrics_history', []))} steps")
