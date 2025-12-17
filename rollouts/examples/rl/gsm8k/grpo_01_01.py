"""GSM8K GRPO baseline experiment.

Run with:
    python examples/rl/gsm8k/grpo_01_01.py
"""

from examples.rl.gsm8k.base_config import train
from rollouts.training.grpo import GRPOConfig

# Base GSM8K config following Miles recipe
config = GRPOConfig(
    experiment_name="gsm8k_grpo_01",
    model_name="Qwen/Qwen3-0.6B",
    num_steps=100,
    checkpoint_every=20,
    batch_size=8,  # 8 unique prompts
    n_samples_per_prompt=8,  # 8 samples per prompt = 64 total
    temperature=0.8,
    lr=1e-6,  # Miles uses 1e-6
    num_minibatches=8,  # 64 / 8 = 8 samples per minibatch
)

if __name__ == "__main__":
    results = train(config=config, max_samples=1000)
    print(f"Training complete. {len(results.get('metrics_history', []))} steps")
