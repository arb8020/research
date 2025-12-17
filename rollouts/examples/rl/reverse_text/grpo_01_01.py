"""Reverse Text GRPO baseline experiment.

Run with:
    python examples/rl/reverse_text/grpo_01_01.py
"""

from examples.rl.reverse_text.base_config import train
from rollouts.training.grpo import GRPOConfig

# Prime-RL reverse text config
config = GRPOConfig(
    experiment_name="reverse_text_grpo_01",
    model_name="Qwen/Qwen3-0.6B",
    num_steps=20,
    checkpoint_every=10,
    batch_size=8,
    n_samples_per_prompt=16,
    temperature=0.7,
    lr=3e-6,
    max_seq_len=256,
    max_tokens=128,
)

if __name__ == "__main__":
    results = train(config=config, num_samples=1000)
    print(f"Training complete. {len(results.get('metrics_history', []))} steps")
