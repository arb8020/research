"""Alphabet Sort GRPO baseline experiment.

Run with:
    python examples/rl/alphabet_sort/grpo_01_01.py
"""

from examples.rl.alphabet_sort.base_config import train
from rollouts.training.grpo import GRPOConfig

# Prime-RL alphabet sort config
config = GRPOConfig(
    experiment_name="alphabet_sort_grpo_01",
    model_name="Qwen/Qwen3-0.6B",
    num_steps=100,
    checkpoint_every=20,
    batch_size=4,
    n_samples_per_prompt=8,
    temperature=0.7,
    lr=1e-6,
    num_minibatches=4,
    max_seq_len=1024,
    max_tokens=256,
)

if __name__ == "__main__":
    results = train(config=config, num_episodes=500)
    print(f"Training complete. {len(results.get('metrics_history', []))} steps")
