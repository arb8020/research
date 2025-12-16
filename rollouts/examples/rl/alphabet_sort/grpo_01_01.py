"""Alphabet Sort GRPO baseline.

Run: python examples/rl/alphabet_sort/grpo_01_01.py
"""

from examples.rl.alphabet_sort.base_config import RLConfig, train

config = RLConfig(
    experiment_name="alphabet_sort_grpo_01",
    num_steps=100,
)

if __name__ == "__main__":
    train(config)
