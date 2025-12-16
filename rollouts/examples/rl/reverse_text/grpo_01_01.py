"""Reverse Text GRPO baseline.

Run: python examples/rl/reverse_text/grpo_01_01.py
"""

from examples.rl.reverse_text.base_config import RLConfig, train

config = RLConfig(
    experiment_name="reverse_text_grpo_01",
    num_steps=20,
)

if __name__ == "__main__":
    train(config)
