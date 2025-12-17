"""Calculator GRPO baseline experiment.

Run with:
    python examples/rl/calculator/grpo_01_01.py
"""

from examples.rl.calculator.base_config import train

from rollouts.training.grpo import GRPOConfig

# Calculator GRPO config
config = GRPOConfig(
    experiment_name="calculator_grpo_01",
    model_name="Qwen/Qwen3-0.6B",
    num_steps=10,
    checkpoint_every=5,
    batch_size=4,
    n_samples_per_prompt=4,
    temperature=0.7,
    lr=1e-5,
    max_seq_len=2048,
    max_tokens=512,
    max_turns=10,  # Multi-turn for tool use
)

if __name__ == "__main__":
    results = train(config=config, max_samples=12)
    print(f"Training complete. {len(results.get('metrics_history', []))} steps")
