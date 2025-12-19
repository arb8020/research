"""Reverse Text GRPO baseline experiment.

Run with:
    # Local (requires GPU + SGLang)
    python examples/rl/reverse_text/grpo_01_01.py

    # Remote (provisions GPU automatically)
    python examples/rl/reverse_text/grpo_01_01.py --remote

    # Reuse existing GPU
    python examples/rl/reverse_text/grpo_01_01.py --node-id runpod:abc123
"""

from examples.rl.reverse_text.base_config import train
from rollouts.training.grpo import GRPOConfig

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
    import argparse

    parser = argparse.ArgumentParser(description="Reverse Text GRPO training")
    parser.add_argument("--remote", action="store_true", help="Run on remote GPU")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument("--node-id", type=str, help="Reuse existing instance ID")
    parser.add_argument("--tui", action="store_true", help="Show TUI monitor")
    parser.add_argument("--tui-debug", action="store_true", help="Print raw JSONL")
    args = parser.parse_args()

    if args.remote or args.node_id:
        from examples.rl.base_config import run_remote

        run_remote(
            __file__,
            keep_alive=args.keep_alive,
            node_id=args.node_id,
            use_tui=args.tui,
            tui_debug=args.tui_debug,
        )
    else:
        results = train(config=config, num_samples=1000)
        print(f"Training complete. {len(results.get('metrics_history', []))} steps")
