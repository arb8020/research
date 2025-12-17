"""Calculator GRPO baseline experiment.

Run with:
    # Local (requires GPU + SGLang)
    python examples/rl/calculator/grpo_01_01.py

    # Remote (provisions GPU automatically)
    python examples/rl/calculator/grpo_01_01.py --remote

    # Reuse existing GPU
    python examples/rl/calculator/grpo_01_01.py --gpu-id runpod:abc123
"""

from examples.rl.calculator.base_config import train
from rollouts.training.grpo import GRPOConfig

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
    max_turns=10,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculator GRPO training")
    parser.add_argument("--remote", action="store_true", help="Run on remote GPU")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument("--gpu-id", type=str, help="Reuse existing GPU instance ID")
    parser.add_argument("--tui", action="store_true", help="Show TUI monitor")
    parser.add_argument("--tui-debug", action="store_true", help="Print raw JSONL")
    args = parser.parse_args()

    if args.remote or args.gpu_id:
        from examples.rl.base_config import run_remote

        run_remote(
            __file__,
            keep_alive=args.keep_alive,
            gpu_id=args.gpu_id,
            use_tui=args.tui,
            tui_debug=args.tui_debug,
        )
    else:
        results = train(config=config, max_samples=12)
        print(f"Training complete. {len(results.get('metrics_history', []))} steps")
