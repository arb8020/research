"""Reverse Text GRPO baseline experiment.

Run with:
    # Local (requires GPU + SGLang) - uses Prime's SFT model by default
    python examples/rl/reverse_text/grpo_01_01.py

    # Use base model (will likely fail - no SFT warmup!)
    python examples/rl/reverse_text/grpo_01_01.py --base-model

    # Remote (provisions GPU automatically)
    python examples/rl/reverse_text/grpo_01_01.py --provision

    # Reuse existing GPU
    python examples/rl/reverse_text/grpo_01_01.py --node-id runpod:abc123

Note:
    Using the base Qwen3-0.6B model without SFT warmup typically achieves
    only ~5% reward because the model doesn't know how to reverse text.

    Prime's SFT model (PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT) was
    trained on willcb/R1-reverse-wikipedia-paragraphs-v1-1000 and starts
    at ~50% reward, which RL can then improve to ~80%.

    For the full SFT → RL pipeline, see sft_then_grpo.py
"""

from examples.rl.reverse_text.base_config import train
from rollouts.training.grpo import GRPOConfig

# Default: Use Prime's pre-trained SFT model (recommended)
# This model already knows how to reverse text, so RL can refine it
DEFAULT_MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

# Alternative: Base model (will struggle without SFT warmup)
BASE_MODEL = "Qwen/Qwen3-0.6B"

# Prime-RL uses:
#   - max_steps=20 (prime-rl) or 100 (verifiers)
#   - max_tokens=128
#   - batch_size=128 total (8 prompts × 16 rollouts)
#   - lr=3e-6
#   - seq_len=512-2048
config = GRPOConfig(
    experiment_name="reverse_text_grpo_01",
    model_name=DEFAULT_MODEL,
    num_steps=100,  # verifiers uses 100, prime-rl uses 20
    checkpoint_every=1,  # Sync weights every step for on-policy training
    batch_size=8,  # prompts per step (× 16 rollouts = 128 total)
    n_samples_per_prompt=16,
    temperature=1.0,  # Prime uses default (1.0), not 0.7
    lr=3e-6,
    max_seq_len=512,  # verifiers uses 512
    max_tokens=128,  # both use 128
)

if __name__ == "__main__":
    import argparse
    from dataclasses import replace

    parser = argparse.ArgumentParser(description="Reverse Text GRPO training")
    parser.add_argument("--provision", action="store_true", help="Provision new GPU instance")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument("--node-id", type=str, help="Reuse existing instance ID")
    parser.add_argument("--tui", action="store_true", help="Show TUI monitor")
    parser.add_argument("--tui-debug", action="store_true", help="Print raw JSONL")
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Use base Qwen3-0.6B instead of SFT model (will likely fail)",
    )
    args = parser.parse_args()

    # Update config if using base model
    run_config = config
    if args.base_model:
        print("WARNING: Using base model without SFT warmup - expect ~5% reward")
        run_config = replace(config, model_name=BASE_MODEL)

    if args.provision or args.node_id:
        from examples.rl.base_config import run_remote

        run_remote(
            __file__,
            keep_alive=args.keep_alive,
            node_id=args.node_id,
            use_tui=args.tui,
            tui_debug=args.tui_debug,
        )
    else:
        results = train(config=run_config, num_samples=1000)
        print(f"Training complete. {len(results.get('metrics_history', []))} steps")
