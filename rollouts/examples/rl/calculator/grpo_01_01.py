#!/usr/bin/env python3
"""Calculator GRPO - Base config.

Naming: grpo_01_01.py
- grpo: GRPO algorithm
- 01: experiment ID
- 01: parent ID (self = base config)

10 steps, small batch. Integration test for RL loop.

Usage:
    # Local (requires CUDA + inference server running)
    python examples/rl/calculator/grpo_01_01.py

    # Remote (provisions GPU automatically)
    python examples/rl/calculator/grpo_01_01.py --remote

    # Reuse existing GPU
    python examples/rl/calculator/grpo_01_01.py --gpu-id runpod:abc123
"""

from base_config import RLConfig, train, run_remote

# Base config - all defaults
config = RLConfig()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculator GRPO training")
    parser.add_argument("--remote", action="store_true", help="Run on remote GPU")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument("--gpu-id", type=str, help="Reuse existing GPU instance ID")
    args = parser.parse_args()

    if args.remote or args.gpu_id:
        run_remote(__file__, keep_alive=args.keep_alive, gpu_id=args.gpu_id)
    else:
        metrics = train(config)
        if metrics:
            # Basic sanity check: training should produce some reward signal
            print(f"\nFinal mean_reward: {metrics[-1]['mean_reward']:.3f}")
            print("PASSED" if len(metrics) == config.num_steps else "INCOMPLETE")
