#!/usr/bin/env python3
"""Qwen 0.5B on Reverse-Text - debug config.

Quick CI test: 10 steps, 100 samples. ~2 min on GPU.

Usage:
    python examples/sft/qwen_reverse_01_01.py
    python examples/sft/qwen_reverse_01_01.py --remote
"""

from dataclasses import replace

from base_config import BaseConfig, train, run_remote

config = replace(
    BaseConfig(),
    num_steps=10,
    max_samples=100,
    log_every=2,
    checkpoint_every=5,
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", help="Run on remote GPU")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument("--gpu-id", type=str, help="Reuse existing GPU instance ID")
    args = parser.parse_args()

    if args.remote or args.gpu_id:
        run_remote(__file__, keep_alive=args.keep_alive, gpu_id=args.gpu_id)
    else:
        metrics = train(config)
        if metrics:
            assert metrics[-1]["loss"] < metrics[0]["loss"], "Loss didn't decrease"
            print("PASSED")
