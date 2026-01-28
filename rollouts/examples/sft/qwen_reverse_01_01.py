#!/usr/bin/env python3
"""Qwen 0.5B on Reverse-Text.

100 steps, 1000 samples. ~5 min on GPU.

Usage:
    python examples/sft/qwen_reverse_01_01.py
    python examples/sft/qwen_reverse_01_01.py --provision
"""

from dataclasses import replace

from base_config import BaseConfig, run_remote, train

from rollouts.training.configs import CheckpointConfig

config = replace(
    BaseConfig(),
    checkpoint=CheckpointConfig(num_steps=100, log_every=10, checkpoint_every=50),
    dataset=replace(BaseConfig().dataset, max_samples=1000),
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provision", action="store_true", help="Provision new GPU instance")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument("--node-id", type=str, help="Reuse existing instance ID")
    args = parser.parse_args()

    if args.provision or args.node_id:
        run_remote(__file__, keep_alive=args.keep_alive, node_id=args.node_id)
    else:
        metrics = train(config)
        if metrics:
            assert metrics[-1]["loss"] < metrics[0]["loss"], "Loss didn't decrease"
            print("PASSED")
