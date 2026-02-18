"""Reverse Text GRPO with NCCL in-flight weight sync.

This is the same as grpo_01_01.py but uses NCCL for weight sync instead of disk.
This enables PipelineRL-style in-flight updates (GPU-to-GPU broadcast).

Run with:
    # Modal (recommended for testing - has NCCL support)
    python examples/rl/reverse_text/grpo_nccl_01.py --modal

    # Local (requires GPU + SGLang with NCCL support)
    python examples/rl/reverse_text/grpo_nccl_01.py
"""

from dataclasses import replace

from examples.rl.reverse_text.grpo_01_01 import config as base_config
from rollouts.training.grpo import CheckpointConfig

# Enable NCCL in-flight weight sync (PipelineRL-style)
config = replace(
    base_config,
    checkpoint=replace(
        base_config.checkpoint,
        weight_sync_mode="nccl",  # GPU-to-GPU broadcast instead of disk
    ),
)

if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
