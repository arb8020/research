"""Reverse Text GRPO with full PipelineRL-style training.

This uses:
- NCCL for weight sync (GPU-to-GPU, no disk I/O)
- Async pipeline (sampling runs in background while training)
- max_lag=2 (samples up to 2 weight versions behind are used)

Run with:
    # Modal (recommended for testing - has NCCL + multi-GPU support)
    python examples/rl/reverse_text/grpo_pipeline_01.py --modal

    # Local (requires GPU + SGLang with NCCL support)
    python examples/rl/reverse_text/grpo_pipeline_01.py
"""

from dataclasses import replace

from examples.rl.reverse_text.grpo_01_01 import config as base_config

# Enable full PipelineRL-style training:
# - NCCL weight sync (GPU-to-GPU)
# - Async pipeline (background sampling)
# - max_lag=2 (allow slightly stale samples)
config = replace(
    base_config,
    checkpoint=replace(
        base_config.checkpoint,
        weight_sync_mode="nccl",  # GPU-to-GPU weight broadcast
        pipeline_mode="async",  # Background sampling
        max_lag=2,  # Allow samples up to 2 versions behind
        pipeline_queue_size=512,  # Buffer size for samples
    ),
)

if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
