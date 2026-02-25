"""Reverse Text GRPO with TRUE PipelineRL mode.

This is the most aggressive pipelining option:
- NCCL for weight sync (GPU-to-GPU, no disk I/O)
- Background sampling (inference never stops)
- Non-blocking weight sync (training never waits)
- max_lag=2 (samples up to 2 weight versions behind are used)

This matches ServiceNow's PipelineRL paper exactly.

Warning:
    During weight sync, some layers may have new weights while others
    have old weights. PipelineRL accepts this for throughput.

Run with:
    # Modal (recommended for testing)
    python examples/rl/reverse_text/grpo_true_pipeline_01.py --modal

    # Local (requires GPU + SGLang with NCCL support)
    python examples/rl/reverse_text/grpo_true_pipeline_01.py
"""

from dataclasses import replace

from examples.rl.reverse_text.base_config import train  # noqa: F401 (used by runner)
from examples.rl.reverse_text.grpo_01_01 import config as base_config

# Enable TRUE PipelineRL-style training:
# - NCCL weight sync (GPU-to-GPU)
# - Async pipeline (background sampling)
# - Non-blocking weight sync (training never waits for sync)
config = replace(
    base_config,
    checkpoint=replace(
        base_config.checkpoint,
        weight_sync_mode="nccl",  # GPU-to-GPU weight broadcast
        pipeline_mode="true_pipeline",  # Both sampling AND sync non-blocking
        max_lag=2,  # Allow samples up to 2 versions behind
        pipeline_queue_size=512,  # Buffer size for samples
    ),
)

if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
