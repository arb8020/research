"""KernelBench GRPO with TRUE PipelineRL mode.

This is the most aggressive pipelining option:
- NCCL for weight sync (GPU-to-GPU, no disk I/O)
- Background sampling (inference never stops)
- Non-blocking weight sync (training never waits)
- max_lag=2 (samples up to 2 weight versions behind are used)

Run with:
    # Modal (recommended)
    python examples/rl/kernelbench/grpo_true_pipeline_01.py --modal

    # Local (requires GPU + SGLang with NCCL support)
    python examples/rl/kernelbench/grpo_true_pipeline_01.py

Note:
    PipelineRL trades some staleness for throughput. During weight sync,
    some layers may have new weights while others have old weights.
    This is generally fine for RL training.
"""

from dataclasses import replace

from examples.rl.kernelbench.base_config import train  # noqa: F401 (used by runner)
from examples.rl.kernelbench.grpo_01_01 import config as base_config

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
        pipeline_queue_size=256,  # Buffer size for samples (smaller than reverse_text due to longer sequences)
    ),
)

if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
