"""Planned throughput/latency benchmark for the trtllm realization.

This file is intentionally not runnable yet.

Why it is still a stub:
  - the benchmark workload/SLA surface is now shared with sglang/vllm
  - but the repo runtime spec for ``trtllm`` still requires a separate
    inference environment that the eval launcher does not honestly realize yet

Keep this file aligned with the real benchmark input model so the remaining gap
is explicit: runtime realization, not config shape.
"""

from __future__ import annotations

from examples.inference.bench_config_lib import (
    InferenceBenchSLA,
    InferenceBenchWorkload,
)

MODEL = "Qwen/Qwen3-0.6B"
PORT = 30003

WORKLOAD = InferenceBenchWorkload(
    kind="random",
    num_prompts=200,
    input_len=512,
    output_len=256,
    seed=42,
    max_concurrent=16,
)

SLA = InferenceBenchSLA()

NOT_YET_RUNNABLE_REASON = (
    "trtllm benchmark is stubbed only. The benchmark data model is shared, but "
    "the current eval launcher does not yet realize the separate inference env "
    "that trtllm requires."
)
