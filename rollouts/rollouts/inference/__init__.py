"""nano-inference: Minimal inference engine for RL training.

See docs/design/nano_inference.md for design documentation.
"""

from rollouts.inference.attention import (
    Attention,
    AttentionBackend,
    CacheConfig,
    FlexAttentionBackend,
    create_causal_block_mask,
)
from rollouts.inference.context import (
    allocate_and_build_context,
    build_decode_context,
    build_prefill_context,
    extend_and_build_context,
)
from rollouts.inference.engine import InferenceEngine
from rollouts.inference.sampling import sample_with_logprobs
from rollouts.inference.scheduler import schedule
from rollouts.inference.types import (
    EngineConfig,
    InferenceContext,
    SamplingParams,
    SchedulerConfig,
    SchedulerOutput,
    TrainingSample,
)

__all__ = [
    # Config
    "EngineConfig",
    "SamplingParams",
    "SchedulerConfig",
    "InferenceContext",
    "CacheConfig",
    # Output
    "TrainingSample",
    "SchedulerOutput",
    # Pure functions
    "sample_with_logprobs",
    "schedule",
    "create_causal_block_mask",
    "build_prefill_context",
    "build_decode_context",
    "allocate_and_build_context",
    "extend_and_build_context",
    # Protocols
    "AttentionBackend",
    # Classes (own state)
    "InferenceEngine",
    "FlexAttentionBackend",
    "Attention",
]
