"""nano-inference: Minimal inference engine for RL training.

See docs/design/nano_inference.md for design documentation.
"""

from rollouts.inference.types import (
    EngineConfig,
    SamplingParams,
    TrainingSample,
    SchedulerConfig,
    SchedulerOutput,
)
from rollouts.inference.sampling import sample_with_logprobs
from rollouts.inference.scheduler import schedule
from rollouts.inference.engine import InferenceEngine

__all__ = [
    # Config
    "EngineConfig",
    "SamplingParams",
    "SchedulerConfig",
    # Output
    "TrainingSample",
    "SchedulerOutput",
    # Pure functions
    "sample_with_logprobs",
    "schedule",
    # Engine
    "InferenceEngine",
]
