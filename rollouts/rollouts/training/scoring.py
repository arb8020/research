from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .types import RolloutConfig, RolloutRuntime, Sample, SampleScorer


@dataclass(frozen=True)
class FunctionSampleScorer:
    """Adapter that turns the legacy score_fn into an explicit scoring stage."""

    score_fn: Callable[[Sample], Any]

    async def score_samples(self, samples: list[Sample]) -> list[Sample]:
        is_async = inspect.iscoroutinefunction(self.score_fn)

        for sample in samples:
            score = await self.score_fn(sample) if is_async else self.score_fn(sample)
            sample.score = score
            sample.reward = score.reward

        return samples


def resolve_sample_scorer(
    *,
    config: RolloutConfig | None = None,
    runtime: RolloutRuntime | None = None,
    sample_scorer: SampleScorer | None = None,
    score_fn: Callable[[Sample], Any] | None = None,
) -> SampleScorer | None:
    """Resolve the scoring stage with explicit precedence.

    Precedence:
    1. Explicit sample_scorer argument
    2. RolloutRuntime.sample_scorer
    3. RolloutConfig.sample_scorer
    4. Legacy score_fn argument
    5. RolloutConfig.score_fn
    """

    if sample_scorer is not None:
        return sample_scorer
    if runtime is not None and runtime.sample_scorer is not None:
        return runtime.sample_scorer
    if config is not None and config.sample_scorer is not None:
        return config.sample_scorer
    if score_fn is not None:
        return FunctionSampleScorer(score_fn)
    if config is not None and config.score_fn is not None:
        return FunctionSampleScorer(config.score_fn)
    return None
