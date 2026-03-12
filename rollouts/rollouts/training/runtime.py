from __future__ import annotations

from .scoring import FunctionSampleScorer
from .types import RolloutConfig, RolloutRuntime


def resolve_rollout_runtime(
    *,
    config: RolloutConfig | None = None,
    runtime: RolloutRuntime | None = None,
) -> RolloutRuntime | None:
    """Resolve runtime wiring with explicit precedence.

    Precedence:
    1. Explicit runtime argument
    2. Legacy callable fields on RolloutConfig
    """

    if runtime is not None:
        return runtime
    if config is None or config.generate_fn is None:
        return None

    sample_scorer = config.sample_scorer
    if sample_scorer is None and config.score_fn is not None:
        sample_scorer = FunctionSampleScorer(config.score_fn)

    return RolloutRuntime(
        generate_fn=config.generate_fn,
        filter_fn=config.filter_fn,
        sample_scorer=sample_scorer,
    )
