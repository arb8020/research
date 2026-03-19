from __future__ import annotations

from .scoring import resolve_scorer
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

    return RolloutRuntime(
        generate_fn=config.generate_fn,
        filter_fn=config.filter_fn,
        scorer=resolve_scorer(config=config),
    )
