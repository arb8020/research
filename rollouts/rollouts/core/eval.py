from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..training.types import AttemptRow, SampleScorer

if TYPE_CHECKING:
    from ..agents import RunConfig
    from ..dtypes import Endpoint, Environment, Trajectory


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    metrics: tuple[Metric, ...]

    @property
    def reward(self) -> float:
        weighted = [(metric.value, metric.weight) for metric in self.metrics if metric.weight > 0]
        if not weighted:
            return 0.0
        total_weight = sum(weight for _, weight in weighted)
        return sum(value * weight for value, weight in weighted) / total_weight


# TODO: Revisit the eval-stage contract. Current ScoreFn takes (trajectory, row),
# but the more honest contract may be a scorer over a raw execution result plus
# explicit scoring context. Keep the lower-level scoring primitive exposed even
# if higher-level convenience APIs materialize richer scored records on top.
ScoreFn = (
    Callable[["Trajectory", dict[str, Any]], Score]
    | Callable[["Trajectory", dict[str, Any]], Awaitable[Score]]
)
PrepareMessagesFn = Callable[[dict[str, Any]], list[Any]]
EnvironmentFactory = Callable[[dict[str, Any]], Any]
# TODO: AttemptExecutor currently returns AttemptRow for compatibility, but the
# intended stage split may be row -> AttemptResult, then scorer attaches
# evaluation / richer row semantics afterward.
AttemptExecutor = Callable[
    [dict[str, Any], str, "Environment | None", "RunConfig"],
    AttemptRow | Awaitable[AttemptRow],
]


@dataclass(frozen=True)
class EvalConfig:
    endpoint: Endpoint | None
    prepare_messages: PrepareMessagesFn | None
    # TODO: A true scored eval should probably require an explicit scoring stage
    # (score_fn, sample_scorer, or equivalent). Open-ended / attempt-only runs
    # may deserve a separate top-level API instead of weakening EvalConfig.
    score_fn: ScoreFn | None = None
    sample_scorer: SampleScorer | None = None
    environment: Environment | None = None
    # environment_factory is the preferred construction path for task environments.
    # It should be a thin wrapper around row_to_state + Environment.deserialize:
    #
    #   environment_factory = lambda row: MyEnvironment.deserialize(row_to_state(row))
    #
    # row_to_state is a pure function that maps dataset row columns to the state dict
    # the environment needs. It can be nearly identity (thin pointer to a registry) or
    # do real setup work (clone repo, create worktree) and return the resulting paths.
    # All I/O that constructs live resources belongs in Environment.deserialize, not here.
    environment_factory: EnvironmentFactory | None = None
    attempt_executor: AttemptExecutor | None = None
    run_config: RunConfig | None = None
    max_samples: int | None = None
    max_concurrent: int = 1
    output_dir: Path | None = None
    eval_name: str = "evaluation"
    config_path: str | None = None
    verbose: bool = True
    show_progress: bool = False
    stream_tokens: bool = False
    max_sample_retries: int = 5
    max_api_concurrent: int | None = None
    max_tool_concurrent: int | None = None
    resume_dir: Path | None = None
    report_batch_size: int = 1
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.prepare_messages is None and self.attempt_executor is None:
            raise ValueError("EvalConfig requires either prepare_messages or attempt_executor")
