from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..training.types import AttemptRow, SampleScorer

if TYPE_CHECKING:
    from ..agents import RunConfig
    from ..dtypes import Endpoint, Environment


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


ScoreFn = Callable[[AttemptRow], Score] | Callable[[AttemptRow], Awaitable[Score]]
PrepareMessagesFn = Callable[[dict[str, Any]], list[Any]]
EnvironmentFactory = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class EvalConfig:
    endpoint: Endpoint
    prepare_messages: PrepareMessagesFn
    score_fn: ScoreFn | None = None
    sample_scorer: SampleScorer | None = None
    environment: Environment | None = None
    environment_factory: EnvironmentFactory | None = None
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
        if self.score_fn is None and self.sample_scorer is None:
            raise ValueError("EvalConfig requires either score_fn or sample_scorer")
