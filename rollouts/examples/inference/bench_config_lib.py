"""Shared config surface for inference throughput/latency benchmarks.

The benchmark entrypoints for sglang/vllm/trtllm should agree on:
  - what workload is being driven
  - which summary metrics operators watch
  - which SLA thresholds define pass/fail

Cache hit rate is intentionally left as a TODO. Today it is best treated as an
observed backend metric, not as a controlled workload invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from examples.inference.bench_workload_lib import make_random_tasks, make_sharegpt_tasks

BenchWorkloadKind = Literal["random", "sharegpt"]

SCALAR_WATCH_METRICS = (
    "requests_per_sec",
    "total_output_tokens_per_sec",
    "llm_output_tokens_per_sec_mean",
    "output_tokens_per_min_per_gpu",
)

SUMMARY_DISTRIBUTION_PERCENTILES = {
    "llm_ttft_ms": (50, 95),
    "llm_tpot_ms": (50, 95),
    "llm_itl_ms": (50, 95),
    "llm_duration_ms": (50, 95),
}


def materialize_distribution_metric_names(
    distribution_percentiles: dict[str, tuple[int, ...]],
) -> tuple[str, ...]:
    names: list[str] = []
    for metric_name, percentiles in distribution_percentiles.items():
        for percentile in percentiles:
            names.append(f"{metric_name}_p{percentile}")
    return tuple(names)


WATCH_METRICS = SCALAR_WATCH_METRICS + materialize_distribution_metric_names(
    SUMMARY_DISTRIBUTION_PERCENTILES
)


@dataclass(frozen=True)
class InferenceBenchWorkload:
    """The input workload driven against an inference endpoint."""

    kind: BenchWorkloadKind
    num_prompts: int
    input_len: int
    output_len: int
    dataset_path: str | None = None
    seed: int = 42
    max_concurrent: int = 16
    # TODO(bench): cache hit rate belongs to workload semantics, but the current
    # harness does not yet construct or verify prefix-sharing regimes honestly.
    target_cache_hit_rate: float | None = None

    def __post_init__(self) -> None:
        if self.num_prompts <= 0:
            raise ValueError("InferenceBenchWorkload.num_prompts must be positive")
        if self.input_len <= 0:
            raise ValueError("InferenceBenchWorkload.input_len must be positive")
        if self.output_len <= 0:
            raise ValueError("InferenceBenchWorkload.output_len must be positive")
        if self.max_concurrent <= 0:
            raise ValueError("InferenceBenchWorkload.max_concurrent must be positive")
        if self.kind == "sharegpt" and self.dataset_path is None:
            raise ValueError("sharegpt workload requires dataset_path")
        if self.kind == "random" and self.dataset_path is not None:
            raise ValueError("random workload does not use dataset_path")


@dataclass(frozen=True)
class InferenceBenchSLA:
    """Optional performance thresholds checked against report.json summary_metrics."""

    minimums: dict[str, float] = field(default_factory=dict)
    maximums: dict[str, float] = field(default_factory=dict)


def build_bench_tasks(workload: InferenceBenchWorkload) -> list[dict]:
    """Materialize benchmark task rows from the shared workload spec."""
    if workload.kind == "random":
        return make_random_tasks(
            num_prompts=workload.num_prompts,
            input_len=workload.input_len,
            output_len=workload.output_len,
            seed=workload.seed,
        )
    if workload.kind == "sharegpt":
        assert workload.dataset_path is not None
        return make_sharegpt_tasks(
            workload.dataset_path,
            num_prompts=workload.num_prompts,
            output_len=workload.output_len,
            seed=workload.seed,
        )
    raise ValueError(f"Unknown workload kind {workload.kind!r}")
