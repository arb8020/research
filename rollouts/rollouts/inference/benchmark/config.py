"""Benchmark configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class WorkloadConfig:
    """Workload specification for benchmarking.

    Matches SGLang/vLLM benchmark formats.
    """

    type: Literal["random", "sharegpt", "burst"] = "random"
    num_prompts: int = 1000
    input_len: int = 512
    output_len: int = 128

    # Request rate control
    request_rate: float | Literal["inf"] = "inf"  # requests/sec, "inf" = burst

    # Burst workload specific
    concurrency: int = 64

    def __post_init__(self) -> None:
        assert self.num_prompts > 0
        assert self.input_len > 0
        assert self.output_len > 0
        if self.type == "burst":
            assert self.concurrency > 0


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for an inference benchmark run.

    Example:
        config = BenchmarkConfig(
            model="Qwen/Qwen3-0.6B",
            backend="engine_v2",
            workload=WorkloadConfig(type="random", num_prompts=1000),
        )
    """

    # Model
    model: str = "Qwen/Qwen3-0.6B"

    # Backend to benchmark
    backend: Literal["sglang", "vllm", "engine_v2"] = "engine_v2"

    # Workload
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)

    # Engine settings
    max_batch_size: int = 256
    mem_fraction: float = 0.9
    tensor_parallel_size: int = 1

    # Output
    output_dir: str = "results/benchmark"
    experiment_name: str = "benchmark"

    def __post_init__(self) -> None:
        assert self.model
        assert self.backend in ("sglang", "vllm", "engine_v2")


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    # Config
    backend: str
    model: str
    gpu: str
    workload: dict[str, Any]

    # Throughput
    requests_per_second: float
    tokens_per_second: float
    output_tokens_per_second: float

    # Latency (ms)
    ttft_mean: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float

    tpot_mean: float
    tpot_p50: float
    tpot_p95: float
    tpot_p99: float

    e2e_mean: float
    e2e_p50: float
    e2e_p95: float
    e2e_p99: float

    # Resources
    gpu_memory_peak_mb: float
    gpu_utilization_mean: float

    # Events file path
    events_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict matching design doc format."""
        return {
            "backend": self.backend,
            "model": self.model,
            "gpu": self.gpu,
            "workload": self.workload,
            "throughput": {
                "requests_per_second": self.requests_per_second,
                "tokens_per_second": self.tokens_per_second,
                "output_tokens_per_second": self.output_tokens_per_second,
            },
            "latency": {
                "ttft_ms": {
                    "mean": self.ttft_mean,
                    "p50": self.ttft_p50,
                    "p95": self.ttft_p95,
                    "p99": self.ttft_p99,
                },
                "tpot_ms": {
                    "mean": self.tpot_mean,
                    "p50": self.tpot_p50,
                    "p95": self.tpot_p95,
                    "p99": self.tpot_p99,
                },
                "e2e_ms": {
                    "mean": self.e2e_mean,
                    "p50": self.e2e_p50,
                    "p95": self.e2e_p95,
                    "p99": self.e2e_p99,
                },
            },
            "resource": {
                "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
                "gpu_utilization_mean": self.gpu_utilization_mean,
            },
            "events_file": self.events_file,
        }
