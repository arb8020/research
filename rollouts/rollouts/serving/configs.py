from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..eval.configs import EvalTaskSpec, InferenceEndpoint, InferenceServerConfig
from ..training.configs import HardwareConfig


@dataclass(frozen=True)
class EvalServingWorkload:
    """One named eval workload served against a shared endpoint."""

    name: str
    eval_task: EvalTaskSpec
    concurrency: int
    max_samples: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("EvalServingWorkload.name must be non-empty")
        if self.concurrency <= 0:
            raise ValueError("EvalServingWorkload.concurrency must be positive")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("EvalServingWorkload.max_samples must be positive when set")


@dataclass(frozen=True)
class ToolCallVerifierWorkload:
    """K2VV-shape tool-call conformance workload served against a shared endpoint.

    Unlike EvalServingWorkload this does not run through the agent loop — each
    row is a full OpenAI chat-completions request body, sent once, response
    classified (finish_reason + jsonschema over declared tools). Produces
    engine.jsonl + engine_report.json whose summary fields match KVV's
    tool_calls_eval.py so numbers are directly comparable with KVV publications.

    corpus_path=None means "download the K2VV public sample tarball and cache
    it on the node" (model-weights-style caching).
    """

    name: str
    concurrency: int
    max_samples: int | None = None
    corpus_path: Path | None = None
    extra_body: dict[str, Any] | None = None
    request_timeout_s: float = 600.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolCallVerifierWorkload.name must be non-empty")
        if self.concurrency <= 0:
            raise ValueError("ToolCallVerifierWorkload.concurrency must be positive")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("ToolCallVerifierWorkload.max_samples must be positive when set")
        if self.request_timeout_s <= 0:
            raise ValueError("ToolCallVerifierWorkload.request_timeout_s must be positive")


ServingWorkload = EvalServingWorkload | ToolCallVerifierWorkload


@dataclass(frozen=True)
class ServingOutputConfig:
    """Serving scenario output settings."""

    experiment_name: str = "serving"
    output_dir: Path | None = None
    save_report: bool = True
    save_manifest: bool = True


@dataclass(frozen=True)
class ServingScenario:
    """Run several eval workloads against one shared endpoint."""

    endpoint: InferenceEndpoint
    workloads: list[ServingWorkload]
    output: ServingOutputConfig = field(default_factory=ServingOutputConfig)
    hardware: HardwareConfig | None = None
    server: InferenceServerConfig = field(default_factory=InferenceServerConfig)

    def __post_init__(self) -> None:
        if not self.workloads:
            raise ValueError("ServingScenario requires at least one workload")
        names = [workload.name for workload in self.workloads]
        if len(set(names)) != len(names):
            raise ValueError("ServingScenario workload names must be unique")


def resolve_serving_scenario(config_module: Any) -> ServingScenario:
    serving_scenario = getattr(config_module, "serving_scenario", None)
    if not isinstance(serving_scenario, ServingScenario):
        raise ValueError("Serving config must export serving_scenario: ServingScenario")
    return serving_scenario
