from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from ..eval.configs import EvalTaskSpec, InferenceEndpoint, InferenceServerConfig
from ..training.configs import HardwareConfig


@dataclass(frozen=True)
class EvalServingWorkload:
    """One named eval workload served against a shared endpoint.

    loop=True means "when the sample set is exhausted, start over." Used for
    long-running load simulations where the workload's job is to keep the
    traffic mix realistic, not to exhaust a fixed sample count.
    """

    name: str
    eval_task: EvalTaskSpec
    concurrency: int
    max_samples: int | None = None
    loop: bool = False

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

    loop=True means "when the corpus is exhausted, start over." Same rationale
    as EvalServingWorkload.loop.
    """

    name: str
    concurrency: int
    max_samples: int | None = None
    corpus_path: Path | None = None
    extra_body: dict[str, Any] | None = None
    request_timeout_s: float = 600.0
    loop: bool = False

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
    """Serving run output settings."""

    experiment_name: str = "serving"
    output_dir: Path | None = None
    save_report: bool = True
    save_manifest: bool = True


# on_engine_crash semantics:
#   "fail"    — first crash ends the run (useful for smokes, CI, debugging)
#   "restart" — supervisor restarts the endpoint and keeps going (prod shape)
OnEngineCrash = Literal["fail", "restart"]


@dataclass(frozen=True)
class ServingRun:
    """A running serving deployment: one endpoint, N workloads, defined lifetime.

    The shape is: engine supervisor keeps the endpoint alive, eval supervisor
    runs the workload fleet against it, outer scope enforces the run's duration
    and shutdown semantics. See rollouts.serving.run._run_scenario for the
    concrete trio nursery topology.

    Fields:
      endpoint          — the inference endpoint the workloads share
      workloads         — the traffic mix (each runs concurrently, own schedule)
      duration          — None means "run until workloads complete"; set for
                          long-running load simulations
      on_engine_crash   — policy for endpoint crashes during the run
      drain_timeout     — on SIGTERM: wait up to this long for in-flight samples
                          to finish before force-cancelling. None = cancel
                          immediately on signal.

    Back-compat alias: ServingScenario = ServingRun. Configs that still export
    `serving_scenario = ServingScenario(...)` continue to work.
    """

    endpoint: InferenceEndpoint
    workloads: list[ServingWorkload]
    output: ServingOutputConfig = field(default_factory=ServingOutputConfig)
    hardware: HardwareConfig | None = None
    server: InferenceServerConfig = field(default_factory=InferenceServerConfig)
    duration: timedelta | None = None
    on_engine_crash: OnEngineCrash = "fail"
    drain_timeout: timedelta | None = None

    def __post_init__(self) -> None:
        if not self.workloads:
            raise ValueError("ServingRun requires at least one workload")
        names = [workload.name for workload in self.workloads]
        if len(set(names)) != len(names):
            raise ValueError("ServingRun workload names must be unique")
        if self.duration is not None and self.duration.total_seconds() <= 0:
            raise ValueError("ServingRun.duration must be positive when set")
        if self.drain_timeout is not None and self.drain_timeout.total_seconds() <= 0:
            raise ValueError("ServingRun.drain_timeout must be positive when set")


# Back-compat alias. Deprecated — prefer ServingRun in new configs.
# TODO(deprecate-serving-scenario): remove once courier + all examples updated.
ServingScenario = ServingRun


def resolve_serving_scenario(config_module: Any) -> ServingRun:
    """Find the ServingRun (or legacy ServingScenario) exported by config_module.

    Accepts either `serving_run` or the legacy `serving_scenario` attribute
    name. The former is preferred; the latter is kept for back-compat while
    callers migrate.
    """
    serving_run = getattr(config_module, "serving_run", None)
    if serving_run is None:
        serving_run = getattr(config_module, "serving_scenario", None)
    if not isinstance(serving_run, ServingRun):
        raise ValueError(
            "Serving config must export serving_run: ServingRun "
            "(or the legacy serving_scenario: ServingScenario)"
        )
    return serving_run
