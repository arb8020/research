"""Core training boundary types.

These types are the clean semantic boundary for training code.

They are intentionally separate from legacy bridge objects like `AttemptRow`.
Those bridge types can be converted into these contracts at the edge, but they
should not define the core training ontology.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

TensorLike: TypeAlias = Any
MetricMap: TypeAlias = dict[str, float]
EventRecord: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class WeightVersion:
    """Explicit weight version identifier."""

    value: int


@dataclass(frozen=True)
class WeightPublication:
    """Record of one checkpoint publication to inference engines."""

    step: int
    version: WeightVersion
    checkpoint_path: str
    engine_count: int
    sync_responses: tuple[EventRecord, ...] = ()


@dataclass(frozen=True)
class WeightSyncPolicy:
    """Semantic policy for when and how weights become visible to inference."""

    every_n_steps: int
    mode: str = "disk"
    enabled: bool = True

    def __post_init__(self) -> None:
        assert self.every_n_steps > 0, "every_n_steps must be > 0"
        assert self.mode in {"disk", "nccl"}, f"unknown weight sync mode: {self.mode!r}"

    @staticmethod
    def every_n_steps_policy(
        every_n_steps: int,
        *,
        mode: str = "disk",
    ) -> "WeightSyncPolicy":
        return WeightSyncPolicy(every_n_steps=every_n_steps, mode=mode, enabled=True)

    @staticmethod
    def disabled(
        *,
        mode: str = "disk",
    ) -> "WeightSyncPolicy":
        return WeightSyncPolicy(every_n_steps=1, mode=mode, enabled=False)

    def should_publish(self, step: int) -> bool:
        if not self.enabled:
            return False
        if step <= 0:
            return False
        return step % self.every_n_steps == 0


@dataclass(frozen=True)
class StalenessPolicy:
    """Semantic policy for how much rollout-version lag training tolerates."""

    max_version_lag: int = 0
    drop_stale: bool = True
    require_exact_version: bool = False

    @staticmethod
    def synchronous() -> "StalenessPolicy":
        return StalenessPolicy(max_version_lag=0, drop_stale=True, require_exact_version=True)

    def allows_version(self, *, batch_version: int, current_train_version: int) -> bool:
        version_lag = max(current_train_version - batch_version, 0)
        if self.require_exact_version:
            return version_lag == 0
        return version_lag <= self.max_version_lag


@dataclass(frozen=True)
class WeightVisibilityPolicy:
    """Semantic policy for when new training weights become visible to inference."""

    publish_mode: str = "disk"
    atomic_visibility: bool = True
    drain_before_publish: bool = True

    def __post_init__(self) -> None:
        assert self.publish_mode in {"disk", "nccl"}, f"unknown publish_mode: {self.publish_mode!r}"

    @staticmethod
    def synchronous(
        *,
        publish_mode: str = "disk",
    ) -> "WeightVisibilityPolicy":
        return WeightVisibilityPolicy(
            publish_mode=publish_mode,
            atomic_visibility=True,
            drain_before_publish=True,
        )


@dataclass(frozen=True)
class AdmissionPolicy:
    """Semantic policy for rollout admissions while training/sync is active."""

    pause_on_sync: bool = True
    allow_inflight_completion: bool = True
    max_inflight_batches: int | None = None

    @staticmethod
    def synchronous() -> "AdmissionPolicy":
        return AdmissionPolicy(
            pause_on_sync=True,
            allow_inflight_completion=True,
            max_inflight_batches=1,
        )

    @staticmethod
    def stream_style_default() -> "AdmissionPolicy":
        return AdmissionPolicy(
            pause_on_sync=False,
            allow_inflight_completion=True,
            max_inflight_batches=None,
        )


@dataclass(frozen=True)
class OverloadPolicy:
    """Safety-valve policy for stream-style async backlog growth."""

    cancel_stale_inflight: bool = False
    spill_to_disk: bool = False
    block_generation_as_last_resort: bool = False
    queue_pressure_threshold: int | None = None


@dataclass(frozen=True)
class TrainingRuntimeState:
    """Explicit runtime state for a training loop."""

    step: int
    weight_version: WeightVersion


@dataclass(frozen=True)
class PipelineRuntimeState:
    """Explicit runtime state for rollout/training/sync coordination."""

    current_train_version: int
    current_serving_version: int
    sync_in_progress: bool = False
    admissions_paused: bool = False
    inflight_batches: int = 0


@dataclass(frozen=True)
class VersionedRolloutBatch:
    """Rollout batch tagged with the weight version that produced it."""

    batch: Any
    weight_version: int
    created_at_step: int
    version_lag: int = 0


@dataclass(frozen=True)
class TrainableParameterPolicy:
    """Semantic policy for which parameter families are allowed to update."""

    mode: str
    extra_trainable_patterns: tuple[str, ...] = ()
    extra_frozen_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        valid_modes = {"full_weight", "lora_only", "router_only", "experts_only"}
        assert self.mode in valid_modes, f"unknown trainable parameter policy: {self.mode!r}"

    @staticmethod
    def full_weight() -> "TrainableParameterPolicy":
        return TrainableParameterPolicy(mode="full_weight")

    @staticmethod
    def lora_only() -> "TrainableParameterPolicy":
        return TrainableParameterPolicy(mode="lora_only")

    @staticmethod
    def router_only() -> "TrainableParameterPolicy":
        return TrainableParameterPolicy(mode="router_only")

    @staticmethod
    def experts_only() -> "TrainableParameterPolicy":
        return TrainableParameterPolicy(mode="experts_only")

    def allows_param(self, param_name: str) -> bool:
        if any(pattern in param_name for pattern in self.extra_frozen_patterns):
            return False
        if any(pattern in param_name for pattern in self.extra_trainable_patterns):
            return True
        if self.mode == "full_weight":
            return True
        if self.mode == "lora_only":
            return any(pattern in param_name for pattern in ("lora", "lora_A", "lora_B", "adapter"))
        if self.mode == "router_only":
            return "router" in param_name
        if self.mode == "experts_only":
            return "expert" in param_name
        return False


@dataclass(frozen=True)
class PrecisionPolicy:
    """Semantic precision choices for a training path."""

    param: str
    compute: str
    reduce: str
    router: str | None = None
    expert_compute: str | None = None
    optimizer_state: str | None = None


@dataclass(frozen=True)
class ModelInput:
    """Model-facing inputs for one training step."""

    tokens: TensorLike
    positions: TensorLike | None = None
    attention_mask: TensorLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingDatum:
    """Semantic training datum for one step.

    Splits model-facing input from objective-facing supervision/signals.
    """

    model_input: ModelInput
    objective_inputs: Mapping[str, Any]
    precision_policy: PrecisionPolicy | None = None
    trainable_parameter_policy: TrainableParameterPolicy | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardProducts:
    """Products returned by the model forward pass before loss computation."""

    logits: TensorLike | None = None
    values: TensorLike | None = None
    hidden_states: TensorLike | None = None
    aux: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Structured result of one training step."""

    backprop_loss: TensorLike
    losses: MetricMap
    other_metrics: MetricMap = field(default_factory=dict)
    events: tuple[EventRecord, ...] = ()


class LossFn(Protocol):
    """Loss contract for backend-facing training code."""

    def __call__(self, products: ForwardProducts, datum: TrainingDatum) -> StepResult: ...


LossFnLike: TypeAlias = Callable[[ForwardProducts, TrainingDatum], StepResult]
