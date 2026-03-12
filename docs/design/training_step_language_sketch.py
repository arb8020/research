"""Sketch of an explicit training-step language.

This is not production code. It is a design artifact for reasoning about the
internal training semantics we want before lowering into Megatron/TorchTitan.

Goals:
- make objective semantics explicit
- make partition/layout transitions explicit
- keep backend/runtime machinery out of the source-level worldview
- allow LoRA / full finetune / expert-only variants as parameter-ownership policy

seqax-inspired idea:
- layout is part of the semantic surface
- communication is expressed as a before/after layout transition string
- local compute should not secretly imply global semantics

Examples of layout strings:
- "batch/d seq hidden/tp"
- "batch seq vocab/tp -> batch seq vocab"
- "tokens ep -> tokens"
- "tokens -> tokens/d"

These strings are intentionally lightweight and authorial. They are a sketch of
the internal worldview, not a final parser or static type system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

LayoutExpr = str
CollectiveExpr = str
TensorLike = Any


@dataclass(frozen=True)
class ModelState:
    """Opaque model state handle.

    Lowerings may back this with Megatron chunks, TorchTitan modules, DTensors,
    or something cleaner later.
    """

    opaque: Any


@dataclass(frozen=True)
class OptimizerState:
    opaque: Any


@dataclass(frozen=True)
class SchedulerState:
    opaque: Any


@dataclass(frozen=True)
class ParameterPolicy:
    """Which parameter subsets are trainable for this run."""

    trainable_sets: tuple[str, ...] = ("all",)
    frozen_sets: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrecisionPolicy:
    """Precision is semantic, not cosmetic."""

    param_dtype: str = "bf16"
    compute_dtype: str = "bf16"
    reduce_dtype: str = "fp32"
    optimizer_state_dtype: str = "fp32"
    router_dtype: str | None = None
    expert_compute_dtype: str | None = None


@dataclass(frozen=True)
class ParallelPlan:
    """Logical parallel intent, not backend topology.

    These are the logical axes and the intended at-rest layouts we want the
    training program to reason in.
    """

    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    default_batch_layout: LayoutExpr = "batch/d seq"
    default_hidden_layout: LayoutExpr = "batch/d seq hidden/tp"
    default_vocab_layout: LayoutExpr = "batch/d seq vocab/tp"


@dataclass(frozen=True)
class TrainingState:
    model: ModelState
    optimizer: OptimizerState
    scheduler: SchedulerState | None
    parameter_policy: ParameterPolicy
    precision: PrecisionPolicy
    parallel: ParallelPlan
    step: int = 0
    weight_version: int = 0


@dataclass(frozen=True)
class LossTerm:
    name: str
    value: float


@dataclass(frozen=True)
class Metric:
    name: str
    value: float


@dataclass(frozen=True)
class Event:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    losses: tuple[LossTerm, ...]
    metrics: tuple[Metric, ...]
    events: tuple[Event, ...] = ()


@dataclass(frozen=True)
class SFTBatch:
    tokens: TensorLike
    labels: TensorLike
    loss_mask: TensorLike
    token_layout: LayoutExpr = "batch/d seq"


@dataclass(frozen=True)
class RLBatch:
    tokens: TensorLike
    rollout_logprobs: TensorLike
    advantages: TensorLike
    returns: TensorLike | None = None
    group_ids: TensorLike | None = None
    token_layout: LayoutExpr = "batch/d seq"


@dataclass(frozen=True)
class DistillBatch:
    tokens: TensorLike
    teacher_logits: TensorLike | None
    teacher_logprobs: TensorLike | None
    loss_mask: TensorLike
    labels: TensorLike | None = None
    token_layout: LayoutExpr = "batch/d seq"


@dataclass(frozen=True)
class ForwardOutput:
    """Semantic forward products.

    These are the things an objective may request from the model execution
    layer. Lowerings decide how to materialize them under the current layout.
    """

    logits: TensorLike | None = None
    values: TensorLike | None = None
    router_aux: TensorLike | None = None


def forward_local(
    model: ModelState,
    *,
    tokens: TensorLike,
    token_layout: LayoutExpr,
    require: tuple[Literal["logits", "values", "router_aux"], ...] = ("logits",),
) -> ForwardOutput:
    """Run local/sharded model compute.

    The meaning here is intentionally local: outputs may remain sharded in their
    at-rest layout. This call should not secretly materialize globally replicated
    values unless the implementation says so explicitly.
    """
    raise NotImplementedError


def materialize(tensor: TensorLike, transition: CollectiveExpr) -> TensorLike:
    """Explicitly change layout via a collective/materialization step.

    Examples:
    - "batch/d seq vocab/tp -> batch seq vocab"
    - "tokens ep -> tokens"
    """
    raise NotImplementedError


def reduce_mean(tensor: TensorLike, transition: CollectiveExpr) -> TensorLike:
    """Explicit mean reduction with an explicit before/after layout expression.

    Examples:
    - "batch/d -> scalar"
    - "batch/d seq -> batch seq"
    """
    raise NotImplementedError


def backward(loss: TensorLike) -> None:
    """Backward through the current graph for the current trainable parameter sets."""
    raise NotImplementedError


def optimizer_step(state: TrainingState) -> tuple[TrainingState, tuple[Metric, ...]]:
    """Apply updates and return new state plus optimizer/runtime metrics."""
    raise NotImplementedError


def zero() -> TensorLike:
    raise NotImplementedError


def gather_token_logprobs(logits: TensorLike, tokens: TensorLike) -> TensorLike:
    raise NotImplementedError


def masked_cross_entropy(
    *,
    logits: TensorLike,
    labels: TensorLike,
    loss_mask: TensorLike,
) -> TensorLike:
    raise NotImplementedError


def policy_objective(
    *,
    new_logprobs: TensorLike,
    old_logprobs: TensorLike,
    advantages: TensorLike,
    group_ids: TensorLike | None,
) -> TensorLike:
    raise NotImplementedError


def value_objective(values: TensorLike, returns: TensorLike) -> TensorLike:
    raise NotImplementedError


def distillation_kl(
    *,
    student_logits: TensorLike,
    teacher_logits: TensorLike | None,
    teacher_logprobs: TensorLike | None,
    loss_mask: TensorLike,
) -> TensorLike:
    raise NotImplementedError


def sft_step(state: TrainingState, batch: SFTBatch) -> tuple[TrainingState, StepResult]:
    """Authorial SFT step with explicit materialization/reduction semantics."""

    outputs = forward_local(
        state.model,
        tokens=batch.tokens,
        token_layout=batch.token_layout,
        require=("logits",),
    )
    assert outputs.logits is not None

    # seqax-like transition: local vocab shard -> materialized vocab view
    logits = materialize(
        outputs.logits,
        "batch/d seq vocab/tp -> batch seq vocab",
    )

    xent = masked_cross_entropy(
        logits=logits,
        labels=batch.labels,
        loss_mask=batch.loss_mask,
    )
    loss = reduce_mean(xent, "batch/d -> scalar")

    backward(loss)
    state, opt_metrics = optimizer_step(state)

    return state, StepResult(
        losses=(
            LossTerm("total", float(loss)),
            LossTerm("xent", float(loss)),
        ),
        metrics=opt_metrics,
    )


def rl_step(state: TrainingState, batch: RLBatch) -> tuple[TrainingState, StepResult]:
    """Authorial RL step.

    The objective says what semantic products it needs. The lowering/runtime
    decides how to realize those under the current partitioned layout.
    """

    outputs = forward_local(
        state.model,
        tokens=batch.tokens,
        token_layout=batch.token_layout,
        require=("logits", "values", "router_aux"),
    )
    assert outputs.logits is not None

    logits = materialize(
        outputs.logits,
        "batch/d seq vocab/tp -> batch seq vocab",
    )
    new_logprobs = gather_token_logprobs(logits, batch.tokens)

    policy_loss = policy_objective(
        new_logprobs=new_logprobs,
        old_logprobs=batch.rollout_logprobs,
        advantages=batch.advantages,
        group_ids=batch.group_ids,
    )

    if outputs.values is not None and batch.returns is not None:
        v_loss = value_objective(outputs.values, batch.returns)
    else:
        v_loss = zero()

    total = reduce_mean(policy_loss + v_loss, "batch/d -> scalar")

    backward(total)
    state, opt_metrics = optimizer_step(state)

    losses = [
        LossTerm("total", float(total)),
        LossTerm("policy", float(reduce_mean(policy_loss, "batch/d -> scalar"))),
        LossTerm("value", float(reduce_mean(v_loss, "batch/d -> scalar"))),
    ]
    if outputs.router_aux is not None:
        losses.append(
            LossTerm(
                "router_aux",
                float(reduce_mean(outputs.router_aux, "batch/d -> scalar")),
            )
        )

    return state, StepResult(
        losses=tuple(losses),
        metrics=opt_metrics,
    )


def distill_step(
    state: TrainingState,
    batch: DistillBatch,
) -> tuple[TrainingState, StepResult]:
    """Authorial distillation step."""

    outputs = forward_local(
        state.model,
        tokens=batch.tokens,
        token_layout=batch.token_layout,
        require=("logits",),
    )
    assert outputs.logits is not None

    student_logits = materialize(
        outputs.logits,
        "batch/d seq vocab/tp -> batch seq vocab",
    )

    kl = distillation_kl(
        student_logits=student_logits,
        teacher_logits=batch.teacher_logits,
        teacher_logprobs=batch.teacher_logprobs,
        loss_mask=batch.loss_mask,
    )

    if batch.labels is not None:
        ce = masked_cross_entropy(
            logits=student_logits,
            labels=batch.labels,
            loss_mask=batch.loss_mask,
        )
    else:
        ce = zero()

    total = reduce_mean(kl + ce, "batch/d -> scalar")

    backward(total)
    state, opt_metrics = optimizer_step(state)

    return state, StepResult(
        losses=(
            LossTerm("total", float(total)),
            LossTerm("kl", float(reduce_mean(kl, "batch/d -> scalar"))),
            LossTerm("ce", float(reduce_mean(ce, "batch/d -> scalar"))),
        ),
        metrics=opt_metrics,
    )


#
# Notes
# -----
#
# 1. LoRA vs non-LoRA should probably not be a separate training worldview.
#    It should likely be expressed via ParameterPolicy / trainable parameter sets.
#
# 2. Pipeline parallelism is probably a lowering/runtime concern first, not a
#    source-level semantic object, unless we decide we need explicit authorial
#    control over stage-local losses, microbatch waves, or inter-stage state.
#
# 3. The string DSL here is a sketch, not a commitment to the exact syntax.
#    The main idea is that partition state and collective transitions should be
#    visible in the source-level training semantics.
#
# 4. Ragged routing is intentionally not first-class in this first pass.
#    That does not mean it is unimportant. All of our likely lowering targets
#    (nmoe, Megatron, TorchTitan) have real uneven per-expert token counts in
#    MoE dispatch and then regularize them with counts/offsets, permutation,
#    capacity policy, and padding/alignment for execution.
#
#    When we need to model this explicitly, the likely extension points are:
#    - ForwardOutput / router-related outputs
#    - materialize(...) transitions around MoE dispatch/combine
#    - a future RoutedTokenBuffer / RoutingResult / DispatchPlan layer
#
#    We should treat raggedness as a first-class MoE/dispatch concern, not
#    necessarily a universal first-class property of every tensor in the core
#    layout language.
