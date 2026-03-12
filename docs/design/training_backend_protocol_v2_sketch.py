"""Sketch of a v2 training backend protocol.

This is a deliberately small evolution of the current backend protocol.

What it keeps:
- `forward_backward`
- `optim_step`
- future-returning backend operations

What it changes:
- replace `batch: dict[str, Any]` with `TrainingDatum`
- make the `loss_fn` contract more explicit
- replace flat metrics dicts with a more structured `StepResult`

This is meant to be a practical migration target, not the final perfect model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

TensorLike = Any
T = TypeVar("T")


class TrainFuture(Protocol[T]):
    """Minimal future-like interface for backend operations."""

    async def result(self) -> T: ...


@dataclass(frozen=True)
class ModelInput:
    """Model-facing input boundary."""

    tokens: TensorLike
    positions: TensorLike | None = None
    attention_mask: TensorLike | None = None
    token_layout: str = "batch/d seq"


@dataclass(frozen=True)
class TrainingDatum:
    """Semantic training datum / batch.

    Inspired by Tinker's split between model-facing input and objective-facing
    tensors. We keep objective inputs flexible for now rather than forcing a
    heavy typed hierarchy too early.
    """

    model_input: ModelInput
    objective_inputs: dict[str, TensorLike] = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardProducts:
    """Semantic outputs available to the loss/objective layer."""

    logits: TensorLike | None = None
    values: TensorLike | None = None
    hidden_states: TensorLike | None = None
    router_aux: TensorLike | None = None
    aux: dict[str, TensorLike] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Structured result of one training substep.

    `losses` are the named scalar loss terms the objective thinks are central.
    `other_metrics` are everything else: grad norm, entropy, KL, router load
    balance stats, overflow flags encoded numerically, etc.
    """

    losses: dict[str, float]
    other_metrics: dict[str, float] = field(default_factory=dict)
    events: tuple[dict[str, Any], ...] = ()


LossFn = Callable[[ForwardProducts, TrainingDatum], StepResult]


class TrainingBackendV2(Protocol):
    """Practical next-step backend protocol.

    The backend still owns the operational/runtime machinery and returns futures.
    The semantic boundary is improved by requiring:
    - a typed datum (`TrainingDatum`)
    - a clearer loss/objective contract (`LossFn`)
    - a structured result (`StepResult`)
    """

    def forward_backward(
        self,
        datum: TrainingDatum,
        *,
        loss_fn: LossFn,
    ) -> TrainFuture[StepResult]:
        """Run forward + backward for one datum/batch and return structured results.

        The objective/loss function consumes semantic forward products plus the
        original datum, rather than a vague logits+batch convention.
        """
        ...

    def optim_step(self) -> TrainFuture[StepResult]:
        """Apply accumulated gradients / optimizer state update.

        Returns optimizer/runtime metrics in `other_metrics`. Losses will usually
        be empty at this stage.
        """
        ...


# ---------------------------------------------------------------------------
# Example loss/objective functions
# ---------------------------------------------------------------------------


def sft_loss(products: ForwardProducts, datum: TrainingDatum) -> StepResult:
    """Example contract shape for SFT.

    `objective_inputs` is expected to contain:
    - `labels`
    - `loss_mask`
    """
    raise NotImplementedError


def distill_loss(products: ForwardProducts, datum: TrainingDatum) -> StepResult:
    """Example contract shape for distillation.

    `objective_inputs` may contain:
    - `teacher_logits`
    - `teacher_logprobs`
    - `labels`
    - `loss_mask`
    """
    raise NotImplementedError


def rl_loss(products: ForwardProducts, datum: TrainingDatum) -> StepResult:
    """Example contract shape for RL.

    `objective_inputs` may contain:
    - `rollout_logprobs`
    - `advantages`
    - `returns`
    - `group_ids`
    """
    raise NotImplementedError


def pretrain_loss(products: ForwardProducts, datum: TrainingDatum) -> StepResult:
    """Example contract shape for pretraining.

    `objective_inputs` is expected to contain:
    - `labels`
    Optionally:
    - `weights`
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
#
# 1. This protocol is intentionally outer-boundary oriented and Tinker-like.
#    It is meant to be the clean interface exposed to higher-level systems.
#
# 2. The inner realization semantics can still be more seqax-like and explicit
#    about partitioning/materialization/collectives.
#
# 3. `loss_fn` is still a callback here because that is the smallest practical
#    improvement over the current API. If needed later, it can harden into a
#    richer `ObjectiveSpec` / objective object.
#
# 4. `objective_inputs: dict[str, TensorLike]` is intentionally flexible for
#    now. If this starts to drift into string soup, the next move is typed
#    wrappers per objective family rather than reverting to raw backend-shaped
#    batches.
#
# 5. LoRA vs full finetune does not need to change this outer protocol. That
#    should likely live in trainable parameter policy and backend/runtime state.
