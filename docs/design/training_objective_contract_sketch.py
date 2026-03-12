"""Sketch of a smaller objective/training contract.

This file is intentionally narrower than training_step_language_sketch.py.
The goal is to pressure-test the common semantic boundary across:

- pretraining
- SFT
- distillation
- RL

before we commit too hard to lowering/runtime details.

The key question:
- what does an objective need to tell the training system?
- what does the training system need to give back?

Current intended role of this file:

- this is the Tinker-like outer boundary we may want to expose to other systems
- it should stay cleaner and more protocol-like than the explicit step-language
- it should not itself encode backend collectives or runtime machinery
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TensorLike = Any


@dataclass(frozen=True)
class ModelInput:
    """What the model consumes.

    This is the model-facing side of the contract.
    Objectives should not need to smuggle supervision into this object unless
    the model itself semantically requires it.
    """

    tokens: TensorLike
    positions: TensorLike | None = None
    attention_mask: TensorLike | None = None
    token_layout: str = "batch/d seq"


@dataclass(frozen=True)
class ObjectiveInputs:
    """What the objective consumes in addition to the model input.

    This is the objective-facing side of the contract. These fields should be
    semantically meaningful signals like labels, masks, rewards, teacher logits,
    rollout logprobs, and so on.
    """

    tensors: dict[str, TensorLike] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingDatum:
    """One semantic training example or batch boundary.

    Inspired partly by Tinker's split between model-facing input and
    objective-facing tensors.
    """

    model_input: ModelInput
    objective_inputs: ObjectiveInputs


@dataclass(frozen=True)
class ForwardProducts:
    """Semantic outputs the objective may request from model execution."""

    logits: TensorLike | None = None
    values: TensorLike | None = None
    hidden_states: TensorLike | None = None
    router_aux: TensorLike | None = None
    aux: dict[str, TensorLike] = field(default_factory=dict)


@dataclass(frozen=True)
class LossTerm:
    name: str
    value: float


@dataclass(frozen=True)
class Metric:
    name: str
    value: float


@dataclass(frozen=True)
class LossOutputs:
    """Named loss terms and optional objective-specific scalar summaries."""

    terms: tuple[LossTerm, ...]
    metrics: tuple[Metric, ...] = ()


@dataclass(frozen=True)
class StepResult:
    """What the training step gives back to orchestration/observability."""

    loss_outputs: LossOutputs
    runtime_metrics: tuple[Metric, ...] = ()
    events: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ObjectiveSpec:
    """Semantic objective boundary.

    The objective says:
    - what forward products it needs
    - how it interprets model outputs + objective inputs

    The execution/lowering layer decides how to realize those products under the
    current partition/layout/runtime.
    """

    name: str
    required_forward_products: tuple[
        Literal["logits", "values", "hidden_states", "router_aux"],
        ...,
    ]
    required_objective_inputs: tuple[str, ...]


# ---------------------------------------------------------------------------
# Example objective shapes
# ---------------------------------------------------------------------------


PRETRAIN_OBJECTIVE = ObjectiveSpec(
    name="pretrain_next_token",
    required_forward_products=("logits",),
    required_objective_inputs=("labels",),
)


SFT_OBJECTIVE = ObjectiveSpec(
    name="sft_next_token",
    required_forward_products=("logits",),
    required_objective_inputs=("labels", "loss_mask"),
)


DISTILL_OBJECTIVE = ObjectiveSpec(
    name="distillation",
    required_forward_products=("logits",),
    required_objective_inputs=("loss_mask",),
)


RL_OBJECTIVE = ObjectiveSpec(
    name="rl_policy_value",
    required_forward_products=("logits", "values"),
    required_objective_inputs=("rollout_logprobs", "advantages"),
)


# ---------------------------------------------------------------------------
# Example data constructors
# ---------------------------------------------------------------------------


def make_pretrain_datum(
    *,
    tokens: TensorLike,
    labels: TensorLike,
    positions: TensorLike | None = None,
    attention_mask: TensorLike | None = None,
) -> TrainingDatum:
    return TrainingDatum(
        model_input=ModelInput(
            tokens=tokens,
            positions=positions,
            attention_mask=attention_mask,
        ),
        objective_inputs=ObjectiveInputs(
            tensors={
                "labels": labels,
            }
        ),
    )


def make_sft_datum(
    *,
    tokens: TensorLike,
    labels: TensorLike,
    loss_mask: TensorLike,
    positions: TensorLike | None = None,
    attention_mask: TensorLike | None = None,
) -> TrainingDatum:
    return TrainingDatum(
        model_input=ModelInput(
            tokens=tokens,
            positions=positions,
            attention_mask=attention_mask,
        ),
        objective_inputs=ObjectiveInputs(
            tensors={
                "labels": labels,
                "loss_mask": loss_mask,
            }
        ),
    )


def make_distill_datum(
    *,
    tokens: TensorLike,
    loss_mask: TensorLike,
    teacher_logits: TensorLike | None = None,
    teacher_logprobs: TensorLike | None = None,
    labels: TensorLike | None = None,
    positions: TensorLike | None = None,
    attention_mask: TensorLike | None = None,
) -> TrainingDatum:
    tensors: dict[str, TensorLike] = {
        "loss_mask": loss_mask,
    }
    if teacher_logits is not None:
        tensors["teacher_logits"] = teacher_logits
    if teacher_logprobs is not None:
        tensors["teacher_logprobs"] = teacher_logprobs
    if labels is not None:
        tensors["labels"] = labels

    return TrainingDatum(
        model_input=ModelInput(
            tokens=tokens,
            positions=positions,
            attention_mask=attention_mask,
        ),
        objective_inputs=ObjectiveInputs(tensors=tensors),
    )


def make_rl_datum(
    *,
    tokens: TensorLike,
    rollout_logprobs: TensorLike,
    advantages: TensorLike,
    returns: TensorLike | None = None,
    group_ids: TensorLike | None = None,
    positions: TensorLike | None = None,
    attention_mask: TensorLike | None = None,
) -> TrainingDatum:
    tensors: dict[str, TensorLike] = {
        "rollout_logprobs": rollout_logprobs,
        "advantages": advantages,
    }
    if returns is not None:
        tensors["returns"] = returns
    if group_ids is not None:
        tensors["group_ids"] = group_ids

    return TrainingDatum(
        model_input=ModelInput(
            tokens=tokens,
            positions=positions,
            attention_mask=attention_mask,
        ),
        objective_inputs=ObjectiveInputs(tensors=tensors),
    )


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
#
# 1. This contract is intentionally objective-centered and backend-agnostic.
#    It does not say how logits or values are realized under partitioning.
#
# 2. Pretraining is just another objective in this model, not a separate
#    training universe.
#
# 3. This sketch currently uses string-keyed objective inputs for flexibility.
#    If this hardens, we may want more typed wrappers for individual objectives.
#
# 4. LoRA vs full finetune does not need to change this contract. That should
#    likely live in parameter ownership / trainable-set policy.
#
# 5. The intended relationship to training_step_language_sketch.py is:
#    - this file = outer typed contract (Tinker-like)
#    - training_step_language_sketch.py = inner explicit realization language
#      (more seqax-like, with explicit partition/materialization/reduction semantics)
