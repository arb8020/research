"""Witness helpers for the contract-native training path."""

import torch
import torch.nn.functional as F

from ..training.contracts import (
    ForwardProducts,
    ModelInput,
    StepResult,
    TrainingDatum,
)
from ..training.types import TrainingSample


def legacy_supervised_batch_to_training_datum(batch: dict[str, torch.Tensor]) -> TrainingDatum:
    """Convert the current dense supervised batch dict into TrainingDatum."""
    assert "input_ids" in batch, "batch must have 'input_ids'"
    assert "labels" in batch, "batch must have 'labels'"

    return TrainingDatum(
        model_input=ModelInput(
            tokens=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        ),
        objective_inputs={
            "labels": batch["labels"],
            "loss_mask": batch.get("loss_mask"),
        },
    )


def training_sample_to_supervised_datum(sample: TrainingSample) -> TrainingDatum:
    """Convert a TrainingSample into a single-example supervised datum."""
    tokens = torch.tensor(sample.tokens, dtype=torch.long).unsqueeze(0)
    labels = tokens.clone()
    loss_mask = torch.tensor(sample.loss_mask, dtype=torch.float32).unsqueeze(0)
    return TrainingDatum(
        model_input=ModelInput(tokens=tokens),
        objective_inputs={
            "labels": labels,
            "loss_mask": loss_mask,
        },
        metadata=sample.metadata,
    )


def pretrain_batch_to_datum(input_ids: torch.Tensor, labels: torch.Tensor) -> TrainingDatum:
    """Convert dense next-token tensors into a contract-native datum."""
    return TrainingDatum(
        model_input=ModelInput(tokens=input_ids),
        objective_inputs={
            "labels": labels,
        },
    )


def pretrain_contract_loss(products: ForwardProducts, datum: TrainingDatum) -> StepResult:
    """Dense next-token pretraining loss using the contract boundary."""
    assert products.logits is not None, "pretrain loss requires logits"
    assert "labels" in datum.objective_inputs, "pretrain loss requires labels"

    logits = products.logits
    labels = datum.objective_inputs["labels"]

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    with torch.no_grad():
        perplexity = torch.exp(loss.detach()).item()

    return StepResult(
        backprop_loss=loss,
        losses={
            "total": loss.detach().item(),
            "xent": loss.detach().item(),
        },
        other_metrics={
            "perplexity": perplexity,
        },
    )


def supervised_contract_loss(products: ForwardProducts, datum: TrainingDatum) -> StepResult:
    """Dense supervised loss using the new contract boundary."""
    assert products.logits is not None, "supervised loss requires logits"
    assert "labels" in datum.objective_inputs, "supervised loss requires labels"

    logits = products.logits
    labels = datum.objective_inputs["labels"]
    loss_mask = datum.objective_inputs.get("loss_mask")

    if loss_mask is None:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
    else:
        per_token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
        masked_loss = per_token_loss * loss_mask
        num_tokens = loss_mask.sum().clamp(min=1.0)
        loss = masked_loss.sum() / num_tokens

    with torch.no_grad():
        perplexity = torch.exp(loss.detach()).item()

    return StepResult(
        backprop_loss=loss,
        losses={
            "total": loss.detach().item(),
            "xent": loss.detach().item(),
        },
        other_metrics={
            "perplexity": perplexity,
        },
    )


def rl_training_batch_to_datum(batch: dict[str, torch.Tensor]) -> TrainingDatum:
    """Convert a dense RL batch dict into TrainingDatum."""
    assert "input_ids" in batch, "batch must have 'input_ids'"
    assert "labels" in batch, "batch must have 'labels'"
    assert "advantages" in batch, "batch must have 'advantages'"

    objective_inputs = {
        "labels": batch["labels"],
        "advantages": batch["advantages"],
    }
    if "loss_mask" in batch:
        objective_inputs["loss_mask"] = batch["loss_mask"]
    if "old_logprobs" in batch:
        objective_inputs["old_logprobs"] = batch["old_logprobs"]
    if "returns" in batch:
        objective_inputs["returns"] = batch["returns"]
    if "group_ids" in batch:
        objective_inputs["group_ids"] = batch["group_ids"]

    return TrainingDatum(
        model_input=ModelInput(
            tokens=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        ),
        objective_inputs=objective_inputs,
    )


def rl_contract_loss(products: ForwardProducts, datum: TrainingDatum) -> StepResult:
    """Simple GRPO-style RL witness loss using the contract boundary."""
    assert products.logits is not None, "rl loss requires logits"
    assert "labels" in datum.objective_inputs, "rl loss requires labels"
    assert "advantages" in datum.objective_inputs, "rl loss requires advantages"

    logits = products.logits
    labels = datum.objective_inputs["labels"]
    loss_mask = datum.objective_inputs.get("loss_mask")
    advantages = datum.objective_inputs["advantages"]

    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    if loss_mask is None:
        seq_logprobs = token_logprobs.mean(dim=1)
        entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
    else:
        masked_logprobs = token_logprobs * loss_mask
        seq_logprobs = masked_logprobs.sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)
        per_token_entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
        entropy = (per_token_entropy * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    pg_loss = -(seq_logprobs * advantages).mean()

    return StepResult(
        backprop_loss=pg_loss,
        losses={
            "total": pg_loss.detach().item(),
            "policy": pg_loss.detach().item(),
            "pg_loss": pg_loss.detach().item(),
        },
        other_metrics={
            "entropy": entropy.detach().item(),
            "avg_logprob": seq_logprobs.detach().mean().item(),
            "avg_advantage": advantages.detach().mean().item(),
        },
    )


def training_sample_to_distill_datum(sample: TrainingSample) -> TrainingDatum:
    """Convert a TrainingSample with teacher logprobs into a distillation datum."""
    assert sample.teacher_log_probs is not None, "distillation sample requires teacher_log_probs"

    tokens = torch.tensor(sample.tokens, dtype=torch.long).unsqueeze(0)
    labels = tokens.clone()
    loss_mask = torch.tensor(sample.loss_mask, dtype=torch.float32).unsqueeze(0)
    teacher_logprobs = torch.tensor(sample.teacher_log_probs, dtype=torch.float32).unsqueeze(0)

    return TrainingDatum(
        model_input=ModelInput(tokens=tokens),
        objective_inputs={
            "labels": labels,
            "loss_mask": loss_mask,
            "teacher_logprobs": teacher_logprobs,
        },
        metadata=sample.metadata,
    )


def distillation_contract_loss(products: ForwardProducts, datum: TrainingDatum) -> StepResult:
    """On-policy distillation witness loss using teacher token logprobs."""
    assert products.logits is not None, "distillation loss requires logits"
    assert "labels" in datum.objective_inputs, "distillation loss requires labels"
    assert "loss_mask" in datum.objective_inputs, "distillation loss requires loss_mask"
    assert "teacher_logprobs" in datum.objective_inputs, (
        "distillation loss requires teacher_logprobs"
    )

    logits = products.logits
    labels = datum.objective_inputs["labels"]
    loss_mask = datum.objective_inputs["loss_mask"]
    teacher_logprobs = datum.objective_inputs["teacher_logprobs"]

    log_probs = F.log_softmax(logits, dim=-1)
    student_logprobs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    advantages = (teacher_logprobs - student_logprobs) * loss_mask

    num_tokens = loss_mask.sum().clamp(min=1.0)
    pg_loss = -(student_logprobs * advantages.detach() * loss_mask).sum() / num_tokens

    with torch.no_grad():
        per_token_entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
        entropy = (per_token_entropy * loss_mask).sum() / num_tokens
        avg_student_lp = (student_logprobs * loss_mask).sum() / num_tokens
        avg_teacher_lp = (teacher_logprobs * loss_mask).sum() / num_tokens
        avg_advantage = (advantages * loss_mask).sum() / num_tokens
        kl_div = ((student_logprobs - teacher_logprobs) * loss_mask).sum() / num_tokens

    return StepResult(
        backprop_loss=pg_loss,
        losses={
            "total": pg_loss.detach().item(),
            "policy": pg_loss.detach().item(),
        },
        other_metrics={
            "entropy": entropy.detach().item(),
            "avg_student_logprob": avg_student_lp.detach().item(),
            "avg_teacher_logprob": avg_teacher_lp.detach().item(),
            "avg_advantage": avg_advantage.detach().item(),
            "kl_div": kl_div.detach().item(),
            "num_tokens": num_tokens.detach().item(),
        },
    )


def moe_supervised_contract_loss(products: ForwardProducts, datum: TrainingDatum) -> StepResult:
    """MoE supervised witness loss with optional router auxiliary term."""
    assert products.logits is not None, "moe supervised loss requires logits"
    assert "labels" in datum.objective_inputs, "moe supervised loss requires labels"

    logits = products.logits
    labels = datum.objective_inputs["labels"]
    loss_mask = datum.objective_inputs.get("loss_mask")

    if loss_mask is None:
        xent = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        num_tokens = torch.tensor(float(labels.numel()), device=logits.device)
    else:
        per_token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
        masked_loss = per_token_loss * loss_mask
        num_tokens = loss_mask.sum().clamp(min=1.0)
        xent = masked_loss.sum() / num_tokens

    router_aux = products.aux.get("router_aux_loss")
    if router_aux is None:
        router_aux = torch.tensor(0.0, device=logits.device)

    total = xent + router_aux

    with torch.no_grad():
        perplexity = torch.exp(xent.detach()).item()
        metrics = {
            "perplexity": perplexity,
            "num_tokens": num_tokens.detach().item(),
            "router_aux_present": 1.0 if "router_aux_loss" in products.aux else 0.0,
        }
        if datum.precision_policy is not None:
            metrics["uses_router_fp32"] = 1.0 if datum.precision_policy.router == "fp32" else 0.0
            metrics["uses_lowp_experts"] = (
                1.0 if datum.precision_policy.expert_compute not in (None, "bf16", "fp32") else 0.0
            )

    return StepResult(
        backprop_loss=total,
        losses={
            "total": total.detach().item(),
            "xent": xent.detach().item(),
            "router_aux": router_aux.detach().item(),
        },
        other_metrics=metrics,
    )
