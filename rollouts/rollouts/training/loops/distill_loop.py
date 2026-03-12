"""Pure function implementation of distillation training loop."""

import logging

from ...training.backends import PyTorchTrainingBackend
from ...training.contract_witnesses import (
    distillation_contract_loss,
)
from ...training.contracts import ModelInput, TrainableParameterPolicy, TrainingDatum
from ...training.metrics import MetricsLogger
from ...training.types import SFTTrainingConfig, TrainingSample

logger = logging.getLogger(__name__)


async def run_distill_training(
    backend: PyTorchTrainingBackend,
    samples: list[TrainingSample],
    config: SFTTrainingConfig,
    metrics_logger: MetricsLogger | None = None,
) -> list[dict[str, float]]:
    """Run distillation training over samples carrying teacher logprobs."""
    assert len(samples) > 0, "samples cannot be empty"
    assert config.num_steps > 0, "num_steps must be > 0"
    assert all(s.teacher_log_probs is not None for s in samples), (
        "all distillation samples must carry teacher_log_probs"
    )

    metrics_history = []

    logger.info("Starting distillation training...")
    logger.info(f"  Samples: {len(samples)}")
    logger.info(f"  Steps: {config.num_steps}")
    logger.info(f"  Batch size: {config.batch_size}")

    for step in range(config.num_steps):
        datum = collate_distill_batch(samples, config.batch_size, step)

        fwd_result = await backend.forward_backward(
            datum,
            loss_fn=distillation_contract_loss,
        ).result()
        opt_metrics = await backend.optim_step().result()

        step_metrics = {
            **fwd_result.losses,
            **fwd_result.other_metrics,
            **opt_metrics,
            "step": step,
        }
        metrics_history.append(step_metrics)

        if step % config.log_every == 0:
            logger.info(
                f"Step {step}: "
                f"loss={fwd_result.losses['total']:.4f}, "
                f"kl_div={fwd_result.other_metrics['kl_div']:.4f}, "
                f"lr={opt_metrics['lr']:.4e}"
            )

        if metrics_logger and step % config.log_every == 0:
            metrics_logger.log(step_metrics, step=step)

        if step % config.checkpoint_every == 0 and step > 0:
            ckpt_path = await backend.save_checkpoint(step, step_metrics)
            logger.info(f"  Saved checkpoint to {ckpt_path}")

    logger.info("Distillation training complete!")

    if metrics_logger:
        metrics_logger.finish()

    return metrics_history


def collate_distill_batch(
    samples: list[TrainingSample],
    batch_size: int,
    step: int,
) -> TrainingDatum:
    """Pure function: collate distillation samples into a packed TrainingDatum."""
    assert batch_size > 0, "batch_size must be > 0"

    start_idx = (step * batch_size) % len(samples)
    end_idx = start_idx + batch_size

    if end_idx <= len(samples):
        batch_samples = samples[start_idx:end_idx]
    else:
        batch_samples = samples[start_idx:] + samples[: end_idx - len(samples)]

    return prepare_distill_batch(batch_samples)


def prepare_distill_batch(samples: list[TrainingSample]) -> TrainingDatum:
    """Pure function: pack distillation samples into a single TrainingDatum."""
    import torch

    flat_tokens = []
    flat_masks = []
    flat_teacher_logprobs = []
    flat_position_ids = []
    cu_seqlens = [0]

    for sample in samples:
        assert sample.teacher_log_probs is not None, (
            "distillation sample requires teacher_log_probs"
        )

        tokens = sample.tokens if isinstance(sample.tokens, list) else sample.tokens.tolist()
        loss_mask = (
            sample.loss_mask if isinstance(sample.loss_mask, list) else sample.loss_mask.tolist()
        )
        teacher_logprobs = (
            sample.teacher_log_probs
            if isinstance(sample.teacher_log_probs, list)
            else sample.teacher_log_probs.tolist()
        )

        assert len(tokens) == len(loss_mask) == len(teacher_logprobs), (
            "tokens, loss_mask, and teacher_log_probs must align"
        )

        flat_tokens.extend(tokens)
        flat_masks.extend(loss_mask)
        flat_teacher_logprobs.extend(teacher_logprobs)
        flat_position_ids.extend(range(len(tokens)))
        cu_seqlens.append(cu_seqlens[-1] + len(tokens))

    return TrainingDatum(
        model_input=ModelInput(
            tokens=torch.tensor(flat_tokens, dtype=torch.long).unsqueeze(0),
            positions=torch.tensor(flat_position_ids, dtype=torch.long).unsqueeze(0),
        ),
        objective_inputs={
            "labels": torch.tensor(flat_tokens, dtype=torch.long).unsqueeze(0),
            "loss_mask": torch.tensor(flat_masks, dtype=torch.float32).unsqueeze(0),
            "teacher_logprobs": torch.tensor(flat_teacher_logprobs, dtype=torch.float32).unsqueeze(
                0
            ),
        },
        metadata={
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.long),
        },
        trainable_parameter_policy=TrainableParameterPolicy.lora_only(),
    )
