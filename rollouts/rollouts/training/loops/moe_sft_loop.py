"""Pure function implementation of MoE supervised training witness loop."""

import logging

from ...training.backends import PyTorchTrainingBackend
from ...training.contract_witnesses import moe_supervised_contract_loss
from ...training.contracts import PrecisionPolicy, TrainableParameterPolicy, TrainingDatum
from ...training.metrics import MetricsLogger
from ...training.types import SFTTrainingConfig, TrainingSample

logger = logging.getLogger(__name__)


async def run_moe_sft_training(
    backend: PyTorchTrainingBackend,
    samples: list[TrainingSample],
    config: SFTTrainingConfig,
    precision_policy: PrecisionPolicy,
    metrics_logger: MetricsLogger | None = None,
) -> list[dict[str, float]]:
    """Run MoE supervised training with explicit precision policy."""
    assert len(samples) > 0, "samples cannot be empty"
    assert config.num_steps > 0, "num_steps must be > 0"
    assert config.batch_size > 0, "batch_size must be > 0"

    from ...training.loops.sft_loop import collate_batch

    metrics_history = []

    logger.info("Starting MoE supervised training...")
    logger.info(f"  Samples: {len(samples)}")
    logger.info(f"  Steps: {config.num_steps}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(
        "  Precision: "
        f"param={precision_policy.param}, "
        f"compute={precision_policy.compute}, "
        f"reduce={precision_policy.reduce}, "
        f"router={precision_policy.router}, "
        f"expert_compute={precision_policy.expert_compute}"
    )

    for step in range(config.num_steps):
        datum = collate_batch(samples, config.batch_size, step)
        moe_datum = TrainingDatum(
            model_input=datum.model_input,
            objective_inputs=datum.objective_inputs,
            precision_policy=precision_policy,
            trainable_parameter_policy=TrainableParameterPolicy.full_weight(),
            metadata=datum.metadata,
        )

        fwd_result = await backend.forward_backward(
            moe_datum,
            loss_fn=moe_supervised_contract_loss,
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
                f"xent={fwd_result.losses['xent']:.4f}, "
                f"router_aux={fwd_result.losses['router_aux']:.4f}, "
                f"grad_norm={fwd_result.other_metrics['grad_norm']:.4f}"
            )

        if metrics_logger and step % config.log_every == 0:
            metrics_logger.log(step_metrics, step=step)

        if step % config.checkpoint_every == 0 and step > 0:
            ckpt_path = await backend.save_checkpoint(step, step_metrics)
            logger.info(f"  Saved checkpoint to {ckpt_path}")

    logger.info("MoE supervised training complete!")

    if metrics_logger:
        metrics_logger.finish()

    return metrics_history
