"""Pure function implementation of RL training loop.

SLIME-inspired: Generation → Training → Weight Sync loop.
No classes, no hidden state - just explicit orchestration.
"""

import logging

from ...training.backends import PyTorchTrainingBackend
from ...training.contract_witnesses import rl_contract_loss
from ...training.contracts import (
    ModelInput,
    TrainableParameterPolicy,
    TrainingDatum,
    TrainingRuntimeState,
    WeightPublication,
    WeightSyncPolicy,
    WeightVersion,
)
from ...training.datasets.data_buffer import DataBuffer
from ...training.metrics import MetricsLogger
from ...training.rollout_gen.async_rollout_manager import AsyncRolloutManager
from ...training.types import RLTrainingConfig, RolloutBatch
from ...training.weight_sync import InferenceEngine, sync_weights_to_engines

logger = logging.getLogger(__name__)


async def run_rl_training(
    backend: PyTorchTrainingBackend,
    data_buffer: DataBuffer,
    rollout_manager: AsyncRolloutManager,
    inference_engines: list[InferenceEngine],
    config: RLTrainingConfig,
    metrics_logger: MetricsLogger | None = None,
) -> list[dict[str, float]]:
    """Run RL training (pure function, no hidden state).

    Args:
        backend: Training backend (stateful)
        data_buffer: Data buffer (stateful)
        rollout_manager: Rollout manager (stateful)
        inference_engines: Inference engines for weight sync
        config: RL training configuration (immutable)
        metrics_logger: Optional metrics logger (Casey: explicit parameter)

    Returns:
        List of metrics dicts (one per step)

    Example:
        >>> from ...training.metrics import JSONLLogger
        >>>
        >>> backend = PyTorchTrainingBackend(...)
        >>> data_buffer = DataBuffer(prompts=[...])
        >>> rollout_manager = AsyncRolloutManager(data_buffer, rollout_config)
        >>> engines = [SGLangEngine(...)]
        >>> config = RLTrainingConfig(num_steps=1000, sync_every=10)
        >>> logger = JSONLLogger(Path("logs/exp_001"))
        >>>
        >>> metrics = await run_rl_training(
        ...     backend, data_buffer, rollout_manager, engines, config, logger
        ... )

    SLIME-inspired: Generation → Training → Weight Sync loop.
    Casey Muratori: No retention, explicit flow.
    """
    # Tiger Style: Assert preconditions
    assert config.num_steps > 0, "num_steps must be > 0"
    assert config.sync_every > 0, "sync_every must be > 0"

    metrics_history = []
    weight_sync_policy = WeightSyncPolicy.every_n_steps_policy(
        config.sync_every,
        mode="disk",
    )
    state = TrainingRuntimeState(
        step=0,
        weight_version=WeightVersion(getattr(backend, "weight_version", 0)),
    )

    logger.info("Starting RL training...")
    logger.info(f"  Steps: {config.num_steps}")
    logger.info(
        f"  Weight sync policy: mode={weight_sync_policy.mode}, "
        f"every={weight_sync_policy.every_n_steps}, enabled={weight_sync_policy.enabled}"
    )
    logger.info(f"  Inference engines: {len(inference_engines)}")
    logger.info(f"  Initial weight version: {state.weight_version.value}")

    async with rollout_manager:  # Context manager for cleanup
        for step in range(config.num_steps):
            # SLIME Step 1: Generate rollouts (rewards computed during generation)
            batch = await rollout_manager.generate_batch()

            # SLIME Step 2: Use pre-computed rewards from batch
            rewards = batch.rewards

            # SLIME Step 3: Prepare RL batch (pure function)
            rl_batch = prepare_grpo_batch(batch, rewards, config)

            # SLIME Step 4: Train
            fwd_result = await backend.forward_backward(rl_batch, loss_fn=rl_contract_loss).result()
            opt_metrics = await backend.optim_step().result()
            state = TrainingRuntimeState(
                step=step,
                weight_version=WeightVersion(
                    getattr(backend, "weight_version", state.weight_version.value)
                ),
            )

            # Combine metrics
            step_metrics = {
                **fwd_result.losses,
                **fwd_result.other_metrics,
                **opt_metrics,
                "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "max_reward": max(rewards) if rewards else 0.0,
                "min_reward": min(rewards) if rewards else 0.0,
                "weight_version": state.weight_version.value,
                "step": step,
            }
            metrics_history.append(step_metrics)

            # SLIME Step 5: Sync weights to inference engines (D5)
            if weight_sync_policy.should_publish(step):
                ckpt_path = await backend.save_checkpoint(step, step_metrics)
                state = TrainingRuntimeState(
                    step=step,
                    weight_version=WeightVersion(
                        getattr(backend, "weight_version", state.weight_version.value)
                    ),
                )
                sync_responses = await sync_weights_to_engines(inference_engines, str(ckpt_path))
                publication = WeightPublication(
                    step=step,
                    version=state.weight_version,
                    checkpoint_path=str(ckpt_path),
                    engine_count=len(inference_engines),
                    sync_responses=tuple(sync_responses),
                )
                step_metrics["published_weight_version"] = publication.version.value
                step_metrics["published_checkpoint_path"] = publication.checkpoint_path
                step_metrics["published_engine_count"] = float(publication.engine_count)
                step_metrics["published"] = 1.0
                step_metrics["weight_sync_policy_every_n_steps"] = float(
                    weight_sync_policy.every_n_steps
                )
                logger.info(f"  Synced weights to {len(inference_engines)} engines")
            else:
                step_metrics["published"] = 0.0
                step_metrics["weight_sync_policy_every_n_steps"] = float(
                    weight_sync_policy.every_n_steps
                )

            # ═══════════════════════════════════════════════════
            # ERROR LOGGING: Events (sporadic)
            # ═══════════════════════════════════════════════════
            if step % config.log_every == 0:
                logger.info(
                    f"Step {step}: "
                    f"reward={step_metrics['mean_reward']:.2f}, "
                    f"loss={fwd_result.losses['total']:.4f}, "
                    f"grad_norm={fwd_result.other_metrics['grad_norm']:.4f}, "
                    f"weight_version={step_metrics['weight_version']}"
                )

            # ═══════════════════════════════════════════════════
            # METRICS LOGGING: Timeseries (regular)
            # ═══════════════════════════════════════════════════
            if metrics_logger and step % config.log_every == 0:
                metrics_logger.log(step_metrics, step=step)

            # Checkpoint
            if step % config.checkpoint_every == 0 and step > 0:
                ckpt_path = await backend.save_checkpoint(step, step_metrics)
                logger.info(f"  Saved checkpoint to {ckpt_path}")

    logger.info("RL training complete!")

    # Finish metrics logging
    if metrics_logger:
        metrics_logger.finish()

    return metrics_history


def prepare_grpo_batch(
    batch: RolloutBatch,
    rewards: list[float],
    config: RLTrainingConfig,
) -> TrainingDatum:
    """Pure function: Prepare GRPO training batch.

    Args:
        batch: Rollout batch
        rewards: Rewards for each sample
        config: RL config with baseline

    Returns:
        RL training datum with advantages
    """
    import torch

    # Compute advantages (pure function)
    advantages = compute_advantages(rewards, config.baseline)

    # Convert to tensors
    advantage_tensor = torch.tensor(advantages, dtype=torch.float32)

    # Convert tokens and masks to tensors
    tokens_list = []
    loss_masks_list = []

    for i, sample_tokens in enumerate(batch.tokens):
        if isinstance(sample_tokens, list):
            tokens_list.append(torch.tensor(sample_tokens, dtype=torch.long))
            loss_masks_list.append(torch.tensor(batch.loss_masks[i], dtype=torch.float))
        else:
            tokens_list.append(sample_tokens)
            loss_masks_list.append(batch.loss_masks[i])

    # Stack into batch
    input_ids = torch.stack(tokens_list)
    labels = torch.stack(tokens_list)  # Same as input for causal LM
    loss_mask = torch.stack(loss_masks_list)

    return TrainingDatum(
        model_input=ModelInput(tokens=input_ids),
        objective_inputs={
            "labels": labels,
            "loss_mask": loss_mask,
            "advantages": advantage_tensor,
            "group_ids": torch.tensor(batch.group_indices, dtype=torch.long),
        },
        trainable_parameter_policy=TrainableParameterPolicy.full_weight(),
        metadata=batch.metadata,
    )


def compute_advantages(
    rewards: list[float],
    baseline: float = 0.0,
) -> list[float]:
    """Pure function: Compute advantages from rewards.

    Args:
        rewards: List of rewards
        baseline: Baseline for advantage computation

    Returns:
        List of advantages (rewards - baseline)
    """
    return [r - baseline for r in rewards]
