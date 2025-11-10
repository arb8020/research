"""Pure function implementation of RL training loop.

SLIME-inspired: Generation → Training → Weight Sync loop.
No classes, no hidden state - just explicit orchestration.
"""

import logging
import trio
from typing import List, Dict, Optional
from pathlib import Path

from rollouts.training.backends import PyTorchTrainingBackend
from rollouts.training.metrics import MetricsLogger
from training.data_buffer import DataBuffer
from training.async_rollout_manager import AsyncRolloutManager
from training.weight_sync import sync_weights_to_engines, InferenceEngine
from training.types import RLTrainingConfig, Sample, RolloutBatch

logger = logging.getLogger(__name__)


async def run_rl_training(
    backend: PyTorchTrainingBackend,
    data_buffer: DataBuffer,
    rollout_manager: AsyncRolloutManager,
    inference_engines: List[InferenceEngine],
    config: RLTrainingConfig,
    metrics_logger: Optional[MetricsLogger] = None,
) -> List[Dict[str, float]]:
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
        >>> from rollouts.training.metrics import JSONLLogger
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

    logger.info("Starting RL training...")
    logger.info(f"  Steps: {config.num_steps}")
    logger.info(f"  Weight sync every: {config.sync_every} steps")
    logger.info(f"  Inference engines: {len(inference_engines)}")

    async with rollout_manager:  # Context manager for cleanup
        for step in range(config.num_steps):
            # SLIME Step 1: Generate rollouts
            batch = await rollout_manager.generate_batch()

            # SLIME Step 2: Compute rewards (pure function)
            rewards = [compute_reward(s) for s in batch.samples]

            # SLIME Step 3: Prepare RL batch (pure function)
            rl_batch = prepare_grpo_batch(batch, rewards, config)

            # SLIME Step 4: Train
            fwd_metrics = await backend.forward_backward(rl_batch).result()
            opt_metrics = await backend.optim_step().result()

            # Combine metrics
            step_metrics = {
                **fwd_metrics,
                **opt_metrics,
                "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "max_reward": max(rewards) if rewards else 0.0,
                "min_reward": min(rewards) if rewards else 0.0,
                "step": step,
            }
            metrics_history.append(step_metrics)

            # SLIME Step 5: Sync weights to inference engines (D5)
            if step % config.sync_every == 0 and step > 0:
                ckpt_path = await backend.save_checkpoint(step, step_metrics)
                await sync_weights_to_engines(inference_engines, str(ckpt_path))
                logger.info(f"  Synced weights to {len(inference_engines)} engines")

            # ═══════════════════════════════════════════════════
            # ERROR LOGGING: Events (sporadic)
            # ═══════════════════════════════════════════════════
            if step % config.log_every == 0:
                logger.info(
                    f"Step {step}: "
                    f"reward={step_metrics['mean_reward']:.2f}, "
                    f"loss={fwd_metrics['loss']:.4f}, "
                    f"grad_norm={fwd_metrics['grad_norm']:.4f}"
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


def compute_reward(sample: Sample) -> float:
    """Pure function: Compute reward from sample.

    Uses environment grading (for now, reward model later).

    Args:
        sample: Sample with metadata containing grading result

    Returns:
        Reward (1.0 if correct, 0.0 otherwise)
    """
    # Simple environment-based grading
    if sample.metadata.get("correct", False):
        return 1.0
    return 0.0


def prepare_grpo_batch(
    batch: RolloutBatch,
    rewards: List[float],
    config: RLTrainingConfig,
) -> Dict[str, any]:
    """Pure function: Prepare GRPO training batch.

    Args:
        batch: Rollout batch
        rewards: Rewards for each sample
        config: RL config with baseline

    Returns:
        RL training batch with advantages
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

    # Prepare batch (similar to SFT, but with advantages)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "advantages": advantage_tensor,
    }


def compute_advantages(
    rewards: List[float],
    baseline: float = 0.0,
) -> List[float]:
    """Pure function: Compute advantages from rewards.

    Args:
        rewards: List of rewards
        baseline: Baseline for advantage computation

    Returns:
        List of advantages (rewards - baseline)
    """
    return [r - baseline for r in rewards]
