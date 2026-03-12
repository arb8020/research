"""Generic training loop (algorithm-agnostic).

Separates orchestration (steps, checkpoint cadence, weight sync cadence)
from algorithm-specific batch processing (loss/advantage computation).

This is the core refactor described in rollouts/docs/training_architecture.md.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..training.configs import CheckpointConfig
from ..training.metrics import MetricsLogger
from ..training.weight_sync import WeightSyncer

if TYPE_CHECKING:
    from ..training.backends.protocol import TrainingBackend


@dataclass(frozen=True)
class TrainResult:
    """Result of a training run."""

    metrics_history: list[dict[str, Any]]
    last_checkpoint: Path | None = None


def should_checkpoint(step: int, config: CheckpointConfig) -> bool:
    """True if this step should trigger a checkpoint save."""
    assert step >= 0, f"step must be >= 0, got {step}"
    assert config.checkpoint_every > 0, (
        f"checkpoint_every must be > 0, got {config.checkpoint_every}"
    )
    return (step + 1) % config.checkpoint_every == 0


def should_sync(step: int, config: CheckpointConfig) -> bool:
    """True if this step should trigger weight sync to inference."""
    assert step >= 0, f"step must be >= 0, got {step}"
    assert config.sync_weights_every > 0, (
        f"sync_weights_every must be > 0, got {config.sync_weights_every}"
    )
    return (step + 1) % config.sync_weights_every == 0


async def train(
    config: CheckpointConfig,
    backend: TrainingBackend,
    batch_iterator: AsyncIterator[Any],
    process_batch: Callable[[int, Any, TrainingBackend], Awaitable[dict[str, Any] | None]],
    *,
    weight_syncer: WeightSyncer | None,
    save_checkpoint: Callable[[int, dict[str, Any]], Awaitable[Path]] | None,
    metrics_logger: MetricsLogger | None,
    logger: logging.Logger,
    before_weight_sync: Callable[[], Awaitable[None]] | None = None,
    after_weight_sync: Callable[[], Awaitable[None]] | None = None,
) -> TrainResult:
    """Run training for a fixed number of steps.

    Args:
        config: Step counts + cadence knobs (checkpoint/sync/log).
        backend: Training backend (owns model + optimizer).
        batch_iterator: Async iterator yielding batches (rollouts or dataset batches).
        process_batch: Algorithm-specific processing for a single step.
        weight_syncer: Optional weight syncer (RL), None for offline training (SFT/pretrain).
        save_checkpoint: Optional checkpoint saver. If None, checkpointing is disabled.
        metrics_logger: Optional metrics logger (timeseries).
        logger: Python logger (wide events).

    Returns:
        TrainResult with metrics history and last checkpoint path (if any).
    """
    assert config.num_steps > 0, f"num_steps must be > 0, got {config.num_steps}"
    assert config.log_every > 0, f"log_every must be > 0, got {config.log_every}"

    metrics_history: list[dict[str, Any]] = []
    last_checkpoint: Path | None = None

    try:
        for step in range(config.num_steps):
            try:
                batch = await anext(batch_iterator)
            except StopAsyncIteration as e:
                raise RuntimeError(
                    f"Batch iterator exhausted early at step {step + 1}/{config.num_steps}"
                ) from e

            # Log step header *after* fetching the next batch.
            # This ensures any "between-step" work done by the batch iterator (e.g., pipeline updates)
            # is ordered before the next step's log block.
            logger.info(f"\n--- Step {step + 1}/{config.num_steps} ---")

            process_start = time.perf_counter()
            step_metrics = await process_batch(step, batch, backend)
            process_batch_ms = (time.perf_counter() - process_start) * 1000

            if step_metrics is None:
                continue

            # Always record history (used by callers for summaries + checks).
            metrics_history.append({"step": step + 1, **step_metrics})

            # Timeseries metrics logging
            if metrics_logger and (step + 1) % config.log_every == 0:
                numeric_metrics: dict[str, float] = {}
                for k, v in step_metrics.items():
                    if isinstance(v, (int, float)):
                        numeric_metrics[k] = float(v)
                metrics_logger.log(numeric_metrics, step=step + 1)

            # Checkpoint
            checkpoint_ms = 0.0
            if save_checkpoint is not None and should_checkpoint(step, config):
                ckpt_start = time.perf_counter()
                last_checkpoint = await save_checkpoint(step + 1, step_metrics)
                checkpoint_ms = (time.perf_counter() - ckpt_start) * 1000

            # Weight sync (if doing RL with separate inference)
            weight_sync_ms = 0.0
            if weight_syncer is not None and should_sync(step, config):
                sync_start = time.perf_counter()
                try:
                    if before_weight_sync is not None:
                        await before_weight_sync()
                    await weight_syncer.sync()
                finally:
                    if after_weight_sync is not None:
                        await after_weight_sync()
                weight_sync_ms = (time.perf_counter() - sync_start) * 1000

            step_total_ms = process_batch_ms + checkpoint_ms + weight_sync_ms

            # Wide event: one log line per step with all context
            if (step + 1) % config.log_every == 0:
                wide_event = {
                    "event": "train_step_complete",
                    "step": step + 1,
                    "process_batch_ms": round(process_batch_ms, 1),
                    "checkpoint_ms": round(checkpoint_ms, 1),
                    "weight_sync_ms": round(weight_sync_ms, 1),
                    "step_total_ms": round(step_total_ms, 1),
                    **{k: v for k, v in step_metrics.items() if isinstance(v, (int, float))},
                }
                logger.info("train_step_complete", extra=wide_event)
    finally:
        # Best-effort cleanup even on exceptions (prevents leaked NCCL groups / open files).
        if metrics_logger:
            metrics_logger.finish()
        if weight_syncer is not None:
            await weight_syncer.close()

    return TrainResult(metrics_history=metrics_history, last_checkpoint=last_checkpoint)
