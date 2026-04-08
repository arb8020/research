"""Eval loop - rl_loop.py without the training step.

MIGRATION TARGET: eval/native.py's _evaluate_batch should be replaced by this.

The core insight: eval and RL are the same pipeline. Both:
  1. Pull prompts from a DataBuffer
  2. Run generate_fn (rollout worker hits inference engine, gets Trajectory)
  3. Score results via scorer

RL additionally:
  4. Compute gradients + update weights (trainer step)
  5. Sync weights to inference engine

So eval is rl_loop.py with steps 4-5 removed. They share rollout collection
infrastructure (AsyncRolloutManager) rather than maintaining two separate loops.

Current state:
  - eval/native.py has its own _evaluate_batch loop (duplicates RL rollout logic)
  - rl_loop.py uses AsyncRolloutManager correctly (slime-inspired)
  - eval/native.py produces AttemptResult; RL produces AttemptRow (see types.py)
  - Observability (events.jsonl, StreamChunk, progress) lives only in eval/native.py
    and is not available in the RL rollout path

Target state:
  - This file replaces eval/native.py's core loop
  - Uses AsyncRolloutManager.generate_batch() as the shared rollout primitive
  - eval/native.py becomes a thin reporting layer on top of RolloutBatch.attempts
  - RL rollout path gains eval's observability by tapping AsyncRolloutManager callbacks

Migration steps:
  1. Done: run_eval() implemented below using AsyncRolloutManager
  2. TODO: Port observability hooks from eval/native.py._evaluate_batch into
     AsyncRolloutManager (on_sample_start/end callbacks, events.jsonl emission)
  3. TODO: Update eval/run.py to call run_eval() instead of eval/native.py's loop
  4. TODO: Update grpo.py to use rl_loop.run_rl_training() for the main loop
  5. TODO: Delete the duplicate loop from eval/native.py

Data flow (same for eval and RL):

    DataBuffer
        ↓ prompts
    generate_fn(prompts)          ← user-provided; calls run_agent(); hits inference engine
        ↓ list[AttemptRow]        ← trajectory + score + (optionally) tokens/loss_mask
    scorer(attempt_row)           ← reward attached inline
        ↓ RolloutBatch
    [eval] report results         ← eval stops here
    [RL]   trainer.forward_backward() + sync_weights_to_engines()

AttemptRow vs AttemptResult: see types.py deprecation note on AttemptResult.
AttemptRow is the shared type. Eval leaves training_sample=None.

Environment/sandbox pattern (already works, no changes needed):
    generate_fn receives batch_prompts: list[dict]
    Each prompt flows to agent_rollout_to_sample(environment_factory=...)
    environment_factory(sample_data) called once per rollout
    Handles Docker containers, Modal sandboxes, subprocesses, etc.
    Lifecycle: factory.start() before loop, factory.stop() after
"""

import logging
from typing import Any

from ...training.datasets.data_buffer import DataBuffer
from ...training.metrics import MetricsLogger
from ...training.rollout_gen.async_rollout_manager import AsyncRolloutManager
from ...training.types import RolloutBatch, Scorer

logger = logging.getLogger(__name__)


async def run_eval(
    data_buffer: DataBuffer,
    rollout_manager: AsyncRolloutManager,
    *,
    max_batches: int | None = None,
    scorer: Scorer | None = None,
    metrics_logger: MetricsLogger | None = None,
) -> list[RolloutBatch]:
    """Run eval loop - collect rollouts and score them, no weight updates.

    This is rl_loop.run_rl_training() with steps 4-5 (train + weight sync) removed.
    Use this instead of eval/native.py's _evaluate_batch for new evals.

    Args:
        data_buffer: Prompt source (manages epoch/offset state)
        rollout_manager: Handles concurrent rollout collection + scoring.
            Constructed with RolloutRuntime(generate_fn=..., scorer=...).
            generate_fn receives list[dict] prompts, returns list[AttemptRow].
            Each AttemptRow carries trajectory + score (training_sample=None for eval).
        max_batches: Stop after N batches. None = run until data_buffer exhausted.
        scorer: Override scorer (takes precedence over rollout_manager's scorer).
        metrics_logger: Optional structured logging.

    Returns:
        List of RolloutBatch. Each batch has:
            .attempts: list[AttemptRow] - trajectories with scores attached
            .rewards: list[float] - scalar rewards per sample
            .rollout_log_probs: per-token logprobs (for inference engine verification)
            .tokens / .loss_masks: populated only if generate_fn produces TrainingSample

    Example:
        >>> from rollouts.training.datasets.data_buffer import DataBuffer
        >>> from rollouts.training.rollout_gen.async_rollout_manager import AsyncRolloutManager
        >>> from rollouts.training.types import RolloutConfig, RolloutRuntime
        >>>
        >>> async def my_generate_fn(prompts):
        ...     # calls run_agent() against your inference engine
        ...     return [AttemptRow(trajectory=..., reward=0.0) for p in prompts]
        >>>
        >>> buffer = DataBuffer(prompts=[{"text": "hello"}, ...])
        >>> config = RolloutConfig(batch_size=4)
        >>> runtime = RolloutRuntime(generate_fn=my_generate_fn, scorer=my_scorer)
        >>> manager = AsyncRolloutManager(buffer, config, runtime=runtime)
        >>>
        >>> batches = await run_eval(buffer, manager, max_batches=10)
        >>> for batch in batches:
        ...     for attempt in batch.attempts:
        ...         print(attempt.reward, attempt.trajectory)
    """
    assert max_batches is None or max_batches > 0, "max_batches must be > 0 or None"
    # DataBuffer wraps around indefinitely (designed for RL). For eval, use max_batches
    # to bound the run. "Run through buffer once" = max_batches = len(prompts) // batch_size.

    batches: list[RolloutBatch] = []

    async with rollout_manager:
        batch_idx = 0
        while max_batches is None or batch_idx < max_batches:
            batch = await rollout_manager.generate_batch(scorer=scorer)
            batches.append(batch)

            n_attempts = len(batch.attempts)
            mean_reward = sum(batch.rewards) / len(batch.rewards) if batch.rewards else 0.0
            logger.info(
                "eval batch %d: %d attempts, mean_reward=%.3f",
                batch_idx,
                n_attempts,
                mean_reward,
            )

            if metrics_logger:
                metrics_logger.log(
                    {
                        "batch": batch_idx,
                        "n_attempts": n_attempts,
                        "mean_reward": mean_reward,
                        "max_reward": max(batch.rewards) if batch.rewards else 0.0,
                        "min_reward": min(batch.rewards) if batch.rewards else 0.0,
                    },
                    step=batch_idx,
                )

            batch_idx += 1

    logger.info(
        "eval complete: %d batches, %d total attempts",
        len(batches),
        sum(len(b.attempts) for b in batches),
    )
    return batches


def summarize_batches(batches: list[RolloutBatch]) -> dict[str, Any]:
    """Aggregate metrics across all batches.

    Pure function - no side effects. Matches the summary format from
    eval/native.py so the reporting layer doesn't need to change.

    Args:
        batches: Output of run_eval()

    Returns:
        Summary dict with mean/min/max reward, total attempts, etc.
    """
    all_attempts = [a for b in batches for a in b.attempts]
    all_rewards = [a.reward for a in all_attempts]

    if not all_rewards:
        return {
            "total_attempts": 0,
            "mean_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "n_batches": len(batches),
        }

    return {
        "total_attempts": len(all_attempts),
        "mean_reward": sum(all_rewards) / len(all_rewards),
        "min_reward": min(all_rewards),
        "max_reward": max(all_rewards),
        "n_batches": len(batches),
        "success_rate": sum(1 for a in all_attempts if a.status.value == "success")
        / len(all_attempts),
    }
