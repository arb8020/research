"""Eval loop - rl_loop.py without the training step.

MIGRATION TARGET: eval/native.py's _evaluate_batch should be replaced by this.

The core insight: eval and RL are the same pipeline. Both:
  1. Pull prompts from a buffer
  2. Run generate_fn (rollout worker hits inference engine, gets trajectory)
  3. Score results

RL additionally:
  4. Compute gradients + update weights (trainer step)
  5. Sync weights to inference engine

So eval is rl_loop.py with steps 4-5 removed. They should share
rollout collection infrastructure (AsyncRolloutManager, observability,
concurrency management) rather than maintaining two separate loops.

Current state:
  - eval/native.py has its own _evaluate_batch loop (duplicates RL rollout logic)
  - rl_loop.py uses AsyncRolloutManager correctly (slime-inspired)
  - eval/native.py produces AttemptResult; RL produces AttemptRow via RolloutBatch
  - Observability (events.jsonl, StreamChunk, progress) lives only in eval/native.py
    and is not available in the RL rollout path

Target state:
  - This file (eval_loop.py) replaces eval/native.py's core loop
  - Uses AsyncRolloutManager.generate_batch() as the shared rollout primitive
  - eval/native.py becomes a thin reporting/output layer on top of RolloutBatch
  - RL rollout path gains eval's observability (events.jsonl, per-sample progress)
    by tapping AsyncRolloutManager at the same points

Migration steps:
  1. Implement run_eval() below using AsyncRolloutManager (mirrors rl_loop.run_rl_training)
  2. Port eval/native.py's observability hooks into AsyncRolloutManager callbacks
     (on_sample_start, on_sample_end, on_chunk) so both paths emit the same events
  3. Update eval/run.py to call run_eval() instead of eval/native.py's loop
  4. Update grpo.py to use rl_loop.run_rl_training() for the main loop
     (grpo.py currently has its own inline version; rl_loop.py is the clean version)
  5. Delete the duplicate loop from eval/native.py

Data flow (same for eval and RL):

    DataBuffer
        ↓ prompts
    generate_fn(prompts)          ← user-provided; calls run_agent(); hits inference engine
        ↓ list[AttemptRow]        ← trajectory + tokens + loss_mask + logprobs
    scorer(attempt_row)           ← reward attached inline (same scorer protocol as RL)
        ↓ RolloutBatch
    [eval] report results         ← eval stops here
    [RL]   trainer.forward_backward() + sync_weights_to_engines()

Key types (already exist, no new types needed):
    DataBuffer          - training/datasets/data_buffer.py
    AsyncRolloutManager - training/rollout_gen/async_rollout_manager.py
    RolloutBatch        - training/types.py (has .attempts: list[AttemptRow])
    AttemptRow          - training/types.py (trajectory + tokens + reward)
    RolloutRuntime      - training/types.py (generate_fn + scorer + filter_fn)

Environment/sandbox pattern (already works, no changes needed):
    generate_fn receives batch_prompts: list[dict]
    Each prompt flows to agent_rollout_to_sample(environment_factory=...)
    environment_factory(sample_data) called once per rollout
    Handles Docker containers, Modal sandboxes, subprocesses, etc.
    Lifecycle: factory.start() before loop, factory.stop() after
"""

import logging

from ...training.datasets.data_buffer import DataBuffer
from ...training.metrics import MetricsLogger
from ...training.rollout_gen.async_rollout_manager import AsyncRolloutManager
from ...training.types import RolloutBatch, RolloutConfig

logger = logging.getLogger(__name__)


async def run_eval(
    data_buffer: DataBuffer,
    rollout_manager: AsyncRolloutManager,
    config: RolloutConfig,
    metrics_logger: MetricsLogger | None = None,
    max_batches: int | None = None,
) -> list[RolloutBatch]:
    """Run eval loop - collect rollouts and score them, no weight updates.

    This is rl_loop.run_rl_training() with steps 4-5 (train + weight sync) removed.
    Use this instead of eval/native.py's _evaluate_batch for new evals.

    Args:
        data_buffer: Prompt source (manages epoch/offset state)
        rollout_manager: Handles concurrent rollout collection + scoring
        config: Batch size, oversampling, etc.
        metrics_logger: Optional structured logging (same as RL path)
        max_batches: Stop after N batches (None = run through buffer once)

    Returns:
        List of RolloutBatch objects. Each batch has:
            .attempts: list[AttemptRow] - trajectories with scores attached
            .rewards: list[float] - scalar rewards
            .tokens: list[list[int]] - token ids (for logprob analysis)
            .rollout_log_probs: per-token logprobs (for inference engine verification)

    Example:
        >>> rollout_manager = AsyncRolloutManager(
        ...     data_buffer=buffer,
        ...     config=config,
        ...     runtime=RolloutRuntime(
        ...         generate_fn=my_generate_fn,   # calls run_agent() → inference engine
        ...         scorer=my_scorer,              # same scorer protocol as RL
        ...     ),
        ... )
        >>> batches = await run_eval(buffer, rollout_manager, config)
        >>> for batch in batches:
        ...     for attempt in batch.attempts:
        ...         print(attempt.reward, attempt.trajectory)

    TODO: implement this using AsyncRolloutManager.generate_batch().
    The shape mirrors rl_loop.run_rl_training() - see that file for the pattern.
    The key difference: no backend.forward_backward(), no sync_weights_to_engines().
    """
    raise NotImplementedError(
        "eval_loop.run_eval: implement using AsyncRolloutManager.generate_batch(). "
        "See rl_loop.run_rl_training() for the pattern - this is the same loop "
        "without steps 4-5 (train + weight sync). "
        "Port observability hooks from eval/native.py._evaluate_batch into "
        "AsyncRolloutManager callbacks so both paths emit the same events."
    )
