"""Pipelined Rollout Manager for versioned background sampling.

Runs sampling in background while training proceeds. Key features:
- Background sampling task that continuously generates rollouts
- Weight version tracking (samples know which model version generated them)
- max_lag parameter to discard stale samples
- Non-blocking batch retrieval
- Optional admission pause around a blocking weight-application boundary

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    PipelinedRolloutManager                  │
    │                                                              │
    │  ┌──────────────┐    ┌─────────────┐    ┌───────────────┐   │
    │  │ Background   │───▶│   Queue     │───▶│  get_batch()  │   │
    │  │ Sampler      │    │ (samples)   │    │  (filter lag) │   │
    │  └──────────────┘    └─────────────┘    └───────────────┘   │
    │         ▲                                       │           │
    │         │                                       ▼           │
    │  weight_version ◀──────── update_weight_version() ◀─────    │
    │  (from trainer)                                             │
    └─────────────────────────────────────────────────────────────┘

Usage:
    async with PipelinedRolloutManager(buffer, config) as manager:
        # Start background sampling
        await manager.start_sampling(initial_weight_version=0)

        for step in range(num_steps):
            # Get batch (filters stale samples)
            batch = await manager.get_batch(current_weight_version=step)

            # Train...

            # Update weight version (tells sampler to use new weights)
            manager.update_weight_version(step + 1)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import trio

from ...training.datasets.data_buffer import DataBuffer
from ...training.group_assembly import (
    assemble_groups,
    collect_incomplete_groups,
    count_complete_groups,
)
from ...training.rollout_gen.rollout_generation import convert_to_batch
from ...training.runtime import resolve_rollout_runtime
from ...training.scoring import resolve_scorer, score_rows
from ...training.types import (
    AttemptRow,
    IncompleteGroupPolicy,
    RolloutBatch,
    RolloutConfig,
    RolloutRuntime,
    Scorer,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelinedRolloutManager:
    """Pipelined rollout manager with background sampling (PipelineRL-inspired).

    Runs sampling continuously in background, training pulls batches when ready.
    Samples are tagged with weight_version and filtered by max_lag.

    Attributes:
        data_buffer: DataBuffer for prompt iteration
        config: RolloutConfig with batch_size, generate_fn, etc.
        max_lag: Maximum weight version difference before discarding samples.
            If current_version - sample_version > max_lag, sample is discarded.
            Set to 0 for strict on-policy (only use samples from current version).
            Set to float('inf') for off-policy (use all samples).
        queue_size: Maximum number of samples to buffer in queue.
            Set to 0 or a negative value to disable producer-side queue blocking.
        current_weight_version: Current model weight version (updated by trainer)
    """

    data_buffer: DataBuffer
    config: RolloutConfig
    runtime: RolloutRuntime | None = None
    max_lag: int = 2  # Allow samples up to 2 versions behind
    queue_size: int = 0  # 0 = no producer-side queue cap

    # Sync status callback - when True, pause sampling (inference is blocked)
    # Set this to weight_sync_manager.sync_in_progress for true_pipeline
    sync_in_progress_fn: Callable[[], bool] | None = None

    # Internal state
    _sample_queue: list[AttemptRow] = field(default_factory=list)
    _queue_lock: trio.Lock = field(default_factory=trio.Lock)
    _current_weight_version: int = 0
    _sampling_task_scope: trio.CancelScope | None = None
    _nursery: trio.Nursery | None = None
    _step_count: int = 0
    _shutdown_requested: bool = False
    _next_group_index: int = 0
    _group_prompts: dict[int, str | dict[str, Any]] = field(default_factory=dict)
    _admission_paused_reason: str | None = None

    # Stats
    _samples_generated: int = 0
    _samples_discarded_stale: int = 0
    _refill_requests: int = 0
    _sync_pause_count: int = 0
    _max_queue_length_seen: int = 0
    _admission_pause_events: int = 0

    # Rollout kwargs passed to generate_fn
    rollout_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        runtime = resolve_rollout_runtime(config=self.config, runtime=self.runtime)
        if runtime is None:
            raise ValueError("Rollout runtime must provide generate_fn")
        self.runtime = runtime
        if self.config.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.config.batch_size}")
        if self.config.max_refill_rounds < 0:
            raise ValueError(f"max_refill_rounds must be >= 0, got {self.config.max_refill_rounds}")

    async def __aenter__(self) -> "PipelinedRolloutManager":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        """Async context manager exit - stop sampling."""
        await self.stop_sampling()
        return False

    @property
    def current_weight_version(self) -> int:
        """Current weight version (read by sampler, updated by trainer)."""
        return self._current_weight_version

    def update_weight_version(self, version: int) -> None:
        """Update weight version (called by trainer after weight sync).

        This tells the sampler that new weights are available.
        Samples generated after this will be tagged with the new version.

        Args:
            version: New weight version number
        """
        assert version >= self._current_weight_version, (
            f"Weight version must be monotonically increasing: "
            f"{version} < {self._current_weight_version}"
        )
        self._current_weight_version = version
        logger.debug(f"Weight version updated to {version}")

    def pause_new_admissions(self, reason: str = "manual") -> None:
        """Stop admitting new prompt groups while allowing in-flight work to drain."""
        if self._admission_paused_reason is None:
            self._admission_pause_events += 1
        self._admission_paused_reason = reason

    def resume_new_admissions(self) -> None:
        """Resume admitting new prompt groups after a drain/sync boundary."""
        self._admission_paused_reason = None

    @property
    def admission_paused(self) -> bool:
        """True when the manager is intentionally not admitting new work."""
        return self._admission_paused_reason is not None

    async def start_sampling(
        self,
        nursery: trio.Nursery,
        initial_weight_version: int = 0,
    ) -> None:
        """Start background sampling task.

        Args:
            nursery: Trio nursery to spawn sampling task in
            initial_weight_version: Starting weight version
        """
        self._current_weight_version = initial_weight_version
        self._nursery = nursery
        self._shutdown_requested = False

        # Start sampling task
        nursery.start_soon(self._sampling_loop)
        logger.info(
            "Started pipelined sampling (max_lag=%s, queue_size=%s)",
            self.max_lag,
            self.queue_size,
        )

    async def stop_sampling(self) -> None:
        """Stop background sampling task."""
        self._shutdown_requested = True

        # Log stats
        if self._samples_generated > 0:
            discard_rate = self._samples_discarded_stale / self._samples_generated * 100
            logger.info(
                f"Pipelined sampling stopped: "
                f"generated={self._samples_generated}, "
                f"discarded_stale={self._samples_discarded_stale} ({discard_rate:.1f}%)"
            )

    async def get_batch(
        self,
        current_weight_version: int,
        scorer: Scorer | None = None,
        timeout: float = 60.0,
    ) -> RolloutBatch:
        """Get a batch of samples, filtering stale ones.

        Blocks until enough fresh samples are available.

        Args:
            current_weight_version: Current training weight version
            scorer: Explicit scoring stage over raw attempt results.
            timeout: Max seconds to wait for batch

        Returns:
            RolloutBatch ready for training

        Raises:
            trio.TooSlowError: If batch not ready within timeout
        """
        target_group_count = self.config.batch_size
        refill_rounds = 0

        with trio.fail_after(timeout):
            while True:
                # Try to collect enough fresh samples
                async with self._queue_lock:
                    fresh_samples = []
                    remaining_queue = []
                    stale_group_counts: dict[int, int] = {}

                    for sample in self._sample_queue:
                        lag = current_weight_version - sample.weight_version
                        if lag <= self.max_lag:
                            fresh_samples.append(sample)
                        else:
                            # Discard stale sample
                            self._samples_discarded_stale += 1
                            if sample.group_index is not None:
                                stale_group_counts[sample.group_index] = (
                                    stale_group_counts.get(sample.group_index, 0) + 1
                                )
                            logger.debug(
                                f"Discarding stale sample (lag={lag} > max_lag={self.max_lag})"
                            )

                    if (
                        count_complete_groups(
                            fresh_samples,
                            self.config.n_samples_per_prompt,
                        )
                        >= target_group_count
                    ):
                        assembly_result = assemble_groups(
                            fresh_samples,
                            target_num_groups=target_group_count,
                            samples_per_group=self.config.n_samples_per_prompt,
                            incomplete_group_policy=self.config.incomplete_group_policy,
                        )
                        collected = assembly_result.ready_samples
                        remaining_queue = assembly_result.overflow_samples
                        self._sample_queue = remaining_queue
                    else:
                        incomplete_groups = collect_incomplete_groups(
                            fresh_samples,
                            self.config.n_samples_per_prompt,
                        )
                        # Not enough fresh samples, keep waiting
                        # Keep only admissible samples in queue. Incomplete groups
                        # are still kept until there are enough complete groups to
                        # assemble a batch; a future refill policy may choose to
                        # drop or regenerate them earlier.
                        self._sample_queue = fresh_samples
                        collected = None
                        assembly_result = None
                        should_refill = (
                            self.config.incomplete_group_policy
                            == IncompleteGroupPolicy.REQUEST_MORE
                            and bool(incomplete_groups or stale_group_counts)
                            and refill_rounds < self.config.max_refill_rounds
                        )
                        if should_refill:
                            refill_requests = []
                            group_sample_counts = {
                                group_idx: len(group_samples)
                                for group_idx, group_samples in incomplete_groups.items()
                            }
                            refill_group_ids = set(incomplete_groups) | set(stale_group_counts)
                            for group_idx in sorted(refill_group_ids):
                                prompt = self._group_prompts.get(group_idx)
                                if prompt is None:
                                    logger.debug(
                                        "Skipping refill for group %s because prompt identity is unknown",
                                        group_idx,
                                    )
                                    continue
                                missing_count = (
                                    self.config.n_samples_per_prompt
                                    - group_sample_counts.get(group_idx, 0)
                                )
                                for _ in range(missing_count):
                                    refill_requests.append((prompt, group_idx))
                            self._refill_requests += len(refill_requests)
                        else:
                            refill_requests = []

                if collected is not None:
                    break

                if refill_requests:
                    refill_rounds += 1
                    refill_samples = await self._refill_groups(
                        refill_requests,
                        weight_version=current_weight_version,
                    )
                    async with self._queue_lock:
                        self._sample_queue.extend(refill_samples)
                        self._samples_generated += len(refill_samples)
                    continue

                # Wait a bit for more samples
                await trio.sleep(0.1)

        # Score samples via the explicit scoring stage if configured.
        scorer = resolve_scorer(
            config=self.config,
            runtime=self.runtime,
            scorer=scorer,
        )
        if scorer is not None:
            await score_rows(scorer, collected)

        # Convert to batch
        batch = convert_to_batch(
            collected,
            epoch_id=self.data_buffer.epoch_id,
            step_id=self._step_count,
        )
        self._prune_group_prompts(remaining_queue, collected)
        assert assembly_result is not None, "assembly_result must exist once a batch is collected"
        batch.metadata.update({
            "assembled_groups": assembly_result.ready_group_count,
            "buffered_samples": len(assembly_result.overflow_samples),
            "dropped_incomplete_groups": len(assembly_result.dropped_incomplete_group_sizes),
            "dropped_incomplete_samples": len(assembly_result.dropped_incomplete_samples),
            "incomplete_group_policy": self.config.incomplete_group_policy.value,
            "refill_rounds": refill_rounds,
            "rollout_stats": self.stats(),
        })

        self._step_count += 1
        return batch

    async def _sampling_loop(self) -> None:
        """Background sampling loop.

        Continuously generates samples and adds them to queue.
        Samples are tagged with current weight_version.
        """
        logger.debug("Sampling loop started")

        while not self._shutdown_requested:
            # Soft pause means: do not admit new prompts, let the current generation
            # round finish naturally, then hold here until the sync boundary clears.
            if self.admission_paused:
                self._sync_pause_count += 1
                await trio.sleep(0.1)
                continue
            if self.sync_in_progress_fn is not None and self.sync_in_progress_fn():
                self._sync_pause_count += 1
                logger.debug("External sync gate active, pausing new admissions...")
                await trio.sleep(0.1)
                continue

            # Check if queue has room
            async with self._queue_lock:
                queue_len = len(self._sample_queue)
                if queue_len > self._max_queue_length_seen:
                    self._max_queue_length_seen = queue_len

            if self.queue_size > 0 and queue_len >= self.queue_size:
                # Queue full, wait a bit
                await trio.sleep(0.1)
                continue

            # Generate some samples
            try:
                # Get prompts
                num_prompts = self.config.batch_size
                prompts = self.data_buffer.get_prompts(num_prompts)

                # Capture current weight version before generation
                generation_weight_version = self._current_weight_version

                # Generate samples
                samples = await self._generate_samples_parallel(
                    prompts,
                    weight_version=generation_weight_version,
                )

                # Add to queue
                async with self._queue_lock:
                    self._sample_queue.extend(samples)
                    self._samples_generated += len(samples)
                    queue_len = len(self._sample_queue)
                    if queue_len > self._max_queue_length_seen:
                        self._max_queue_length_seen = queue_len

                logger.debug(
                    f"Generated {len(samples)} samples (v={generation_weight_version}), "
                    f"queue={queue_len}"
                )

            except Exception as e:
                logger.warning(f"Sampling error: {e}")
                await trio.sleep(1.0)  # Back off on error

        logger.debug("Sampling loop stopped")

    async def _generate_samples_parallel(
        self,
        prompts: list[str | dict[str, Any]],
        weight_version: int,
    ) -> list[AttemptRow]:
        """Generate samples for prompts in parallel.

        Tags all samples with weight_version.
        """

        async def generate_for_prompt(
            prompt: str | dict[str, Any],
            group_idx: int,
        ) -> list[AttemptRow]:
            """Generate samples for a single prompt with group index."""
            samples = await self._call_user_generate_fn([prompt])
            for sample in samples:
                sample.group_index = group_idx
                sample.weight_version = weight_version
            return samples

        group_indices = list(range(self._next_group_index, self._next_group_index + len(prompts)))
        self._next_group_index += len(prompts)
        for group_idx, prompt in zip(group_indices, prompts, strict=False):
            self._group_prompts[group_idx] = prompt

        # Launch all tasks in parallel
        async with trio.open_nursery() as nursery:
            results: list[AttemptRow] = []
            results_lock = trio.Lock()

            async def run_task(prompt: str | dict[str, Any], group_idx: int) -> None:
                samples = await generate_for_prompt(prompt, group_idx)
                async with results_lock:
                    results.extend(samples)

            for prompt_idx, prompt in enumerate(prompts):
                for _ in range(self.config.n_samples_per_prompt):
                    nursery.start_soon(run_task, prompt, group_indices[prompt_idx])

        return results

    async def _refill_groups(
        self,
        refill_requests: list[tuple[str | dict[str, Any], int]],
        weight_version: int,
    ) -> list[AttemptRow]:
        """Generate additional samples for existing group ids."""
        async with trio.open_nursery() as nursery:
            results: list[AttemptRow] = []
            results_lock = trio.Lock()

            async def run_task(prompt: str | dict[str, Any], group_idx: int) -> None:
                samples = await self._call_user_generate_fn([prompt])
                for sample in samples:
                    sample.group_index = group_idx
                    sample.weight_version = weight_version
                async with results_lock:
                    results.extend(samples)

            for prompt, group_idx in refill_requests:
                nursery.start_soon(run_task, prompt, group_idx)

        return results

    async def _call_user_generate_fn(
        self,
        prompts: list[str | dict[str, Any]],
    ) -> list[AttemptRow]:
        """Call user-provided generate function (async or sync)."""
        import inspect

        assert self.runtime is not None, "runtime must be provided"
        generate_fn = self.runtime.generate_fn

        if inspect.iscoroutinefunction(generate_fn):
            samples = await generate_fn(prompts, **self.rollout_kwargs)
        else:
            samples = await trio.to_thread.run_sync(
                lambda: generate_fn(prompts, **self.rollout_kwargs)
            )

        assert isinstance(samples, list)
        assert all(isinstance(s, AttemptRow) for s in samples)

        return samples

    def _prune_group_prompts(
        self, active_samples: list[AttemptRow], consumed_samples: list[AttemptRow]
    ) -> None:
        """Forget prompt identities for groups that have fully left the pipeline."""
        active_group_ids = {
            sample.group_index for sample in active_samples if sample.group_index is not None
        }
        consumed_group_ids = {
            sample.group_index for sample in consumed_samples if sample.group_index is not None
        }
        for group_idx in consumed_group_ids:
            if group_idx not in active_group_ids:
                self._group_prompts.pop(group_idx, None)

    def stats(self) -> dict[str, Any]:
        """Get sampling statistics."""
        return {
            "mode": "pipelined_rollout_manager",
            "samples_generated": self._samples_generated,
            "samples_discarded_stale": self._samples_discarded_stale,
            "refill_requests": self._refill_requests,
            "queue_length": len(self._sample_queue),
            "queue_limit": self.queue_size,
            "queue_limit_enabled": self.queue_size > 0,
            "max_queue_length_seen": self._max_queue_length_seen,
            "current_weight_version": self._current_weight_version,
            "sync_pause_count": self._sync_pause_count,
            "admission_paused": self.admission_paused,
            "admission_pause_events": self._admission_pause_events,
            "known_group_prompts": len(self._group_prompts),
            "discard_rate": (
                self._samples_discarded_stale / self._samples_generated * 100
                if self._samples_generated > 0
                else 0.0
            ),
        }
