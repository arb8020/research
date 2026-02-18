"""Pipelined Rollout Manager (PipelineRL-style in-flight updates).

Runs sampling in background while training proceeds. Key features:
- Background sampling task that continuously generates rollouts
- Weight version tracking (samples know which model version generated them)
- max_lag parameter to discard stale samples
- Non-blocking batch retrieval

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
from ...training.rollout_gen.rollout_generation import convert_to_batch
from ...training.types import RolloutBatch, RolloutConfig, Sample

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
        queue_size: Maximum number of samples to buffer in queue
        current_weight_version: Current model weight version (updated by trainer)
    """

    data_buffer: DataBuffer
    config: RolloutConfig
    max_lag: int = 2  # Allow samples up to 2 versions behind
    queue_size: int = 1024  # Buffer up to 1024 samples

    # Internal state
    _sample_queue: list[Sample] = field(default_factory=list)
    _queue_lock: trio.Lock = field(default_factory=trio.Lock)
    _current_weight_version: int = 0
    _sampling_task_scope: trio.CancelScope | None = None
    _nursery: trio.Nursery | None = None
    _step_count: int = 0
    _shutdown_requested: bool = False

    # Stats
    _samples_generated: int = 0
    _samples_discarded_stale: int = 0

    # Rollout kwargs passed to generate_fn
    rollout_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.config.generate_fn is None:
            raise ValueError("RolloutConfig.generate_fn must be provided")
        if self.config.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.config.batch_size}")

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
            f"Started pipelined sampling (max_lag={self.max_lag}, "
            f"queue_size={self.queue_size})"
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
        score_fn: Callable[[Sample], Any] | None = None,
        timeout: float = 60.0,
    ) -> RolloutBatch:
        """Get a batch of samples, filtering stale ones.

        Blocks until enough fresh samples are available.

        Args:
            current_weight_version: Current training weight version
            score_fn: Optional score function (Sample -> Score)
            timeout: Max seconds to wait for batch

        Returns:
            RolloutBatch ready for training

        Raises:
            trio.TooSlowError: If batch not ready within timeout
        """
        import inspect

        target_size = self.config.batch_size * self.config.n_samples_per_prompt

        with trio.fail_after(timeout):
            while True:
                # Try to collect enough fresh samples
                async with self._queue_lock:
                    fresh_samples = []
                    remaining_queue = []

                    for sample in self._sample_queue:
                        lag = current_weight_version - sample.weight_version
                        if lag <= self.max_lag:
                            fresh_samples.append(sample)
                        else:
                            # Discard stale sample
                            self._samples_discarded_stale += 1
                            logger.debug(
                                f"Discarding stale sample (lag={lag} > max_lag={self.max_lag})"
                            )

                    if len(fresh_samples) >= target_size:
                        # Take what we need, keep the rest
                        collected = fresh_samples[:target_size]
                        remaining_queue = fresh_samples[target_size:]
                        self._sample_queue = remaining_queue
                    else:
                        # Not enough fresh samples, keep waiting
                        # Keep fresh samples in queue
                        self._sample_queue = fresh_samples
                        collected = None

                if collected is not None:
                    break

                # Wait a bit for more samples
                await trio.sleep(0.1)

        # Compute rewards from score_fn if provided
        if score_fn is not None:
            is_async = inspect.iscoroutinefunction(score_fn)
            for sample in collected:
                if is_async:
                    score = await score_fn(sample)
                else:
                    score = score_fn(sample)
                sample.reward = score.reward

        # Convert to batch
        batch = convert_to_batch(
            collected,
            epoch_id=self.data_buffer.epoch_id,
            step_id=self._step_count,
        )

        self._step_count += 1
        return batch

    async def _sampling_loop(self) -> None:
        """Background sampling loop.

        Continuously generates samples and adds them to queue.
        Samples are tagged with current weight_version.
        """
        logger.debug("Sampling loop started")

        while not self._shutdown_requested:
            # Check if queue has room
            async with self._queue_lock:
                queue_len = len(self._sample_queue)

            if queue_len >= self.queue_size:
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

                logger.debug(
                    f"Generated {len(samples)} samples (v={generation_weight_version}), "
                    f"queue={len(self._sample_queue)}"
                )

            except Exception as e:
                logger.warning(f"Sampling error: {e}")
                await trio.sleep(1.0)  # Back off on error

        logger.debug("Sampling loop stopped")

    async def _generate_samples_parallel(
        self,
        prompts: list[str | dict[str, Any]],
        weight_version: int,
    ) -> list[Sample]:
        """Generate samples for prompts in parallel.

        Tags all samples with weight_version.
        """
        async def generate_for_prompt(
            prompt: str | dict[str, Any],
            group_idx: int,
        ) -> list[Sample]:
            """Generate samples for a single prompt with group index."""
            samples = await self._call_user_generate_fn([prompt])
            for sample in samples:
                sample.group_index = group_idx
                sample.weight_version = weight_version
            return samples

        # Launch all tasks in parallel
        async with trio.open_nursery() as nursery:
            results: list[Sample] = []
            results_lock = trio.Lock()

            async def run_task(prompt: str | dict[str, Any], group_idx: int) -> None:
                samples = await generate_for_prompt(prompt, group_idx)
                async with results_lock:
                    results.extend(samples)

            for prompt_idx, prompt in enumerate(prompts):
                for _ in range(self.config.n_samples_per_prompt):
                    nursery.start_soon(run_task, prompt, prompt_idx)

        return results

    async def _call_user_generate_fn(
        self,
        prompts: list[str | dict[str, Any]],
    ) -> list[Sample]:
        """Call user-provided generate function (async or sync)."""
        import inspect

        assert self.config.generate_fn is not None
        generate_fn = self.config.generate_fn

        if inspect.iscoroutinefunction(generate_fn):
            samples = await generate_fn(prompts, **self.rollout_kwargs)
        else:
            samples = await trio.to_thread.run_sync(
                lambda: generate_fn(prompts, **self.rollout_kwargs)
            )

        assert isinstance(samples, list)
        assert all(isinstance(s, Sample) for s in samples)

        return samples

    def stats(self) -> dict[str, Any]:
        """Get sampling statistics."""
        return {
            "samples_generated": self._samples_generated,
            "samples_discarded_stale": self._samples_discarded_stale,
            "queue_length": len(self._sample_queue),
            "current_weight_version": self._current_weight_version,
            "discard_rate": (
                self._samples_discarded_stale / self._samples_generated * 100
                if self._samples_generated > 0 else 0.0
            ),
        }
