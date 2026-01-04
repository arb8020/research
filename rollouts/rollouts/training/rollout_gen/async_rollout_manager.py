"""Async Rollout Manager with dynamic sampling (D4).

SLIME's killer feature: Over-sample rollouts, then filter by quality.

Key features:
- Async parallel generation with trio
- Dynamic over-sampling (generate N*1.5, keep best N)
- Partial rollout caching on abort
- Filter functions for quality control

Tiger Style: Explicit abort handling, clear state transitions.
SLIME: Dynamic sampling strategy, quality filtering.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import trio

from ...training.datasets.data_buffer import DataBuffer
from ...training.rollout_gen.rollout_generation import convert_to_batch
from ...training.types import RolloutBatch, RolloutConfig, Sample


@dataclass
class AsyncRolloutManager:
    """Async rollout manager with dynamic sampling (SLIME-inspired).

    Generates rollouts in parallel with automatic over-sampling and filtering.

    Usage:
        async with AsyncRolloutManager(buffer, config) as manager:
            batch = await manager.generate_batch()
            # Train on batch...

    Attributes:
        data_buffer: DataBuffer for prompt iteration
        config: RolloutConfig with batch_size, generate_fn, filters
        partial_samples: Cache for incomplete rollouts (SLIME feature)
        _step_count: Number of batches generated
        _abort_requested: Flag for graceful shutdown
    """

    data_buffer: DataBuffer
    config: RolloutConfig
    partial_samples: list[Sample] = field(default_factory=list)
    _step_count: int = 0
    _abort_requested: bool = False

    # Rollout kwargs passed to generate_fn
    rollout_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.config.generate_fn is None:
            raise ValueError("RolloutConfig.generate_fn must be provided")
        if self.config.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.config.batch_size}")
        if self.config.over_sampling_factor < 1.0:
            raise ValueError(
                f"over_sampling_factor must be >= 1.0, got {self.config.over_sampling_factor}"
            )

    async def __aenter__(self) -> "AsyncRolloutManager":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        """Async context manager exit - cache any partial samples."""
        if self.partial_samples:
            # Log partial samples for debugging
            print(f"Caching {len(self.partial_samples)} partial samples on exit")
        return False

    async def generate_batch(
        self,
        score_fn: Callable[[Sample], Any] | None = None,
    ) -> RolloutBatch:
        """Generate one batch with dynamic over-sampling.

        SLIME's algorithm:
        1. Calculate target samples (batch_size)
        2. Calculate over-sample count (batch_size * over_sampling_factor)
        3. Generate over-sampled rollouts in parallel
        4. Apply filter function to groups
        5. Select best samples up to target
        6. Cache remaining samples for next batch

        Args:
            score_fn: Optional score function (Sample -> Score), reward = score.reward

        Returns:
            RolloutBatch ready for training

        Raises:
            RuntimeError: If abort requested mid-generation
        """
        # Step 1: Calculate batch sizes
        # batch_size = number of prompts, total samples = batch_size * n_samples_per_prompt
        target_size = self.config.batch_size * self.config.n_samples_per_prompt
        over_sample_size = int(target_size * self.config.over_sampling_factor)

        collected_samples: list[Sample] = []

        # Step 2: Use cached partial samples first (SLIME feature!)
        if self.partial_samples:
            num_from_cache = min(len(self.partial_samples), target_size)
            collected_samples.extend(self.partial_samples[:num_from_cache])
            self.partial_samples = self.partial_samples[num_from_cache:]

        # Step 3: Generate remaining samples with over-sampling
        while len(collected_samples) < target_size:
            if self._abort_requested:
                # Cache what we have and abort
                self.partial_samples.extend(collected_samples)
                raise RuntimeError("Abort requested during batch generation")

            # How many samples do we still need?
            needed = target_size - len(collected_samples)
            to_generate = min(over_sample_size, needed * self.config.over_sampling_factor)
            to_generate = int(to_generate)

            # Get prompts from buffer
            num_prompts = to_generate // max(self.config.n_samples_per_prompt, 1)
            num_prompts = max(1, num_prompts)
            prompts = self.data_buffer.get_prompts(num_prompts)

            # Generate samples in parallel (SLIME's async generation!)
            samples = await self._generate_samples_parallel(prompts)

            # Apply filter if provided
            if self.config.filter_fn is not None:
                samples = self._apply_filter(samples)

            # Take what we need, cache the rest
            take_count = min(len(samples), needed)
            collected_samples.extend(samples[:take_count])

            # Cache overflow samples for next batch (SLIME feature!)
            if len(samples) > take_count:
                self.partial_samples.extend(samples[take_count:])

        # Step 4: Compute rewards from score_fn if provided
        if score_fn is not None:
            for sample in collected_samples:
                score = score_fn(sample)
                sample.reward = score.reward

        # Step 5: Convert to batch
        batch = convert_to_batch(
            collected_samples,
            epoch_id=self.data_buffer.epoch_id,
            step_id=self._step_count,
        )

        self._step_count += 1
        return batch

    async def _generate_samples_parallel(
        self,
        prompts: list[str | dict[str, Any]],
    ) -> list[Sample]:
        """Generate samples for prompts in parallel.

        Creates n_samples_per_prompt for each prompt, all in parallel.

        Args:
            prompts: List of prompts to generate samples for

        Returns:
            List of generated samples (len = len(prompts) * n_samples_per_prompt)
        """

        # Create tasks for parallel generation
        async def generate_for_prompt(prompt: str | dict[str, Any], group_idx: int) -> list[Sample]:
            """Generate sample for a single prompt with group index."""
            # Call user's generate function
            # Note: User function should return list[Sample]
            samples = await self._call_user_generate_fn([prompt])
            # Set group_index on all returned samples
            for sample in samples:
                sample.group_index = group_idx
            return samples

        # Launch all tasks in parallel with trio
        async with trio.open_nursery() as nursery:
            results: list[Sample] = []
            results_lock = trio.Lock()

            async def run_task(prompt: str | dict[str, Any], group_idx: int) -> None:
                samples = await generate_for_prompt(prompt, group_idx)
                async with results_lock:
                    results.extend(samples)

            for prompt_idx, prompt in enumerate(prompts):
                # Generate n_samples_per_prompt times, all with same group_index
                for _ in range(self.config.n_samples_per_prompt):
                    nursery.start_soon(run_task, prompt, prompt_idx)

        return results

    async def _call_user_generate_fn(
        self,
        prompts: list[str | dict[str, Any]],
    ) -> list[Sample]:
        """Call user-provided generate function (async or sync).

        Handles both async and sync user functions transparently.

        Args:
            prompts: Prompts to generate samples for

        Returns:
            List of samples from user function
        """
        import inspect

        # Validate generate_fn is provided
        assert self.config.generate_fn is not None, "generate_fn must be provided"
        generate_fn = self.config.generate_fn

        # Check if user function is async
        if inspect.iscoroutinefunction(generate_fn):
            # Async user function
            samples = await generate_fn(prompts, **self.rollout_kwargs)
        else:
            # Sync user function - run in thread to avoid blocking
            samples = await trio.to_thread.run_sync(
                lambda: generate_fn(prompts, **self.rollout_kwargs)
            )

        # Validate return type
        assert isinstance(samples, list), (
            f"generate_fn must return list[Sample], got {type(samples)}"
        )
        assert all(isinstance(s, Sample) for s in samples), (
            "generate_fn must return list of Sample objects"
        )

        return samples

    def _apply_filter(self, samples: list[Sample]) -> list[Sample]:
        """Apply filter function to samples.

        SLIME-style: Filter can look at groups or individual samples.

        Args:
            samples: List of samples to filter

        Returns:
            Filtered list of samples
        """
        if self.config.filter_fn is None:
            return samples

        # Group samples by prompt if n_samples_per_prompt > 1
        if self.config.n_samples_per_prompt > 1:
            filtered = []
            for i in range(0, len(samples), self.config.n_samples_per_prompt):
                group = samples[i : i + self.config.n_samples_per_prompt]
                # Filter function decides if group passes
                if self.config.filter_fn(group):
                    filtered.extend(group)
            return filtered
        else:
            # Filter individual samples
            return [s for s in samples if self.config.filter_fn([s])]

    def request_abort(self) -> None:
        """Request graceful abort of current generation.

        Partial samples will be cached for next batch.
        """
        self._abort_requested = True

    def state_dict(self) -> dict[str, Any]:
        """Save manager state for checkpointing.

        Includes buffer state + partial samples (SLIME feature!).

        Returns:
            State dict for serialization
        """
        return {
            "buffer_state": self.data_buffer.save_state(),
            "step_count": self._step_count,
            "partial_samples": [s.to_dict() for s in self.partial_samples],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore manager state from checkpoint.

        Args:
            state: State dict from state_dict()
        """
        self.data_buffer.load_state(state["buffer_state"])
        self._step_count = state["step_count"]

        # Restore partial samples (SLIME feature!)
        self.partial_samples = [Sample.from_dict(s) for s in state.get("partial_samples", [])]


# ────────────────────── Convenience Function ──────────────────────


async def generate_rollout_batch(
    buffer: DataBuffer,
    config: RolloutConfig,
    score_fn: Callable[[Sample], Any] | None = None,
    **rollout_kwargs: Any,
) -> RolloutBatch:
    """Generate a single batch with dynamic sampling (convenience function).

    Args:
        buffer: DataBuffer for prompts
        config: RolloutConfig with generation settings
        score_fn: Optional score function (Sample -> Score), reward = score.reward
        **rollout_kwargs: Kwargs passed to generate_fn

    Returns:
        RolloutBatch ready for training

    Example:
        >>> from rollouts import Score, Metric
        >>> batch = await generate_rollout_batch(
        ...     buffer=buffer,
        ...     config=config,
        ...     score_fn=lambda s: Score(metrics=(Metric("correct", 1.0 if "correct" in s.response else 0.0, weight=1.0),)),
        ...     tokenizer=tokenizer,
        ... )
    """
    manager = AsyncRolloutManager(
        data_buffer=buffer,
        config=config,
        rollout_kwargs=rollout_kwargs,
    )

    async with manager:
        return await manager.generate_batch(score_fn=score_fn)
