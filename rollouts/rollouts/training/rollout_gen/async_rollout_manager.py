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

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import trio

logger = logging.getLogger(__name__)

from ...training.datasets.data_buffer import DataBuffer
from ...training.group_assembly import (
    assemble_groups,
    collect_incomplete_groups,
    count_complete_groups,
)
from ...training.rollout_gen.rollout_generation import convert_to_batch
from ...training.runtime import resolve_rollout_runtime
from ...training.scoring import resolve_sample_scorer
from ...training.types import (
    AttemptRow,
    IncompleteGroupPolicy,
    RolloutBatch,
    RolloutConfig,
    RolloutRuntime,
    SampleScorer,
)


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
        buffered_samples: Complete overflow groups buffered for reuse
        _step_count: Number of batches generated
        _abort_requested: Flag for graceful shutdown
    """

    data_buffer: DataBuffer
    config: RolloutConfig
    runtime: RolloutRuntime | None = None
    buffered_samples: list[AttemptRow] = field(default_factory=list)
    _step_count: int = 0
    _abort_requested: bool = False
    _next_group_index: int = 0
    _samples_generated: int = 0
    _refill_requests: int = 0
    _max_buffered_samples: int = 0

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
        if self.config.over_sampling_factor < 1.0:
            raise ValueError(
                f"over_sampling_factor must be >= 1.0, got {self.config.over_sampling_factor}"
            )
        if self.config.max_refill_rounds < 0:
            raise ValueError(f"max_refill_rounds must be >= 0, got {self.config.max_refill_rounds}")

    async def __aenter__(self) -> "AsyncRolloutManager":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        """Async context manager exit - retain any buffered overflow groups."""
        if self.buffered_samples:
            logger.info("Buffering %s samples on exit", len(self.buffered_samples))
        return False

    async def generate_batch(
        self,
        sample_scorer: SampleScorer | None = None,
        score_fn: Callable[[AttemptRow], Any] | None = None,
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
            sample_scorer: Explicit scoring stage owned by the buffer/data side.
            score_fn: Legacy score function (Sample -> Score), adapted into a
                sample_scorer when provided.

        Returns:
            RolloutBatch ready for training

        Raises:
            RuntimeError: If abort requested mid-generation
        """
        # Step 1: Calculate batch sizes
        # batch_size = number of prompts, total samples = batch_size * n_samples_per_prompt
        target_size = self.config.batch_size * self.config.n_samples_per_prompt
        over_sample_size = int(target_size * self.config.over_sampling_factor)

        target_group_count = self.config.batch_size
        working_samples = list(self.buffered_samples)
        self.buffered_samples = []
        refill_rounds = 0
        self._max_buffered_samples = max(self._max_buffered_samples, len(working_samples))

        # Step 3: Generate remaining samples with over-sampling
        while (
            count_complete_groups(working_samples, self.config.n_samples_per_prompt)
            < target_group_count
        ):
            if self._abort_requested:
                # Keep complete overflow groups across shutdown. Incomplete groups
                # are not reusable without an explicit refill path.
                self.buffered_samples = assemble_groups(
                    working_samples,
                    target_num_groups=0,
                    samples_per_group=self.config.n_samples_per_prompt,
                    incomplete_group_policy=self.config.incomplete_group_policy,
                ).overflow_samples
                raise RuntimeError("Abort requested during batch generation")

            complete_group_count = count_complete_groups(
                working_samples,
                self.config.n_samples_per_prompt,
            )
            incomplete_groups = collect_incomplete_groups(
                working_samples,
                self.config.n_samples_per_prompt,
            )
            if (
                self.config.incomplete_group_policy == IncompleteGroupPolicy.REQUEST_MORE
                and incomplete_groups
                and refill_rounds < self.config.max_refill_rounds
            ):
                refill_rounds += 1
                self._refill_requests += sum(
                    self.config.n_samples_per_prompt - len(group_samples)
                    for group_samples in incomplete_groups.values()
                )
                refill_samples = await self._refill_incomplete_groups(incomplete_groups)
                working_samples.extend(refill_samples)
                self._samples_generated += len(refill_samples)
                continue

            missing_groups = target_group_count - complete_group_count
            missing_samples = missing_groups * self.config.n_samples_per_prompt
            to_generate = min(over_sample_size, missing_samples * self.config.over_sampling_factor)
            to_generate = int(to_generate)

            # Get prompts from buffer
            num_prompts = to_generate // max(self.config.n_samples_per_prompt, 1)
            num_prompts = max(1, num_prompts)
            prompts = self.data_buffer.get_prompts(num_prompts)

            # Generate samples in parallel (SLIME's async generation!)
            samples = await self._generate_samples_parallel(prompts)
            self._samples_generated += len(samples)

            # Apply filter if provided
            assert self.runtime is not None, "runtime must be provided"
            if self.runtime.filter_fn is not None:
                samples = self._apply_filter(samples)

            working_samples.extend(samples)

        assembly_result = assemble_groups(
            working_samples,
            target_num_groups=target_group_count,
            samples_per_group=self.config.n_samples_per_prompt,
            incomplete_group_policy=self.config.incomplete_group_policy,
        )
        collected_samples = assembly_result.ready_samples
        self.buffered_samples = assembly_result.overflow_samples
        self._max_buffered_samples = max(self._max_buffered_samples, len(self.buffered_samples))
        dropped_incomplete_groups = len(assembly_result.dropped_incomplete_group_sizes)

        if dropped_incomplete_groups > 0:
            logger.info(
                "Dropped %s incomplete groups during assembly",
                dropped_incomplete_groups,
            )

        # Step 4: Score samples via the explicit scoring stage if configured.
        scorer = resolve_sample_scorer(
            config=self.config,
            runtime=self.runtime,
            sample_scorer=sample_scorer,
            score_fn=score_fn,
        )
        if scorer is not None:
            await scorer.score_samples(collected_samples)

        # Step 5: Convert the explicitly assembled groups into a training batch.
        batch = convert_to_batch(
            collected_samples,
            epoch_id=self.data_buffer.epoch_id,
            step_id=self._step_count,
        )
        batch.metadata.update({
            "assembled_groups": assembly_result.ready_group_count,
            "buffered_samples": len(self.buffered_samples),
            "dropped_incomplete_groups": dropped_incomplete_groups,
            "dropped_incomplete_samples": len(assembly_result.dropped_incomplete_samples),
            "incomplete_group_policy": self.config.incomplete_group_policy.value,
            "refill_rounds": refill_rounds,
            "rollout_stats": self.stats(),
        })

        self._step_count += 1
        return batch

    async def _generate_samples_parallel(
        self,
        prompts: list[str | dict[str, Any]],
    ) -> list[AttemptRow]:
        """Generate samples for prompts in parallel.

        Creates n_samples_per_prompt for each prompt, all in parallel.

        Args:
            prompts: List of prompts to generate samples for

        Returns:
            List of generated samples (len = len(prompts) * n_samples_per_prompt)
        """

        group_indices = list(range(self._next_group_index, self._next_group_index + len(prompts)))
        self._next_group_index += len(prompts)

        # Create tasks for parallel generation
        async def generate_for_prompt(
            prompt: str | dict[str, Any], group_idx: int
        ) -> list[AttemptRow]:
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
            results: list[AttemptRow] = []
            results_lock = trio.Lock()

            async def run_task(prompt: str | dict[str, Any], group_idx: int) -> None:
                samples = await generate_for_prompt(prompt, group_idx)
                async with results_lock:
                    results.extend(samples)

            for prompt_idx, prompt in enumerate(prompts):
                # Generate n_samples_per_prompt times, all with same group_index
                for _ in range(self.config.n_samples_per_prompt):
                    nursery.start_soon(run_task, prompt, group_indices[prompt_idx])

        return results

    async def _refill_incomplete_groups(
        self,
        incomplete_groups: dict[int, list[AttemptRow]],
    ) -> list[AttemptRow]:
        """Request additional samples for the same prompts to complete groups."""
        refill_requests: list[tuple[str | dict[str, Any], int]] = []
        for group_idx, group_samples in incomplete_groups.items():
            prompt = group_samples[0].prompt
            missing_count = self.config.n_samples_per_prompt - len(group_samples)
            for _ in range(missing_count):
                refill_requests.append((prompt, group_idx))

        if not refill_requests:
            return []

        async with trio.open_nursery() as nursery:
            refill_results: list[AttemptRow] = []
            results_lock = trio.Lock()

            async def run_task(prompt: str | dict[str, Any], group_idx: int) -> None:
                samples = await self._call_user_generate_fn([prompt])
                for sample in samples:
                    sample.group_index = group_idx
                async with results_lock:
                    refill_results.extend(samples)

            for prompt, group_idx in refill_requests:
                nursery.start_soon(run_task, prompt, group_idx)

        return refill_results

    async def _call_user_generate_fn(
        self,
        prompts: list[str | dict[str, Any]],
    ) -> list[AttemptRow]:
        """Call user-provided generate function (async or sync).

        Handles both async and sync user functions transparently.

        Args:
            prompts: Prompts to generate samples for

        Returns:
            List of samples from user function
        """
        import inspect

        # Validate generate_fn is provided
        assert self.runtime is not None, "runtime must be provided"
        generate_fn = self.runtime.generate_fn

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
            f"generate_fn must return list[AttemptRow], got {type(samples)}"
        )
        assert all(isinstance(s, AttemptRow) for s in samples), (
            "generate_fn must return list of AttemptRow objects"
        )

        return samples

    def _apply_filter(self, samples: list[AttemptRow]) -> list[AttemptRow]:
        """Apply filter function to samples.

        SLIME-style: Filter can look at groups or individual samples.

        Args:
            samples: List of samples to filter

        Returns:
            Filtered list of samples
        """
        assert self.runtime is not None, "runtime must be provided"
        if self.runtime.filter_fn is None:
            return samples

        # Group samples by explicit group identity instead of list stride because
        # parallel generation can complete out of order.
        if self.config.n_samples_per_prompt > 1:
            filtered = []
            grouped_samples: dict[int, list[AttemptRow]] = {}
            for sample in samples:
                assert sample.group_index is not None, "group_index required for grouped filtering"
                if sample.group_index not in grouped_samples:
                    grouped_samples[sample.group_index] = []
                grouped_samples[sample.group_index].append(sample)

            for group in grouped_samples.values():
                # Filter function decides if group passes
                if self.runtime.filter_fn(group):
                    filtered.extend(group)
            return filtered
        else:
            # Filter individual samples
            return [s for s in samples if self.runtime.filter_fn([s])]

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
            "buffered_samples": [s.to_dict() for s in self.buffered_samples],
            "next_group_index": self._next_group_index,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore manager state from checkpoint.

        Args:
            state: State dict from state_dict()
        """
        self.data_buffer.load_state(state["buffer_state"])
        self._step_count = state["step_count"]
        self._next_group_index = state.get("next_group_index", 0)
        buffered_samples = state.get("buffered_samples", state.get("partial_samples", []))
        self.buffered_samples = [AttemptRow.from_dict(s) for s in buffered_samples]

    def stats(self) -> dict[str, Any]:
        """Return explicit rollout-generation statistics."""
        return {
            "mode": "sync_async_manager",
            "step_count": self._step_count,
            "samples_generated": self._samples_generated,
            "refill_requests": self._refill_requests,
            "buffered_samples": len(self.buffered_samples),
            "max_buffered_samples": self._max_buffered_samples,
            "next_group_index": self._next_group_index,
        }


# ────────────────────── Convenience Function ──────────────────────


async def generate_rollout_batch(
    buffer: DataBuffer,
    config: RolloutConfig,
    runtime: RolloutRuntime | None = None,
    sample_scorer: SampleScorer | None = None,
    score_fn: Callable[[AttemptRow], Any] | None = None,
    **rollout_kwargs: Any,
) -> RolloutBatch:
    """Generate a single batch with dynamic sampling (convenience function).

    Args:
        buffer: DataBuffer for prompts
        config: RolloutConfig with generation settings
        runtime: Optional explicit rollout runtime wiring.
        sample_scorer: Explicit scoring stage owned by the buffer/data side.
        score_fn: Legacy score function (Sample -> Score), adapted into a
            sample_scorer when provided.
        **rollout_kwargs: Kwargs passed to generate_fn

    Returns:
        RolloutBatch ready for training

    Example:
        >>> from rollouts.core import Score, Metric
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
        runtime=resolve_rollout_runtime(config=config, runtime=runtime),
        rollout_kwargs=rollout_kwargs,
    )

    async with manager:
        return await manager.generate_batch(
            sample_scorer=sample_scorer,
            score_fn=score_fn,
        )
