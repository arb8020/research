"""Rollout Manager - DEPRECATED.

⚠️ DEPRECATION WARNING ⚠️

RolloutManager is deprecated and will be removed in a future version.

Use generate_rollout_batches() from training.rollout_generation instead:

    # OLD (deprecated):
    manager = RolloutManager(data_buffer, config)
    for batch in manager:
        train_on_batch(batch)

    # NEW (recommended):
    from training.rollout_generation import generate_rollout_batches
    batches = generate_rollout_batches(data_buffer, config)
    for batch in batches:
        train_on_batch(batch)

For async rollout generation with over-sampling, use AsyncRolloutManager instead.

See docs/ROLLOUTMANAGER_DEPRECATION.md for migration guide.

---

This is the main orchestration layer that connects:
- DataBuffer (prompt iteration)
- User rollout function (prompt → samples)
- Batch conversion (samples → RolloutBatch)

Following Casey Muratori's principles:
- Minimal state (just wraps DataBuffer)
- Explicit iteration (no callbacks)
- No hidden coupling
"""

import warnings
from typing import Any, Callable, Iterator, Optional

from training.data_buffer import DataBuffer
from training.types import Sample, RolloutConfig, RolloutBatch
from training.rollout_generation import (
    apply_sample_transforms,
    convert_to_batch,
    extract_sample_fields,
    compute_response_lengths,
    build_batch_metadata,
)


class RolloutManager:
    """Iterator that orchestrates data buffer + rollout generation.

    Usage:
        manager = RolloutManager(buffer, config)
        for batch in manager:
            # batch is a RolloutBatch ready for training
            train_step(batch)
    """

    def __init__(
        self,
        data_buffer: DataBuffer,
        config: RolloutConfig,
        **rollout_kwargs: Any,
    ):
        """Initialize rollout manager.

        ⚠️ DEPRECATED: Use generate_rollout_batches() from training.rollout_generation instead.

        Args:
            data_buffer: DataBuffer for prompt iteration
            config: RolloutConfig with batch_size and generate_fn
            **rollout_kwargs: Additional kwargs passed to generate_fn
                             (e.g., tokenizer, dataset, etc.)
        """
        warnings.warn(
            "RolloutManager is deprecated and will be removed in a future version. "
            "Use generate_rollout_batches() from training.rollout_generation instead. "
            "For async rollout generation with over-sampling, use AsyncRolloutManager.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.data_buffer = data_buffer
        self.config = config
        self.rollout_kwargs = rollout_kwargs

        # Validate config
        if config.generate_fn is None:
            raise ValueError("RolloutConfig.generate_fn must be provided")
        if config.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {config.batch_size}")

        # Iteration state
        self._step_count = 0

    def __iter__(self) -> Iterator[RolloutBatch]:
        """Iterate over rollout batches indefinitely."""
        return self

    def __next__(self) -> RolloutBatch:
        """Generate next rollout batch.

        Returns:
            RolloutBatch ready for training backend

        Process:
            1. Get prompts from DataBuffer
            2. Call user rollout function
            3. Apply optional transforms (filter/reward)
            4. Convert to RolloutBatch
        """
        # Get prompts from buffer (handles epoch wraparound)
        prompts = self.data_buffer.get_prompts(self.config.batch_size)
        assert len(prompts) == self.config.batch_size, "Buffer must return requested batch size"

        # Call user-provided rollout function
        samples = self.config.generate_fn(prompts, **self.rollout_kwargs)
        assert isinstance(samples, list), f"generate_fn must return list[Sample], got {type(samples)}"
        assert len(samples) > 0, "generate_fn must return non-empty sample list"

        # Apply optional transforms
        samples = apply_sample_transforms(samples, self.config)

        # Convert to batch
        batch = convert_to_batch(
            samples,
            epoch_id=self.data_buffer.epoch_id,
            step_id=self._step_count,
        )

        self._step_count += 1
        return batch

    def state_dict(self) -> dict[str, Any]:
        """Save manager state for checkpointing.

        Returns:
            State dict with buffer state + step count
        """
        return {
            "buffer_state": self.data_buffer.save_state(),
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore manager state from checkpoint.

        Args:
            state: State dict from state_dict()
        """
        self.data_buffer.load_state(state["buffer_state"])
        self._step_count = state["step_count"]
