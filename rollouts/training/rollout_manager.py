"""Rollout Manager - orchestrates DataBuffer + user rollout functions.

This is the main orchestration layer that connects:
- DataBuffer (prompt iteration)
- User rollout function (prompt → samples)
- Batch conversion (samples → RolloutBatch)

Following Casey Muratori's principles:
- Minimal state (just wraps DataBuffer)
- Explicit iteration (no callbacks)
- No hidden coupling
"""

from typing import Any, Callable, Iterator, Optional

from training.data_buffer import DataBuffer
from training.types import Sample, RolloutConfig, RolloutBatch


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

        Args:
            data_buffer: DataBuffer for prompt iteration
            config: RolloutConfig with batch_size and generate_fn
            **rollout_kwargs: Additional kwargs passed to generate_fn
                             (e.g., tokenizer, dataset, etc.)
        """
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
        samples = _apply_sample_transforms(samples, self.config)

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


def _apply_sample_transforms(samples: list[Sample], config: RolloutConfig) -> list[Sample]:
    """Apply optional filter and reward functions to samples.

    Isolates optional transform logic from main iteration flow.
    """
    if config.filter_fn is not None:
        samples = config.filter_fn(samples)
        assert len(samples) > 0, "filter_fn must not filter out all samples"

    if config.reward_fn is not None:
        samples = config.reward_fn(samples)
        assert len(samples) > 0, "reward_fn must not remove all samples"

    return samples


def convert_to_batch(
    samples: list[Sample],
    epoch_id: int = 0,
    step_id: int = 0,
) -> RolloutBatch:
    """Convert list of samples to RolloutBatch.

    Pure function - no side effects.

    Args:
        samples: List of Sample objects
        epoch_id: Current epoch number
        step_id: Current step number

    Returns:
        RolloutBatch ready for training backend

    Example:
        >>> samples = [Sample(...), Sample(...)]
        >>> batch = convert_to_batch(samples, epoch_id=0, step_id=42)
        >>> batch.tokens  # list of token lists
        >>> batch.loss_masks  # list of loss masks
    """
    # Preconditions
    assert len(samples) > 0, "Cannot convert empty sample list to batch"
    assert all(len(s.tokens) == len(s.loss_mask) for s in samples), \
        "All samples must have matching token/loss_mask lengths"

    # Extract fields
    tokens, loss_masks, rewards = _extract_sample_fields(samples)
    response_lengths = _compute_response_lengths(loss_masks)
    metadata = _build_batch_metadata(samples, epoch_id, step_id)

    # Postcondition
    assert len(tokens) == len(loss_masks) == len(rewards) == len(response_lengths), \
        "All batch fields must have same length"

    return RolloutBatch(
        tokens=tokens,
        loss_masks=loss_masks,
        rewards=rewards,
        response_lengths=response_lengths,
        metadata=metadata,
    )


def _extract_sample_fields(samples: list[Sample]) -> tuple[list, list, list]:
    """Extract tokens, loss_masks, and rewards from samples.

    Pure extraction - no computation.
    """
    tokens = [s.tokens for s in samples]
    loss_masks = [s.loss_mask for s in samples]
    rewards = [s.reward for s in samples]
    return tokens, loss_masks, rewards


def _compute_response_lengths(loss_masks: list[list[float]]) -> list[int]:
    """Compute response lengths from loss masks.

    Response tokens are where loss_mask > 0.
    Pure computation - no side effects.
    """
    response_lengths = []
    for mask in loss_masks:
        response_len = sum(1 for m in mask if m > 0.0)
        response_lengths.append(response_len)
    return response_lengths


def _build_batch_metadata(
    samples: list[Sample],
    epoch_id: int,
    step_id: int,
) -> dict[str, Any]:
    """Build metadata dict for batch.

    Aggregates sample-level metadata and adds batch-level info.
    """
    metadata = {
        "epoch_id": epoch_id,
        "step_id": step_id,
        "batch_size": len(samples),
        "prompts": [s.prompt for s in samples],
        "responses": [s.response for s in samples],
    }

    # Aggregate custom metadata if present
    if samples[0].metadata:
        for key in samples[0].metadata.keys():
            values = [s.metadata.get(key) for s in samples]
            metadata[f"sample_{key}"] = values

    return metadata
