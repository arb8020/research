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
            3. Apply optional filters
            4. Convert to RolloutBatch
        """
        # 1. Get prompts from buffer (handles epoch wraparound)
        prompts = self.data_buffer.get_prompts(self.config.batch_size)

        # 2. Call user-provided rollout function
        samples = self.config.generate_fn(
            prompts,
            **self.rollout_kwargs,
        )

        # Validate output
        if not isinstance(samples, list):
            raise TypeError(
                f"generate_fn must return list[Sample], got {type(samples)}"
            )

        # 3. Apply optional filter
        if self.config.filter_fn is not None:
            samples = self.config.filter_fn(samples)

        # 4. Apply optional reward function (for RL rollouts)
        if self.config.reward_fn is not None:
            samples = self.config.reward_fn(samples)

        # 5. Convert to RolloutBatch
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
    if not samples:
        raise ValueError("Cannot convert empty sample list to batch")

    # Extract fields from samples
    tokens = [s.tokens for s in samples]
    loss_masks = [s.loss_mask for s in samples]
    rewards = [s.reward for s in samples]

    # Calculate response lengths from tokens and loss masks
    # Response tokens are where loss_mask > 0
    response_lengths = []
    for mask in loss_masks:
        response_len = sum(1 for m in mask if m > 0.0)
        response_lengths.append(response_len)

    # Collect metadata
    metadata = {
        "epoch_id": epoch_id,
        "step_id": step_id,
        "batch_size": len(samples),
        # Store prompts/responses in metadata for inspection
        "prompts": [s.prompt for s in samples],
        "responses": [s.response for s in samples],
    }

    # Add any custom metadata from samples
    if samples[0].metadata:
        # Aggregate sample-level metadata
        for key in samples[0].metadata.keys():
            values = [s.metadata.get(key) for s in samples]
            metadata[f"sample_{key}"] = values

    return RolloutBatch(
        tokens=tokens,
        loss_masks=loss_masks,
        rewards=rewards,
        response_lengths=response_lengths,
        metadata=metadata,
    )
