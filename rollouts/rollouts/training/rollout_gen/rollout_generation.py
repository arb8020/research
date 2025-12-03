"""Pure functions for rollout generation.

Replaces the stateful RolloutManager class with pure functions.
Following Casey Muratori: minimize state, maximize pure functions.

Design Philosophy:
- RolloutManager had no legitimate mutable state (just loop counter)
- All the real work was done by pure functions
- Generator pattern is cleaner than class-based iteration

See docs/ROLLOUTMANAGER_DEPRECATION.md for migration guide.
"""

from collections.abc import Iterator
from typing import Any

from rollouts.training.datasets.data_buffer import DataBuffer
from rollouts.training.types import RolloutBatch, RolloutConfig, Sample


def generate_rollout_batches(
    data_buffer: DataBuffer,
    config: RolloutConfig,
    **rollout_kwargs: Any,
) -> Iterator[RolloutBatch]:
    """Generate rollout batches indefinitely (generator).

    Args:
        data_buffer: DataBuffer for prompt iteration (has state)
        config: RolloutConfig with batch_size and generate_fn
        **rollout_kwargs: Additional kwargs for generate_fn

    Yields:
        RolloutBatch objects ready for training

    Example:
        >>> data_buffer = DataBuffer(prompts=[...])
        >>> config = RolloutConfig(batch_size=4, generate_fn=my_fn)
        >>> batches = generate_rollout_batches(data_buffer, config)
        >>> for batch in batches:
        ...     train_on_batch(batch)

    Design Notes:
        - No class needed - generator pattern is simpler
        - data_buffer manages its own state (epoch_id, sample_offset)
        - step counter is local variable (no instance state)
        - All helper functions are pure (no side effects)
    """
    # Validate config (Tiger Style: assert preconditions)
    if config.generate_fn is None:
        raise ValueError("RolloutConfig.generate_fn must be provided")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {config.batch_size}")

    step = 0
    while True:
        # Get prompts (data_buffer manages its own state)
        prompts = data_buffer.get_prompts(config.batch_size)
        assert len(prompts) == config.batch_size, "Buffer must return requested batch size"

        # Call user-provided rollout function
        samples = config.generate_fn(prompts, **rollout_kwargs)
        assert isinstance(samples, list), f"generate_fn must return list[Sample], got {type(samples)}"
        assert len(samples) > 0, "generate_fn must return non-empty sample list"

        # Apply optional transforms (pure function)
        samples = apply_sample_transforms(samples, config)

        # Convert to batch (pure function)
        batch = convert_to_batch(
            samples,
            epoch_id=data_buffer.epoch_id,
            step_id=step,
        )

        yield batch
        step += 1


def apply_sample_transforms(samples: list[Sample], config: RolloutConfig) -> list[Sample]:
    """Apply optional filter and reward functions to samples.

    Pure function - no side effects.

    Args:
        samples: List of Sample objects
        config: RolloutConfig with optional filter_fn and reward_fn

    Returns:
        Transformed samples
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
    # Tiger Style: Assert preconditions
    assert len(samples) > 0, "Cannot convert empty sample list to batch"
    assert all(len(s.tokens) == len(s.loss_mask) for s in samples), \
        "All samples must have matching token/loss_mask lengths"

    # Extract fields (pure function)
    tokens, loss_masks, rewards = extract_sample_fields(samples)
    response_lengths = compute_response_lengths(loss_masks)
    metadata = build_batch_metadata(samples, epoch_id, step_id)

    # Tiger Style: Assert postconditions
    assert len(tokens) == len(loss_masks) == len(rewards) == len(response_lengths), \
        "All batch fields must have same length"

    return RolloutBatch(
        tokens=tokens,
        loss_masks=loss_masks,
        rewards=rewards,
        response_lengths=response_lengths,
        metadata=metadata,
    )


def extract_sample_fields(samples: list[Sample]) -> tuple[list, list, list]:
    """Extract tokens, loss_masks, and rewards from samples.

    Pure extraction - no computation.

    Args:
        samples: List of Sample objects

    Returns:
        Tuple of (tokens, loss_masks, rewards)
    """
    tokens = [s.tokens for s in samples]
    loss_masks = [s.loss_mask for s in samples]
    rewards = [s.reward for s in samples]
    return tokens, loss_masks, rewards


def compute_response_lengths(loss_masks: list[list[float]]) -> list[int]:
    """Compute response lengths from loss masks.

    Response tokens are where loss_mask > 0.
    Pure computation - no side effects.

    Args:
        loss_masks: List of loss mask lists

    Returns:
        List of response lengths (one per sample)
    """
    response_lengths = []
    for mask in loss_masks:
        response_len = sum(1 for m in mask if m > 0.0)
        response_lengths.append(response_len)
    return response_lengths


def build_batch_metadata(
    samples: list[Sample],
    epoch_id: int,
    step_id: int,
) -> dict[str, Any]:
    """Build metadata dict for batch.

    Aggregates sample-level metadata and adds batch-level info.

    Args:
        samples: List of Sample objects
        epoch_id: Current epoch number
        step_id: Current step number

    Returns:
        Metadata dict with batch and sample info
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
