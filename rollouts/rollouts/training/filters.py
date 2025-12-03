"""Dynamic sampling filters for rollout quality control.

Based on SLIME's filter_hub (references/slime/slime/rollout/filter_hub/).

Filter functions decide whether to keep or discard sample groups during
over-sampling. Used by AsyncRolloutManager to maintain data quality.

Tiger Style: Pure functions, explicit criteria.
SLIME: Dynamic sampling with quality control.
"""

import torch

from rollouts.training.types import Sample

# ────────────────────── SLIME's Default Filter ──────────────────────


def check_reward_nonzero_std(samples: list[Sample]) -> bool:
    """Keep only if reward standard deviation > 0.

    Based on SLIME's check_reward_nonzero_std (the default filter).
    Reference: references/slime/slime/rollout/filter_hub/dynamic_sampling_filters.py:9-15

    Why this matters:
    - GRPO/PPO compute advantages as (reward - mean) / std
    - If all rewards are identical (std=0), advantages are all 0
    - Zero advantages = no learning signal = wasted compute
    - Better to discard and generate new prompt

    Args:
        samples: Group of samples (same prompt, different responses)

    Returns:
        True if reward variance > 0 (keep), False otherwise (discard)

    Example:
        >>> # All samples have same reward (no learning signal)
        >>> samples = [Sample(prompt="Q", response="A1", reward=1.0),
        ...            Sample(prompt="Q", response="A2", reward=1.0)]
        >>> check_reward_nonzero_std(samples)
        False  # Discard - no variance

        >>> # Samples have different rewards (useful for learning)
        >>> samples = [Sample(prompt="Q", response="A1", reward=1.0),
        ...            Sample(prompt="Q", response="A2", reward=0.0)]
        >>> check_reward_nonzero_std(samples)
        True  # Keep - has variance
    """
    assert len(samples) > 0, "samples required"

    rewards = [sample.reward for sample in samples]
    std = torch.tensor(rewards, dtype=torch.float).std()

    return std.item() > 0.0


# ────────────────────── Additional Filters ──────────────────────


def check_min_reward(samples: list[Sample], threshold: float = 0.5) -> bool:
    """Keep if at least one sample exceeds reward threshold.

    Useful for filtering out groups where all attempts failed.

    Args:
        samples: Group of samples
        threshold: Minimum reward threshold

    Returns:
        True if any sample.reward > threshold

    Example:
        >>> samples = [Sample(prompt="Q", response="A1", reward=0.0),
        ...            Sample(prompt="Q", response="A2", reward=0.8)]
        >>> check_min_reward(samples, threshold=0.5)
        True  # Keep - has one good sample

        >>> samples = [Sample(prompt="Q", response="A1", reward=0.2),
        ...            Sample(prompt="Q", response="A2", reward=0.3)]
        >>> check_min_reward(samples, threshold=0.5)
        False  # Discard - all below threshold
    """
    assert len(samples) > 0, "samples required"
    assert 0.0 <= threshold <= 1.0, f"threshold must be in [0,1], got {threshold}"

    return any(sample.reward > threshold for sample in samples)


def check_response_diversity(samples: list[Sample], min_unique_ratio: float = 0.5) -> bool:
    """Keep if responses are sufficiently diverse.

    Avoids training on repetitive responses (e.g., model collapse).

    Args:
        samples: Group of samples
        min_unique_ratio: Minimum ratio of unique responses (0.5 = 50% unique)

    Returns:
        True if unique_responses / total_responses >= min_unique_ratio

    Example:
        >>> samples = [Sample(prompt="Q", response="A"),
        ...            Sample(prompt="Q", response="A"),  # Duplicate
        ...            Sample(prompt="Q", response="B"),
        ...            Sample(prompt="Q", response="C")]
        >>> check_response_diversity(samples, min_unique_ratio=0.5)
        True  # 3 unique / 4 total = 75% >= 50%

        >>> samples = [Sample(prompt="Q", response="A"),
        ...            Sample(prompt="Q", response="A"),
        ...            Sample(prompt="Q", response="A"),
        ...            Sample(prompt="Q", response="A")]
        >>> check_response_diversity(samples, min_unique_ratio=0.5)
        False  # 1 unique / 4 total = 25% < 50%
    """
    assert len(samples) > 0, "samples required"
    assert 0.0 <= min_unique_ratio <= 1.0, \
        f"min_unique_ratio must be in [0,1], got {min_unique_ratio}"

    responses = [sample.response for sample in samples]
    unique_count = len(set(responses))
    total_count = len(responses)

    return unique_count / total_count >= min_unique_ratio


def check_reasonable_length(
    samples: list[Sample],
    min_tokens: int = 10,
    max_tokens: int = 2048
) -> bool:
    """Keep if average response length is reasonable.

    Filters out degenerate cases (too short or too long).

    Args:
        samples: Group of samples
        min_tokens: Minimum average tokens
        max_tokens: Maximum average tokens

    Returns:
        True if min_tokens <= avg_length <= max_tokens

    Example:
        >>> samples = [Sample(prompt="Q", response="A", tokens=[1,2,3]),
        ...            Sample(prompt="Q", response="B", tokens=[1,2,3,4])]
        >>> check_reasonable_length(samples, min_tokens=2, max_tokens=10)
        True  # avg=3.5, within [2, 10]

        >>> samples = [Sample(prompt="Q", response="", tokens=[]),
        ...            Sample(prompt="Q", response="A", tokens=[1])]
        >>> check_reasonable_length(samples, min_tokens=2, max_tokens=10)
        False  # avg=0.5, below min_tokens=2
    """
    assert len(samples) > 0, "samples required"
    assert min_tokens > 0, f"min_tokens must be positive, got {min_tokens}"
    assert max_tokens >= min_tokens, \
        f"max_tokens ({max_tokens}) must be >= min_tokens ({min_tokens})"

    total_tokens = sum(len(sample.tokens) for sample in samples)
    avg_length = total_tokens / len(samples)

    return min_tokens <= avg_length <= max_tokens


def check_any_success(samples: list[Sample]) -> bool:
    """Keep if at least one sample completed successfully.

    Checks Sample.status for COMPLETED state.

    Args:
        samples: Group of samples

    Returns:
        True if any sample has status=COMPLETED

    Example:
        >>> samples = [
        ...     Sample(prompt="Q", response="A", status=Sample.Status.COMPLETED),
        ...     Sample(prompt="Q", response="B", status=Sample.Status.ABORTED),
        ... ]
        >>> check_any_success(samples)
        True  # Keep - has one successful sample

        >>> samples = [
        ...     Sample(prompt="Q", response="", status=Sample.Status.ABORTED),
        ...     Sample(prompt="Q", response="", status=Sample.Status.TRUNCATED),
        ... ]
        >>> check_any_success(samples)
        False  # Discard - all failed
    """
    assert len(samples) > 0, "samples required"

    return any(sample.status == Sample.Status.COMPLETED for sample in samples)


# ────────────────────── Composite Filters ──────────────────────


def check_quality_and_diversity(samples: list[Sample]) -> bool:
    """Composite filter: reward variance AND response diversity.

    Combines SLIME's default with diversity check.

    Args:
        samples: Group of samples

    Returns:
        True if both conditions pass

    Example:
        >>> samples = [Sample(prompt="Q", response="A", reward=1.0),
        ...            Sample(prompt="Q", response="B", reward=0.0)]
        >>> check_quality_and_diversity(samples)
        True  # Has variance AND diversity

        >>> samples = [Sample(prompt="Q", response="A", reward=1.0),
        ...            Sample(prompt="Q", response="A", reward=0.0)]
        >>> check_quality_and_diversity(samples)
        False  # Has variance but NO diversity (same response)
    """
    return (
        check_reward_nonzero_std(samples)
        and check_response_diversity(samples, min_unique_ratio=0.5)
    )


# ────────────────────── Filter Utilities ──────────────────────


def make_threshold_filter(threshold: float) -> callable:
    """Create a filter with custom threshold (Casey: redundancy).

    Args:
        threshold: Reward threshold

    Returns:
        Filter function

    Example:
        >>> strict_filter = make_threshold_filter(0.8)
        >>> lenient_filter = make_threshold_filter(0.3)
    """
    def filter_fn(samples: list[Sample]) -> bool:
        return check_min_reward(samples, threshold=threshold)

    return filter_fn


def make_length_filter(min_tokens: int, max_tokens: int) -> callable:
    """Create a length filter with custom bounds (Casey: redundancy).

    Args:
        min_tokens: Minimum tokens
        max_tokens: Maximum tokens

    Returns:
        Filter function

    Example:
        >>> short_filter = make_length_filter(5, 100)
        >>> long_filter = make_length_filter(100, 2048)
    """
    def filter_fn(samples: list[Sample]) -> bool:
        return check_reasonable_length(samples, min_tokens, max_tokens)

    return filter_fn
