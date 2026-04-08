from __future__ import annotations

from dataclasses import dataclass, field

from .types import IncompleteGroupPolicy, RowAttempt


@dataclass(frozen=True)
class GroupAssemblyResult:
    """Result of assembling trainable groups from sample-level input."""

    ready_samples: list[RowAttempt]
    overflow_samples: list[RowAttempt]
    dropped_incomplete_samples: list[RowAttempt]
    ready_group_count: int
    dropped_incomplete_group_sizes: dict[int, int] = field(default_factory=dict)


def count_complete_groups(samples: list[RowAttempt], samples_per_group: int) -> int:
    """Count how many complete groups are present in sample order."""
    assert samples_per_group > 0, "samples_per_group must be positive"
    if samples_per_group == 1:
        return len(samples)

    complete_group_count = 0
    for group_samples in _ordered_group_buckets(samples).values():
        if len(group_samples) == samples_per_group:
            complete_group_count += 1
        elif len(group_samples) > samples_per_group:
            raise ValueError(
                f"Group {group_samples[0].group_index} has {len(group_samples)} samples, "
                f"expected at most {samples_per_group}"
            )
    return complete_group_count


def collect_incomplete_groups(
    samples: list[RowAttempt],
    samples_per_group: int,
) -> dict[int, list[RowAttempt]]:
    """Collect incomplete groups keyed by stable group index."""
    assert samples_per_group > 0, "samples_per_group must be positive"
    if samples_per_group == 1:
        return {}

    incomplete_groups: dict[int, list[RowAttempt]] = {}
    for group_key, group_samples in _ordered_group_buckets(samples).items():
        group_size = len(group_samples)
        if group_size > samples_per_group:
            raise ValueError(
                f"Group {group_key} has {group_size} samples, expected at most {samples_per_group}"
            )
        if group_size < samples_per_group:
            incomplete_groups[group_key] = group_samples
    return incomplete_groups


def assemble_groups(
    samples: list[RowAttempt],
    target_num_groups: int,
    samples_per_group: int,
    incomplete_group_policy: IncompleteGroupPolicy = IncompleteGroupPolicy.DROP_INCOMPLETE,
) -> GroupAssemblyResult:
    """Assemble complete groups and explicitly drop incomplete ones.

    This stage is intentionally honest about the current implementation:
    complete groups can be buffered for later use, but incomplete groups are
    dropped because there is not yet a refill path that can regenerate missing
    samples for the same prompt.
    """
    assert target_num_groups >= 0, "target_num_groups must be >= 0"
    assert samples_per_group > 0, "samples_per_group must be positive"

    if samples_per_group == 1:
        ready_samples = list(samples[:target_num_groups])
        overflow_samples = list(samples[target_num_groups:])
        return GroupAssemblyResult(
            ready_samples=ready_samples,
            overflow_samples=overflow_samples,
            dropped_incomplete_samples=[],
            ready_group_count=len(ready_samples),
        )

    ready_samples: list[RowAttempt] = []
    overflow_samples: list[RowAttempt] = []
    dropped_incomplete_samples: list[RowAttempt] = []
    dropped_incomplete_group_sizes: dict[int, int] = {}
    ready_group_count = 0

    for group_key, group_samples in _ordered_group_buckets(samples).items():
        group_size = len(group_samples)
        if group_size > samples_per_group:
            raise ValueError(
                f"Group {group_key} has {group_size} samples, expected at most {samples_per_group}"
            )
        if group_size < samples_per_group:
            if incomplete_group_policy == IncompleteGroupPolicy.ERROR:
                raise ValueError(
                    f"Group {group_key} is incomplete with {group_size}/{samples_per_group} samples"
                )
            dropped_incomplete_samples.extend(group_samples)
            dropped_incomplete_group_sizes[group_key] = group_size
            continue
        if ready_group_count < target_num_groups:
            ready_samples.extend(group_samples)
            ready_group_count += 1
        else:
            overflow_samples.extend(group_samples)

    return GroupAssemblyResult(
        ready_samples=ready_samples,
        overflow_samples=overflow_samples,
        dropped_incomplete_samples=dropped_incomplete_samples,
        ready_group_count=ready_group_count,
        dropped_incomplete_group_sizes=dropped_incomplete_group_sizes,
    )


def _ordered_group_buckets(samples: list[RowAttempt]) -> dict[int, list[RowAttempt]]:
    """Bucket samples by stable group index, preserving first-seen order."""
    grouped_samples: dict[int, list[RowAttempt]] = {}
    next_fallback_group = -1

    for sample in samples:
        group_key = sample.group_index
        if group_key is None:
            group_key = next_fallback_group
            next_fallback_group -= 1

        if group_key not in grouped_samples:
            grouped_samples[group_key] = []
        grouped_samples[group_key].append(sample)

    return grouped_samples
