from __future__ import annotations

from collections import defaultdict

import pytest

from rollouts.training.datasets.data_buffer import DataBuffer
from rollouts.training.group_assembly import assemble_groups, count_complete_groups
from rollouts.training.rollout_gen.async_rollout_manager import AsyncRolloutManager
from rollouts.training.rollout_gen.pipelined_rollout_manager import PipelinedRolloutManager
from rollouts.training.runtime import resolve_rollout_runtime
from rollouts.training.types import (
    IncompleteGroupPolicy,
    RolloutConfig,
    RolloutRuntime,
    Sample,
)


def test_assemble_groups_drops_incomplete_groups_and_keeps_complete_overflow() -> None:
    samples = [
        Sample(group_index=10),
        Sample(group_index=10),
        Sample(group_index=20),
        Sample(group_index=30),
        Sample(group_index=30),
    ]

    result = assemble_groups(samples, target_num_groups=1, samples_per_group=2)

    assert count_complete_groups(samples, 2) == 2
    assert [sample.group_index for sample in result.ready_samples] == [10, 10]
    assert [sample.group_index for sample in result.overflow_samples] == [30, 30]
    assert [sample.group_index for sample in result.dropped_incomplete_samples] == [20]
    assert result.dropped_incomplete_group_sizes == {20: 1}


def test_assemble_groups_can_fail_closed_on_incomplete_groups() -> None:
    samples = [
        Sample(group_index=10),
        Sample(group_index=10),
        Sample(group_index=20),
    ]

    with pytest.raises(ValueError, match="Group 20 is incomplete"):
        assemble_groups(
            samples,
            target_num_groups=1,
            samples_per_group=2,
            incomplete_group_policy=IncompleteGroupPolicy.ERROR,
        )


def test_rollout_config_roundtrips_incomplete_group_policy() -> None:
    config = RolloutConfig(
        batch_size=4,
        n_samples_per_prompt=8,
        incomplete_group_policy=IncompleteGroupPolicy.ERROR,
    )

    roundtrip = RolloutConfig.from_dict(config.to_dict())

    assert roundtrip.incomplete_group_policy == IncompleteGroupPolicy.ERROR


def test_resolve_rollout_runtime_prefers_explicit_runtime() -> None:
    explicit_runtime = RolloutRuntime(generate_fn=lambda prompts: [])
    legacy_config = RolloutConfig(
        batch_size=1,
        generate_fn=lambda prompts: [Sample() for _ in prompts],
    )

    resolved_runtime = resolve_rollout_runtime(
        config=legacy_config,
        runtime=explicit_runtime,
    )

    assert resolved_runtime is explicit_runtime


@pytest.mark.trio
async def test_async_rollout_manager_refills_incomplete_groups() -> None:
    call_counts: dict[str, int] = defaultdict(int)

    async def generate_fn(prompts: list[str]) -> list[Sample]:
        prompt = prompts[0]
        call_counts[prompt] += 1
        return [Sample(prompt=prompt, tokens=[1], loss_mask=[1.0], reward=0.0)]

    config = RolloutConfig(
        batch_size=1,
        n_samples_per_prompt=2,
        incomplete_group_policy=IncompleteGroupPolicy.REQUEST_MORE,
        max_refill_rounds=1,
    )
    runtime = RolloutRuntime(generate_fn=generate_fn)
    manager = AsyncRolloutManager(
        data_buffer=DataBuffer(prompts=["unused"]),
        config=config,
        runtime=runtime,
        buffered_samples=[Sample(prompt="prompt-a", group_index=7, tokens=[1], loss_mask=[1.0])],
    )

    async with manager:
        batch = await manager.generate_batch()

    assert batch.metadata["assembled_groups"] == 1
    assert batch.metadata["refill_rounds"] == 1
    assert batch.metadata["rollout_stats"]["refill_requests"] == 1
    assert [sample.group_index for sample in batch.samples] == [7, 7]
    assert [sample.prompt for sample in batch.samples] == ["prompt-a", "prompt-a"]
    assert call_counts["prompt-a"] == 1


@pytest.mark.trio
async def test_pipelined_rollout_manager_refills_fully_stale_group_from_prompt_registry() -> None:
    call_counts: dict[str, int] = defaultdict(int)

    async def generate_fn(prompts: list[str]) -> list[Sample]:
        prompt = prompts[0]
        call_counts[prompt] += 1
        return [Sample(prompt=prompt, tokens=[1], loss_mask=[1.0], reward=0.0)]

    config = RolloutConfig(
        batch_size=1,
        n_samples_per_prompt=2,
        incomplete_group_policy=IncompleteGroupPolicy.REQUEST_MORE,
        max_refill_rounds=1,
    )
    manager = PipelinedRolloutManager(
        data_buffer=DataBuffer(prompts=["unused"]),
        config=config,
        runtime=RolloutRuntime(generate_fn=generate_fn),
        max_lag=0,
        queue_size=0,
    )
    manager._group_prompts[7] = "prompt-a"
    manager._sample_queue = [
        Sample(
            prompt="prompt-a",
            group_index=7,
            weight_version=0,
            tokens=[1],
            loss_mask=[1.0],
            reward=0.0,
        )
    ]

    async with manager:
        batch = await manager.get_batch(current_weight_version=1, timeout=2.0)

    assert batch.metadata["assembled_groups"] == 1
    assert batch.metadata["refill_rounds"] == 1
    assert batch.metadata["rollout_stats"]["refill_requests"] == 2
    assert [sample.group_index for sample in batch.samples] == [7, 7]
    assert [sample.prompt for sample in batch.samples] == ["prompt-a", "prompt-a"]
    assert call_counts["prompt-a"] == 2
