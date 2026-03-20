from __future__ import annotations

import pytest

from rollouts.config import ElasticPool, PoolSpec, ResourceProfile


def test_elastic_pool_fixed_helper_marks_pool_fixed() -> None:
    pool = ElasticPool.fixed(replicas=4)

    assert pool.min_replicas == 4
    assert pool.max_replicas == 4
    assert pool.is_fixed is True
    assert pool.is_cold_startable is False


def test_elastic_pool_min_zero_represents_cold_startable_capacity() -> None:
    pool = ElasticPool(min_replicas=0, max_replicas=1)

    assert pool.is_fixed is False
    assert pool.is_cold_startable is True


def test_pool_spec_carries_named_profile_and_capacity() -> None:
    profile = ResourceProfile(
        name="inference_h100",
        accelerator_class="train_large",
        gpu_count=1,
        image="ghcr.io/openai/sglang:latest",
    )
    spec = PoolSpec(
        name="policy_inference",
        profile=profile,
        capacity=ElasticPool(min_replicas=1, max_replicas=8),
        kind="service",
    )

    assert spec.name == "policy_inference"
    assert spec.profile.name == "inference_h100"
    assert spec.capacity.min_replicas == 1
    assert spec.capacity.max_replicas == 8
    assert spec.kind == "service"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ResourceProfile(name="", gpu_count=1), "name cannot be empty"),
        (lambda: ResourceProfile(name="bad", gpu_count=-1), "gpu_count cannot be negative"),
        (
            lambda: ElasticPool(min_replicas=2, max_replicas=1),
            "max_replicas must be >=",
        ),
        (
            lambda: PoolSpec(
                name="",
                profile=ResourceProfile(name="cpu_sandbox"),
                capacity=ElasticPool.fixed(replicas=1),
            ),
            "name cannot be empty",
        ),
    ],
)
def test_pool_types_reject_invalid_state(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
