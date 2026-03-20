"""Resource pool product types for workload topology.

These types are intentionally small:

- `ResourceProfile` describes one kind of allocatable resource.
- `ElasticPool` describes standing capacity, including fixed pools when
  `min_replicas == max_replicas`.
- `PoolSpec` gives a named pool in a workload topology.

Provider realization, scheduling policy, and live execution are still owned by
other layers. This module only captures workload-side resource intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ResourceProfile:
    """Named resource shape for a workload pool."""

    name: str
    image: str | None = None
    accelerator_class: str | None = None
    gpu_count: int = 0
    cpu: int | None = None
    memory_gb: int | None = None
    volume_key: str | None = None
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ResourceProfile.name cannot be empty")
        if self.gpu_count < 0:
            raise ValueError("ResourceProfile.gpu_count cannot be negative")
        if self.cpu is not None and self.cpu <= 0:
            raise ValueError("ResourceProfile.cpu must be positive when set")
        if self.memory_gb is not None and self.memory_gb <= 0:
            raise ValueError("ResourceProfile.memory_gb must be positive when set")


@dataclass(frozen=True)
class ElasticPool:
    """Standing capacity policy for a resource pool.

    Fixed pools are represented by `min_replicas == max_replicas`.
    Cold or on-demand-ish pools are represented by `min_replicas == 0`.
    """

    min_replicas: int
    max_replicas: int
    idle_timeout_s: int | None = None

    def __post_init__(self) -> None:
        if self.min_replicas < 0:
            raise ValueError("ElasticPool.min_replicas cannot be negative")
        if self.max_replicas < self.min_replicas:
            raise ValueError("ElasticPool.max_replicas must be >= min_replicas")
        if self.idle_timeout_s is not None and self.idle_timeout_s <= 0:
            raise ValueError("ElasticPool.idle_timeout_s must be positive when set")

    @classmethod
    def fixed(cls, *, replicas: int) -> ElasticPool:
        if replicas < 0:
            raise ValueError("ElasticPool.fixed replicas cannot be negative")
        return cls(min_replicas=replicas, max_replicas=replicas)

    @property
    def is_fixed(self) -> bool:
        return self.min_replicas == self.max_replicas

    @property
    def is_cold_startable(self) -> bool:
        return self.min_replicas == 0


@dataclass(frozen=True)
class PoolSpec:
    """Named pool in a workload topology."""

    name: str
    profile: ResourceProfile
    capacity: ElasticPool
    kind: Literal["sandbox", "service", "job"] = "service"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PoolSpec.name cannot be empty")
