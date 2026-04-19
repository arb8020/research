from __future__ import annotations

from dataclasses import dataclass, field

from rollouts.environments.harbor_environment import (
    HarborHostConfig,
    LocalHarborHost,
    ModalHarborHost,
    SSHHarborHost,
)

__all__ = [
    "HarborTaskEnvironmentConfig",
    "HarborHostConfig",
    "LocalHarborHost",
    "ModalHarborHost",
    "SSHHarborHost",
]


@dataclass(frozen=True)
class HarborTaskEnvironmentConfig:
    host: HarborHostConfig = field(default_factory=LocalHarborHost)
