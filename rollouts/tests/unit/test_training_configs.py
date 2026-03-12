import pytest

from rollouts.training.configs import DepsConfig, HardwareConfig


def test_ssh_provider_requires_explicit_deps() -> None:
    with pytest.raises(ValueError, match="requires explicit deps"):
        HardwareConfig(provider="runpod", deps=None)


def test_ssh_provider_accepts_explicit_deps() -> None:
    hardware = HardwareConfig(provider="runpod", deps=DepsConfig())

    assert hardware.deps is not None
