import pytest

from rollouts.training.configs import DepsConfig, HardwareConfig


def test_legacy_remote_bootstrap_is_rejected_for_modal() -> None:
    with pytest.raises(ValueError, match="legacy_remote_bootstrap"):
        HardwareConfig(provider="modal", deps=DepsConfig(), legacy_remote_bootstrap=True)


def test_legacy_remote_bootstrap_is_allowed_for_ssh_provider() -> None:
    hardware = HardwareConfig(provider="runpod", legacy_remote_bootstrap=True)

    assert hardware.legacy_remote_bootstrap is True
