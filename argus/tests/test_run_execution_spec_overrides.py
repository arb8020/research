from __future__ import annotations

from types import SimpleNamespace

import pytest

from argus import run as argus_run
from argus.run import ExecutionSpecOverrides
from rollouts.training.configs import DepsConfig, HardwareConfig


def test_execution_spec_overrides_tracks_local_flag() -> None:
    args = SimpleNamespace(local=True)

    overrides = ExecutionSpecOverrides.from_args(args)

    assert overrides.used_flags() == ["--local"]


def test_execution_spec_overrides_apply_local_override_to_hardware() -> None:
    hardware = HardwareConfig(provider="runpod", deps=DepsConfig())
    overrides = ExecutionSpecOverrides(force_local=True)

    resolved = overrides.apply_to_hardware(hardware)

    assert resolved.provider == "local"
    assert resolved.gpu_type == hardware.gpu_type
    assert resolved.gpu_count == hardware.gpu_count


@pytest.mark.parametrize(
    "argv",
    [
        ["--config", "dummy.py", "--provider", "runpod"],
        ["--config", "dummy.py", "--gpu-type", "H100"],
        ["--config", "dummy.py", "--container-disk-gb", "200"],
        ["--config", "dummy.py", "--hf-cache-dir", "/cache/hf"],
        ["--config", "dummy.py", "--persistent-volume-id", "vol-123"],
        ["--config", "dummy.py", "--modal"],
        ["--config", "dummy.py", "--provision"],
    ],
)
def test_run_main_rejects_removed_execution_spec_flags(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        argus_run.main(argv)

    assert excinfo.value.code == 2
