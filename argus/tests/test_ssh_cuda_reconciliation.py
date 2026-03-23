from __future__ import annotations

from argus import run as argus_run
from rollouts.image_spec import ImageSpec


def test_ssh_cuda_reconciliation_skips_image_owned_runtime() -> None:
    image = ImageSpec.from_registry(
        "slimerl/slime:v0.2.3",
        python_runtime="image_owned",
    )

    assert argus_run._should_reconcile_ssh_cuda_toolkit(image) is False


def test_ssh_cuda_reconciliation_keeps_managed_runtime() -> None:
    image = ImageSpec.from_registry(
        "runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204",
        python_runtime="managed_venv",
    )

    assert argus_run._should_reconcile_ssh_cuda_toolkit(image) is False


def test_ssh_cuda_reconciliation_defaults_true_without_image() -> None:
    assert argus_run._should_reconcile_ssh_cuda_toolkit(None) is True
