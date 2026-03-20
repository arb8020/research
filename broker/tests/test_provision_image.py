import pytest
from bifrost.provision import GPUQuery

from broker.types import ProvisionImage, ProvisionRequest


def test_provision_request_creates_default_registry_boot_image() -> None:
    request = ProvisionRequest(image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel")

    assert request.boot_image is not None
    assert request.boot_image.source_type == "registry"
    assert request.boot_image.reference == request.image


def test_provision_request_normalizes_registry_boot_image_to_image_field() -> None:
    request = ProvisionRequest(
        image="",
        boot_image=ProvisionImage(
            source_type="registry",
            reference="ghcr.io/example/custom:latest",
        ),
    )

    assert request.image == "ghcr.io/example/custom:latest"


def test_gpu_query_creates_default_registry_boot_image() -> None:
    query = GPUQuery(image="ghcr.io/example/custom:latest")

    assert query.boot_image is not None
    assert query.boot_image.source_type == "registry"
    assert query.boot_image.reference == query.image


def test_provision_request_rejects_non_positive_max_lifetime() -> None:
    with pytest.raises(AssertionError, match="max_lifetime_seconds"):
        ProvisionRequest(max_lifetime_seconds=0)
