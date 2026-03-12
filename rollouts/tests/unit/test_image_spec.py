import pytest

from rollouts.image_spec import (
    ImageManifest,
    ImageSpec,
    RegistryImage,
    RuntimeOverlay,
    default_cuda_image,
    resolve_image_for_provisioning,
    stable_feature_name,
)
from rollouts.training.configs import DepsConfig


def test_deps_config_resolves_default_cuda_image() -> None:
    deps = DepsConfig(
        pip_packages=("torch>=2.4",),
        pip_index_url="https://download.pytorch.org/whl/cu128",
    )

    spec = deps.resolved_image("H100")

    assert spec.source_type == "registry"
    assert spec.source_ref == default_cuda_image("H100", deps.pip_index_url)
    assert "torch>=2.4" in spec.pip_packages


def test_deps_config_extends_explicit_image_spec() -> None:
    deps = DepsConfig(
        image=ImageSpec.from_dockerfile_path("./Dockerfile", context_dir="."),
        system_packages=("git",),
        pip_packages=("numpy",),
        bootstrap_commands=("echo hello",),
        runtime_overlay=RuntimeOverlay(commands=("echo runtime",)),
    )

    spec = deps.resolved_image("A100")
    overlay = deps.resolved_runtime_overlay()

    assert spec.source_type == "dockerfile_path"
    assert spec.system_packages == ("git",)
    assert spec.pip_packages == ("numpy",)
    assert spec.build_commands == ("echo hello",)
    assert overlay.commands == ("echo runtime",)


def test_image_manifest_round_trip() -> None:
    feature_name = stable_feature_name("overlay-pip-packages", ("numpy", "scipy"))
    manifest = ImageManifest(
        image_name="test-image",
        source_type="registry",
        source_ref="ghcr.io/acme/rollouts:latest",
        resolved_image_ref="ghcr.io/acme/rollouts@sha256:abc123",
        python_version="3.12",
        features=(feature_name,),
        installed_groups=("rollouts-training",),
        env={"HF_HOME": "/workspace/.cache/huggingface"},
    )

    decoded = ImageManifest.from_json(manifest.to_json())

    assert decoded.image_name == "test-image"
    assert decoded.has_feature(feature_name)
    assert decoded.has_installed_group("rollouts-training")
    assert decoded.resolved_image_ref == "ghcr.io/acme/rollouts@sha256:abc123"


def test_registry_image_preserves_original_ref() -> None:
    image = RegistryImage.from_ref("nvidia/cuda:12.8.0-devel-ubuntu22.04")

    assert image.registry == "docker.io"
    assert image.repository == "nvidia/cuda"
    assert image.tag == "12.8.0-devel-ubuntu22.04"
    assert image.to_ref() == "nvidia/cuda:12.8.0-devel-ubuntu22.04"
    assert image.resolved_ref == "docker.io/nvidia/cuda:12.8.0-devel-ubuntu22.04"


def test_resolve_image_for_provisioning_requires_registry_source() -> None:
    registry_image = resolve_image_for_provisioning(
        ImageSpec.from_registry("ghcr.io/acme/rollouts:latest")
    )

    assert registry_image.to_ref() == "ghcr.io/acme/rollouts:latest"

    with pytest.raises(ValueError, match="Build/push"):
        resolve_image_for_provisioning(ImageSpec.from_dockerfile_path("./Dockerfile"))
