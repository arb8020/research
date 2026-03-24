from argus.run import REMOTE_SYSTEM_TOOLS_FEATURE, REMOTE_UV_FEATURE, _build_ssh_bootstrap_plan
from rollouts.image_spec import ImageManifest, ImageSpec, RuntimeOverlay


def test_build_ssh_bootstrap_plan_skips_satisfied_base_features() -> None:
    manifest = ImageManifest(
        features=(
            REMOTE_SYSTEM_TOOLS_FEATURE,
            REMOTE_UV_FEATURE,
            "ssh-managed-venv-python-3.12",
        )
    )

    plan = _build_ssh_bootstrap_plan(
        remote_manifest=manifest,
        custom_image=None,
        custom_overlay=None,
        runtime_feature_scope="managed-venv-python-3.12",
        image_owned_runtime=False,
        runtime_python="/root/.bifrost/venvs/rollouts-rl/bin/python",
        managed_venv_ready=False,
        needs_cuda_upgrade=False,
        cuda_req=None,
        extra_python_project_roots=(),
    )

    labels = [label for label, _ in plan.steps]
    assert "Installing system deps" not in labels
    assert "Installing uv" not in labels
    assert "Creating managed Python runtime" not in labels


def test_build_ssh_bootstrap_plan_preserves_overlay_groups_and_extra_projects() -> None:
    image = ImageSpec.from_registry(
        "ghcr.io/example/train:latest",
        system_packages=("git",),
        pip_packages=("flashinfer",),
    )
    overlay = RuntimeOverlay(
        pip_packages=("vllm",),
        features=("overlay-ready",),
        installed_groups=("train",),
    )

    plan = _build_ssh_bootstrap_plan(
        remote_manifest=None,
        custom_image=image,
        custom_overlay=overlay,
        runtime_feature_scope="managed-venv-python-3.12",
        image_owned_runtime=False,
        runtime_python="/root/.bifrost/venvs/rollouts-rl/bin/python",
        managed_venv_ready=True,
        needs_cuda_upgrade=False,
        cuda_req=None,
        extra_python_project_roots=("/workspace/charisma",),
    )

    labels = [label for label, _ in plan.steps]
    assert "Installing image system packages" in labels
    assert "Installing image Python packages" in labels
    assert "Installing runtime Python packages" in labels
    assert "Installing extra project Python packages" in labels
    assert "overlay-ready" in plan.manifest_features
    assert plan.manifest_groups == ("train",)
