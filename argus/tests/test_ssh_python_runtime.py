from __future__ import annotations

from argus import run as argus_run
from rollouts.image_spec import ImageSpec


def test_ssh_runtime_python_uses_image_python_for_image_owned() -> None:
    image = ImageSpec.from_registry(
        "slimerl/slime:v0.2.3",
        python_runtime="image_owned",
        python_executable="python3",
    )

    assert argus_run._ssh_runtime_python(image) == "python3"


def test_ssh_runtime_python_uses_managed_venv_for_managed_runtime() -> None:
    image = ImageSpec.from_registry(
        "slimerl/slime:v0.2.3",
        python_runtime="managed_venv",
        python_executable="python3",
    )

    assert argus_run._ssh_runtime_python(image) == "/root/.bifrost/venvs/rollouts-rl/bin/python"


def test_ssh_runtime_feature_scope_distinguishes_managed_venv() -> None:
    image = ImageSpec.from_registry(
        "slimerl/slime:v0.2.3",
        python_runtime="managed_venv",
        python_version="3.12",
    )

    assert argus_run._ssh_runtime_feature_scope(image) == "managed-venv-python-3.12"


def test_uv_pip_install_command_targets_requested_python() -> None:
    command = argus_run._uv_pip_install_command(
        ("torch", "trio"),
        python_bin="/root/.bifrost/venvs/rollouts-rl/bin/python",
    )

    assert "--python /root/.bifrost/venvs/rollouts-rl/bin/python" in command
    assert " torch trio" in command


def test_uv_pip_install_editable_command_supports_system_installs() -> None:
    command = argus_run._uv_pip_install_editable_command(
        ("/tmp/charisma",),
        system=True,
    )

    assert "--system" in command
    assert "-e /tmp/charisma" in command


def test_uv_pip_install_editable_command_can_skip_deps() -> None:
    command = argus_run._uv_pip_install_editable_command(
        ("/tmp/charisma",),
        python_bin="/root/.bifrost/venvs/rollouts-rl/bin/python",
        no_deps=True,
    )

    assert "--python /root/.bifrost/venvs/rollouts-rl/bin/python" in command
    assert "--no-deps" in command
    assert "-e /tmp/charisma" in command
