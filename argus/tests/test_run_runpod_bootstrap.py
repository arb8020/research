from __future__ import annotations

from argus import run as argus_run


def test_runpod_custom_images_get_ssh_startup_script() -> None:
    script = argus_run._runpod_custom_image_ssh_startup_script("slimerl/slime:v0.2.3")

    assert script is not None
    assert "openssh-server" in script
    assert "/usr/sbin/sshd" in script


def test_runpod_official_images_do_not_get_ssh_startup_script() -> None:
    script = argus_run._runpod_custom_image_ssh_startup_script(
        "runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204"
    )

    assert script is None
