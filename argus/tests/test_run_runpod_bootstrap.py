from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch

from argus import run as argus_run


def test_runpod_custom_images_get_ssh_startup_script() -> None:
    script = argus_run._runpod_custom_image_ssh_startup_script("slimerl/slime:v0.2.3")

    assert script is not None
    assert "openssh-server" in script
    assert script == argus_run.RUNPOD_DOCS_CUSTOM_IMAGE_SSH_STARTUP_SCRIPT
    assert "service ssh start" in script
    assert "sleep infinity" in script


def test_runpod_official_images_do_not_get_ssh_startup_script() -> None:
    script = argus_run._runpod_custom_image_ssh_startup_script(
        "runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204"
    )

    assert script is None


def test_runpod_template_id_lowering_reads_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_SSH_TEMPLATE_ID", "runpod-torch-v280")

    assert argus_run._runpod_template_id_for_custom_image() == "runpod-torch-v280"
