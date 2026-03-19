from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from broker.providers.modal import _build_image_from_deps
from rollouts.environments.modal_sandbox_resource import ModalSandboxResource, ModalSandboxResourceConfig


@dataclass
class _FakeImage:
    source: str
    actions: list[tuple[str, object]] = field(default_factory=list)

    def apt_install(self, *packages: str) -> "_FakeImage":
        self.actions.append(("apt_install", packages))
        return self

    def pip_install(self, *packages: str, **kwargs: object) -> "_FakeImage":
        self.actions.append(("pip_install", (packages, kwargs)))
        return self

    def run_commands(self, *commands: str) -> "_FakeImage":
        self.actions.append(("run_commands", commands))
        return self

    def env(self, env_vars: dict[str, str]) -> "_FakeImage":
        self.actions.append(("env", env_vars))
        return self


class _FakeModal:
    class Image:
        @staticmethod
        def from_registry(source_ref: str, add_python: str) -> _FakeImage:
            image = _FakeImage(source=f"registry:{source_ref}:{add_python}")
            image.actions.append(("from_registry", (source_ref, add_python)))
            return image

        @staticmethod
        def debian_slim(python_version: str) -> _FakeImage:
            image = _FakeImage(source=f"debian:{python_version}")
            image.actions.append(("debian_slim", python_version))
            return image


def test_build_image_from_deps_preserves_registry_source_and_env() -> None:
    deps = SimpleNamespace(
        source_type="registry",
        source_ref="nvidia/cuda:13.0.0-devel-ubuntu22.04",
        python_version="3.10",
        system_packages=("git",),
        pip_packages=("torch",),
        pip_index_url=None,
        pip_extra_index_url=None,
        pip_prerelease=False,
        bootstrap_commands=("echo hello",),
        env={"THUNDERKITTENS_ROOT": "/root/ThunderKittens"},
    )

    image = _build_image_from_deps(_FakeModal, deps)

    assert image.source == "registry:nvidia/cuda:13.0.0-devel-ubuntu22.04:3.10"
    assert ("env", {"THUNDERKITTENS_ROOT": "/root/ThunderKittens"}) in image.actions


@pytest.mark.trio
async def test_modal_sandbox_resource_provisions_and_terminates_via_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broker.providers import modal as broker_modal

    calls: dict[str, object] = {}

    async def fake_provision_instance(request, ssh_startup_script=None, api_key=None):
        del ssh_startup_script, api_key
        calls["request"] = request
        return SimpleNamespace(id="sb-broker-123")

    async def fake_terminate_instance(instance_id: str, api_key=None) -> bool:
        del api_key
        calls["terminated"] = instance_id
        return True

    monkeypatch.setattr(broker_modal, "provision_instance", fake_provision_instance)
    monkeypatch.setattr(broker_modal, "terminate_instance", fake_terminate_instance)

    resource = ModalSandboxResource(
        ModalSandboxResourceConfig(
            app_name="charisma-kernelbench-v3-smoke",
            gpu="A100",
            image_registry="nvidia/cuda:13.0.0-devel-ubuntu22.04",
            env={"THUNDERKITTENS_ROOT": "/root/ThunderKittens"},
        )
    )

    sandbox_id = await resource._provision_sandbox_via_broker()
    assert sandbox_id == "sb-broker-123"

    request = calls["request"]
    assert request.provider == "modal"
    assert request.gpu_type == "A100"
    assert request.raw_data["deps"].source_ref == "nvidia/cuda:13.0.0-devel-ubuntu22.04"
    assert request.raw_data["deps"].env["THUNDERKITTENS_ROOT"] == "/root/ThunderKittens"

    resource._sandbox = object()
    resource._sandbox_id = "sb-broker-123"
    await resource.close()
    assert calls["terminated"] == "sb-broker-123"
