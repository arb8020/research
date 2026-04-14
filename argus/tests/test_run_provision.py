import asyncio
from types import SimpleNamespace

from argus.run import _provision_remote_node


def test_provision_remote_node_normalizes_reused_instance() -> None:
    captured: dict[str, object] = {}
    bifrost = object()
    instance = SimpleNamespace(provider="runpod", id="node-123")

    async def fake_acquire_node(**kwargs: object) -> tuple[object, object]:
        captured.update(kwargs)
        return bifrost, instance

    result = asyncio.run(
        _provision_remote_node(
            acquire_node_fn=fake_acquire_node,
            node_id="runpod:node-123",
            provider="runpod",
            gpu_type="A100",
            gpu_count=1,
            logs_port=9100,
            container_disk_gb=100,
            provision_image="example:image",
            provision_boot_image=None,
            provision_template_id=None,
            provision_docker_args=None,
            persistent_volume_id=None,
            persistent_volume_mount_path="/workspace",
            persistent_volume_location=None,
            run_name="run_20260324-120000",
            image_ref="example:image",
        )
    )

    assert captured == {"node_id": "runpod:node-123"}
    assert result.bifrost is bifrost
    assert result.instance is instance
    assert result.reused is True
    assert result.node_ref == "runpod:node-123"
    assert result.image_ref == "example:image"


def test_provision_remote_node_builds_gpu_query_for_new_instance() -> None:
    captured: dict[str, object] = {}
    bifrost = object()
    instance = SimpleNamespace(provider="modal", id="node-456")

    async def fake_acquire_node(**kwargs: object) -> tuple[object, object]:
        captured.update(kwargs)
        return bifrost, instance

    result = asyncio.run(
        _provision_remote_node(
            acquire_node_fn=fake_acquire_node,
            node_id=None,
            provider="modal",
            gpu_type="H100",
            gpu_count=4,
            logs_port=9200,
            container_disk_gb=250,
            provision_image="registry/image:tag",
            provision_boot_image=None,
            provision_template_id="tmpl-123",
            provision_docker_args="--ipc=host",
            persistent_volume_id="vol-123",
            persistent_volume_mount_path="/data",
            persistent_volume_location="us-west",
            run_name="run_20260324-120001",
            image_ref="registry/image:tag",
        )
    )

    provision = captured["provision"]
    assert provision.type == "H100"
    assert provision.count == 4
    assert provision.exposed_ports == (9200,)
    assert provision.container_disk_gb == 250
    assert provision.provider == "modal"
    assert provision.image == "registry/image:tag"
    assert provision.template_id == "tmpl-123"
    assert provision.docker_args == "--ipc=host"
    assert provision.persistent_volume_id == "vol-123"
    assert provision.persistent_volume_mount_path == "/data"
    assert provision.persistent_volume_location == "us-west"
    assert provision.name == "rollouts/run_20260324-120001"

    assert result.bifrost is bifrost
    assert result.instance is instance
    assert result.reused is False
    assert result.node_ref == "modal:node-456"
