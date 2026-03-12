from __future__ import annotations

import os
import shutil
import socket
import subprocess
import uuid
from pathlib import Path

import pytest

from rollouts.image_publisher import DockerBuildPublishConfig, DockerImagePublisher
from rollouts.image_spec import ImageSpec

pytestmark = pytest.mark.live


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_docker_image_publisher_with_local_registry(tmp_path: Path) -> None:
    if os.getenv("ROLLOUTS_RUN_DOCKER_PUBLISHER_LIVE") != "1":
        pytest.skip("Set ROLLOUTS_RUN_DOCKER_PUBLISHER_LIVE=1 to run Docker publisher live test")
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI not installed")

    port = _free_port()
    container_name = f"rollouts-test-registry-{uuid.uuid4().hex[:8]}"
    run_result = _docker("run", "-d", "-p", f"{port}:5000", "--name", container_name, "registry:2")
    if run_result.returncode != 0:
        pytest.skip(f"Could not start local registry container: {run_result.stderr.strip()}")

    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\nLABEL rollouts.test=true\n")

    try:
        publisher = DockerImagePublisher(
            DockerBuildPublishConfig(repository_prefix=f"localhost:{port}/rollouts-live")
        )
        image = publisher.build_or_resolve(ImageSpec.from_dockerfile_path(str(dockerfile)))

        assert image.to_ref().startswith(f"localhost:{port}/rollouts-live/dockerfile@sha256:")
    finally:
        _docker("rm", "-f", container_name)
