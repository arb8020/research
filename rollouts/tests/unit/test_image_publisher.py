from __future__ import annotations

from pathlib import Path

import pytest

from rollouts.image_publisher import (
    DockerBuildPublishConfig,
    DockerImagePublisher,
    build_or_resolve_image,
)
from rollouts.image_spec import ImageSpec


class _FakeDockerRunner:
    def __init__(
        self,
        *,
        inspect_output: str = '["ghcr.io/acme/images/dockerfile@sha256:deadbeef"]',
        fail_on: tuple[str, ...] | None = None,
    ) -> None:
        self.inspect_output = inspect_output
        self.fail_on = fail_on or ()
        self.built_ref: str | None = None
        self.pushed_ref: str | None = None
        self.inspected_ref: str | None = None

    def __call__(self, args: list[str]) -> str:
        if args[:2] == ["docker", "build"]:
            self.built_ref = args[args.index("-t") + 1]
            if "build" in self.fail_on:
                raise RuntimeError("Command failed (1): docker build\nbuild failed")
            return ""

        if args[:2] == ["docker", "push"]:
            self.pushed_ref = args[2]
            if "push" in self.fail_on:
                raise RuntimeError("Command failed (1): docker push\npush failed")
            return ""

        if args[:3] == ["docker", "image", "inspect"]:
            self.inspected_ref = args[3]
            if "inspect" in self.fail_on:
                raise RuntimeError("Command failed (1): docker image inspect\ninspect failed")
            return self.inspect_output

        raise AssertionError(f"unexpected docker command: {args}")


def test_build_or_resolve_image_keeps_registry_refs() -> None:
    image = build_or_resolve_image(ImageSpec.from_registry("ghcr.io/acme/rollouts:latest"))

    assert image.to_ref() == "ghcr.io/acme/rollouts:latest"
    assert image.resolved_ref == "ghcr.io/acme/rollouts:latest"


def test_docker_image_publisher_builds_and_pushes(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n")
    runner = _FakeDockerRunner()

    publisher = DockerImagePublisher(
        DockerBuildPublishConfig(repository_prefix="ghcr.io/acme/images"),
        command_runner=runner,
    )

    image = publisher.build_or_resolve(ImageSpec.from_dockerfile_path(str(dockerfile)))

    assert runner.built_ref is not None
    assert runner.pushed_ref == runner.built_ref
    assert runner.inspected_ref == runner.built_ref
    assert image.to_ref() == "ghcr.io/acme/images/dockerfile@sha256:deadbeef"


def test_docker_image_publisher_surfaces_build_failures(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n")
    runner = _FakeDockerRunner(fail_on=("build",))

    publisher = DockerImagePublisher(
        DockerBuildPublishConfig(repository_prefix="ghcr.io/acme/images"),
        command_runner=runner,
    )

    with pytest.raises(RuntimeError, match="docker build"):
        publisher.build_or_resolve(ImageSpec.from_dockerfile_path(str(dockerfile)))


def test_docker_image_publisher_surfaces_push_failures(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n")
    runner = _FakeDockerRunner(fail_on=("push",))

    publisher = DockerImagePublisher(
        DockerBuildPublishConfig(repository_prefix="ghcr.io/acme/images"),
        command_runner=runner,
    )

    with pytest.raises(RuntimeError, match="docker push"):
        publisher.build_or_resolve(ImageSpec.from_dockerfile_path(str(dockerfile)))


def test_docker_image_publisher_falls_back_to_tag_when_digest_missing(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n")
    runner = _FakeDockerRunner(inspect_output="not-json")

    publisher = DockerImagePublisher(
        DockerBuildPublishConfig(repository_prefix="ghcr.io/acme/images"),
        command_runner=runner,
    )

    image = publisher.build_or_resolve(ImageSpec.from_dockerfile_path(str(dockerfile)))

    assert runner.built_ref is not None
    assert image.to_ref() == runner.built_ref


def test_docker_image_publisher_surfaces_inspect_failures(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n")
    runner = _FakeDockerRunner(fail_on=("inspect",))

    publisher = DockerImagePublisher(
        DockerBuildPublishConfig(repository_prefix="ghcr.io/acme/images"),
        command_runner=runner,
    )

    with pytest.raises(RuntimeError, match="docker image inspect"):
        publisher.build_or_resolve(ImageSpec.from_dockerfile_path(str(dockerfile)))


def test_build_or_resolve_image_requires_publish_config_for_dockerfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n")
    monkeypatch.delenv("ROLLOUTS_IMAGE_REPOSITORY_PREFIX", raising=False)

    with pytest.raises(ValueError, match="ROLLOUTS_IMAGE_REPOSITORY_PREFIX"):
        build_or_resolve_image(ImageSpec.from_dockerfile_path(str(dockerfile)))


def test_publish_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROLLOUTS_IMAGE_REPOSITORY_PREFIX", "ghcr.io/acme/images")
    monkeypatch.setenv("ROLLOUTS_IMAGE_CREDENTIALS_REF", "ghcr-prod")
    monkeypatch.setenv("ROLLOUTS_IMAGE_BUILD_TOOL", "podman")

    config = DockerBuildPublishConfig.from_env()

    assert config.repository_prefix == "ghcr.io/acme/images"
    assert config.credentials_ref == "ghcr-prod"
    assert config.build_tool == "podman"
