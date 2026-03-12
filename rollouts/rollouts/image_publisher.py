"""Resolve source image specs into provisionable registry images."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .image_spec import ImageSpec, RegistryImage, resolve_image_for_provisioning

logger = logging.getLogger(__name__)

IMAGE_REPOSITORY_PREFIX_ENV = "ROLLOUTS_IMAGE_REPOSITORY_PREFIX"
IMAGE_CREDENTIALS_REF_ENV = "ROLLOUTS_IMAGE_CREDENTIALS_REF"
IMAGE_BUILD_TOOL_ENV = "ROLLOUTS_IMAGE_BUILD_TOOL"


class ImagePublisher(Protocol):
    """Build or resolve an image spec into a registry image."""

    def build_or_resolve(self, spec: ImageSpec) -> RegistryImage: ...


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-")
    return slug or "image"


def _dockerfile_hash(spec: ImageSpec) -> str:
    dockerfile_path = Path(spec.source_ref).expanduser()
    payload = {
        "source_type": spec.source_type,
        "source_ref": str(dockerfile_path.resolve()),
        "context_dir": str(
            Path(spec.context_dir).expanduser().resolve()
            if spec.context_dir is not None
            else dockerfile_path.parent.resolve()
        ),
        "python_version": spec.python_version,
        "build_args": spec.build_args,
        "system_packages": spec.system_packages,
        "pip_packages": spec.pip_packages,
        "pip_index_url": spec.pip_index_url,
        "pip_extra_index_url": spec.pip_extra_index_url,
        "build_commands": spec.build_commands,
        "features": spec.features,
        "installed_groups": spec.installed_groups,
        "dockerfile_sha256": hashlib.sha256(dockerfile_path.read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def default_repository_name(spec: ImageSpec) -> str:
    if spec.source_type == "dockerfile_path":
        return _slugify(Path(spec.source_ref).stem)
    return _slugify(spec.source_ref.split("/")[-1])


@dataclass(frozen=True)
class DockerBuildPublishConfig:
    """Settings for Dockerfile build/push publishing."""

    repository_prefix: str
    credentials_ref: str | None = None
    build_tool: str = "docker"

    def __post_init__(self) -> None:
        assert self.repository_prefix.strip(), "repository_prefix cannot be empty"
        assert "://" not in self.repository_prefix, (
            "repository_prefix must be a registry path, not a URL"
        )
        assert self.build_tool.strip(), "build_tool cannot be empty"

    @classmethod
    def from_env(cls) -> DockerBuildPublishConfig:
        repository_prefix = os.getenv(IMAGE_REPOSITORY_PREFIX_ENV)
        if not repository_prefix:
            raise ValueError(
                f"{IMAGE_REPOSITORY_PREFIX_ENV} must be set to publish non-registry image specs"
            )
        return cls(
            repository_prefix=repository_prefix.rstrip("/"),
            credentials_ref=os.getenv(IMAGE_CREDENTIALS_REF_ENV),
            build_tool=os.getenv(IMAGE_BUILD_TOOL_ENV, "docker"),
        )


class DockerImagePublisher:
    """Build/push Dockerfile-backed image specs using the local Docker CLI."""

    def __init__(
        self,
        config: DockerBuildPublishConfig,
        *,
        command_runner: callable | None = None,
    ) -> None:
        self.config = config
        self._command_runner = command_runner

    def build_or_resolve(self, spec: ImageSpec) -> RegistryImage:
        if spec.source_type == "registry":
            return resolve_image_for_provisioning(spec, credentials_ref=self.config.credentials_ref)
        if spec.source_type != "dockerfile_path":
            raise NotImplementedError(
                f"DockerImagePublisher does not support source_type={spec.source_type!r}"
            )

        dockerfile_path = Path(spec.source_ref).expanduser()
        assert dockerfile_path.exists(), f"Dockerfile does not exist: {dockerfile_path}"
        assert dockerfile_path.is_file(), f"Dockerfile path must be a file: {dockerfile_path}"
        context_dir = (
            Path(spec.context_dir).expanduser()
            if spec.context_dir is not None
            else dockerfile_path.parent
        )
        assert context_dir.exists(), f"Build context does not exist: {context_dir}"
        assert context_dir.is_dir(), f"Build context must be a directory: {context_dir}"
        image_name = default_repository_name(spec)
        tag = f"rollouts-{_dockerfile_hash(spec)}"
        target_ref = f"{self.config.repository_prefix}/{image_name}:{tag}"
        assert image_name.strip(), "resolved image_name cannot be empty"
        assert tag.startswith("rollouts-"), "generated image tag must use the rollouts prefix"
        assert target_ref.startswith(f"{self.config.repository_prefix}/"), (
            "target_ref must stay under the configured repository_prefix"
        )

        build_cmd = [
            self.config.build_tool,
            "build",
            "-f",
            str(dockerfile_path),
            "-t",
            target_ref,
        ]
        for key, value in sorted(spec.build_args.items()):
            build_cmd.extend(["--build-arg", f"{key}={value}"])
        build_cmd.append(str(context_dir))

        logger.info("Building image %s from %s", target_ref, dockerfile_path)
        self._run(build_cmd)
        logger.info("Pushing image %s", target_ref)
        self._run([self.config.build_tool, "push", target_ref])

        digest_ref = self._inspect_repo_digest(target_ref)
        return RegistryImage.from_ref(
            digest_ref or target_ref,
            credentials_ref=self.config.credentials_ref,
        )

    def _inspect_repo_digest(self, target_ref: str) -> str | None:
        assert target_ref.strip(), "target_ref cannot be empty"
        target_repo = target_ref.rsplit(":", 1)[0]
        output = self._run([
            self.config.build_tool,
            "image",
            "inspect",
            target_ref,
            "--format",
            "{{json .RepoDigests}}",
        ])
        try:
            repo_digests = json.loads(output)
        except json.JSONDecodeError:
            return None
        if not isinstance(repo_digests, list):
            return None
        for repo_digest in repo_digests:
            if repo_digest.startswith(target_repo + "@"):
                return repo_digest
        if not repo_digests:
            return None
        first_repo_digest = repo_digests[0]
        assert isinstance(first_repo_digest, str), "RepoDigests entries must be strings"
        assert "@" in first_repo_digest, "RepoDigests entries must include a digest"
        return first_repo_digest

    def _run(self, args: list[str]) -> str:
        if self._command_runner is not None:
            return self._command_runner(args)

        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {' '.join(args)}\n{result.stderr.strip()}"
            )
        return result.stdout.strip()


def build_or_resolve_image(
    spec: ImageSpec,
    *,
    publisher: ImagePublisher | None = None,
    credentials_ref: str | None = None,
) -> RegistryImage:
    """Return a registry image for provisioning.

    Registry image specs resolve directly. Dockerfile-backed specs use the
    configured publisher or the default Docker CLI publisher.
    """
    if spec.source_type == "registry":
        return resolve_image_for_provisioning(spec, credentials_ref=credentials_ref)

    if publisher is None:
        publisher = DockerImagePublisher(DockerBuildPublishConfig.from_env())
    return publisher.build_or_resolve(spec)
