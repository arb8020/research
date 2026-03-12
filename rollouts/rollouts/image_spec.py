"""Shared image/bootstrapping contracts.

These types separate:
- the desired image contents (`ImageSpec`)
- run-specific additions (`RuntimeOverlay`)
- machine-readable image capabilities (`ImageManifest`)

The same contract is intentionally backend-agnostic so Modal, SSH, Dockerfile,
and future Nix-based flows can all converge on one shape.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Literal

DEFAULT_IMAGE_MANIFEST_PATH = "/etc/rollouts-image.json"
USER_IMAGE_MANIFEST_PATH = "~/.rollouts-image.json"


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def infer_cuda_version(gpu_type: str, pip_index_url: str | None = None) -> str:
    """Infer CUDA version from the requested GPU or wheel index."""
    if gpu_type in ("B200", "GB200"):
        return "12.8.0"
    if pip_index_url:
        if "cu128" in pip_index_url:
            return "12.8.0"
        if "cu126" in pip_index_url:
            return "12.6.0"
        if "cu124" in pip_index_url:
            return "12.4.0"
    return "12.4.0"


def default_cuda_image(gpu_type: str, pip_index_url: str | None = None) -> str:
    """Choose the default CUDA devel image for training workloads."""
    cuda_version = infer_cuda_version(gpu_type, pip_index_url)
    return f"nvidia/cuda:{cuda_version}-devel-ubuntu22.04"


def stable_feature_name(prefix: str, values: tuple[str, ...]) -> str:
    """Create a stable feature marker from the exact contents of a step."""
    if not values:
        return prefix
    digest = hashlib.sha1("\n".join(values).encode()).hexdigest()[:12]
    return f"{prefix}-{digest}"


@dataclass(frozen=True)
class RegistryImage:
    """Resolved registry image reference for provisioning.

    `original_ref` preserves the exact user-provided string when available so
    downstream provisioning can avoid accidental normalization changes.
    """

    registry: str
    repository: str
    tag: str | None = None
    digest: str | None = None
    credentials_ref: str | None = None
    original_ref: str | None = None

    def __post_init__(self) -> None:
        assert self.registry.strip(), "registry cannot be empty"
        assert self.repository.strip(), "repository cannot be empty"
        if self.digest is not None:
            assert self.digest.strip(), "digest cannot be empty"

    @classmethod
    def from_ref(
        cls,
        ref: str,
        *,
        credentials_ref: str | None = None,
    ) -> RegistryImage:
        assert ref.strip(), "registry ref cannot be empty"

        ref_without_digest, digest = ref, None
        if "@" in ref:
            ref_without_digest, digest = ref.rsplit("@", 1)

        last_slash = ref_without_digest.rfind("/")
        last_colon = ref_without_digest.rfind(":")
        tag = None
        if last_colon > last_slash:
            ref_without_tag = ref_without_digest[:last_colon]
            tag = ref_without_digest[last_colon + 1 :]
        else:
            ref_without_tag = ref_without_digest

        if "/" in ref_without_tag:
            first_segment, remainder = ref_without_tag.split("/", 1)
            if "." in first_segment or ":" in first_segment or first_segment == "localhost":
                registry = first_segment
                repository = remainder
            else:
                registry = "docker.io"
                repository = ref_without_tag
        else:
            registry = "docker.io"
            repository = ref_without_tag

        return cls(
            registry=registry,
            repository=repository,
            tag=tag,
            digest=digest,
            credentials_ref=credentials_ref,
            original_ref=ref,
        )

    def to_ref(self, *, prefer_original: bool = True) -> str:
        if prefer_original and self.original_ref:
            return self.original_ref

        ref = f"{self.registry}/{self.repository}"
        if self.digest:
            return f"{ref}@{self.digest}"
        if self.tag:
            return f"{ref}:{self.tag}"
        return ref

    @property
    def resolved_ref(self) -> str:
        return self.to_ref(prefer_original=False)


@dataclass(frozen=True)
class ImageSpec:
    """Desired image contents before a run starts."""

    source_type: Literal["registry", "dockerfile_path", "nix"] = "registry"
    source_ref: str = "debian:bookworm-slim"
    python_version: str = "3.12"
    context_dir: str | None = None
    build_args: dict[str, str] = field(default_factory=dict)
    system_packages: tuple[str, ...] = ()
    pip_packages: tuple[str, ...] = ()
    pip_index_url: str | None = None
    pip_extra_index_url: str | None = None
    build_commands: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    features: tuple[str, ...] = ()
    installed_groups: tuple[str, ...] = ()
    manifest_path: str = DEFAULT_IMAGE_MANIFEST_PATH

    def __post_init__(self) -> None:
        assert self.source_ref, "source_ref cannot be empty"
        assert self.python_version, "python_version cannot be empty"
        assert self.manifest_path, "manifest_path cannot be empty"

    @classmethod
    def from_registry(
        cls,
        tag: str,
        *,
        python_version: str = "3.12",
        **kwargs: object,
    ) -> ImageSpec:
        return cls(source_type="registry", source_ref=tag, python_version=python_version, **kwargs)

    @classmethod
    def from_dockerfile_path(
        cls,
        path: str,
        *,
        context_dir: str | None = None,
        python_version: str = "3.12",
        **kwargs: object,
    ) -> ImageSpec:
        return cls(
            source_type="dockerfile_path",
            source_ref=path,
            context_dir=context_dir,
            python_version=python_version,
            **kwargs,
        )

    @classmethod
    def from_nix(
        cls,
        reference: str,
        *,
        python_version: str = "3.12",
        **kwargs: object,
    ) -> ImageSpec:
        return cls(source_type="nix", source_ref=reference, python_version=python_version, **kwargs)

    def extended(
        self,
        *,
        system_packages: tuple[str, ...] = (),
        pip_packages: tuple[str, ...] = (),
        pip_index_url: str | None = None,
        pip_extra_index_url: str | None = None,
        build_commands: tuple[str, ...] = (),
        env: dict[str, str] | None = None,
        features: tuple[str, ...] = (),
        installed_groups: tuple[str, ...] = (),
    ) -> ImageSpec:
        merged_env = {**self.env, **(env or {})}
        return replace(
            self,
            system_packages=_dedupe(self.system_packages + system_packages),
            pip_packages=_dedupe(self.pip_packages + pip_packages),
            pip_index_url=pip_index_url or self.pip_index_url,
            pip_extra_index_url=pip_extra_index_url or self.pip_extra_index_url,
            build_commands=self.build_commands + build_commands,
            env=merged_env,
            features=_dedupe(self.features + features),
            installed_groups=_dedupe(self.installed_groups + installed_groups),
        )

    def resolve_registry_image(self, *, credentials_ref: str | None = None) -> RegistryImage:
        if self.source_type != "registry":
            raise ValueError(
                f"Image source_type={self.source_type!r} is not directly provisionable. "
                "Build/push it to a registry first."
            )
        return RegistryImage.from_ref(self.source_ref, credentials_ref=credentials_ref)


@dataclass(frozen=True)
class RuntimeOverlay:
    """Run-time additions on top of a base image or machine."""

    system_packages: tuple[str, ...] = ()
    pip_packages: tuple[str, ...] = ()
    pip_index_url: str | None = None
    pip_extra_index_url: str | None = None
    commands: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    features: tuple[str, ...] = ()
    installed_groups: tuple[str, ...] = ()

    def extended(
        self,
        *,
        system_packages: tuple[str, ...] = (),
        pip_packages: tuple[str, ...] = (),
        pip_index_url: str | None = None,
        pip_extra_index_url: str | None = None,
        commands: tuple[str, ...] = (),
        env: dict[str, str] | None = None,
        features: tuple[str, ...] = (),
        installed_groups: tuple[str, ...] = (),
    ) -> RuntimeOverlay:
        merged_env = {**self.env, **(env or {})}
        return replace(
            self,
            system_packages=_dedupe(self.system_packages + system_packages),
            pip_packages=_dedupe(self.pip_packages + pip_packages),
            pip_index_url=pip_index_url or self.pip_index_url,
            pip_extra_index_url=pip_extra_index_url or self.pip_extra_index_url,
            commands=self.commands + commands,
            env=merged_env,
            features=_dedupe(self.features + features),
            installed_groups=_dedupe(self.installed_groups + installed_groups),
        )


@dataclass(frozen=True)
class ImageManifest:
    """Machine-readable capabilities available in an image or reused node."""

    schema_version: int = 1
    image_name: str | None = None
    source_type: str | None = None
    source_ref: str | None = None
    resolved_image_ref: str | None = None
    python_version: str | None = None
    cuda_version: str | None = None
    features: tuple[str, ...] = ()
    installed_groups: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    def has_feature(self, feature: str) -> bool:
        return feature in self.features

    def has_installed_group(self, group: str) -> bool:
        return group in self.installed_groups

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "image_name": self.image_name,
                "source_type": self.source_type,
                "source_ref": self.source_ref,
                "resolved_image_ref": self.resolved_image_ref,
                "python_version": self.python_version,
                "cuda_version": self.cuda_version,
                "features": list(self.features),
                "installed_groups": list(self.installed_groups),
                "env": self.env,
                "paths": self.paths,
            },
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str) -> ImageManifest:
        data = json.loads(raw)
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            image_name=data.get("image_name"),
            source_type=data.get("source_type"),
            source_ref=data.get("source_ref"),
            resolved_image_ref=data.get("resolved_image_ref"),
            python_version=data.get("python_version"),
            cuda_version=data.get("cuda_version"),
            features=tuple(data.get("features", ())),
            installed_groups=tuple(data.get("installed_groups", ())),
            env=dict(data.get("env", {})),
            paths=dict(data.get("paths", {})),
        )

    def extended(
        self,
        *,
        features: tuple[str, ...] = (),
        installed_groups: tuple[str, ...] = (),
        env: dict[str, str] | None = None,
        paths: dict[str, str] | None = None,
        resolved_image_ref: str | None = None,
        python_version: str | None = None,
        cuda_version: str | None = None,
    ) -> ImageManifest:
        return replace(
            self,
            resolved_image_ref=resolved_image_ref or self.resolved_image_ref,
            python_version=python_version or self.python_version,
            cuda_version=cuda_version or self.cuda_version,
            features=_dedupe(self.features + features),
            installed_groups=_dedupe(self.installed_groups + installed_groups),
            env={**self.env, **(env or {})},
            paths={**self.paths, **(paths or {})},
        )


def image_manifest_for_spec(
    spec: ImageSpec,
    *,
    image_name: str | None = None,
    cuda_version: str | None = None,
    features: tuple[str, ...] = (),
    installed_groups: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    paths: dict[str, str] | None = None,
    resolved_image_ref: str | None = None,
) -> ImageManifest:
    return ImageManifest(
        image_name=image_name,
        source_type=spec.source_type,
        source_ref=spec.source_ref,
        resolved_image_ref=resolved_image_ref,
        python_version=spec.python_version,
        cuda_version=cuda_version,
        features=_dedupe(spec.features + features),
        installed_groups=_dedupe(spec.installed_groups + installed_groups),
        env={**spec.env, **(env or {})},
        paths=dict(paths or {}),
    )


def resolve_image_for_provisioning(
    spec: ImageSpec,
    *,
    credentials_ref: str | None = None,
) -> RegistryImage:
    """Resolve an ImageSpec into a registry image that providers can pull."""
    return spec.resolve_registry_image(credentials_ref=credentials_ref)


def manifest_write_command(manifest: ImageManifest, path: str = DEFAULT_IMAGE_MANIFEST_PATH) -> str:
    """Return a shell command that writes a manifest file."""
    import base64

    payload = manifest.to_json(indent=2)
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    return (
        "python3 -c "
        f"\"import base64; from pathlib import Path; path = Path({path!r}).expanduser(); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        f"path.write_text(base64.b64decode({encoded!r}).decode('utf-8'))\""
    )
