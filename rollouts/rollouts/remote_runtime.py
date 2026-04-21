"""Shared remote runtime/materialization/source-sync contracts.

This is the first compression step toward the intended layering:

- broker: runtime-ready resource contract
- bifrost: source/materialization/process execution on that resource
- argus: detached run semantics and event supervision

The existing `HardwareConfig` / `DepsConfig` still mix some concerns.
This module does not try to solve that fully yet. It provides one shared
surface for the runners so launch semantics stop being duplicated.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

from .training.configs import DepsConfig, HardwareConfig

if TYPE_CHECKING:
    from bifrost import PythonProjectMaterialization


@dataclass(frozen=True)
class RuntimeContract:
    """Runtime contract that a remote resource must satisfy."""

    provider: Literal["modal", "runpod", "lambdalabs", "vast", "local"]
    gpu_type: str
    gpu_count: int
    deps: DepsConfig | None
    container_disk_gb: int
    hf_cache_dir: str
    persistent_volume_id: str | None
    persistent_volume_mount_path: str
    persistent_volume_location: str | None
    use_torchrun: bool


@dataclass(frozen=True)
class MaterializationPlan:
    """Project/workspace materialization on top of the runtime contract."""

    workspace_root: str = "~/.bifrost/workspaces/rollouts-rl"
    bootstrap_commands: tuple[str, ...] = ()
    source_mode: Literal["git_bundle_committed"] = "git_bundle_committed"


@dataclass(frozen=True)
class SourceSyncPolicy:
    """Policy for how local source is allowed to differ from deployed source."""

    source_mode: Literal["git_bundle_committed"] = "git_bundle_committed"
    dirty_action: Literal["fail", "warn"] = "fail"

    @classmethod
    def committed_only(cls, *, dirty_action: Literal["fail", "warn"] = "fail") -> SourceSyncPolicy:
        return cls(source_mode="git_bundle_committed", dirty_action=dirty_action)


def runtime_contract_from_hardware(hardware: HardwareConfig) -> RuntimeContract:
    return RuntimeContract(
        provider=hardware.provider,
        gpu_type=hardware.gpu_type,
        gpu_count=hardware.gpu_count,
        deps=hardware.deps,
        container_disk_gb=hardware.container_disk_gb,
        hf_cache_dir=hardware.hf_cache_dir,
        persistent_volume_id=hardware.persistent_volume_id,
        persistent_volume_mount_path=hardware.persistent_volume_mount_path,
        persistent_volume_location=hardware.persistent_volume_location,
        use_torchrun=hardware.use_torchrun,
    )


def materialization_plan_from_runtime(
    runtime: RuntimeContract,
    *,
    workspace_root: str = "~/.bifrost/workspaces/rollouts-rl",
) -> MaterializationPlan:
    bootstrap_commands = runtime.deps.bootstrap_commands if runtime.deps is not None else ()
    return MaterializationPlan(
        workspace_root=workspace_root,
        bootstrap_commands=bootstrap_commands,
        source_mode="git_bundle_committed",
    )


def _collect_dirty_paths(repo_root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return (), ()

    lines = [line for line in result.stdout.splitlines() if line]
    if not lines:
        return (), ()

    modified = tuple(line[3:] for line in lines if not line.startswith("??"))
    untracked = tuple(line[3:] for line in lines if line.startswith("??"))
    return modified, untracked


def enforce_source_sync_policy(
    policy: SourceSyncPolicy,
    *,
    repo_root: Path,
    stream: TextIO = sys.stderr,
) -> None:
    """Check local source state against the deployment/source policy."""

    modified, untracked = _collect_dirty_paths(repo_root)
    if not modified and not untracked:
        return

    if policy.dirty_action == "fail":
        print(
            "\n❌ Deploy uses committed-only source sync; dirty changes will not be deployed.\n",
            file=stream,
        )
    else:
        print(
            "\n⚠️  Deploy uses committed-only source sync; dirty changes will not be deployed.\n",
            file=stream,
        )

    total = len(modified) + len(untracked)
    print(f"{total} file(s) will NOT be deployed:\n", file=stream)
    for path in modified[:5]:
        print(f"   - {path} (modified)", file=stream)
    if len(modified) > 5:
        print(f"   ... and {len(modified) - 5} more modified", file=stream)
    for path in untracked[:5]:
        print(f"   - {path} (untracked)", file=stream)
    if len(untracked) > 5:
        print(f"   ... and {len(untracked) - 5} more untracked", file=stream)

    if policy.dirty_action == "fail":
        print(
            "\nCommit your changes or use '--force-deploy-committed' to proceed anyway.\n",
            file=stream,
        )
        raise SystemExit(1)

    print("\nProceeding with committed-only source sync.\n", file=stream)


def resolve_consumer_project(config_path: Path) -> PythonProjectMaterialization:
    """Resolve the consumer project that owns this config file.

    The consumer project is the Python project that *depends on* rollouts/argus
    (i.e. the top-level project the user is working in), not rollouts itself.

    Discovery walks up from the config file:
      1. Keep climbing past any pyproject whose directory is declared as a
         ``[tool.uv.workspace].members`` entry in a parent pyproject. The
         parent is the workspace root and owns results / deploy semantics.
      2. Stop at the first pyproject that is itself a workspace root
         (``[tool.uv.workspace]`` present) or a non-member project.
      3. Fall back to the nearest ``.git`` dir, then to the config file's
         parent, if no pyproject was found.

    Rationale: in the research monorepo the nearest pyproject to an example
    config is ``rollouts/pyproject.toml`` (rollouts is a workspace member).
    The "consumer project" we actually want is ``~/research/`` — the
    workspace root. In courier, the nearest pyproject is courier's own and
    rollouts lives under ``third_party/research_deps/``, so the first hit is
    already correct.

    Returned as a ``bifrost.PythonProjectMaterialization`` so it can be
    handed directly to bifrost materialize specs. Callers that just want the
    path can read ``.local_root``.
    """
    import tomllib

    from bifrost import PythonProjectMaterialization

    resolved = config_path.expanduser().resolve()

    def _parse(path: Path) -> dict | None:
        try:
            with path.open("rb") as f:
                return tomllib.load(f)
        except Exception:
            return None

    def _is_workspace_root(data: dict) -> bool:
        return "workspace" in data.get("tool", {}).get("uv", {})

    def _workspace_members(data: dict) -> tuple[str, ...]:
        members = data.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
        return tuple(str(m) for m in members)

    # Walk up from the config file, collecting pyproject.toml locations.
    candidates = [resolved.parent, *resolved.parents]
    pyproject_hits: list[Path] = [c for c in candidates if (c / "pyproject.toml").exists()]

    # Look for a workspace root that declares one of the hits as a member.
    for hit_idx, hit in enumerate(pyproject_hits):
        # Check parents for a workspace that lists `hit` as a member.
        for parent in pyproject_hits[hit_idx + 1 :]:
            parent_data = _parse(parent / "pyproject.toml")
            if parent_data is None:
                continue
            if not _is_workspace_root(parent_data):
                continue
            members = _workspace_members(parent_data)
            for member in members:
                member_path = (parent / member).resolve()
                if member_path == hit.resolve():
                    return PythonProjectMaterialization(local_root=str(parent))
        # No workspace parent declared `hit` as a member -> `hit` wins.
        return PythonProjectMaterialization(local_root=str(hit))

    # No pyproject found; fall back to .git then to config parent.
    for c in candidates:
        if (c / ".git").exists():
            return PythonProjectMaterialization(local_root=str(c))
    return PythonProjectMaterialization(local_root=str(resolved.parent))
