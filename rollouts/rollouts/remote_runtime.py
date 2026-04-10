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
from typing import Literal, TextIO

from .training.configs import DepsConfig, HardwareConfig


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

    # Modal-specific storage. Ignored for non-Modal providers.
    # TODO(broker): once PersistentVolumeAttachment supports Modal's name-based
    # volume semantics, fold modal_volume_mounts into persistent_volume and
    # modal_snapshot_registry into a BootImageSnapshot field here.
    modal_volume_mounts: tuple[tuple[str, str], ...] = ()
    modal_snapshot_registry: tuple[str, str] | None = None


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
        modal_volume_mounts=hardware.modal_volume_mounts,
        modal_snapshot_registry=hardware.modal_snapshot_registry,
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
