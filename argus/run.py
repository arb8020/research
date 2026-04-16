#!/usr/bin/env python3
"""Argus-owned workload launcher implementation.

Public control-plane entry should come through `python -m argus run`.
This module owns launch orchestration while the lower execution/session layers
are still being cleaned up.

Important ownership boundary:

- Argus owns run identity, launch orchestration, and event/journal recording.
- Rollouts owns workload semantics.

That means Argus may record workload events such as stage markers, but it should
not define what those stages mean. Backend-specific preflights, training stage
names, and validity/invariant checks belong in Rollouts and should be emitted
into Argus as opaque workload events.

Usage:
    # Preferred public entrypoint
    python -m argus run --config examples/rl/kernelbench/grpo_01_01.py
    python -m argus run --config configs/trusted/eval_api.py

    # Local dev override
    python -m argus run --config ... --local  # Force local execution

The config file should export one of:
    - training contract:
      - config: A training config (e.g., GRPOConfig)
      - hardware: HardwareConfig (optional, defaults to local execution)
      - train(config, **kwargs): Function to run local training
    - eval contract:
      - tasks or tasks_path
      - run_spec or prepare_messages
      - score_fn or sample_scorer

Execution modes (determined by hardware.provider, with optional local dev override):
    - "local":     Run on local GPU
    - "modal":     Run on Modal sandbox (fast ~30s cold start)
    - "runpod":    Provision GPU via RunPod SSH
    - "lambdalabs": Provision GPU via Lambda Labs
    - "vast":      Provision GPU via Vast.ai

Options:
    --tui:       Launch TUI after submitting (default: fire-and-forget) [SSH only]
    --tail:      Stream logs to stdout (default: fire-and-forget) [SSH only]
    --keep-alive: Keep instance running after completion [SSH only]
    --node-id:   Reuse existing SSH instance (provider:id format)

Attach to running job:
    python -m argus monitor --attach <run_name>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shlex
import sys
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomllib

from argus.active_run import ActiveRun
from argus.model import RunKind, RunStatus


@contextmanager
def _quiet_spinner(msg: str) -> Generator[None, None, None]:
    """No-op context manager for quiet mode.

    Inner operations (like bifrost.acquire_node) log their own progress,
    so we just yield without adding wrapper messages.
    """
    yield None


@contextmanager
def _prepend_sys_path(entries: list[str]) -> Generator[None, None, None]:
    original_sys_path = list(sys.path)
    try:
        for entry in reversed(entries):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        yield None
    finally:
        sys.path[:] = original_sys_path


def _workspace_member_roots(workspace_root: Path) -> tuple[Path, ...]:
    """Return concrete package roots for uv workspace members.

    The important distinction is:
    - `workspace_root` creates namespace-package ambiguity for sibling projects
    - each member root (`.../argus`, `.../bifrost`, `.../miniray`, ...) is an
      honest import root for that package
    """

    pyproject_path = workspace_root / "pyproject.toml"
    if not pyproject_path.exists():
        return ()

    data = tomllib.loads(pyproject_path.read_text())
    members = data.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
    if not isinstance(members, list):
        return ()

    roots: list[Path] = []
    for member in members:
        if not isinstance(member, str):
            continue
        member_root = (workspace_root / member).resolve()
        if not (member_root / "pyproject.toml").exists():
            continue
        roots.append(member_root)
    return tuple(roots)


def _workspace_pythonpath_entries(
    workspace_root: Path,
    *,
    extra_entries: tuple[str, ...] = (),
) -> list[str]:
    entries = [str(root) for root in _workspace_member_roots(workspace_root)]
    entries.extend(extra_entries)
    # Keep the workspace root as a last-resort compatibility fallback for
    # subprocesses that still assume the old monorepo-root import shape. Member
    # roots stay first so real packages win before namespace-package ambiguity.
    entries.append(str(workspace_root))
    # Preserve order while deduplicating.
    return list(dict.fromkeys(entries))


def _remote_workspace_pythonpath_entries(
    *,
    local_workspace_root: Path,
    remote_workspace_root: str,
    extra_entries: tuple[str, ...] = (),
) -> list[str]:
    remote_entries = [
        f"{remote_workspace_root}/{root.relative_to(local_workspace_root).as_posix()}"
        for root in _workspace_member_roots(local_workspace_root)
    ]
    remote_entries.extend(extra_entries)
    remote_entries.append(remote_workspace_root)
    return list(dict.fromkeys(remote_entries))


def _remote_broker_credentials_env() -> dict[str, str]:
    """Project active broker credentials onto the remote job env.

    Some remote workloads legitimately provision nested broker-backed resources,
    for example KernelBench codeblock evaluators that need their own workspace
    GPU host. Keep that dependency explicit at the launch boundary instead of
    making remote workload code rediscover local credential state.
    """
    from broker.credentials import ENV_VAR_MAP, get_credentials

    provider_credentials = get_credentials()
    env_var_by_provider = {provider: env_var for env_var, provider in ENV_VAR_MAP.items()}
    forwarded: dict[str, str] = {}
    for provider, api_key in provider_credentials.items():
        env_var = env_var_by_provider.get(provider)
        if env_var is None:
            continue
        forwarded[env_var] = api_key
    return forwarded


if TYPE_CHECKING:
    from bifrost import BifrostClient, PythonProjectMaterialization
    from broker import ClientGPUInstance
    from rollouts.training.configs import DepsConfig
    from rollouts.training.multi_node import MultiNodeConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1] / "rollouts"
ARGUS_STATE_DIR = Path.home() / ".argus"
LAUNCHES_DIR = ARGUS_STATE_DIR / "launches"
REMOTE_SYSTEM_TOOLS_FEATURE = "remote-system-tools-v1"
REMOTE_UV_FEATURE = "uv"
SSH_MANAGED_VENV_DIR = "/root/.bifrost/venvs/rollouts-rl"


@dataclass(frozen=True)
class _SshBootstrapPlan:
    steps: tuple[tuple[str, str], ...]
    manifest_features: tuple[str, ...]
    manifest_groups: tuple[str, ...]


# sys.path hack: make sibling packages (miniray, bifrost, broker, etc.) importable.
#
# WHY THIS EXISTS
# ---------------
# The workspace has multiple Python packages (argus, rollouts, bifrost, broker,
# miniray, infra_utils) installed via `uv sync --extra deploy` from the workspace
# root. Locally this works. But when argus deploys a job remotely, it git-pushes
# the workspace as a bundle and then runs `argus run --local` inside it. At that
# point the remote process has no venv and no installed packages — just a directory
# tree. So the code manually inserts workspace member roots into sys.path to
# make siblings importable without installation.
#
# This also shows up in the PYTHONPATH set on the remote env (see `_deploy_and_submit`
# near `env_vars`), which is doing the same thing for the remote training process.
#
# WHAT NEEDS TO MOVE
# ------------------
# When argus owns the full run lifecycle, the launch flow becomes:
#
#   1. argus emits ATTEMPT_CREATED + ALLOCATION_BOUND events (via broker)
#   2. argus calls bifrost to deploy the workspace and run the job
#   3. bifrost installs the workspace packages properly on the remote (uv sync)
#      instead of relying on PYTHONPATH
#   4. the remote training process imports cleanly from an installed venv
#
# Step 3 is the key change: bifrost should run `uv sync` (or equivalent) as part
# of workspace setup, making the workspace packages genuinely installed rather than
# path-patched. Once that happens:
#   - This sys.path insert block goes away
#   - The PYTHONPATH line in `_deploy_and_submit` goes away
#   - The TYPE_CHECKING guards for bifrost/broker imports become real imports
#   - The rollouts/run.py __getattr__ forwarding shim can be deleted
#
# The net result: imports become honest, missing dependencies fail loudly at startup
# instead of at the first call site, and the code no longer needs to know the
# directory structure of the remote machine.
_workspace_root = REPO_ROOT.parent
if _workspace_root.exists():
    for entry in reversed(_workspace_pythonpath_entries(_workspace_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

from rollouts.image_publisher import build_or_resolve_image
from rollouts.image_spec import (
    USER_IMAGE_MANIFEST_PATH,
    ImageManifest,
    ImageSpec,
    RuntimeOverlay,
    image_manifest_for_spec,
    infer_cuda_version,
    manifest_write_command,
    stable_feature_name,
)
from rollouts.install_probes import (
    apt_install_probe_command,
    command_looks_like_install,
    python_install_probe_command,
    python_runtime_contract_snapshot_command,
    python_runtime_contract_verify_command,
)
from rollouts.launch_plan import (
    LocalInProcessLaunchPlan,
    LocalSubprocessLaunchPlan,
    build_local_workload_plan,
    resolve_workload_kind,
)
from rollouts.remote_runtime import (
    SourceSyncPolicy,
    enforce_source_sync_policy,
    materialization_plan_from_runtime,
    runtime_contract_from_hardware,
)
from rollouts.training.configs import HardwareConfig, WorkerTopologyConfig

# TODO(chiraag): This import cluster is the current control-plane leak. Argus
# should choose an execution substrate and own run/attempt lifecycle, but the
# runtime/image/materialization machinery itself should move into that substrate
# layer. See docs/design/runtime_ownership_cleanup.md.


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _new_launcher_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"launch_{timestamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


# TODO(step-2): Move _runpod_image_is_official_ssh_ready, _runpod_template_id_for_custom_image,
# _runpod_custom_image_docker_args, _should_reconcile_ssh_cuda_toolkit, _ssh_runtime_python,
# _ssh_runtime_feature_scope, and _build_ssh_bootstrap_plan into rollouts.remote_runtime
# (or a new rollouts.ssh_bootstrap module). These functions compute what commands to run
# on a remote node — they are bootstrap plan construction, not run supervision. The
# _SshBootstrapPlan dataclass moves with them. Argus becomes a caller: it receives a
# bootstrap plan from rollouts, then executes each step through bifrost.exec().
def _runpod_image_is_official_ssh_ready(image_ref: str) -> bool:
    normalized = image_ref.strip().lower()
    return normalized.startswith("runpod/")


RUNPOD_DOCS_CUSTOM_IMAGE_SSH_DOCKER_ARGS = (
    "bash -c 'apt update; "
    "DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y; "
    "mkdir -p ~/.ssh; "
    "cd ~/.ssh; "
    "chmod 700 ~/.ssh; "
    'echo "$PUBLIC_KEY" >> authorized_keys; '
    "chmod 700 authorized_keys; "
    "service ssh start; "
    "sleep infinity'"
)


def _runpod_template_id_for_custom_image() -> str | None:
    template_id = os.getenv("RUNPOD_SSH_TEMPLATE_ID")
    if template_id is None:
        return None
    normalized = template_id.strip()
    return normalized or None


def _runpod_custom_image_docker_args(image_ref: str) -> str | None:
    if _runpod_image_is_official_ssh_ready(image_ref):
        return None

    # Keep the exact docs version here for now. It is a provider-specific
    # lowering concern for custom images, not a frontend config choice.
    # The docs preconditions are:
    # - the pod must expose TCP port 22
    # - PUBLIC_KEY must be injected
    return RUNPOD_DOCS_CUSTOM_IMAGE_SSH_DOCKER_ARGS


def _should_reconcile_ssh_cuda_toolkit(custom_image: ImageSpec | None) -> bool:
    """Return whether SSH launch should mutate CUDA toolkit state on the remote.

    An explicit image means the system-level CUDA/toolchain contract is already
    part of the chosen boot substrate. The SSH bootstrap path may still create a
    managed Python environment on top, but it should not pretend to own or
    reconcile the image's CUDA toolkit state.
    """
    return custom_image is None


def _ssh_runtime_python(custom_image: ImageSpec | None) -> str:
    if custom_image is not None and custom_image.python_runtime == "image_owned":
        return custom_image.python_executable
    return f"{SSH_MANAGED_VENV_DIR}/bin/python"


def _ssh_runtime_feature_scope(custom_image: ImageSpec | None) -> str:
    if custom_image is not None and custom_image.python_runtime == "image_owned":
        return "image-owned"
    python_version = custom_image.python_version if custom_image is not None else "3.12"
    return f"managed-venv-python-{python_version}"


def _build_ssh_bootstrap_plan(
    *,
    remote_manifest: ImageManifest | None,
    custom_image: ImageSpec | None,
    custom_overlay: RuntimeOverlay | None,
    runtime_feature_scope: str,
    image_owned_runtime: bool,
    runtime_python: str,
    managed_venv_ready: bool,
    needs_cuda_upgrade: bool,
    cuda_req: tuple[int, int, int, str] | None,
    extra_python_project_roots: tuple[str, ...],
) -> _SshBootstrapPlan:
    steps: list[tuple[str, str]] = []
    manifest_features: list[str] = []
    manifest_groups: list[str] = []

    if remote_manifest is None or not remote_manifest.has_feature(REMOTE_SYSTEM_TOOLS_FEATURE):
        steps.append((
            "Installing system deps",
            "apt-get update && apt-get install -y tmux libnuma1 wget",
        ))
        manifest_features.append(REMOTE_SYSTEM_TOOLS_FEATURE)

    if remote_manifest is None or not remote_manifest.has_feature(REMOTE_UV_FEATURE):
        steps.append((
            "Installing uv",
            "curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env",
        ))
        manifest_features.append(REMOTE_UV_FEATURE)

    if not image_owned_runtime and not managed_venv_ready:
        managed_venv_feature = f"ssh-managed-venv-python-{custom_image.python_version if custom_image is not None else '3.12'}"
        if remote_manifest is None or not remote_manifest.has_feature(managed_venv_feature):
            python_version = custom_image.python_version if custom_image is not None else "3.12"
            steps.append((
                "Creating managed Python runtime",
                (
                    f"~/.local/bin/uv python install {shlex.quote(python_version)} && "
                    f"~/.local/bin/uv venv {shlex.quote(SSH_MANAGED_VENV_DIR)} "
                    f"--python {shlex.quote(python_version)}"
                ),
            ))
            manifest_features.append(managed_venv_feature)

    if needs_cuda_upgrade:
        assert cuda_req is not None, "cuda_req required when needs_cuda_upgrade is true"
        _, req_major, req_minor, _ = cuda_req
        cuda_installers = {
            (
                12,
                8,
            ): "https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run",
            (
                12,
                9,
            ): "https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run",
        }
        installer_url = cuda_installers.get((req_major, req_minor))
        if installer_url is None:
            raise RuntimeError(
                f"No CUDA installer URL configured for required toolkit {req_major}.{req_minor}"
            )
        steps.append((
            f"Upgrading CUDA toolkit to {req_major}.{req_minor}",
            f"wget -q {installer_url} -O /tmp/cuda_installer.run && "
            f"sh /tmp/cuda_installer.run --silent --toolkit && "
            f"rm /tmp/cuda_installer.run && "
            f"echo 'export PATH=/usr/local/cuda-{req_major}.{req_minor}/bin:$PATH' >> ~/.bashrc && "
            f"export PATH=/usr/local/cuda-{req_major}.{req_minor}/bin:$PATH",
        ))

    if custom_image is not None and custom_image.system_packages:
        image_apt_feature = stable_feature_name(
            "image-system-packages", custom_image.system_packages
        )
        if remote_manifest is None or not remote_manifest.has_feature(image_apt_feature):
            steps.append((
                "Installing image system packages",
                (
                    f"{_apt_install_command(custom_image.system_packages)} && "
                    f"{apt_install_probe_command('image-system-packages', packages=custom_image.system_packages)}"
                ),
            ))
            manifest_features.append(image_apt_feature)

    if custom_image is not None and custom_image.pip_packages:
        image_pip_feature = stable_feature_name(
            f"image-pip-packages-{runtime_feature_scope}",
            custom_image.pip_packages,
        )
        if remote_manifest is None or not remote_manifest.has_feature(image_pip_feature):
            steps.append((
                "Installing image Python packages",
                (
                    f"{
                        _uv_pip_install_command(
                            custom_image.pip_packages,
                            python_bin=None if image_owned_runtime else runtime_python,
                            system=image_owned_runtime,
                            index_url=custom_image.pip_index_url,
                            extra_index_url=custom_image.pip_extra_index_url,
                            pre=custom_image.pip_prerelease,
                        )
                    } && "
                    f"{python_install_probe_command('image-pip-packages', python_bin=runtime_python)} && "
                    f"{python_runtime_contract_snapshot_command('image-pip-packages', python_bin=runtime_python)}"
                ),
            ))
            manifest_features.append(image_pip_feature)

    if custom_image is not None and custom_image.build_commands:
        image_build_feature = stable_feature_name(
            f"image-build-commands-{runtime_feature_scope}",
            custom_image.build_commands,
        )
        if remote_manifest is None or not remote_manifest.has_feature(image_build_feature):
            for idx, command in enumerate(custom_image.build_commands, start=1):
                if command_looks_like_install(command):
                    command = (
                        f"{command} && "
                        f"{python_install_probe_command(f'image-build-command-{idx}', python_bin=runtime_python)} && "
                        f"{python_runtime_contract_verify_command(f'image-build-command-{idx}', python_bin=runtime_python)}"
                    )
                steps.append((f"Running image build command {idx}", command))
            manifest_features.append(image_build_feature)

    if custom_overlay is not None and custom_overlay.system_packages:
        overlay_apt_feature = stable_feature_name(
            "overlay-system-packages", custom_overlay.system_packages
        )
        if remote_manifest is None or not remote_manifest.has_feature(overlay_apt_feature):
            steps.append((
                "Installing runtime system packages",
                (
                    f"{_apt_install_command(custom_overlay.system_packages)} && "
                    f"{apt_install_probe_command('overlay-system-packages', packages=custom_overlay.system_packages)}"
                ),
            ))
            manifest_features.append(overlay_apt_feature)

    if custom_overlay is not None and custom_overlay.pip_packages:
        overlay_pip_feature = stable_feature_name(
            f"overlay-pip-packages-{runtime_feature_scope}",
            custom_overlay.pip_packages,
        )
        if remote_manifest is None or not remote_manifest.has_feature(overlay_pip_feature):
            steps.append((
                "Installing runtime Python packages",
                (
                    f"{
                        _uv_pip_install_command(
                            custom_overlay.pip_packages,
                            python_bin=None if image_owned_runtime else runtime_python,
                            system=image_owned_runtime,
                            index_url=custom_overlay.pip_index_url
                            or (custom_image.pip_index_url if custom_image else None),
                            extra_index_url=custom_overlay.pip_extra_index_url
                            or (custom_image.pip_extra_index_url if custom_image else None),
                            pre=custom_overlay.pip_prerelease
                            or (custom_image.pip_prerelease if custom_image else False),
                        )
                    } && "
                    f"{python_install_probe_command('overlay-pip-packages', python_bin=runtime_python)} && "
                    f"{python_runtime_contract_snapshot_command('overlay-pip-packages', python_bin=runtime_python)}"
                ),
            ))
            manifest_features.append(overlay_pip_feature)

    if custom_overlay is not None and custom_overlay.commands:
        overlay_cmd_feature = stable_feature_name(
            f"overlay-commands-{runtime_feature_scope}",
            custom_overlay.commands,
        )
        if remote_manifest is None or not remote_manifest.has_feature(overlay_cmd_feature):
            for idx, command in enumerate(custom_overlay.commands, start=1):
                if command_looks_like_install(command):
                    command = (
                        f"{command} && "
                        f"{python_install_probe_command(f'overlay-command-{idx}', python_bin=runtime_python)} && "
                        f"{python_runtime_contract_verify_command(f'overlay-command-{idx}', python_bin=runtime_python)}"
                    )
                steps.append((f"Running runtime overlay command {idx}", command))
            manifest_features.append(overlay_cmd_feature)

    if custom_overlay is not None:
        manifest_features.extend(custom_overlay.features)
        manifest_groups.extend(custom_overlay.installed_groups)

    if extra_python_project_roots:
        steps.append((
            "Installing extra project Python packages",
            (
                f"{_uv_pip_install_editable_command(extra_python_project_roots, python_bin=None if image_owned_runtime else runtime_python, system=image_owned_runtime, no_deps=True)} && "
                f"{python_install_probe_command('extra-python-projects', python_bin=runtime_python)} && "
                f"{python_runtime_contract_snapshot_command('extra-python-projects', python_bin=runtime_python)}"
            ),
        ))

    return _SshBootstrapPlan(
        steps=tuple(steps),
        manifest_features=tuple(manifest_features),
        manifest_groups=tuple(manifest_groups),
    )


# TODO(step-2): Move _find_config_project_root, _external_config_project_roots,
# _external_config_projects, _remote_materialized_path, and _normalize_remote_workspace_root
# into rollouts.remote_runtime. These functions resolve which project a config belongs to
# and how its paths map onto the remote workspace — that is workload compilation, not
# run supervision. They have no business being in argus.
def _find_config_project_root(config_path: Path) -> Path:
    search_roots = [config_path.parent, *config_path.parents]
    for candidate in search_roots:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return config_path.parent


def _external_config_project_roots(config_path: Path) -> tuple[Path, ...]:
    return tuple(Path(project.local_root) for project in _external_config_projects(config_path))


def _external_config_projects(config_path: Path) -> tuple[PythonProjectMaterialization, ...]:
    workspace_root = REPO_ROOT.parent.resolve()
    resolved_config = config_path.resolve()
    try:
        resolved_config.relative_to(workspace_root)
    except ValueError:
        from bifrost import PythonProjectMaterialization

        return (
            PythonProjectMaterialization(
                local_root=str(_find_config_project_root(resolved_config)),
                primary_workspace_local_root=str(workspace_root),
            ),
        )
    return ()


def _remote_materialized_path(
    *,
    local_path: Path,
    workspace_root: str,
    extra_python_projects: tuple[PythonProjectMaterialization, ...],
) -> str:
    resolved_local = local_path.resolve()
    primary_workspace_root = REPO_ROOT.parent.resolve()
    try:
        relative_to_primary = resolved_local.relative_to(primary_workspace_root)
    except ValueError as err:
        for project in extra_python_projects:
            project_root = Path(project.local_root).expanduser().resolve()
            try:
                project_relative = resolved_local.relative_to(project_root)
            except ValueError:
                continue
            return f"{project.remote_source_root(workspace_root)}/{project_relative.as_posix()}"
        raise ValueError(
            f"Path {resolved_local} is not inside the primary workspace or any extra project"
        ) from err
    return f"{workspace_root}/{relative_to_primary.as_posix()}"


def _normalize_remote_workspace_root(bifrost: BifrostClient, workspace_root: str) -> str:
    """Expand remote workspace roots before using them as argv values."""
    assert workspace_root, "workspace_root cannot be empty"
    return bifrost.expand_path(workspace_root)


def _active_launches() -> list[dict[str, Any]]:
    if not LAUNCHES_DIR.exists():
        return []
    launches: list[dict[str, Any]] = []
    for path in sorted(LAUNCHES_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        payload["_path"] = str(path)
        launches.append(payload)
    return launches


def _write_launch_record(payload: dict[str, Any]) -> Path:
    LAUNCHES_DIR.mkdir(parents=True, exist_ok=True)
    path = LAUNCHES_DIR / f"{payload['launcher_id']}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _remove_launch_record(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


from argus.event_log import (
    RunEventSinks,
    build_jsonl_run_event_sinks,
    emit_run_event,
)


@dataclass(frozen=True)
class ExecutionSpecOverrides:
    """Transitional CLI patches to the config-owned execution spec.

    `--local` remains as an explicit dev/operator override. The rest of the
    execution spec should be expressed in config-owned product types rather than
    patched from the CLI.
    """

    force_local: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ExecutionSpecOverrides:
        return cls(force_local=args.local)

    def used_flags(self) -> list[str]:
        flags: list[str] = []
        if self.force_local:
            flags.append("--local")
        return flags

    def apply_to_hardware(self, hardware: HardwareConfig) -> HardwareConfig:
        if self.force_local:
            hardware = replace(hardware, provider="local")
        return hardware

    def warn_if_used(self) -> None:
        used_flags = self.used_flags()
        if not used_flags:
            return
        print(
            "Warning: --local overrides the config-owned runtime and should stay a dev-only "
            f"escape hatch: {', '.join(used_flags)}",
            file=sys.stderr,
        )


def _read_remote_manifest(bifrost: BifrostClient) -> ImageManifest | None:
    """Load a baked or previously bootstrapped manifest from the remote node."""
    result = bifrost.exec(
        "if [ -f /etc/rollouts-image.json ]; then cat /etc/rollouts-image.json; "
        f"elif [ -f {USER_IMAGE_MANIFEST_PATH} ]; then cat {USER_IMAGE_MANIFEST_PATH}; fi"
    )
    if not result.success:
        raise RuntimeError(
            "Remote image manifest probe failed: "
            f"exit={result.exit_code} stderr={result.stderr.strip()}"
        )
    raw = result.stdout
    raw = raw.strip()
    if not raw:
        return None
    try:
        return ImageManifest.from_json(raw)
    except Exception as exc:
        raise RuntimeError(f"Remote image manifest is unreadable: {exc}") from exc


# TODO(step-2): Move _uv_pip_install_command, _uv_pip_install_editable_command,
# _apt_install_command, and _read_remote_manifest into rollouts.image_spec or
# bifrost.bootstrap. These functions build shell commands for remote package
# installation — they are part of the bootstrap plan, not argus control-plane
# logic. The move is mechanical: no callers outside _build_ssh_bootstrap_plan
# and _deploy_and_submit.
def _uv_pip_install_command(
    packages: tuple[str, ...],
    *,
    python_bin: str | None = None,
    system: bool = False,
    index_url: str | None = None,
    extra_index_url: str | None = None,
    pre: bool = False,
    extra_options: str | None = None,
) -> str:
    quoted_packages = " ".join(shlex.quote(package) for package in packages)
    parts = ["~/.local/bin/uv", "pip", "install", "--upgrade"]
    if system:
        parts.append("--system")
    elif python_bin is not None:
        parts.extend(["--python", shlex.quote(python_bin)])
    if index_url:
        parts.extend(["--index-url", shlex.quote(index_url)])
    if extra_index_url:
        parts.extend(["--extra-index-url", shlex.quote(extra_index_url)])
    if pre:
        parts.append("--pre")
    if extra_options:
        parts.append(extra_options)
    parts.append(quoted_packages)
    return " ".join(parts)


def _uv_pip_install_editable_command(
    project_roots: tuple[str, ...],
    *,
    python_bin: str | None = None,
    system: bool = False,
    no_deps: bool = False,
    extra_options: str | None = None,
) -> str:
    quoted_projects = " ".join(f"-e {shlex.quote(project_root)}" for project_root in project_roots)
    parts = ["~/.local/bin/uv", "pip", "install", "--upgrade"]
    if system:
        parts.append("--system")
    elif python_bin is not None:
        parts.extend(["--python", shlex.quote(python_bin)])
    if no_deps:
        parts.append("--no-deps")
    if extra_options:
        parts.append(extra_options)
    parts.append(quoted_projects)
    return " ".join(parts)


def _apt_install_command(packages: tuple[str, ...]) -> str:
    quoted_packages = " ".join(shlex.quote(package) for package in packages)
    return f"apt-get update && apt-get install -y {quoted_packages}"


def load_config_module(config_path: Path) -> Any:
    """Load a config module from path."""
    spec = importlib.util.spec_from_file_location("_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_config"] = module
    extra_import_roots = [str(path) for path in _external_config_project_roots(config_path)]
    with _prepend_sys_path(extra_import_roots):
        spec.loader.exec_module(module)
    return module


def _modal_workload_tags(config: Any) -> dict[str, str]:
    """Opaque workload tags supplied by rollouts config semantics."""
    tags: dict[str, str] = {}
    tags["workload"] = type(config).__name__.removesuffix("Config").lower() or "unknown"

    model = getattr(config, "model", None)
    model_name = getattr(model, "name", None)
    if model_name:
        tags["model"] = str(model_name)

    trainer = getattr(config, "trainer", None)
    trainer_backend = getattr(trainer, "backend", None)
    if trainer_backend:
        tags["backend"] = str(trainer_backend)

    inference = getattr(config, "inference", None)
    inference_backend = getattr(inference, "backend", None)
    if inference_backend:
        tags["inference_backend"] = str(inference_backend)

    return tags


def _modal_workload_request_fields(config: Any) -> tuple[str | None, str | None]:
    """Opaque workload fields that affect Modal-side caching/materialization."""
    model = getattr(config, "model", None)
    model_name = getattr(model, "name", None)

    pruning_recipe = None
    recipe_path_value = getattr(model, "pruning_recipe", None)
    if recipe_path_value:
        recipe_path = Path(recipe_path_value)
        if not recipe_path.is_absolute():
            recipe_path = REPO_ROOT / recipe_path
        if recipe_path.exists():
            pruning_recipe = recipe_path.read_text()
        else:
            logger.warning("Pruning recipe not found for Modal request: %s", recipe_path)

    return model_name, pruning_recipe


def _normalize_worker_topology(config_module: Any) -> WorkerTopologyConfig | None:
    topology = getattr(config_module, "worker_topology", None)
    if topology is not None:
        if not isinstance(topology, WorkerTopologyConfig):
            raise ValueError("config_module.worker_topology must be a WorkerTopologyConfig")
        return topology

    workload_config = getattr(config_module, "config", None)
    if workload_config is None:
        return None

    topology = getattr(workload_config, "topology", None)
    if topology is not None:
        if not isinstance(topology, WorkerTopologyConfig):
            raise ValueError("config.topology must be a WorkerTopologyConfig")
        return topology

    return None


def _argus_modal_tags(*, launcher_id: str, run_name: str, config_path: Path) -> dict[str, str]:
    """Control-plane identity tags for Modal sandboxes."""
    return {
        "launcher_id": launcher_id,
        "run_name": run_name,
        "config_basename": config_path.name,
        "provider": "modal",
        "control_plane": "argus",
    }


def _setup_run_logging(
    run_dir: Path,
    *,
    journal_name: str = "run.jsonl",
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> RunEventSinks:
    """Create run directory and return the canonical run event sinks.

    Workload code should use this one object for structured run events. Argus
    owns the durable JSONL sink; other projections can be attached later without
    changing workload call sites.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    return build_jsonl_run_event_sinks(run_dir / journal_name, on_event=on_event)


def _emit_run(log: RunEventSinks, event: str, **data: Any) -> None:
    emit_run_event(log, event, **data)


def _job_status_from_run_status(status: RunStatus) -> str:
    if status == RunStatus.PENDING:
        return "starting"
    if status == RunStatus.RUNNING:
        return "running"
    if status == RunStatus.SUCCEEDED:
        return "completed"
    if status in {RunStatus.FAILED, RunStatus.CANCELLED}:
        return "failed"
    raise AssertionError(status)


def _active_run_log_fields(active_run: ActiveRun) -> dict[str, str | None]:
    return {
        "argus_run_id": active_run.run_id,
        "argus_status": active_run.status.value,
        "argus_stage": active_run.stage,
        "argus_attempt_id": active_run.attempt_id,
    }


def _sync_registered_run(active_run: ActiveRun) -> None:
    from rollouts.jobs import update_job_node, update_job_status

    allocation = active_run.snapshot.allocation
    if allocation is not None:
        update_job_node(active_run.run_id, allocation.provider, allocation.node_id)
    update_job_status(active_run.run_id, _job_status_from_run_status(active_run.status))


def _apply_modal_run_event(active_run: ActiveRun, event: str, data: dict[str, Any]) -> None:
    """Project provider-owned Modal events into Argus' logical run state."""
    if event == "modal_sandbox_created":
        sandbox_id = data.get("sandbox_id")
        if sandbox_id:
            active_run.bind_allocation(provider="modal", node_id=str(sandbox_id))
        return
    if event == "modal_repo_sync_start":
        active_run.update_stage("modal_repo_sync")
        return
    if event == "modal_repo_synced":
        active_run.update_stage("modal_repo_synced")
        return
    if event == "modal_training_start":
        active_run.mark_running(stage="modal_training")
        return
    if event == "workload_entrypoint_started":
        active_run.update_stage("workload_entrypoint_started")
        return
    if event == "remote_artifact_projection_started":
        active_run.update_stage("artifact_projection")
        return


def _spawn_local_subprocess(
    *,
    plan: LocalSubprocessLaunchPlan,
    log: RunEventSinks,
) -> int:
    """Launch one local workload subprocess from a Rollouts-owned plan."""
    # TODO(argus-run): Extract local launch paths (eval + local training) into
    # a separate launcher module. This is a distinct state machine from remote
    # provisioning/bootstrap and should not stay interleaved in `run.py`.
    import subprocess

    output_dir = plan.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = output_dir / "stdout.log"
    stderr_log = output_dir / "stderr.log"
    command = list(plan.command)

    stdout_handle = stdout_log.open("a")
    stderr_handle = stderr_log.open("a")
    env = os.environ.copy()
    env["ROLLOUTS_OUTPUT_DIR"] = str(output_dir)
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT.parent),
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            env=env,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()

    _emit_run(
        log,
        "submit_done",
        kind=plan.kind,
        pid=proc.pid,
        output_dir=str(output_dir),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        command=command,
    )
    return proc.pid


def _run_local_entrypoint(plan: LocalInProcessLaunchPlan) -> Any:
    """Execute one Rollouts-owned in-process plan."""
    if plan.emit_startup_sentinel and os.getenv("ARGUS_EMIT_STARTUP_SENTINEL") == "1":
        print("__ARGUS_WORKLOAD_ENTRYPOINT_STARTED__", flush=True)
    return plan.run()


def _report_local_entrypoint_result(plan: LocalInProcessLaunchPlan, result: Any) -> None:
    """Render one Rollouts-owned local result summary when available."""
    if plan.render_result is None:
        return
    rendered = plan.render_result(result)
    if rendered:
        print(rendered)


def _launch_eval_monitor(
    *,
    run_dir: Path,
    tail: bool,
    fmt: str = "pretty",
) -> int:
    if tail:
        # Direct tail — no rollouts/TUI dependency
        from .tail import tail_run

        return tail_run(run_dir, fmt=fmt)

    import subprocess

    monitor_cmd = [sys.executable, "-m", "argus", "monitor", str(run_dir)]
    try:
        return subprocess.run(monitor_cmd, check=False).returncode
    except KeyboardInterrupt:
        return 0


async def _deploy_and_submit(
    script_path: str,
    node_id: str | None,
    gpu_count: int,
    gpu_type: str,
    provider: str | None = None,
    allow_dirty: bool = False,
    quiet: bool = False,
    skip_hf_token_check: bool = False,
    container_disk_gb: int = 100,
    hf_cache_dir: str = "/workspace/.cache/huggingface",
    persistent_volume_id: str | None = None,
    persistent_volume_mount_path: str = "/workspace",
    persistent_volume_location: str | None = None,
    deps: DepsConfig | None = None,
    raw_script: bool = False,
    extra_python_projects: tuple[PythonProjectMaterialization, ...] = (),
    provider_overrides: dict | None = None,
) -> tuple:
    """Provision node, deploy code, submit training job.

    Returns (bifrost_client, instance, job, run_name, remote_output_dir, workspace, console, local_run_dir).
    """
    from bifrost import GPUQuery, ProcessSpec, ReadinessProbe, ServiceSpec, acquire_node
    from broker import AccountError, ProvisionError
    from broker.types import ProvisionImage
    from pytui import Console
    from rollouts.jobs import register_job

    # Check HF_TOKEN before provisioning (downloads will be slow/rate-limited without it)
    if not skip_hf_token_check and not os.getenv("HF_TOKEN"):
        print(
            "ERROR: HF_TOKEN environment variable not set.\n"
            "\n"
            "Model downloads will be slow and rate-limited without authentication.\n"
            "Set HF_TOKEN to enable faster downloads:\n"
            "\n"
            "    export HF_TOKEN=hf_...\n"
            "\n"
            "Get your token from: https://huggingface.co/settings/tokens\n"
            "\n"
            "To proceed anyway (not recommended): --no-hf-token",
            file=sys.stderr,
        )
        sys.exit(1)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"

    # Create local run directory immediately for logging
    local_run_dir = REPO_ROOT / "results" / "rl" / run_name
    active_run = ActiveRun.create(
        run_id=run_name,
        kind=RunKind.TRAINING,
        entrypoint=script_path,
        run_dir=local_run_dir,
        name=run_name,
    )
    log = _setup_run_logging(local_run_dir)
    _emit_run(
        log,
        "run_start",
        config=script_path,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        node_id=node_id,
        **_active_run_log_fields(active_run),
    )

    # Register job in local registry BEFORE provisioning
    # Use placeholder node_id if reusing, will be updated after provision
    initial_provider = "runpod"  # Default, updated after provision
    initial_node_id = "pending"
    if node_id:
        parts = node_id.split(":", 1)
        if len(parts) == 2:
            initial_provider, initial_node_id = parts
    elif provider:
        initial_provider = provider

    register_job(
        job_id=run_name,
        provider=initial_provider,
        node_id=initial_node_id,
        config_path=script_path,
        log_path=f"results/rl/{run_name}",
    )
    _sync_registered_run(active_run)

    logs_port = 9100
    provision_image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    provision_boot_image: ProvisionImage | None = None
    provision_template_id: str | None = None
    provision_docker_args: str | None = None
    resolved_registry_image_ref: str | None = None
    if deps is None:
        raise ValueError(
            "Remote Argus launch now requires explicit hardware.deps for SSH providers. "
            "Declare a DepsConfig in the config's HardwareConfig."
        )

    if deps is not None:
        requested_image = deps.resolved_image(gpu_type)
        registry_image = None
        if not node_id:
            registry_image = build_or_resolve_image(requested_image)
        elif requested_image.source_type == "registry":
            registry_image = build_or_resolve_image(requested_image)
        else:
            raise ValueError(
                "Reusing an existing SSH node requires a registry-backed image spec. "
                "Build and push the image first, then reference it via ImageSpec.from_registry(...)."
            )

        if registry_image is not None:
            provision_image = registry_image.to_ref()
            resolved_registry_image_ref = registry_image.resolved_ref
            provision_boot_image = ProvisionImage(
                source_type="registry",
                reference=registry_image.to_ref(),
                context_dir=requested_image.context_dir,
                build_args=requested_image.build_args,
                metadata={
                    "python_version": requested_image.python_version,
                    "resolved_image_ref": registry_image.resolved_ref,
                    "credentials_ref": registry_image.credentials_ref,
                },
            )
            if provider == "runpod":
                provision_template_id = _runpod_template_id_for_custom_image()
                provision_docker_args = _runpod_custom_image_docker_args(provision_image)

    # Create console for coordinated spinner + logging output
    # In quiet mode, skip spinners and just use plain logging to stderr
    if quiet:
        console = None
        spinner = _quiet_spinner
        # Ensure logs go to stderr so agents see progress
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stderr,
            force=True,
        )
        # Silence noisy HTTP loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    else:
        console = Console()
        console.install_logging_handler(logging.getLogger())
        spinner = console.spinner  # Use interactive spinner

    source_sync_policy = SourceSyncPolicy.committed_only(
        dirty_action="warn" if allow_dirty else "fail"
    )
    enforce_source_sync_policy(source_sync_policy, repo_root=REPO_ROOT, stream=sys.stderr)

    try:
        # Acquire node - show which credentials profile is being used
        from broker.credentials import get_active_profile

        profile_name, _ = get_active_profile()
        profile_hint = f" [{profile_name}]" if profile_name else ""

        active_run.update_stage("provisioning")
        _sync_registered_run(active_run)
        if node_id:
            provision_msg = "Connecting..."
        elif provider:
            provision_msg = f"Provisioning {gpu_count}x {gpu_type} on {provider}{profile_hint}..."
        else:
            provision_msg = f"Provisioning {gpu_count}x {gpu_type}{profile_hint}..."
        _emit_run(log, "provision_start", msg=provision_msg, **_active_run_log_fields(active_run))
        try:
            with spinner(provision_msg) as spin:
                if node_id:
                    bifrost, instance = await acquire_node(node_id=node_id)
                    if spin:
                        spin.update(f"Connected to {node_id}")
                    if instance:
                        active_run.bind_allocation(
                            provider=instance.provider,
                            node_id=instance.id,
                            image_ref=resolved_registry_image_ref,
                        )
                        _sync_registered_run(active_run)
                    _emit_run(
                        log,
                        "provision_done",
                        node_id=node_id,
                        reused=True,
                        **_active_run_log_fields(active_run),
                    )
                else:
                    bifrost, instance = await acquire_node(
                        provision=GPUQuery(
                            type=gpu_type,
                            count=gpu_count,
                            min_cuda="12.8",
                            exposed_ports=(logs_port,),
                            container_disk_gb=container_disk_gb,
                            image=provision_image,
                            boot_image=provision_boot_image,
                            template_id=provision_template_id,
                            docker_args=provision_docker_args,
                            persistent_volume_id=persistent_volume_id,
                            persistent_volume_mount_path=persistent_volume_mount_path,
                            persistent_volume_location=persistent_volume_location,
                            name=f"rollouts/{run_name}",
                            provider=provider,
                            provider_overrides=provider_overrides or {},
                        )
                    )
                    node_str = f"{instance.provider}:{instance.id}" if instance else "?"
                    if spin:
                        spin.update(f"Provisioned {node_str}")
                    if instance:
                        active_run.bind_allocation(
                            provider=instance.provider,
                            node_id=instance.id,
                            image_ref=resolved_registry_image_ref,
                        )
                        _sync_registered_run(active_run)
                    _emit_run(
                        log,
                        "provision_done",
                        node_id=node_str,
                        provider=instance.provider if instance else None,
                        **_active_run_log_fields(active_run),
                    )
        except AccountError as e:
            logger.debug("AccountError details", exc_info=True)
            active_run.mark_failed(error=e.user_message())
            _sync_registered_run(active_run)
            _emit_run(
                log,
                "run_failed",
                error=e.user_message(),
                **_active_run_log_fields(active_run),
            )
            print(f"\nError: {e.user_message()}", file=sys.stderr)
            sys.exit(1)
        except ProvisionError as e:
            logger.debug("ProvisionError details", exc_info=True)
            # Surface categorized one-liner based on result
            result = e.result
            if result.credential_error:
                user_message = "Invalid API credentials. Check your API keys."
                print("\nError: Invalid API credentials. Check your API keys.", file=sys.stderr)
            elif result.no_offers_found:
                user_message = f"No {gpu_type} GPUs found. Update hardware.gpu_type in the config."
                print(
                    f"\nError: No {gpu_type} GPUs found. Update hardware.gpu_type in the config.",
                    file=sys.stderr,
                )
            elif result.all_unavailable:
                user_message = (
                    f"No {gpu_type} GPUs available right now. Try again later or update "
                    "hardware.gpu_type in the config."
                )
                print(
                    f"\nError: No {gpu_type} GPUs available right now. Try again later or update "
                    "hardware.gpu_type in the config.",
                    file=sys.stderr,
                )
            elif result.network_error:
                user_message = "Network error reaching GPU provider. Try again."
                print("\nError: Network error reaching GPU provider. Try again.", file=sys.stderr)
            else:
                user_message = f"Provisioning failed: {e}"
                print(f"\nError: Provisioning failed: {e}", file=sys.stderr)
            active_run.mark_failed(error=user_message)
            _sync_registered_run(active_run)
            _emit_run(
                log,
                "run_failed",
                error=user_message,
                **_active_run_log_fields(active_run),
            )
            sys.exit(1)

        custom_image = deps.resolved_image(gpu_type) if deps is not None else None

        # Check CUDA toolkit version compatibility and auto-upgrade if needed
        # The driver version (nvidia-smi) may be newer than the toolkit (nvcc)
        # FlashInfer/Triton JIT-compile kernels and need nvcc to support the GPU arch
        from rollouts.training.preflight import get_gpu_cuda_requirement

        cuda_req = get_gpu_cuda_requirement(gpu_type)
        needs_cuda_upgrade = False
        should_reconcile_cuda = _should_reconcile_ssh_cuda_toolkit(custom_image)
        if cuda_req is not None and should_reconcile_cuda:
            sm_version, min_major, min_minor, arch_name = cuda_req
            _emit_run(
                log,
                "cuda_check_start",
                gpu_type=gpu_type,
                arch=arch_name,
                min_cuda=f"{min_major}.{min_minor}",
                **_active_run_log_fields(active_run),
            )
            with spinner(f"Checking CUDA toolkit for {gpu_type} ({arch_name})..."):
                # Get nvcc version from remote
                try:
                    result = bifrost.exec("nvcc --version 2>/dev/null || echo 'nvcc not found'")
                    nvcc_output = result.stdout if hasattr(result, "stdout") else str(result)

                    # Parse "release X.Y" from nvcc output
                    import re

                    match = re.search(r"release (\d+)\.(\d+)", nvcc_output)
                    if match:
                        nvcc_major, nvcc_minor = int(match.group(1)), int(match.group(2))
                        if nvcc_major < min_major or (
                            nvcc_major == min_major and nvcc_minor < min_minor
                        ):
                            logger.warning(
                                f"CUDA toolkit {nvcc_major}.{nvcc_minor} is too old for {gpu_type} "
                                f"(needs {min_major}.{min_minor}+). Will upgrade during bootstrap."
                            )
                            needs_cuda_upgrade = True
                            _emit_run(
                                log,
                                "cuda_check_done",
                                nvcc_version=f"{nvcc_major}.{nvcc_minor}",
                                compatible=False,
                                will_upgrade=True,
                                **_active_run_log_fields(active_run),
                            )
                        else:
                            _emit_run(
                                log,
                                "cuda_check_done",
                                nvcc_version=f"{nvcc_major}.{nvcc_minor}",
                                compatible=True,
                                **_active_run_log_fields(active_run),
                            )
                    else:
                        # nvcc not found - will need to install
                        logger.warning(
                            "nvcc not found on remote. Will install CUDA toolkit during bootstrap."
                        )
                        needs_cuda_upgrade = True
                        _emit_run(
                            log,
                            "cuda_check_done",
                            nvcc_version="not_found",
                            will_upgrade=True,
                            **_active_run_log_fields(active_run),
                        )
                except Exception as e:
                    _emit_run(
                        log,
                        "cuda_check_done",
                        error=str(e),
                        **_active_run_log_fields(active_run),
                    )
                    logger.warning(
                        f"CUDA check failed: {e}. Will attempt toolkit install during bootstrap."
                    )
                    needs_cuda_upgrade = True
        elif cuda_req is not None:
            _, min_major, min_minor, arch_name = cuda_req
            _emit_run(
                log,
                "cuda_check_skipped",
                gpu_type=gpu_type,
                arch=arch_name,
                min_cuda=f"{min_major}.{min_minor}",
                reason="image_owned_runtime",
                **_active_run_log_fields(active_run),
            )

        # Deploy code (git sync only, no bootstrap)
        local_script_path = Path(script_path).resolve()

        active_run.update_stage("deploying")
        _sync_registered_run(active_run)
        _emit_run(log, "deploy_start", **_active_run_log_fields(active_run))
        with spinner("Deploying code..."):
            from bifrost import WorkspaceMaterializationSpec

            workspace_handle = bifrost.materialize(
                WorkspaceMaterializationSpec(
                    requested_root="~/.bifrost/workspaces/rollouts-rl",
                    allow_dirty=allow_dirty,
                    extra_python_projects=extra_python_projects,
                )
            )
            workspace = _normalize_remote_workspace_root(bifrost, workspace_handle.root)
        if active_run.snapshot.allocation is not None:
            allocation = active_run.snapshot.allocation
            active_run.bind_allocation(
                provider=allocation.provider,
                node_id=allocation.node_id,
                image_ref=allocation.image_ref,
                workspace=workspace,
            )
            _sync_registered_run(active_run)
        _emit_run(log, "deploy_done", workspace=workspace, **_active_run_log_fields(active_run))
        remote_script_path = _remote_materialized_path(
            local_path=local_script_path,
            workspace_root=workspace,
            extra_python_projects=extra_python_projects,
        )

        remote_manifest = _read_remote_manifest(bifrost)
        if remote_manifest is not None:
            _emit_run(
                log,
                "image_manifest_loaded",
                features=list(remote_manifest.features),
                installed_groups=list(remote_manifest.installed_groups),
                **_active_run_log_fields(active_run),
            )

        custom_overlay = deps.resolved_runtime_overlay() if deps is not None else None
        image_owned_runtime = (
            custom_image is not None and custom_image.python_runtime == "image_owned"
        )
        runtime_python = _ssh_runtime_python(custom_image)
        runtime_feature_scope = _ssh_runtime_feature_scope(custom_image)
        managed_venv_ready = True
        if not image_owned_runtime:
            managed_venv_ready = bifrost.exec(f"test -x {shlex.quote(runtime_python)}").success

        if deps is not None and deps.image is not None:
            logger.info(
                "Custom image spec provided for SSH runner. Registry-backed images are now passed through to provisioning; non-registry image sources still require a build/push step first."
            )

        bootstrap_plan = _build_ssh_bootstrap_plan(
            remote_manifest=remote_manifest,
            custom_image=custom_image,
            custom_overlay=custom_overlay,
            runtime_feature_scope=runtime_feature_scope,
            image_owned_runtime=image_owned_runtime,
            runtime_python=runtime_python,
            managed_venv_ready=managed_venv_ready,
            needs_cuda_upgrade=needs_cuda_upgrade,
            cuda_req=cuda_req,
            extra_python_project_roots=tuple(
                project.remote_source_root(workspace) for project in extra_python_projects
            ),
        )

        active_run.update_stage("bootstrapping")
        _sync_registered_run(active_run)
        for label, cmd in bootstrap_plan.steps:
            _emit_run(
                log, "bootstrap_step_start", label=label, **_active_run_log_fields(active_run)
            )
            with spinner(f"{label}..."):
                bifrost.exec(cmd, working_dir=workspace)
            _emit_run(log, "bootstrap_step_done", label=label, **_active_run_log_fields(active_run))

        # HuggingFace login for faster authenticated downloads
        # Token is written to ~/.cache/huggingface/token (standard HF location)
        # Using printf to avoid token appearing in shell history or ps output
        if hf_token := os.getenv("HF_TOKEN"):
            _emit_run(
                log,
                "bootstrap_step_start",
                label="HuggingFace login",
                **_active_run_log_fields(active_run),
            )
            with spinner("Logging into HuggingFace..."):
                # Use env var in subshell - token only visible to this process
                bifrost.exec(
                    f"mkdir -p {hf_cache_dir} && printf '%s' \"$HF_TOKEN\" > {hf_cache_dir}/token",
                    env={"HF_TOKEN": hf_token},
                    working_dir=workspace,
                )
            _emit_run(
                log,
                "bootstrap_step_done",
                label="HuggingFace login",
                **_active_run_log_fields(active_run),
            )

        manifest_base = remote_manifest
        if manifest_base is None:
            if custom_image is not None:
                manifest_base = image_manifest_for_spec(
                    custom_image,
                    image_name=f"ssh-{gpu_type.lower()}",
                    cuda_version=infer_cuda_version(gpu_type, custom_image.pip_index_url),
                    resolved_image_ref=resolved_registry_image_ref,
                    env={
                        "HF_HOME": hf_cache_dir,
                        "HF_HUB_ENABLE_HF_TRANSFER": "1",
                        **custom_image.env,
                    },
                    paths={"megatron_root": "/root/Megatron-LM"},
                )
            else:
                manifest_base = ImageManifest(
                    image_name=f"ssh-{gpu_type.lower()}",
                    python_version="3.12",
                    env={
                        "HF_HOME": hf_cache_dir,
                        "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    },
                    paths={"megatron_root": "/root/Megatron-LM"},
                )

        requested_cuda_version = (
            f"{cuda_req[1]}.{cuda_req[2]}" if needs_cuda_upgrade and cuda_req is not None else None
        )
        manifest_to_write = manifest_base.extended(
            features=bootstrap_plan.manifest_features,
            installed_groups=bootstrap_plan.manifest_groups,
            resolved_image_ref=resolved_registry_image_ref,
            env={
                "HF_HOME": hf_cache_dir,
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                **(custom_image.env if custom_image is not None else {}),
                **(custom_overlay.env if custom_overlay is not None else {}),
            },
            paths={"megatron_root": "/root/Megatron-LM"},
            python_version=manifest_base.python_version or "3.12",
            cuda_version=manifest_base.cuda_version or requested_cuda_version,
        )
        _emit_run(
            log,
            "image_manifest_write",
            features=list(manifest_to_write.features),
            installed_groups=list(manifest_to_write.installed_groups),
            **_active_run_log_fields(active_run),
        )
        bifrost.exec(manifest_write_command(manifest_to_write, USER_IMAGE_MANIFEST_PATH))

        # Create run output directory
        remote_output_dir = f"{workspace}/rollouts/results/rl/{run_name}"
        bifrost.exec(f"mkdir -p {remote_output_dir}")
        training_log = f"{remote_output_dir}/training.log"
        active_run.record_artifact(name="remote_training_log", path=training_log, kind="log")

        # Start LogsServer as a detached service BEFORE the training job. This
        # decouples LogsServer lifetime from the training job — if training
        # crashes, LogsServer keeps running and can serve the final logs
        # (including tracebacks).
        logs_port = 9100
        logs_dir = f"{workspace}/rollouts/results/rl/{run_name}"
        logs_service_name = f"logs-{run_name}"
        logs_log_file = f"{logs_dir}/logs_server.log"

        # Kill any stale LogsServer processes from previous runs
        bifrost.exec(f"fuser -k {logs_port}/tcp 2>/dev/null || true")
        bifrost.exec("pkill -f 'miniray.logs_server' 2>/dev/null || true")

        logs_service = bifrost.serve_service(
            ServiceSpec(
                process=ProcessSpec(
                    command="python3",
                    args=("-m", "miniray.logs_server", "--port", str(logs_port), "--dir", logs_dir),
                    cwd=workspace,
                ),
                port=logs_port,
                readiness_probe=ReadinessProbe(kind="process_alive"),
            ),
            name=logs_service_name,
            log_file=logs_log_file,
            workspace=workspace,
        )
        _emit_run(
            log,
            "logs_server_started",
            port=logs_port,
            session=logs_service.handle_id,
            log_file=logs_service.log_file,
            **_active_run_log_fields(active_run),
        )
        active_run.record_artifact(name="logs_server_log", path=logs_log_file, kind="log")
        _sync_registered_run(active_run)

        remote_workspace_member_entries = _remote_workspace_pythonpath_entries(
            local_workspace_root=_workspace_root,
            remote_workspace_root=workspace,
            extra_entries=("/root/Megatron-LM",),
        )
        remote_extra_project_entries = [
            project.remote_source_root(workspace) for project in extra_python_projects
        ]
        pythonpath_entries = list(
            dict.fromkeys([*remote_extra_project_entries, *remote_workspace_member_entries])
        )

        env_vars = {
            "PYTHONUNBUFFERED": "1",
            "ROLLOUTS_RUN_NAME": run_name,
            "ROLLOUTS_OUTPUT_DIR": f"results/rl/{run_name}",
            "ROLLOUTS_JSON_LOGS": "true",
            "HF_HOME": hf_cache_dir,
            # PYTHONPATH includes:
            # - staged extra project roots for external configs like charisma
            # - each workspace member root for sibling package imports
            # - /root/Megatron-LM for megatron.core imports
            "PYTHONPATH": ":".join(pythonpath_entries),
            # NCCL settings for multi-GPU training
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            **(custom_image.env if custom_image is not None else {}),
            **(custom_overlay.env if custom_overlay is not None else {}),
            **_remote_broker_credentials_env(),
        }

        # Submit training job
        process_command = runtime_python
        if raw_script:
            run_args = (remote_script_path,)
        else:
            run_args = (
                "-m",
                "argus.run",
                "--config",
                remote_script_path,
                "--local",
            )

        _emit_run(log, "submit_start", **_active_run_log_fields(active_run))
        with spinner(f"Starting {run_name}...") as spin:
            job = bifrost.submit(
                ProcessSpec(
                    command=process_command,
                    args=run_args,
                    cwd=f"{workspace}/rollouts",
                    env=env_vars,
                ),
                name=run_name,  # Unique per run for tmux session isolation
                log_file=training_log,
                workspace=f"{workspace}/rollouts",
            )
            if spin:
                spin.update(f"Training started ({job.tmux_session})")
        active_run.mark_running(stage="remote_submitted")
        _sync_registered_run(active_run)
        _emit_run(
            log, "submit_done", tmux_session=job.tmux_session, **_active_run_log_fields(active_run)
        )

        return (
            bifrost,
            instance,
            job,
            run_name,
            remote_output_dir,
            workspace,
            console,
            local_run_dir,
        )
    except Exception as exc:
        active_run.mark_failed(error=str(exc))
        _sync_registered_run(active_run)
        _emit_run(log, "run_failed", error=str(exc), **_active_run_log_fields(active_run))
        raise


async def _sync_and_cleanup(
    bifrost: BifrostClient,
    instance: ClientGPUInstance | None,
    run_name: str,
    remote_output_dir: str | None,
    keep_alive: bool,
) -> None:
    """Sync results from remote and optionally terminate instance.

    Currently unused - monitor handles sync/terminate internally.
    Kept for future --detach cleanup support.
    """
    logger.info("Syncing results...")
    local_results = Path("results/rl")
    local_run_dir = local_results / run_name
    local_run_dir.mkdir(parents=True, exist_ok=True)

    if remote_output_dir:
        files_to_sync = [
            "training.log",
            "config.json",
            "rollouts.jsonl",
            "sglang.log",
            "vllm.log",
        ]

        for filename in files_to_sync:
            try:
                result = bifrost.download_files(
                    remote_path=f"{remote_output_dir}/{filename}",
                    local_path=str(local_run_dir / filename),
                    recursive=False,
                )
                if result and result.success:
                    logger.info("Synced: %s/%s", run_name, filename)
            except Exception as exc:
                logger.warning(
                    "Failed to sync file %s for run %s: %s",
                    filename,
                    run_name,
                    exc,
                )

    if instance:
        if not keep_alive:
            logger.info("Terminating instance %s:%s...", instance.provider, instance.id)
            await instance.terminate()
        else:
            logger.info("Instance kept alive: %s:%s", instance.provider, instance.id)
            logger.info("Reuse with: --node-id %s:%s", instance.provider, instance.id)


def _modal_provider_overrides(runtime: Any) -> dict:
    """Build provider_overrides dict for Modal from a RuntimeContract.

    Translates modal_volume_mounts and modal_snapshot_registry from the runtime
    contract into ModalVolumeMount / ModalFilesystemSnapshot objects that
    ProvisionRequest understands, quarantined in provider_overrides so GPUQuery
    stays provider-agnostic.

    Returns empty dict for non-Modal providers or when no Modal-specific config
    is set.

    TODO(broker): when other providers get first-class overrides (RunPod
    template_id, Vast.ai bid_price), add similar helpers here and consolidate
    into a single _provider_overrides(runtime) dispatch.
    """
    if runtime.provider != "modal":
        return {}

    overrides: dict = {}

    try:
        from broker.broker.types import ModalFilesystemSnapshot, ModalVolumeMount
    except ImportError:
        return {}

    if runtime.modal_volume_mounts:
        overrides["modal_volumes"] = [
            ModalVolumeMount(volume_name=name, mount_path=path)
            for name, path in runtime.modal_volume_mounts
        ]

    if runtime.modal_snapshot_registry is not None:
        registry_name, registry_key = runtime.modal_snapshot_registry
        overrides["modal_snapshot"] = ModalFilesystemSnapshot(
            snapshot_registry_name=registry_name,
            snapshot_registry_key=registry_key,
        )

    return overrides


async def run_remote(
    script_path: str,
    keep_alive: bool = False,
    node_id: str | None = None,
    tui: bool = False,
    gpu_count: int = 1,
    gpu_type: str = "A100",
    tail: bool = False,
    provider: str | None = None,
    allow_dirty: bool = False,
    quiet: bool = False,
    skip_hf_token_check: bool = False,
    container_disk_gb: int = 100,
    hf_cache_dir: str = "/workspace/.cache/huggingface",
    persistent_volume_id: str | None = None,
    persistent_volume_mount_path: str = "/workspace",
    persistent_volume_location: str | None = None,
    deps: DepsConfig | None = None,
    raw_script: bool = False,
    block: bool = False,
    extra_python_projects: tuple[PythonProjectMaterialization, ...] = (),
    provider_overrides: dict | None = None,
) -> None:
    """Run training script on remote GPU via bifrost."""
    # TODO(step-2): Move _deploy_and_submit and run_remote into
    # rollouts.training.ssh_launcher (or rollouts.launchers.ssh). After step-2
    # extractions (bootstrap plan, install commands, config project resolution),
    # what remains is: acquire_node → materialize → exec bootstrap steps →
    # submit job → stream logs. That is a bifrost orchestration concern that
    # belongs in rollouts, not in argus. Argus becomes a thin caller:
    # create ActiveRun, call rollouts.launchers.ssh.launch(...), record events.
    (
        bifrost,
        instance,
        job,
        run_name,
        remote_output_dir,
        workspace,
        console,
        local_run_dir,
    ) = await _deploy_and_submit(
        script_path=script_path,
        node_id=node_id,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        provider=provider,
        allow_dirty=allow_dirty,
        quiet=quiet,
        skip_hf_token_check=skip_hf_token_check,
        container_disk_gb=container_disk_gb,
        hf_cache_dir=hf_cache_dir,
        persistent_volume_id=persistent_volume_id,
        persistent_volume_mount_path=persistent_volume_mount_path,
        persistent_volume_location=persistent_volume_location,
        deps=deps,
        raw_script=raw_script,
        extra_python_projects=extra_python_projects,
        provider_overrides=provider_overrides or {},
    )

    assert instance is not None, "run_remote requires a provisioned instance"
    node_id_str = f"{instance.provider}:{instance.id}"

    # LogsServer is now started in _deploy_and_submit() before the training job,
    # so it survives training crashes and can serve final logs/tracebacks.

    logger.info("Training submitted: %s", run_name)
    logger.info("  Node:   %s", node_id_str)
    logger.info("  Local:  results/rl/%s/", run_name)

    # Start background sync daemon so logs appear locally
    # This lets agents `tail -f results/rl/<run_name>/training.log`
    import shutil
    import subprocess

    sync_session = f"sync_{run_name}"

    if shutil.which("tmux"):
        # Start sync daemon in a detached tmux session
        # cd to REPO_ROOT so logs sync to rollouts/results/rl/
        sync_cmd = [
            "tmux",
            "new",
            "-d",
            "-s",
            sync_session,
            f"cd {REPO_ROOT} && {sys.executable} -m argus monitor --attach {run_name} --sync-only",
        ]
        subprocess.run(sync_cmd, check=False, capture_output=True)
        local_log_path = REPO_ROOT / "results" / "rl" / run_name / "training.log"
        logger.info("Syncing to: %s", local_log_path)
    else:
        logger.info("Install tmux for automatic log sync")

    # Default: fire-and-forget (print attach instructions and exit)
    if not tui and not tail and not block:
        if keep_alive:
            logger.info("  (instance will stay alive)")
        else:
            logger.info("  (instance will terminate when job completes)")
        return

    if block:
        tail = True

    # --tui or --tail: launch monitor
    import subprocess

    if tui:
        logger.info("Launching TUI...")
        # Clean up logging handler before launching TUI (it has its own output)
        console.remove_logging_handlers()
        # If no handlers remain, Python's logging.lastResort will still emit WARNING+
        # to stderr, corrupting the TUI. Install a NullHandler to keep the terminal clean.
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            root_logger.addHandler(logging.NullHandler())

    monitor_cmd = [sys.executable, "-m", "argus", "monitor", "--attach", run_name]
    if tail:
        monitor_cmd.append("--tail")
    if keep_alive:
        monitor_cmd.append("--keep-alive")
    else:
        monitor_cmd.append("--terminate")
    subprocess.run(monitor_cmd, check=False)
    # Note: monitor handles final sync and terminate internally


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an Argus workload (training or eval)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Training config
    python -m argus run --config examples/rl/kernelbench/grpo_01_01.py

    # Eval config
    python -m argus run --config configs/prime_ci/reverse_text/eval_api.py

    # Local dev override
    python -m argus run --config ... --local
        """,
    )
    parser.add_argument("--config", required=True, help="Path to config file")

    # `--local` stays as an explicit dev/operator escape hatch. The rest of the
    # execution spec should come from config-owned product types.
    parser.add_argument("--local", action="store_true", help="Force local execution")

    # Remote execution options
    parser.add_argument("--node-id", type=str, help="Reuse existing instance (provider:id)")
    parser.add_argument(
        "--tui", action="store_true", help="Launch TUI after submitting (default: fire-and-forget)"
    )
    parser.add_argument(
        "--tail", action="store_true", help="Stream logs to stdout (default: fire-and-forget)"
    )
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument(
        "--modal-cleanup-scope",
        choices=["app", "tag", "run", "none"],
        default="tag",
        help="Modal pre-create sandbox cleanup scope (default: tag)",
    )
    parser.add_argument(
        "--force-deploy-committed",
        action="store_true",
        help="Proceed despite uncommitted changes (only committed code is deployed)",
    )
    parser.add_argument(
        "--spinners",
        action="store_true",
        help="Enable interactive spinners (default: plain text logging for agents)",
    )
    parser.add_argument(
        "--no-hf-token",
        action="store_true",
        help="Skip HF_TOKEN check (downloads will be slower and rate-limited)",
    )

    # Local execution
    parser.add_argument("--max-samples", type=int, help="Limit dataset size (local only)")

    args = parser.parse_args(argv)
    launcher_id = _new_launcher_id()
    execution_spec_overrides = ExecutionSpecOverrides.from_args(args)
    execution_spec_overrides.warn_if_used()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        repo_relative = REPO_ROOT / config_path
        workspace_relative = REPO_ROOT.parent / config_path
        if repo_relative.exists():
            config_path = repo_relative
        else:
            config_path = workspace_relative

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    # Load config module
    config_module = load_config_module(config_path)
    try:
        workload_kind = resolve_workload_kind(config_module, config_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # Get hardware config (default to local if not specified)
    worker_topology = _normalize_worker_topology(config_module)
    hardware: HardwareConfig = (
        worker_topology.hardware
        if worker_topology is not None
        else getattr(config_module, "hardware", HardwareConfig(provider="local"))
    )
    workload_config = getattr(config_module, "config", None)
    # TODO(step-3): This loader/merge path is assembling a LaunchableExperiment
    # inline from module exports, CLI overrides, and service-scoped deps. Once
    # LaunchableExperiment is an explicit type, this block becomes a single call:
    #   experiment = rollouts.compile(config_module, hardware, overrides)
    # and argus stops needing to know about WorkerTopologyConfig, DepsConfig,
    # service_runtime_layout, or any other workload compilation details.

    trainer_service_deps = None
    inference_service_deps = None
    service_runtime_layout = (
        worker_topology.service_runtime_layout if worker_topology is not None else "shared_env"
    )
    if worker_topology is not None and workload_config is not None:
        trainer = getattr(workload_config, "trainer", None)
        inference = getattr(workload_config, "inference", None)
        if trainer is not None:
            trainer_service_deps = getattr(trainer, "deps", None)
        if inference is not None:
            inference_service_deps = getattr(inference, "deps", None)
    elif workload_config is not None:
        trainer = getattr(workload_config, "trainer", None)
        inference = getattr(workload_config, "inference", None)
        trainer_service_deps = getattr(trainer, "deps", None)
        inference_service_deps = getattr(inference, "deps", None)
        service_runtime_layout = getattr(workload_config, "service_runtime_layout", "shared_env")

    hardware = execution_spec_overrides.apply_to_hardware(hardware)
    if service_runtime_layout not in {"shared_env", "split_env"}:
        raise ValueError(
            f"Unknown service_runtime_layout={service_runtime_layout!r}. "
            "Use 'shared_env' or 'split_env'."
        )

    if service_runtime_layout == "split_env":
        # Trainer and inference server run in separate venvs on the same machine.
        # For same base image: one Modal image with two venvs layered on top.
        # For different base image: two separate sandboxes (only Modal for now).
        # The trainer_service_deps drive the image; inference_service_deps are
        # passed through to bifrost which layers INFERENCE_VENV_DIR on top and
        # sets ROLLOUTS_INFERENCE_PYTHON in the trainer subprocess environment.
        # For RunPod: not yet implemented (falls through to the SSH launcher which
        # will raise if it encounters inference_service_deps without shared_env).
        if trainer_service_deps is not None:
            hardware = replace(hardware, deps=trainer_service_deps)
        # inference_service_deps passed separately to ModalExecutionRequest below.
    elif trainer_service_deps is not None or inference_service_deps is not None:
        if trainer_service_deps is None:
            shared_deps = inference_service_deps
        elif inference_service_deps is None:
            shared_deps = trainer_service_deps
        else:
            shared_deps = trainer_service_deps.merged_with(inference_service_deps)

        if shared_deps is not None:
            hardware = replace(hardware, deps=shared_deps)

    runtime = runtime_contract_from_hardware(hardware)
    materialization = materialization_plan_from_runtime(runtime)
    extra_python_projects = _external_config_projects(config_path)
    extra_source_roots = tuple(Path(project.local_root) for project in extra_python_projects)

    launch_record = {
        "launcher_id": launcher_id,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "argv": sys.argv if argv is None else ["argus", "run", *argv],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "config_path": str(config_path),
        "provider": runtime.provider,
        "gpu_type": runtime.gpu_type,
        "gpu_count": runtime.gpu_count,
    }
    launch_record_path = _write_launch_record(launch_record)

    duplicates = [
        launch
        for launch in _active_launches()
        if launch.get("launcher_id") != launcher_id
        and launch.get("config_path") == str(config_path)
        and launch.get("provider") == runtime.provider
        and isinstance(launch.get("pid"), int)
        and _process_alive(int(launch["pid"]))
    ]

    print(
        f"Launcher: {launcher_id} pid={os.getpid()} record={launch_record_path}",
        flush=True,
    )
    print(f"Config: {config_path}")
    print(f"Hardware: {runtime.gpu_count}x {runtime.gpu_type} on {runtime.provider}")
    if duplicates:
        print("Warning: other active launchers for this config/provider:", file=sys.stderr)
        for launch in duplicates:
            print(
                f"  - {launch['launcher_id']} pid={launch['pid']} started_at={launch.get('started_at', '')}",
                file=sys.stderr,
            )

    # Check for multi-node config (only if not forced to local)
    multi_node: MultiNodeConfig | None = None
    if not args.local:
        multi_node = getattr(config_module, "multi_node", None)

    try:
        # TODO(step-3): Replace this dispatch block with two stages separated by
        # a LaunchableExperiment product type (see docs/design/ownership_and_launch_boundary.md):
        #
        #   stage 1 — compile (rollouts owns):
        #     experiment = rollouts.compile(config_module, hardware, overrides)
        #     → LaunchableExperiment(runtime, source_snapshot, bootstrap, command, artifacts, lifecycle)
        #
        #   stage 2 — supervise (argus owns):
        #     active_run = ActiveRun.create(...)
        #     launcher = select_launcher(experiment.runtime.provider)
        #     launcher.launch(experiment, active_run)
        #
        # Each per-provider branch below does both stages inline and duplicates
        # ActiveRun.create() + event setup. Once LaunchableExperiment exists,
        # the duplication collapses and switching provider changes procurement
        # only (step 3 of ownership_and_launch_boundary.md cleanup sequence).
        # Dispatch based on workload/provider
        if workload_kind == "evaluation":
            if args.node_id or multi_node is not None:
                raise ValueError(
                    "Argus evaluation launch does not yet support explicit node reuse or "
                    "multi-node execution."
                )

        local_workload_plan = build_local_workload_plan(
            config_module=config_module,
            config_path=config_path,
            repo_root=REPO_ROOT,
            max_samples=args.max_samples,
            force_deploy_committed=args.force_deploy_committed,
            python_executable=sys.executable,
            stream_run_events=os.getenv("ARGUS_RUN_EVENT_STREAM") == "1",
            provider=runtime.provider,
            gpu_type=runtime.gpu_type,
            gpu_count=runtime.gpu_count,
        )
        if isinstance(local_workload_plan, LocalSubprocessLaunchPlan):
            local_run_dir = local_workload_plan.run_dir
            run_name = local_workload_plan.run_name
            log = _setup_run_logging(local_run_dir, journal_name=local_workload_plan.journal_name)
            _emit_run(
                log,
                "run_start",
                launcher_id=launcher_id,
                kind="evaluation",
                provider="local",
                config=str(config_path),
                output_dir=str(local_run_dir),
            )
            pid = _spawn_local_subprocess(plan=local_workload_plan, log=log)
            print(f"Evaluation submitted: {run_name}")
            print(f"  PID:    {pid}")
            print(f"  Local:  results/eval/{run_name}/")
            print()

            if not args.tui and not args.tail:
                return 0

            return _launch_eval_monitor(
                run_dir=local_run_dir,
                tail=args.tail,
                fmt=getattr(args, "format", "pretty"),
            )
        if isinstance(local_workload_plan, LocalInProcessLaunchPlan):
            result = _run_local_entrypoint(local_workload_plan)
            _report_local_entrypoint_result(local_workload_plan, result)
            return 0

        if workload_kind == "training":
            train_fn = getattr(config_module, "train", None)
            if not callable(train_fn):
                print(
                    f"Training config {config_path} must export callable train(config, **kwargs)",
                    file=sys.stderr,
                )
                return 1

        if multi_node is not None:
            # Multi-node distributed training
            import trio

            from rollouts.training.multi_node import launch_multi_node_training

            print(f"Multi-node: {multi_node.num_nodes} nodes × {multi_node.gpus_per_node} GPUs")
            print(f"  Inference: {multi_node.total_inference_engines} engines")
            print(f"  Training: {multi_node.total_trainer_gpus} FSDP ranks")

            async def _run_multi_node() -> None:
                allocation = await launch_multi_node_training(
                    config=multi_node,
                    train_config=config_module.config,
                )
                print(f"\nCluster launched: {allocation.fsdp_world_size} FSDP ranks")
                print(f"Inference endpoints: {allocation.all_inference_endpoints}")
                print("\nMonitor with:")
                for node in allocation.nodes:
                    print(f"  ssh root@{node.public_ip} tmux attach -t trainer_0")

            trio.run(_run_multi_node)

        elif runtime.provider == "modal":
            # TODO(step-3): This modal path builds a ModalExecutionRequest inline —
            # that is a LaunchableExperiment with a modal-specific shape. After step-3,
            # rollouts.compile() produces a LaunchableExperiment and argus calls
            # rollouts.launchers.modal.launch(experiment, active_run). Switching
            # provider from "modal" to "runpod" will then change only procurement,
            # not the entire event-recording + supervision stack.
            # Modal execution (fast cold start)
            import json

            import trio

            from rollouts.jobs import register_job

            if workload_kind == "benchmark":
                # Benchmark run
                from rollouts.inference.benchmark.runner import run_benchmark

                assert hardware.deps is not None  # Validated by HardwareConfig
                result = trio.run(
                    run_benchmark,
                    config_module.config,
                    hardware.deps,
                    hardware.gpu_type,
                    hardware.gpu_count,
                )
                # Output results
                print(json.dumps(result.to_dict(), indent=2))
            else:
                # Training run
                from bifrost import ModalExecutionRequest, run_modal_request

                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                run_name = f"run_{timestamp}"
                local_run_dir = REPO_ROOT / "results" / "rl" / run_name
                register_job(
                    job_id=run_name,
                    provider="modal",
                    node_id="pending",
                    config_path=str(config_path),
                    log_path=f"results/rl/{run_name}",
                )
                active_run = ActiveRun.create(
                    run_id=run_name,
                    kind=RunKind.TRAINING,
                    entrypoint=str(config_path),
                    run_dir=local_run_dir,
                    name=run_name,
                )

                def _project_modal_event(event: str, data: dict[str, Any]) -> None:
                    _apply_modal_run_event(active_run, event, data)
                    _sync_registered_run(active_run)

                log = _setup_run_logging(local_run_dir, on_event=_project_modal_event)
                _emit_run(
                    log,
                    "run_start",
                    launcher_id=launcher_id,
                    provider="modal",
                    config=str(config_path),
                    gpu_count=runtime.gpu_count,
                    gpu_type=runtime.gpu_type,
                    argus_run_id=active_run.run_id,
                    argus_status=active_run.status.value,
                    argus_attempt_id=active_run.attempt_id,
                )

                modal_tags = _argus_modal_tags(
                    launcher_id=launcher_id,
                    run_name=run_name,
                    config_path=config_path,
                )
                workload_tags = _modal_workload_tags(config_module.config)
                model_name, pruning_recipe = _modal_workload_request_fields(config_module.config)
                for reserved_key in modal_tags:
                    workload_tags.pop(reserved_key, None)
                modal_request = ModalExecutionRequest(
                    config_path=str(config_path),
                    runtime=runtime,
                    materialization=materialization,
                    extra_source_roots=tuple(
                        project.local_root for project in extra_python_projects
                    ),
                    keep_alive=args.keep_alive,
                    cleanup_scope=args.modal_cleanup_scope,
                    run_name=run_name,
                    model_name=model_name,
                    pruning_recipe=pruning_recipe,
                    run_logger=log,
                    source_sync_policy=SourceSyncPolicy.committed_only(
                        dirty_action="warn" if args.force_deploy_committed else "fail"
                    ),
                    tags={**modal_tags, **workload_tags},
                    inference_deps=inference_service_deps
                    if service_runtime_layout == "split_env"
                    else None,
                )
                for extra_root in extra_source_roots:
                    enforce_source_sync_policy(
                        modal_request.source_sync_policy,
                        repo_root=extra_root,
                        stream=sys.stderr,
                    )
                _emit_run(log, "modal_submit_dispatch")
                results = trio.run(run_modal_request, modal_request)
                if not results.get("success"):
                    active_run.mark_failed(
                        exit_code=results.get("exit_code"),
                        error=str(results.get("stderr") or ""),
                    )
                    _sync_registered_run(active_run)
                    _emit_run(
                        log,
                        "run_failed",
                        exit_code=results.get("exit_code"),
                        stderr=results.get("stderr"),
                        argus_run_id=active_run.run_id,
                        argus_status=active_run.status.value,
                        argus_stage=active_run.stage,
                        argus_attempt_id=active_run.attempt_id,
                    )
                    return 1
                active_run.mark_succeeded(exit_code=results.get("exit_code"))
                _sync_registered_run(active_run)
                _emit_run(
                    log,
                    "run_completed",
                    exit_code=results.get("exit_code"),
                    argus_run_id=active_run.run_id,
                    argus_status=active_run.status.value,
                    argus_stage=active_run.stage,
                    argus_attempt_id=active_run.attempt_id,
                )

        elif runtime.provider in ("runpod", "lambdalabs", "vast") or args.node_id:
            # Remote execution via SSH
            import trio

            # TODO(event-inspection): Make structured event inspection first-class
            # in argus so `--tail` is optional UI sugar, not the debugging
            # substrate. The real source of truth should be the run journal plus
            # bifrost lifecycle JSONL, queryable directly by run/handle id.
            trio.run(
                run_remote,
                str(config_path),
                args.keep_alive,
                args.node_id,
                args.tui,
                runtime.gpu_count,
                runtime.gpu_type,
                args.tail,
                runtime.provider if runtime.provider != "local" else None,
                args.force_deploy_committed,
                not args.spinners,  # quiet=True by default, --spinners to enable
                args.no_hf_token,
                runtime.container_disk_gb,
                runtime.hf_cache_dir,
                runtime.persistent_volume_id,
                runtime.persistent_volume_mount_path,
                runtime.persistent_volume_location,
                runtime.deps,
                False,  # raw_script
                False,  # block
                extra_python_projects,
                _modal_provider_overrides(runtime),
            )

        else:
            raise ValueError(
                f"No Argus launcher available for workload_kind={workload_kind!r} "
                f"on provider={runtime.provider!r}"
            )

        return 0
    finally:
        _remove_launch_record(launch_record_path)


if __name__ == "__main__":
    sys.exit(main())


def run_main(argv: list[str] | None = None) -> int:
    """Compatibility entry for the Argus CLI."""
    return main(argv)
