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

from rollouts.config_contracts import validate_eval_config_module, validate_train_config_module
from rollouts.image_publisher import build_or_resolve_image
from rollouts.image_spec import (
    USER_IMAGE_MANIFEST_PATH,
    ImageManifest,
    ImageSpec,
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
from rollouts.remote_runtime import (
    SourceSyncPolicy,
    enforce_source_sync_policy,
    materialization_plan_from_runtime,
    runtime_contract_from_hardware,
)
from rollouts.training.configs import HardwareConfig

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


from rollouts.run_logger import JsonlEventSink, RunLogger, stream_run_logger


class _RunLogger(RunLogger):
    pass


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
    extra_options: str | None = None,
) -> str:
    quoted_projects = " ".join(f"-e {shlex.quote(project_root)}" for project_root in project_roots)
    parts = ["~/.local/bin/uv", "pip", "install", "--upgrade"]
    if system:
        parts.append("--system")
    elif python_bin is not None:
        parts.extend(["--python", shlex.quote(python_bin)])
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


def _classify_config_module(config_module: Any, config_path: Path) -> str:
    """Classify a config module by the runner contract it satisfies."""
    # TODO(argus-run): Move config loading + contract classification into a
    # dedicated workload-resolution module. `run.py` should parse CLI args and
    # dispatch on an already-resolved workload kind, not own config contract
    # semantics directly.
    train_error: ValueError | None = None
    eval_error: ValueError | None = None

    try:
        validate_train_config_module(config_module, config_path)
        return "training"
    except ValueError as exc:
        train_error = exc

    try:
        validate_eval_config_module(config_module, config_path)
        return "evaluation"
    except ValueError as exc:
        eval_error = exc

    raise ValueError(
        f"Config {config_path} is neither a valid training config nor a valid eval config.\n"
        f"Training contract error: {train_error}\n"
        f"Eval contract error: {eval_error}"
    )


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
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> _RunLogger:
    """Create run directory and return the canonical structured run logger.

    Workload code should use this one object for structured run events. Argus
    owns the durable JSONL sink; other projections can be attached later without
    changing workload call sites.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.jsonl"
    jsonl_sink = JsonlEventSink(log_file)

    def _emit_event(event: str, **data: Any) -> None:
        jsonl_sink(event, **data)
        if on_event is not None:
            on_event(event, data)

    # Provider-owned projections sometimes need to append directly to the
    # canonical parent journal without re-entering this callback path.
    _emit_event.log_file = log_file  # type: ignore[attr-defined]
    return RunLogger(emit_event=_emit_event)


def _spawn_eval_subprocess(
    *,
    config_path: Path,
    output_dir: Path,
    max_samples: int | None,
    log: _RunLogger,
) -> int:
    """Launch rollouts.eval.run as a detached local subprocess."""
    # TODO(argus-run): Extract local launch paths (eval + local training) into
    # a separate launcher module. This is a distinct state machine from remote
    # provisioning/bootstrap and should not stay interleaved in `run.py`.
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = output_dir / "stdout.log"
    stderr_log = output_dir / "stderr.log"
    command = [
        sys.executable,
        "-m",
        "rollouts.eval.run",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    if max_samples is not None:
        command.extend(["--limit", str(max_samples)])

    stdout_handle = stdout_log.open("a")
    stderr_handle = stderr_log.open("a")
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT.parent),
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            env=os.environ.copy(),
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()

    log(
        "submit_done",
        kind="evaluation",
        pid=proc.pid,
        output_dir=str(output_dir),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        command=command,
    )
    return proc.pid


def _launch_eval_monitor(
    *,
    run_dir: Path,
    tail: bool,
) -> int:
    import subprocess

    monitor_cmd = [sys.executable, "-m", "argus", "monitor", str(run_dir)]
    if tail:
        monitor_cmd.append("--tail")
    return subprocess.run(monitor_cmd, check=False).returncode


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
) -> tuple:
    """Provision node, deploy code, submit training job.

    Returns (bifrost_client, instance, job, run_name, remote_output_dir, workspace, console, local_run_dir).
    """
    from bifrost import GPUQuery, ProcessSpec, ReadinessProbe, ServiceSpec, acquire_node
    from broker import AccountError, ProvisionError
    from broker.types import ProvisionImage
    from pytui import Console
    from rollouts.jobs import register_job, update_job_node

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
    log = _setup_run_logging(local_run_dir)
    log("run_start", config=script_path, gpu_count=gpu_count, gpu_type=gpu_type, node_id=node_id)

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

    # Acquire node - show which credentials profile is being used
    from broker.credentials import get_active_profile

    profile_name, _ = get_active_profile()
    profile_hint = f" [{profile_name}]" if profile_name else ""

    if node_id:
        provision_msg = "Connecting..."
    elif provider:
        provision_msg = f"Provisioning {gpu_count}x {gpu_type} on {provider}{profile_hint}..."
    else:
        provision_msg = f"Provisioning {gpu_count}x {gpu_type}{profile_hint}..."
    log("provision_start", msg=provision_msg)
    try:
        with spinner(provision_msg) as spin:
            if node_id:
                bifrost, instance = await acquire_node(node_id=node_id)
                if spin:
                    spin.update(f"Connected to {node_id}")
                log("provision_done", node_id=node_id, reused=True)
                # Update job registry with actual node info
                if instance:
                    update_job_node(run_name, instance.provider, instance.id)
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
                    )
                )
                node_str = f"{instance.provider}:{instance.id}" if instance else "?"
                if spin:
                    spin.update(f"Provisioned {node_str}")
                log(
                    "provision_done",
                    node_id=node_str,
                    provider=instance.provider if instance else None,
                )
                # Update job registry with actual node info
                if instance:
                    update_job_node(run_name, instance.provider, instance.id)
    except AccountError as e:
        logger.debug("AccountError details", exc_info=True)
        print(f"\nError: {e.user_message()}", file=sys.stderr)
        sys.exit(1)
    except ProvisionError as e:
        logger.debug("ProvisionError details", exc_info=True)
        # Surface categorized one-liner based on result
        result = e.result
        if result.credential_error:
            print("\nError: Invalid API credentials. Check your API keys.", file=sys.stderr)
        elif result.no_offers_found:
            print(
                f"\nError: No {gpu_type} GPUs found. Update hardware.gpu_type in the config.",
                file=sys.stderr,
            )
        elif result.all_unavailable:
            print(
                f"\nError: No {gpu_type} GPUs available right now. Try again later or update "
                "hardware.gpu_type in the config.",
                file=sys.stderr,
            )
        elif result.network_error:
            print("\nError: Network error reaching GPU provider. Try again.", file=sys.stderr)
        else:
            print(f"\nError: Provisioning failed: {e}", file=sys.stderr)
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
        log(
            "cuda_check_start",
            gpu_type=gpu_type,
            arch=arch_name,
            min_cuda=f"{min_major}.{min_minor}",
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
                        log(
                            "cuda_check_done",
                            nvcc_version=f"{nvcc_major}.{nvcc_minor}",
                            compatible=False,
                            will_upgrade=True,
                        )
                    else:
                        log(
                            "cuda_check_done",
                            nvcc_version=f"{nvcc_major}.{nvcc_minor}",
                            compatible=True,
                        )
                else:
                    # nvcc not found - will need to install
                    logger.warning(
                        "nvcc not found on remote. Will install CUDA toolkit during bootstrap."
                    )
                    needs_cuda_upgrade = True
                    log("cuda_check_done", nvcc_version="not_found", will_upgrade=True)
            except Exception as e:
                log("cuda_check_done", error=str(e))
                logger.warning(
                    f"CUDA check failed: {e}. Will attempt toolkit install during bootstrap."
                )
                needs_cuda_upgrade = True
    elif cuda_req is not None:
        _, min_major, min_minor, arch_name = cuda_req
        log(
            "cuda_check_skipped",
            gpu_type=gpu_type,
            arch=arch_name,
            min_cuda=f"{min_major}.{min_minor}",
            reason="image_owned_runtime",
        )

    # Deploy code (git sync only, no bootstrap)
    local_script_path = Path(script_path).resolve()

    log("deploy_start")
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
    log("deploy_done", workspace=workspace)
    remote_script_path = _remote_materialized_path(
        local_path=local_script_path,
        workspace_root=workspace,
        extra_python_projects=extra_python_projects,
    )

    remote_manifest = _read_remote_manifest(bifrost)
    if remote_manifest is not None:
        log(
            "image_manifest_loaded",
            features=list(remote_manifest.features),
            installed_groups=list(remote_manifest.installed_groups),
        )

    custom_overlay = deps.resolved_runtime_overlay() if deps is not None else None
    image_owned_runtime = custom_image is not None and custom_image.python_runtime == "image_owned"
    runtime_python = _ssh_runtime_python(custom_image)
    runtime_feature_scope = _ssh_runtime_feature_scope(custom_image)
    managed_venv_ready = True
    if not image_owned_runtime:
        managed_venv_ready = bifrost.exec(f"test -x {shlex.quote(runtime_python)}").success

    if deps is not None and deps.image is not None:
        logger.info(
            "Custom image spec provided for SSH runner. Registry-backed images are now passed through to provisioning; non-registry image sources still require a build/push step first."
        )

    # Bootstrap steps — each gets its own spinner with ✓ on completion
    bootstrap_steps: list[tuple[str, str]] = []
    manifest_features_applied: list[str] = []
    manifest_groups_applied: list[str] = []

    if remote_manifest is None or not remote_manifest.has_feature(REMOTE_SYSTEM_TOOLS_FEATURE):
        bootstrap_steps.append((
            "Installing system deps",
            "apt-get update && apt-get install -y tmux libnuma1 wget",
        ))
        manifest_features_applied.append(REMOTE_SYSTEM_TOOLS_FEATURE)

    if remote_manifest is None or not remote_manifest.has_feature(REMOTE_UV_FEATURE):
        bootstrap_steps.append((
            "Installing uv",
            "curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env",
        ))
        manifest_features_applied.append(REMOTE_UV_FEATURE)

    if not image_owned_runtime and not managed_venv_ready:
        managed_venv_feature = f"ssh-managed-venv-python-{custom_image.python_version if custom_image is not None else '3.12'}"
        if remote_manifest is None or not remote_manifest.has_feature(managed_venv_feature):
            python_version = custom_image.python_version if custom_image is not None else "3.12"
            bootstrap_steps.append((
                "Creating managed Python runtime",
                (
                    f"~/.local/bin/uv python install {shlex.quote(python_version)} && "
                    f"~/.local/bin/uv venv {shlex.quote(SSH_MANAGED_VENV_DIR)} "
                    f"--python {shlex.quote(python_version)}"
                ),
            ))
            manifest_features_applied.append(managed_venv_feature)

    # Add CUDA toolkit upgrade if needed (must happen before Python packages that compile CUDA code)
    if needs_cuda_upgrade and cuda_req is not None:
        _, req_major, req_minor, _ = cuda_req
        # Use runfile installer with --toolkit to upgrade nvcc without touching driver
        # This installs to /usr/local/cuda-X.Y and we update PATH to use it
        # Download URLs from: https://developer.nvidia.com/cuda-12-8-0-download-archive
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
        if installer_url:
            bootstrap_steps.append((
                f"Upgrading CUDA toolkit to {req_major}.{req_minor}",
                f"wget -q {installer_url} -O /tmp/cuda_installer.run && "
                f"sh /tmp/cuda_installer.run --silent --toolkit && "
                f"rm /tmp/cuda_installer.run && "
                f"echo 'export PATH=/usr/local/cuda-{req_major}.{req_minor}/bin:$PATH' >> ~/.bashrc && "
                f"export PATH=/usr/local/cuda-{req_major}.{req_minor}/bin:$PATH",
            ))
        else:
            raise RuntimeError(
                f"No CUDA installer URL configured for required toolkit {req_major}.{req_minor}"
            )

    if custom_image is not None and custom_image.system_packages:
        image_apt_feature = stable_feature_name(
            "image-system-packages", custom_image.system_packages
        )
        if remote_manifest is None or not remote_manifest.has_feature(image_apt_feature):
            bootstrap_steps.append((
                "Installing image system packages",
                (
                    f"{_apt_install_command(custom_image.system_packages)} && "
                    f"{apt_install_probe_command('image-system-packages', packages=custom_image.system_packages)}"
                ),
            ))
            manifest_features_applied.append(image_apt_feature)

    if custom_image is not None and custom_image.pip_packages:
        # TODO: Mirror the Modal path here by creating an image-owned uv venv / runtime
        # contract and keeping heavy Python deps out of per-run reconciliation entirely.
        # This SSH/bootstrap path should eventually verify that runtime, not redefine it.
        image_pip_feature = stable_feature_name(
            f"image-pip-packages-{runtime_feature_scope}",
            custom_image.pip_packages,
        )
        if (
            not image_owned_runtime
            or remote_manifest is None
            or not remote_manifest.has_feature(image_pip_feature)
        ):
            bootstrap_steps.append((
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
            manifest_features_applied.append(image_pip_feature)

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
                bootstrap_steps.append((f"Running image build command {idx}", command))
            manifest_features_applied.append(image_build_feature)

    if custom_overlay is not None and custom_overlay.system_packages:
        overlay_apt_feature = stable_feature_name(
            "overlay-system-packages", custom_overlay.system_packages
        )
        if remote_manifest is None or not remote_manifest.has_feature(overlay_apt_feature):
            bootstrap_steps.append((
                "Installing runtime system packages",
                (
                    f"{_apt_install_command(custom_overlay.system_packages)} && "
                    f"{apt_install_probe_command('overlay-system-packages', packages=custom_overlay.system_packages)}"
                ),
            ))
            manifest_features_applied.append(overlay_apt_feature)

    if custom_overlay is not None and custom_overlay.pip_packages:
        overlay_pip_feature = stable_feature_name(
            f"overlay-pip-packages-{runtime_feature_scope}",
            custom_overlay.pip_packages,
        )
        if (
            not image_owned_runtime
            or remote_manifest is None
            or not remote_manifest.has_feature(overlay_pip_feature)
        ):
            bootstrap_steps.append((
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
            manifest_features_applied.append(overlay_pip_feature)

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
                bootstrap_steps.append((f"Running runtime overlay command {idx}", command))
            manifest_features_applied.append(overlay_cmd_feature)

    if custom_overlay is not None:
        manifest_features_applied.extend(custom_overlay.features)
        manifest_groups_applied.extend(custom_overlay.installed_groups)

    if extra_python_projects:
        extra_project_roots = tuple(
            project.remote_source_root(workspace) for project in extra_python_projects
        )
        bootstrap_steps.append((
            "Installing extra project Python packages",
            (
                f"{_uv_pip_install_editable_command(extra_project_roots, python_bin=None if image_owned_runtime else runtime_python, system=image_owned_runtime)} && "
                f"{python_install_probe_command('extra-python-projects', python_bin=runtime_python)} && "
                f"{python_runtime_contract_snapshot_command('extra-python-projects', python_bin=runtime_python)}"
            ),
        ))

    for label, cmd in bootstrap_steps:
        log("bootstrap_step_start", label=label)
        with spinner(f"{label}..."):
            bifrost.exec(cmd, working_dir=workspace)
        log("bootstrap_step_done", label=label)

    # HuggingFace login for faster authenticated downloads
    # Token is written to ~/.cache/huggingface/token (standard HF location)
    # Using printf to avoid token appearing in shell history or ps output
    if hf_token := os.getenv("HF_TOKEN"):
        log("bootstrap_step_start", label="HuggingFace login")
        with spinner("Logging into HuggingFace..."):
            # Use env var in subshell - token only visible to this process
            bifrost.exec(
                f"mkdir -p {hf_cache_dir} && printf '%s' \"$HF_TOKEN\" > {hf_cache_dir}/token",
                env={"HF_TOKEN": hf_token},
                working_dir=workspace,
            )
        log("bootstrap_step_done", label="HuggingFace login")

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

    manifest_to_write = manifest_base.extended(
        features=tuple(manifest_features_applied),
        installed_groups=tuple(manifest_groups_applied),
        resolved_image_ref=resolved_registry_image_ref,
        env={
            "HF_HOME": hf_cache_dir,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            **(custom_image.env if custom_image is not None else {}),
            **(custom_overlay.env if custom_overlay is not None else {}),
        },
        paths={"megatron_root": "/root/Megatron-LM"},
        python_version=manifest_base.python_version or "3.12",
        cuda_version=manifest_base.cuda_version
        or (f"{req_major}.{req_minor}" if needs_cuda_upgrade and cuda_req is not None else None),
    )
    log(
        "image_manifest_write",
        features=list(manifest_to_write.features),
        installed_groups=list(manifest_to_write.installed_groups),
    )
    bifrost.exec(manifest_write_command(manifest_to_write, USER_IMAGE_MANIFEST_PATH))

    # Create run output directory
    remote_output_dir = f"{workspace}/rollouts/results/rl/{run_name}"
    bifrost.exec(f"mkdir -p {remote_output_dir}")
    training_log = f"{remote_output_dir}/training.log"

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
    log(
        "logs_server_started",
        port=logs_port,
        session=logs_service.handle_id,
        log_file=logs_service.log_file,
    )

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

    log("submit_start")
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
    log("submit_done", tmux_session=job.tmux_session)

    return bifrost, instance, job, run_name, remote_output_dir, workspace, console, local_run_dir


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
) -> None:
    """Run training script on remote GPU via bifrost."""
    # TODO(argus-run): Move the SSH/bifrost remote training launcher into its
    # own module. This path mixes provisioning, bootstrap, sync, tmux/logserver
    # lifecycle, and attach behavior, which overwhelms the top-level CLI file.
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
        workload_kind = _classify_config_module(config_module, config_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # Get hardware config (default to local if not specified)
    hardware: HardwareConfig = getattr(config_module, "hardware", HardwareConfig(provider="local"))
    workload_config = getattr(config_module, "config", None)
    # TODO(boundary): this loader/merge path is reconstructing a launchable
    # execution spec from module exports, CLI overrides, and service-scoped deps.
    # Replace it with one explicit product type that Argus consumes directly.

    trainer_service_deps = None
    inference_service_deps = None
    service_runtime_layout = "shared_env"
    if workload_config is not None:
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

    if service_runtime_layout != "shared_env":
        raise ValueError(
            "The current Argus launcher only realizes service_runtime_layout='shared_env'. "
            "Split service runtimes need a launcher that provisions separate trainer "
            "and inference environments."
        )

    if trainer_service_deps is not None or inference_service_deps is not None:
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
        # TODO(argus-run): Replace this large inline dispatch block with staged
        # launcher selection over a resolved workload product type:
        #   1. resolve workload/config
        #   2. choose local vs modal vs remote launcher
        #   3. invoke the selected launcher
        # The current file still interleaves those state machines.
        # Dispatch based on workload/provider
        from rollouts.inference.benchmark.config import BenchmarkConfig

        if workload_kind == "evaluation":
            if runtime.provider != "local" or args.node_id or multi_node is not None:
                raise ValueError(
                    "Argus evaluation launch currently supports only local orchestration. "
                    "Provider-specific remote lifecycle belongs in the eval workload itself "
                    "(for example via Modal/RunPod resources), not in the training SSH launcher."
                )

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            run_name = f"run_{timestamp}"
            local_run_dir = REPO_ROOT / "results" / "eval" / run_name
            log = _setup_run_logging(local_run_dir)
            log(
                "run_start",
                launcher_id=launcher_id,
                kind="evaluation",
                provider="local",
                config=str(config_path),
                output_dir=str(local_run_dir),
            )
            pid = _spawn_eval_subprocess(
                config_path=config_path,
                output_dir=local_run_dir,
                max_samples=args.max_samples,
                log=log,
            )
            print(f"Evaluation submitted: {run_name}")
            print(f"  PID:    {pid}")
            print(f"  Local:  results/eval/{run_name}/")

            if not args.tui and not args.tail:
                return 0

            return _launch_eval_monitor(run_dir=local_run_dir, tail=args.tail)

        if not isinstance(config_module.config, BenchmarkConfig):
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
            # TODO(chiraag): Unify this Modal sandbox path with the broker/bifrost
            # asset/session model so switching provider from "modal" to "runpod"
            # changes procurement, not the entire execution/supervision stack.
            # Modal execution (fast cold start)
            import json

            import trio

            # Check if this is a benchmark config
            from rollouts.inference.benchmark.config import BenchmarkConfig
            from rollouts.jobs import register_job, update_job_status

            if isinstance(config_module.config, BenchmarkConfig):
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
                from rollouts.jobs import update_job_node

                def _project_modal_event(event: str, data: dict[str, Any]) -> None:
                    if event == "modal_sandbox_created":
                        sandbox_id = data.get("sandbox_id")
                        if sandbox_id:
                            update_job_node(run_name, "modal", str(sandbox_id))
                    elif event == "modal_training_start":
                        update_job_status(run_name, "running")

                log = _setup_run_logging(local_run_dir, on_event=_project_modal_event)
                log(
                    "run_start",
                    launcher_id=launcher_id,
                    provider="modal",
                    config=str(config_path),
                    gpu_count=runtime.gpu_count,
                    gpu_type=runtime.gpu_type,
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
                )
                for extra_root in extra_source_roots:
                    enforce_source_sync_policy(
                        modal_request.source_sync_policy,
                        repo_root=extra_root,
                        stream=sys.stderr,
                    )
                log("modal_submit_dispatch")
                results = trio.run(run_modal_request, modal_request)
                if not results.get("success"):
                    update_job_status(run_name, "failed")
                    log(
                        "run_failed",
                        exit_code=results.get("exit_code"),
                        stderr=results.get("stderr"),
                    )
                    return 1
                update_job_status(run_name, "completed")
                log("run_completed", exit_code=results.get("exit_code"))

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
            )

        else:
            # Local execution
            if isinstance(config_module.config, BenchmarkConfig):
                # Run benchmark locally (we're already on the GPU machine)
                import json

                import trio

                from rollouts.inference.benchmark.runner import run_benchmark_local

                result = trio.run(
                    run_benchmark_local,
                    config_module.config,
                    hardware.gpu_type,
                    hardware.gpu_count,
                )
                print(json.dumps(result.to_dict(), indent=2))
            else:
                # Training run
                if os.getenv("ARGUS_EMIT_STARTUP_SENTINEL") == "1":
                    print("__ARGUS_WORKLOAD_ENTRYPOINT_STARTED__", flush=True)

                kwargs = {}
                if os.getenv("ARGUS_RUN_EVENT_STREAM") == "1":
                    kwargs["run_logger"] = stream_run_logger()
                if args.max_samples is not None:
                    kwargs["max_samples"] = args.max_samples

                results = config_module.train(config=config_module.config, **kwargs)
                print(f"Training complete. {len(results.get('metrics_history', []))} steps")

        return 0
    finally:
        _remove_launch_record(launch_record_path)


if __name__ == "__main__":
    sys.exit(main())


def run_main(argv: list[str] | None = None) -> int:
    """Compatibility entry for the Argus CLI."""
    return main(argv)
