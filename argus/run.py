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

    # CLI overrides (optional, override config values)
    python -m argus run --config ... --gpu-type H100  # Override GPU type
    python -m argus run --config ... --provider modal  # Override provider
    python -m argus run --config ... --local  # Force local execution

    # Legacy CLI flags (still supported for backwards compat)
    python -m argus run --config ... --modal  # Same as --provider modal
    python -m argus run --config ... --provision  # Provision via config.hardware.provider

The config file should export:
    - config: A training config (e.g., GRPOConfig)
    - hardware: HardwareConfig (optional, defaults to local execution)
    - train(config, **kwargs): Function to run local training

Execution modes (determined by hardware.provider or CLI override):
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
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any


@contextmanager
def _quiet_spinner(msg: str) -> Generator[None, None, None]:
    """No-op context manager for quiet mode.

    Inner operations (like bifrost.acquire_node) log their own progress,
    so we just yield without adding wrapper messages.
    """
    yield None


if TYPE_CHECKING:
    from bifrost import BifrostClient
    from broker import ClientGPUInstance

    from rollouts.training.configs import DepsConfig
    from rollouts.training.multi_node import MultiNodeConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1] / "rollouts"
ARGUS_STATE_DIR = Path.home() / ".argus"
LAUNCHES_DIR = ARGUS_STATE_DIR / "launches"
REMOTE_SYSTEM_TOOLS_FEATURE = "remote-system-tools-v1"
REMOTE_UV_FEATURE = "uv"

# sys.path hack: make sibling packages (miniray, bifrost, broker, etc.) importable.
#
# WHY THIS EXISTS
# ---------------
# The workspace has multiple Python packages (argus, rollouts, bifrost, broker,
# miniray, infra_utils) installed via `uv sync --extra deploy` from the workspace
# root. Locally this works. But when argus deploys a job remotely, it git-pushes
# the workspace as a bundle and then runs `argus run --local` inside it. At that
# point the remote process has no venv and no installed packages — just a directory
# tree. So the code manually inserts the workspace root into sys.path to make
# siblings importable without installation.
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
#   - This sys.path.insert block goes away
#   - The PYTHONPATH line in `_deploy_and_submit` goes away
#   - The TYPE_CHECKING guards for bifrost/broker imports become real imports
#   - The rollouts/run.py __getattr__ forwarding shim can be deleted
#
# The net result: imports become honest, missing dependencies fail loudly at startup
# instead of at the first call site, and the code no longer needs to know the
# directory structure of the remote machine.
_workspace_root = REPO_ROOT.parent
if _workspace_root.exists() and str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from rollouts.image_publisher import build_or_resolve_image
from rollouts.config_contracts import validate_train_config_module
from rollouts.image_spec import (
    USER_IMAGE_MANIFEST_PATH,
    ImageManifest,
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


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _new_launcher_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"launch_{timestamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


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


class _RunLogger:
    """Callable logger interface for structured run events."""

    def __call__(self, event: str, **data: Any) -> None: ...


def _read_remote_manifest(bifrost: BifrostClient) -> ImageManifest | None:
    """Load a baked or previously bootstrapped manifest from the remote node."""
    raw = bifrost.exec(
        "if [ -f /etc/rollouts-image.json ]; then cat /etc/rollouts-image.json; "
        f"elif [ -f {USER_IMAGE_MANIFEST_PATH} ]; then cat {USER_IMAGE_MANIFEST_PATH}; fi"
    )
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
    index_url: str | None = None,
    extra_index_url: str | None = None,
    pre: bool = False,
    extra_options: str | None = None,
) -> str:
    quoted_packages = " ".join(shlex.quote(package) for package in packages)
    parts = ["~/.local/bin/uv", "pip", "install", "--upgrade"]
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
    spec.loader.exec_module(module)
    return module


def _setup_run_logging(run_dir: Path) -> _RunLogger:
    """Create run directory and return a generic event logger.

    This logger intentionally knows only about a generic event envelope.
    Workload-specific stage names and semantics belong in Rollouts.
    """
    import json
    from datetime import datetime

    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.jsonl"

    def log_event(event: str, **data: Any) -> None:
        entry = {
            "ts": datetime.now().isoformat(),
            "event": event,
            **data,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    return log_event


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
) -> tuple:
    """Provision node, deploy code, submit training job.

    Returns (bifrost_client, instance, job, run_name, remote_output_dir, workspace, console, local_run_dir).
    """
    from bifrost import GPUQuery, ProcessSpec, acquire_node
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
                f"\nError: No {gpu_type} GPUs found. Try a different --gpu-type.",
                file=sys.stderr,
            )
        elif result.all_unavailable:
            print(
                f"\nError: No {gpu_type} GPUs available right now. Try again later or use --gpu-type to pick a different GPU.",
                file=sys.stderr,
            )
        elif result.network_error:
            print("\nError: Network error reaching GPU provider. Try again.", file=sys.stderr)
        else:
            print(f"\nError: Provisioning failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Check CUDA toolkit version compatibility and auto-upgrade if needed
    # The driver version (nvidia-smi) may be newer than the toolkit (nvcc)
    # FlashInfer/Triton JIT-compile kernels and need nvcc to support the GPU arch
    from rollouts.training.preflight import get_gpu_cuda_requirement

    cuda_req = get_gpu_cuda_requirement(gpu_type)
    needs_cuda_upgrade = False
    if cuda_req is not None:
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

    # Deploy code (git sync only, no bootstrap)
    script_rel_path = Path(script_path).relative_to(REPO_ROOT)

    log("deploy_start")
    with spinner("Deploying code..."):
        workspace = bifrost.push("~/.bifrost/workspaces/rollouts-rl", allow_dirty=allow_dirty)
    log("deploy_done", workspace=workspace)

    remote_manifest = _read_remote_manifest(bifrost)
    if remote_manifest is not None:
        log(
            "image_manifest_loaded",
            features=list(remote_manifest.features),
            installed_groups=list(remote_manifest.installed_groups),
        )

    custom_image = deps.resolved_image(gpu_type) if deps is not None else None
    custom_overlay = deps.resolved_runtime_overlay() if deps is not None else None

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
        image_pip_feature = stable_feature_name("image-pip-packages", custom_image.pip_packages)
        if remote_manifest is None or not remote_manifest.has_feature(image_pip_feature):
            bootstrap_steps.append((
                "Installing image Python packages",
                (
                    f"{_uv_pip_install_command(
                        custom_image.pip_packages,
                        index_url=custom_image.pip_index_url,
                        extra_index_url=custom_image.pip_extra_index_url,
                        pre=custom_image.pip_prerelease,
                    )} && "
                    f"{python_install_probe_command('image-pip-packages')} && "
                    f"{python_runtime_contract_snapshot_command('image-pip-packages')}"
                ),
            ))
            manifest_features_applied.append(image_pip_feature)

    if custom_image is not None and custom_image.build_commands:
        image_build_feature = stable_feature_name(
            "image-build-commands", custom_image.build_commands
        )
        if remote_manifest is None or not remote_manifest.has_feature(image_build_feature):
            for idx, command in enumerate(custom_image.build_commands, start=1):
                if command_looks_like_install(command):
                    command = (
                        f"{command} && "
                        f"{python_install_probe_command(f'image-build-command-{idx}')} && "
                        f"{python_runtime_contract_verify_command(f'image-build-command-{idx}')}"
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
            "overlay-pip-packages", custom_overlay.pip_packages
        )
        if remote_manifest is None or not remote_manifest.has_feature(overlay_pip_feature):
            bootstrap_steps.append((
                "Installing runtime Python packages",
                (
                    f"{_uv_pip_install_command(
                        custom_overlay.pip_packages,
                        index_url=custom_overlay.pip_index_url
                        or (custom_image.pip_index_url if custom_image else None),
                        extra_index_url=custom_overlay.pip_extra_index_url
                        or (custom_image.pip_extra_index_url if custom_image else None),
                        pre=custom_overlay.pip_prerelease
                        or (custom_image.pip_prerelease if custom_image else False),
                    )} && "
                    f"{python_install_probe_command('overlay-pip-packages')} && "
                    f"{python_runtime_contract_snapshot_command('overlay-pip-packages')}"
                ),
            ))
            manifest_features_applied.append(overlay_pip_feature)

    if custom_overlay is not None and custom_overlay.commands:
        overlay_cmd_feature = stable_feature_name("overlay-commands", custom_overlay.commands)
        if remote_manifest is None or not remote_manifest.has_feature(overlay_cmd_feature):
            for idx, command in enumerate(custom_overlay.commands, start=1):
                if command_looks_like_install(command):
                    command = (
                        f"{command} && "
                        f"{python_install_probe_command(f'overlay-command-{idx}')} && "
                        f"{python_runtime_contract_verify_command(f'overlay-command-{idx}')}"
                    )
                bootstrap_steps.append((f"Running runtime overlay command {idx}", command))
            manifest_features_applied.append(overlay_cmd_feature)

    if custom_overlay is not None:
        manifest_features_applied.extend(custom_overlay.features)
        manifest_groups_applied.extend(custom_overlay.installed_groups)

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

    # Start LogsServer in a separate tmux session BEFORE the training job.
    # This decouples LogsServer lifetime from the training job — if training crashes,
    # LogsServer keeps running and can serve the final logs (including tracebacks).
    logs_port = 9100
    logs_dir = f"{workspace}/rollouts/results/rl/{run_name}"
    logs_session = f"logs-{run_name}"

    # Kill any stale LogsServer processes from previous runs
    bifrost.exec(f"fuser -k {logs_port}/tcp 2>/dev/null || true")
    bifrost.exec("pkill -f 'miniray.logs_server' 2>/dev/null || true")
    bifrost.exec(f"tmux kill-session -t {logs_session} 2>/dev/null || true")

    # Start LogsServer in its own tmux session (survives training crashes)
    logs_cmd = (
        f"cd {workspace} && python3 -m miniray.logs_server --port {logs_port} --dir {logs_dir}"
    )
    bifrost.exec(f"tmux new-session -d -s {logs_session} '{logs_cmd}'")
    log("logs_server_started", port=logs_port, session=logs_session)

    env_vars = {
        "PYTHONUNBUFFERED": "1",
        "ROLLOUTS_RUN_NAME": run_name,
        "ROLLOUTS_OUTPUT_DIR": f"results/rl/{run_name}",
        "ROLLOUTS_JSON_LOGS": "true",
        "HF_HOME": hf_cache_dir,
        # PYTHONPATH includes:
        # - workspace root for miniray and other sibling packages
        # - /root/Megatron-LM for megatron.core imports
        "PYTHONPATH": f"{workspace}:/root/Megatron-LM",
        # NCCL settings for multi-GPU training
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        **(custom_image.env if custom_image is not None else {}),
        **(custom_overlay.env if custom_overlay is not None else {}),
    }

    # Submit training job
    if raw_script:
        run_args = (
            "run",
            "python",
            str(script_rel_path),
        )
    else:
        run_args = (
            "run",
            "python",
            "-m",
            "argus.run",
            "--config",
            str(script_rel_path),
            "--local",
        )

    log("submit_start")
    with spinner(f"Starting {run_name}...") as spin:
        job = bifrost.submit(
            ProcessSpec(
                command="/root/.local/bin/uv",
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
) -> None:
    """Run training script on remote GPU via bifrost."""
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

    monitor_cmd = [sys.executable, "-m", "rollouts", "monitor", "--attach", run_name]
    if tail:
        monitor_cmd.append("--tail")
    if keep_alive:
        monitor_cmd.append("--keep-alive")
    else:
        monitor_cmd.append("--terminate")
    subprocess.run(monitor_cmd, check=False)
    # Note: monitor handles final sync and terminate internally


def main(argv: list[str] | None = None) -> int:
    from dataclasses import replace

    from rollouts.training.configs import HardwareConfig

    parser = argparse.ArgumentParser(
        description="Run RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Config-driven (reads hardware from config file)
    python -m argus run --config examples/rl/kernelbench/grpo_01_01.py

    # Override provider via CLI
    python -m argus run --config ... --provider modal
    python -m argus run --config ... --local

    # Legacy flags (still supported)
    python -m argus run --config ... --modal
    python -m argus run --config ... --provision --provider runpod
        """,
    )
    parser.add_argument("--config", required=True, help="Path to config file")

    # Provider/hardware overrides
    parser.add_argument(
        "--provider",
        type=str,
        choices=["local", "modal", "runpod", "lambdalabs", "vast"],
        help="Override hardware provider from config",
    )
    parser.add_argument("--local", action="store_true", help="Force local execution")
    parser.add_argument("--gpu-type", type=str, help="Override GPU type from config")
    parser.add_argument("--gpu-count", type=int, help="Override GPU count from config")
    parser.add_argument("--container-disk-gb", type=int, help="Override container disk size")
    parser.add_argument("--hf-cache-dir", type=str, help="Override remote HuggingFace cache dir")
    parser.add_argument(
        "--persistent-volume-id",
        type=str,
        help="Attach a persistent volume when provisioning remote hardware",
    )
    parser.add_argument(
        "--persistent-volume-mount-path",
        type=str,
        help="Mount path for the attached persistent volume",
    )
    parser.add_argument(
        "--persistent-volume-location",
        type=str,
        help="Provider-specific placement hint for the persistent volume",
    )
    # Legacy flags (for backwards compat)
    parser.add_argument("--modal", action="store_true", help="[Legacy] Same as --provider modal")
    parser.add_argument(
        "--provision", action="store_true", help="[Legacy] Provision using config's provider"
    )

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
        validate_train_config_module(config_module, config_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # Get hardware config (default to local if not specified)
    hardware: HardwareConfig = getattr(config_module, "hardware", HardwareConfig(provider="local"))

    # Apply CLI overrides
    if args.local:
        hardware = replace(hardware, provider="local")
    elif args.modal:
        # Legacy --modal flag
        hardware = replace(hardware, provider="modal")
    elif args.provider:
        hardware = replace(hardware, provider=args.provider)
    elif args.provision and hardware.provider == "local":
        # Legacy --provision without provider: default to runpod
        hardware = replace(hardware, provider="runpod")

    if args.gpu_type:
        hardware = replace(hardware, gpu_type=args.gpu_type)
    if args.gpu_count:
        hardware = replace(hardware, gpu_count=args.gpu_count)
    if args.container_disk_gb:
        hardware = replace(hardware, container_disk_gb=args.container_disk_gb)
    if args.hf_cache_dir:
        hardware = replace(hardware, hf_cache_dir=args.hf_cache_dir)
    if args.persistent_volume_id:
        hardware = replace(hardware, persistent_volume_id=args.persistent_volume_id)
    if args.persistent_volume_mount_path:
        hardware = replace(
            hardware,
            persistent_volume_mount_path=args.persistent_volume_mount_path,
        )
    if args.persistent_volume_location:
        hardware = replace(
            hardware,
            persistent_volume_location=args.persistent_volume_location,
        )
    runtime = runtime_contract_from_hardware(hardware)
    materialization = materialization_plan_from_runtime(runtime)

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
                f"  - {launch['launcher_id']} pid={launch['pid']} started_at={launch.get('started_at','')}",
                file=sys.stderr,
            )

    # Check for multi-node config (only if not forced to local)
    multi_node: MultiNodeConfig | None = None
    if not args.local:
        multi_node = getattr(config_module, "multi_node", None)

    try:
        # Dispatch based on provider
        from rollouts.inference.benchmark.config import BenchmarkConfig

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
                from rollouts.modal_runner import ModalRunConfig, run_modal

                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                run_name = f"run_{timestamp}"
                local_run_dir = REPO_ROOT / "results" / "rl" / run_name
                log = _setup_run_logging(local_run_dir)
                log(
                    "run_start",
                    launcher_id=launcher_id,
                    provider="modal",
                    config=str(config_path),
                    gpu_count=runtime.gpu_count,
                    gpu_type=runtime.gpu_type,
                )
                register_job(
                    job_id=run_name,
                    provider="modal",
                    node_id="pending",
                    config_path=str(config_path),
                    log_path=f"results/rl/{run_name}",
                )

                modal_config = ModalRunConfig(
                    config_path=str(config_path),
                    runtime=runtime,
                    materialization=materialization,
                    run_name=run_name,
                    event_log=log,
                    source_sync_policy=SourceSyncPolicy.committed_only(
                        dirty_action="warn" if args.force_deploy_committed else "fail"
                    ),
                )
                log("modal_submit_dispatch")
                results = trio.run(run_modal, modal_config)
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
                kwargs = {}
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
