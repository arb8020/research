"""
Modal provider implementation — Sandbox-based compute.

Unlike other providers (RunPod, Lambda, etc.), Modal sandboxes don't expose SSH.
Commands execute directly via sandbox.exec(). This means bifrost (SSH-based) is
not in the loop — callers use GPUInstance.exec() directly.

Modal SDK is sync-looking but runs asyncio internally. All blocking calls are
wrapped in trio.to_thread.run_sync() to avoid blocking the trio event loop.
"""

import logging
from typing import Any

import trio

from ..types import (
    GPUInstance,
    GPUOffer,
    InstanceStatus,
    ModalFilesystemSnapshot,
    ModalVolumeMount,
    ProvisionRequest,
    SSHResult,
)
from .modal_image import build_modal_image

logger = logging.getLogger(__name__)

# Modal app name for broker-managed sandboxes
BROKER_APP_NAME = "broker-gpu"

# Known GPU types and approximate pricing ($/hr).
# Modal has no search/pricing API — these are hardcoded from published rates.
# Updated 2025-01. Prices are per-GPU-hour.
GPU_CATALOG: list[dict[str, Any]] = [
    {"gpu_type": "T4", "vram_gb": 16, "price_per_hour": 0.59},
    {"gpu_type": "L4", "vram_gb": 24, "price_per_hour": 0.80},
    {"gpu_type": "A10G", "vram_gb": 24, "price_per_hour": 1.10},
    {"gpu_type": "L40S", "vram_gb": 48, "price_per_hour": 1.95},
    {"gpu_type": "A100-40GB", "vram_gb": 40, "price_per_hour": 2.10},
    {"gpu_type": "A100-80GB", "vram_gb": 80, "price_per_hour": 2.50},
    {"gpu_type": "H100", "vram_gb": 80, "price_per_hour": 3.95},
    {"gpu_type": "H200", "vram_gb": 141, "price_per_hour": 4.54},
    {"gpu_type": "B200", "vram_gb": 192, "price_per_hour": 6.25},
]


def _import_modal():  # noqa: ANN202
    """Import modal lazily to avoid import-time side effects.

    Modal SDK does network I/O on import (reads ~/.modal.toml, sets up gRPC).
    Lazy import keeps broker fast for callers who don't use Modal.
    """
    try:
        import modal

        return modal
    except ImportError as e:
        raise ImportError("Modal SDK not installed. Install with: pip install modal") from e


def _build_image_from_deps(modal: Any, deps: Any, gpu_type: str) -> Any:
    """Build Modal image from a DepsConfig (recipes/schema.py).

    Translates the explicit dependency spec into modal.Image builder calls.
    No guessing — every pip package, system package, and bootstrap command
    comes from the DepsConfig.
    """
    return build_modal_image(modal, deps, gpu_type)


def _build_default_image(modal: Any, gpu_type: str) -> Any:
    """Fallback image when no DepsConfig is provided.

    Installs torch only. Callers should prefer passing a DepsConfig
    with explicit deps for reproducibility.
    """
    if gpu_type in ("B200", "GB200"):
        torch_index = "https://download.pytorch.org/whl/nightly/cu128"
        torch_version = "torch>=2.8.0"
    else:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_version = "torch>=2.4.0"

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("bash", "curl", "git", "build-essential")
        .pip_install(
            torch_version,
            index_url=torch_index,
            extra_index_url="https://pypi.org/simple",
            pre="nightly" in torch_index,
        )
    )
    return image


def _sandbox_exec_sync(sandbox: Any, command: str, timeout: int = 300) -> SSHResult:
    """Execute command on sandbox synchronously. Returns SSHResult for API compat.

    Named SSHResult for historical reasons — it's the standard broker result type.
    Modal sandboxes use sandbox.exec(), not SSH.
    """
    process = sandbox.exec("bash", "-c", command, timeout=timeout)

    stdout_lines = []
    for line in process.stdout:
        stdout_lines.append(line)

    stderr_lines = []
    for line in process.stderr:
        stderr_lines.append(line)

    process.wait()

    return SSHResult(
        success=process.returncode == 0,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
        exit_code=process.returncode,
        command=command,
    )


def _resolve_volumes(
    modal: Any,
    volume_mounts: list[ModalVolumeMount],
) -> dict[str, Any]:
    """Resolve ModalVolumeMount list into the dict Modal's Sandbox.create() expects.

    Args:
        modal: The modal module (lazy-imported).
        volume_mounts: List of named volumes to mount.

    Returns:
        Dict mapping mount paths to modal.Volume instances.

    # TODO(broker): add volume creation helper — right now callers must
    # pre-create volumes manually or via modal CLI. A first-class broker API
    # would be:
    #   broker.modal.ensure_volume(name, size_gb) -> ModalVolumeMount
    # analogous to runpod's create_network_volume().
    """
    volumes = {}
    for mount in volume_mounts:
        vol = modal.Volume.from_name(mount.volume_name, create_if_missing=True)
        if mount.read_only:
            vol = vol.read_only()
        volumes[mount.mount_path] = vol
        logger.info("Modal volume: %s -> %s (read_only=%s)", mount.volume_name, mount.mount_path, mount.read_only)
    return volumes


def _resolve_image(
    modal: Any,
    request: ProvisionRequest,
    deps: Any | None,
    gpu_type: str,
) -> tuple[Any, bool]:
    """Resolve the sandbox base image, preferring a filesystem snapshot if available.

    If request.modal_snapshot is set and resolves to a known image ID, uses
    modal.Image.from_id() to restore from the snapshot. Otherwise builds from
    DepsConfig (or default image). Returns (image, from_snapshot).

    On first run (snapshot not yet created), returns (built_image, False).
    After the first run, snapshot_sandbox() should be called to create the
    snapshot so subsequent runs use (snapshot_image, True).

    # TODO(broker): make snapshot fallback automatic — if snapshot resolves to
    # None, build the image, run a one-shot download function, snapshot, save
    # the image ID, then provision the actual training sandbox from the snapshot.
    # This "lazy snapshot" pattern would make first-run cost transparent.
    """
    snapshot = request.modal_snapshot
    if snapshot is not None:
        image_id = snapshot.resolve_image_id()
        if image_id is not None:
            logger.info("Using filesystem snapshot image: %s", image_id)
            return modal.Image.from_id(image_id), True
        else:
            logger.info(
                "Snapshot not yet created (registry=%s key=%s) — building from deps",
                snapshot.snapshot_registry_name,
                snapshot.snapshot_registry_key,
            )

    # Fall back to building from DepsConfig or default
    if deps is not None:
        return _build_image_from_deps(modal, deps, gpu_type), False
    return _build_default_image(modal, gpu_type), False


def _create_sandbox_sync(
    request: ProvisionRequest,
    deps: Any | None = None,
) -> tuple[Any, Any]:
    """Create Modal sandbox. Blocking — call from trio.to_thread.run_sync().

    Args:
        request: Standard broker provision request.
        deps: Optional DepsConfig (from recipes/schema.py). If provided,
              builds the image from explicit deps. Otherwise falls back
              to a default torch-only image.

    Respects request.modal_volumes (named volume mounts) and
    request.modal_snapshot (filesystem snapshot as base image).

    Returns (sandbox, modal_module) tuple.
    """
    modal = _import_modal()

    # Get or create app
    app = modal.App.lookup(
        BROKER_APP_NAME,
        create_if_missing=True,
    )

    gpu_type = request.gpu_type or "T4"

    # Resolve image — snapshot if available, else build from deps
    image, from_snapshot = _resolve_image(modal, request, deps, gpu_type)

    # Resolve volume mounts
    # TODO(broker): validate that volume mount paths don't overlap with
    # image paths — Modal doesn't error cleanly on path conflicts.
    volumes = _resolve_volumes(modal, request.modal_volumes)

    # Build GPU spec
    gpu_count = request.gpu_count or 1
    if gpu_count > 1:
        gpu_spec = f"{gpu_type}:{gpu_count}"
    else:
        gpu_spec = gpu_type

    # Sandbox name for lookup/reconnect.
    # Always include timestamp — Modal rejects duplicate names, even after termination.
    import time

    ts = int(time.time())
    sandbox_name = f"broker-{request.name or gpu_type}-{ts}"

    # Create sandbox with provider-supported max lifetime.
    sandbox_timeout_seconds = request.max_lifetime_seconds or 60 * 60 * 24

    sandbox_kwargs: dict[str, Any] = {
        "app": app,
        "image": image,
        "gpu": gpu_spec,
        "timeout": sandbox_timeout_seconds,
        "name": sandbox_name,
    }
    if volumes:
        sandbox_kwargs["volumes"] = volumes

    sandbox = modal.Sandbox.create(**sandbox_kwargs)

    assert sandbox is not None, "Sandbox.create() returned None"
    assert sandbox.object_id, "Sandbox missing object_id"

    logger.info(
        "Modal sandbox created: %s (gpu=%s, name=%s, ttl_seconds=%s, "
        "volumes=%s, from_snapshot=%s)",
        sandbox.object_id,
        gpu_spec,
        sandbox_name,
        sandbox_timeout_seconds,
        list(volumes.keys()) if volumes else [],
        from_snapshot,
    )

    return sandbox, modal


# ============================================================================
# Snapshot API
# ============================================================================


async def snapshot_sandbox(
    instance_id: str,
    snapshot: ModalFilesystemSnapshot | None = None,
) -> str:
    """Snapshot a running sandbox's filesystem and return the image ID.

    The snapshot captures the full sandbox filesystem as a Modal Image.
    Use this after a one-shot setup sandbox (e.g. model weight download)
    to create a reusable base image for future sandboxes.

    Args:
        instance_id: The sandbox object_id to snapshot.
        snapshot: Optional ModalFilesystemSnapshot to auto-persist the image ID.
                  If provided, saves the image ID to the snapshot registry so
                  future ProvisionRequests with the same snapshot config will
                  use this image automatically.

    Returns:
        The Modal image ID (pass as ModalFilesystemSnapshot.image_id for reuse).

    Example:
        # One-time setup: download weights + snapshot
        instance = await provision_instance(setup_request)
        await exec_on_sandbox(instance.id, "python download_weights.py")
        image_id = await snapshot_sandbox(instance.id, snapshot=my_snapshot_config)
        await terminate_instance(instance.id)

        # Subsequent runs: start from snapshot automatically
        train_request = ProvisionRequest(modal_snapshot=my_snapshot_config, ...)
        train_instance = await provision_instance(train_request)

    # TODO(broker): make this a first-class broker operation exposed via CLI:
    #   broker modal snapshot <instance-id> [--registry NAME --key KEY]
    # Currently only accessible programmatically.

    # TODO(broker): add snapshot_directory() support for partial snapshots
    # (beta feature in Modal SDK). Useful for snapshotting just /root/.cache
    # without the full filesystem, which would be faster and smaller.
    """

    def _snapshot_sync() -> str:
        modal = _import_modal()
        sandbox = modal.Sandbox.from_id(instance_id)
        image = sandbox.snapshot_filesystem()
        image_id = image.object_id
        logger.info("Snapshot created: sandbox=%s image_id=%s", instance_id, image_id)
        if snapshot is not None:
            snapshot.save_image_id(image_id)
            logger.info(
                "Snapshot image ID saved to registry=%s key=%s",
                snapshot.snapshot_registry_name,
                snapshot.snapshot_registry_key,
            )
        return image_id

    return await trio.to_thread.run_sync(_snapshot_sync)


# ============================================================================
# ProviderModule interface (async)
#
# TODO: Implement the three Modal workload types from modal.com/llm-almanac:
# (1) Offline/batch — vLLM + .spawn()/.spawn_map() for throughput-first evals
# (2) Online/interactive — SGLang + modal.experimental.http_server for low-latency
# (3) Semi-online/bursty — autoscaling web_server + GPU memory snapshots
# Currently only Sandbox-based (1) is supported.
# ============================================================================


async def search_gpu_offers(
    cuda_version: str | None = None,
    manufacturer: str | None = None,
    memory_gb: int | None = None,
    container_disk_gb: int | None = None,
    gpu_count: int = 1,
    api_key: str | None = None,
) -> list[GPUOffer]:
    """Return hardcoded GPU offers from Modal's known catalog.

    Modal has no search/pricing API. We return all known GPU types with
    published pricing. Availability is handled at Sandbox.create() time.
    """
    offers = []

    for entry in GPU_CATALOG:
        # Filter by VRAM if requested
        if memory_gb and entry["vram_gb"] < memory_gb:
            continue

        # Filter by manufacturer (Modal is all NVIDIA)
        if manufacturer and manufacturer.lower() != "nvidia":
            continue

        offers.append(
            GPUOffer(
                id=f"modal-{entry['gpu_type'].lower()}",
                provider="modal",
                gpu_type=entry["gpu_type"],
                gpu_count=gpu_count,
                vcpu=4,  # Modal default
                memory_gb=entry["vram_gb"],
                vram_gb=entry["vram_gb"],
                storage_gb=0,  # Ephemeral
                price_per_hour=entry["price_per_hour"],
                manufacturer="nvidia",
            )
        )

    return offers


async def provision_instance(
    request: ProvisionRequest,
    ssh_startup_script: str | None = None,
    api_key: str | None = None,
) -> GPUInstance | None:
    """Provision a Modal sandbox.

    ssh_startup_script is ignored — Modal sandboxes don't have SSH.
    api_key is ignored — Modal reads from ~/.modal.toml.

    Pass a DepsConfig via request.raw_data["deps"] to build the image
    from explicit dependencies instead of the default torch-only image.
    """
    gpu_type = request.gpu_type or "T4"
    deps = request.raw_data.get("deps") if request.raw_data else None

    try:
        sandbox, _modal = await trio.to_thread.run_sync(
            lambda: _create_sandbox_sync(request, deps=deps)
        )

        # Look up pricing
        price = 0.0
        for entry in GPU_CATALOG:
            if entry["gpu_type"] == gpu_type:
                price = entry["price_per_hour"]
                break

        return GPUInstance(
            id=sandbox.object_id,
            provider="modal",
            status=InstanceStatus.RUNNING,  # Sandboxes are running immediately
            gpu_type=gpu_type,
            gpu_count=request.gpu_count or 1,
            name=request.name,
            price_per_hour=price,
            public_ip=None,  # No SSH
            ssh_port=None,
            ssh_username=None,
            raw_data={"sandbox_id": sandbox.object_id},
        )

    except Exception:
        logger.exception(f"Failed to provision Modal sandbox (gpu={gpu_type})")
        return None


async def get_instance_details(instance_id: str, api_key: str | None = None) -> GPUInstance | None:
    """Get details of a Modal sandbox by ID."""
    assert instance_id, "instance_id cannot be empty"

    def _get_sync() -> GPUInstance | None:
        modal = _import_modal()
        try:
            modal.Sandbox.from_id(instance_id)  # Raises if not found
            return GPUInstance(
                id=instance_id,
                provider="modal",
                status=InstanceStatus.RUNNING,
                gpu_type="unknown",  # Can't determine from ID alone
                gpu_count=1,
                price_per_hour=0.0,
                raw_data={"sandbox_id": instance_id},
            )
        except Exception:
            logger.debug(f"Modal sandbox {instance_id} not found")
            return None

    return await trio.to_thread.run_sync(_get_sync)


async def list_instances(api_key: str | None = None) -> list[GPUInstance]:
    """List Modal sandboxes. Uses modal.Sandbox.list() if available."""

    def _list_sync() -> list[GPUInstance]:
        modal = _import_modal()
        instances = []
        try:
            for sandbox in modal.Sandbox.list():
                instances.append(
                    GPUInstance(
                        id=sandbox.object_id,
                        provider="modal",
                        status=InstanceStatus.RUNNING,
                        gpu_type="unknown",
                        gpu_count=1,
                        price_per_hour=0.0,
                        raw_data={"sandbox_id": sandbox.object_id},
                    )
                )
        except Exception:
            logger.debug("modal.Sandbox.list() not available or failed")

        return instances

    return await trio.to_thread.run_sync(_list_sync)


async def terminate_instance(instance_id: str, api_key: str | None = None) -> bool:
    """Terminate a Modal sandbox."""
    assert instance_id, "instance_id cannot be empty"

    def _terminate_sync() -> bool:
        modal = _import_modal()
        try:
            sandbox = modal.Sandbox.from_id(instance_id)
            sandbox.terminate()
            logger.info(f"Modal sandbox terminated: {instance_id}")
            return True
        except Exception:
            logger.exception(f"Failed to terminate Modal sandbox: {instance_id}")
            return False

    return await trio.to_thread.run_sync(_terminate_sync)


# ============================================================================
# ProviderProtocol (wait_for_ssh_ready / get_fresh_instance)
# ============================================================================


def wait_for_ssh_ready(instance: Any, timeout: int = 900) -> bool:
    """Modal sandboxes are ready immediately — no SSH to wait for."""
    return True


def get_fresh_instance(instance_id: str, api_key: str) -> GPUInstance | None:
    """Alias for get_instance_details (sync version for ProviderProtocol)."""
    modal = _import_modal()
    try:
        modal.Sandbox.from_id(instance_id)  # Raises if not found
        return GPUInstance(
            id=instance_id,
            provider="modal",
            status=InstanceStatus.RUNNING,
            gpu_type="unknown",
            gpu_count=1,
            price_per_hour=0.0,
            raw_data={"sandbox_id": instance_id},
        )
    except Exception:
        return None


# ============================================================================
# Modal-specific exec (used by ModalGPUInstance)
# ============================================================================


async def exec_on_sandbox(instance_id: str, command: str, exec_timeout: int = 300) -> SSHResult:
    """Execute command on an existing Modal sandbox.

    This is the Modal equivalent of GPUInstance.exec() — called by
    ModalGPUInstance to route exec through sandbox.exec() instead of SSH.
    """
    deadline = exec_timeout

    def _exec_sync() -> SSHResult:
        modal = _import_modal()
        sandbox = modal.Sandbox.from_id(instance_id)
        return _sandbox_exec_sync(sandbox, command, timeout=deadline)

    return await trio.to_thread.run_sync(_exec_sync)
