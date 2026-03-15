"""Weight synchronization protocol for training-inference coordination.

Defines how trainer pushes updated weights to inference servers.
Multiple implementations for different deployment scenarios:
- DiskWeightSync: Shared filesystem (same machine or NFS)
- ModalVolumeWeightSync: Modal Volume (for Modal sandboxes, zero setup)
- R2WeightSync: Cloudflare R2 (S3-compatible, free egress)
- NCCLWeightSync: GPU-to-GPU broadcast (high-perf clusters)

Design (Option C from discussion):
- Dataclass holds config (immutable) + connection state (mutable)
- Functions operate on the dataclass
- No hidden state, explicit mutation

Usage:
    config = DiskWeightSyncConfig(sync_dir=Path("/dev/shm/weights"))
    sync = DiskWeightSync(config=config)

    connect_disk(sync, endpoints=["http://localhost:30000"])

    # In training loop:
    save_checkpoint(model, sync.config.sync_dir / f"v{version}")
    await sync_weights_disk(sync, checkpoint_path=..., version=version)

    disconnect_disk(sync)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
import trio

logger = logging.getLogger(__name__)


InferenceSyncTransport = Literal["filesystem", "nccl", "http", "custom_rpc", "in_process"]
InferenceSyncMode = Literal["blocking", "inflight"]
InferenceSyncMechanism = Literal[
    "checkpoint_path_reload",
    "current_model_root_reload",
    "tensor_broadcast",
]


@dataclass(frozen=True)
class InferenceSyncRealization:
    """Concrete runtime sync adapter for one inference backend realization."""

    name: str
    transport: InferenceSyncTransport
    mode: InferenceSyncMode
    mechanism: InferenceSyncMechanism
    requires_custom_server_patch: bool = False
    requires_worker_extension: bool = False
    requires_mutable_model_root: bool = False
    requires_sleep_wake: bool = False
    notes: str = ""


@dataclass(frozen=True)
class InferenceBackendCapabilities:
    """Capabilities of a concrete inference backend/runtime realization."""

    backend_name: str
    supported_sync_realizations: tuple[str, ...] = ()
    default_sync_realization: str | None = None
    supports_blocking_updates: bool = True
    supports_inflight_updates: bool = False
    capability_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.default_sync_realization is not None:
            assert self.default_sync_realization in self.supported_sync_realizations, (
                "default_sync_realization must be included in supported_sync_realizations"
            )


@dataclass(frozen=True)
class InferenceWeightUpdate:
    """Concrete update request issued to an inference backend."""

    checkpoint_path: str | None = None
    version: int | None = None
    realization: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


SGLANG_HTTP_PATH_RELOAD = InferenceSyncRealization(
    name="sglang_http_path_reload",
    transport="http",
    mode="blocking",
    mechanism="checkpoint_path_reload",
    notes="SGLang /update_weights_from_disk with a checkpoint path.",
)

ENGINE_V2_HTTP_PATH_RELOAD = InferenceSyncRealization(
    name="engine_v2_http_path_reload",
    transport="http",
    mode="blocking",
    mechanism="checkpoint_path_reload",
    notes="rollouts engine_v2 /update_weights_from_disk with a checkpoint path.",
)

VLLM_DEV_CURRENT_MODEL_ROOT_RELOAD = InferenceSyncRealization(
    name="vllm_dev_current_model_root_reload",
    transport="custom_rpc",
    mode="blocking",
    mechanism="current_model_root_reload",
    requires_mutable_model_root=True,
    requires_sleep_wake=True,
    notes=(
        "vLLM dev sleep/wake + collective_rpc reload_weights() with no model_path. "
        "Only honest when launched against a mutable local model root."
    ),
)

VLLM_CUSTOM_PATH_RELOAD = InferenceSyncRealization(
    name="vllm_custom_path_reload",
    transport="custom_rpc",
    mode="blocking",
    mechanism="checkpoint_path_reload",
    requires_custom_server_patch=True,
    requires_worker_extension=True,
    requires_sleep_wake=True,
    notes="PRIME-style custom vLLM route/worker extension for arbitrary checkpoint paths.",
)

VLLM_CUSTOM_NCCL_BROADCAST = InferenceSyncRealization(
    name="vllm_custom_nccl_broadcast",
    transport="nccl",
    mode="inflight",
    mechanism="tensor_broadcast",
    requires_custom_server_patch=True,
    requires_worker_extension=True,
    notes="QED-Nano-style direct tensor broadcast into vLLM workers.",
)


INFERENCE_SYNC_REALIZATIONS: dict[str, InferenceSyncRealization] = {
    realization.name: realization
    for realization in (
        SGLANG_HTTP_PATH_RELOAD,
        ENGINE_V2_HTTP_PATH_RELOAD,
        VLLM_DEV_CURRENT_MODEL_ROOT_RELOAD,
        VLLM_CUSTOM_PATH_RELOAD,
        VLLM_CUSTOM_NCCL_BROADCAST,
    )
}


def get_inference_sync_realization(name: str) -> InferenceSyncRealization:
    """Return the named sync realization or fail loudly."""
    try:
        return INFERENCE_SYNC_REALIZATIONS[name]
    except KeyError as exc:
        known = ", ".join(sorted(INFERENCE_SYNC_REALIZATIONS))
        raise ValueError(f"Unknown inference sync realization {name!r}. Known: {known}") from exc


def resolve_inference_sync_realization(
    capabilities: InferenceBackendCapabilities,
    requested: str | None,
) -> InferenceSyncRealization:
    """Resolve and validate the explicit sync realization for one backend."""
    realization_name = requested or capabilities.default_sync_realization
    if realization_name is None:
        notes = (
            " ".join(capabilities.capability_notes).strip()
            or f"{capabilities.backend_name} exposes no default sync realization."
        )
        raise ValueError(
            f"Inference backend {capabilities.backend_name!r} requires an explicit sync realization. "
            f"{notes}"
        )
    if realization_name not in capabilities.supported_sync_realizations:
        supported = ", ".join(capabilities.supported_sync_realizations) or "<none>"
        raise ValueError(
            f"Inference backend {capabilities.backend_name!r} does not support "
            f"sync realization {realization_name!r}. Supported: {supported}"
        )
    return get_inference_sync_realization(realization_name)


# ============================================================================
# Configs (frozen - immutable)
# ============================================================================


@dataclass(frozen=True)
class DiskWeightSyncConfig:
    """Config for disk-based weight sync.

    Uses SGLang/vLLM's update_weights_from_disk endpoint.
    Requires shared filesystem between trainer and inference server.
    """

    sync_dir: Path = field(default_factory=lambda: Path("/dev/shm/rollouts_weight_sync"))
    timeout_seconds: float = 300.0


@dataclass(frozen=True)
class ModalVolumeWeightSyncConfig:
    """Config for Modal Volume-based weight sync.

    Uses Modal's built-in volumes for sharing data between sandboxes.
    Zero external setup required - volumes are created on demand.

    Trainer writes checkpoint to volume, inference server reads from mount path.
    """

    volume_name: str = "rollouts-weight-sync"
    mount_path: str = "/vol/weights"  # Where volume is mounted in inference sandbox
    timeout_seconds: float = 300.0


@dataclass(frozen=True)
class R2WeightSyncConfig:
    """Config for Cloudflare R2-based weight sync.

    S3-compatible API with free egress. Good for remote inference.
    Requires R2 bucket + API credentials.
    """

    account_id: str = ""
    bucket: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    prefix: str = "checkpoints"
    timeout_seconds: float = 600.0


@dataclass(frozen=True)
class NCCLWeightSyncConfig:
    """Config for NCCL-based weight sync.

    Direct GPU-to-GPU broadcast via NCCL.
    Highest performance, requires network connectivity and NCCL setup.
    """

    master_addr: str = ""
    master_port: int = 29500
    timeout_seconds: float = 300.0


# ============================================================================
# State (mutable - holds connection info)
# ============================================================================


@dataclass
class DiskWeightSync:
    """Disk-based weight sync state."""

    config: DiskWeightSyncConfig
    endpoints: list[str] = field(default_factory=list)
    backend: str = "sglang"  # "sglang" or "vllm"


@dataclass
class ModalVolumeWeightSync:
    """Modal Volume-based weight sync state."""

    config: ModalVolumeWeightSyncConfig
    endpoints: list[str] = field(default_factory=list)
    backend: str = "sglang"
    _volume: Any = field(default=None, repr=False)  # modal.Volume


@dataclass
class R2WeightSync:
    """Cloudflare R2-based weight sync state."""

    config: R2WeightSyncConfig
    endpoints: list[str] = field(default_factory=list)
    backend: str = "sglang"
    _s3_client: Any = field(default=None, repr=False)  # boto3 client configured for R2


@dataclass
class NCCLWeightSync:
    """NCCL-based weight sync state."""

    config: NCCLWeightSyncConfig
    endpoints: list[str] = field(default_factory=list)
    backend: str = "sglang"
    # NCCL process group created during connect
    _process_group: Any = field(default=None, repr=False)
    _world_size: int = 0


# ============================================================================
# Functions - DiskWeightSync
# ============================================================================


def connect_disk(sync: DiskWeightSync, endpoints: list[str], backend: str = "sglang") -> None:
    """Connect to inference endpoints for disk-based sync.

    Creates sync directory if needed. No network setup required.
    """
    assert endpoints, "Must provide at least one endpoint"
    assert backend in ("sglang", "vllm"), f"Unknown backend: {backend}"

    sync.endpoints = list(endpoints)
    sync.backend = backend
    sync.config.sync_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"DiskWeightSync connected to {len(endpoints)} endpoints, sync_dir={sync.config.sync_dir}"
    )


async def sync_weights_disk(
    sync: DiskWeightSync,
    checkpoint_path: str | Path,
    version: int,
) -> list[dict[str, Any]]:
    """Sync weights to all connected endpoints via disk.

    Args:
        sync: DiskWeightSync state
        checkpoint_path: Path to checkpoint directory
        version: Weight version number (for logging/debugging)

    Returns:
        List of responses from each endpoint
    """
    assert sync.endpoints, "Must call connect_disk() first"
    checkpoint_path = str(checkpoint_path)

    results = []

    async def sync_one(endpoint: str) -> dict[str, Any]:
        if sync.backend == "sglang":
            return await _update_sglang_weights_from_disk(
                endpoint, checkpoint_path, sync.config.timeout_seconds
            )
        else:
            return await _update_vllm_weights_from_disk(
                endpoint, checkpoint_path, sync.config.timeout_seconds
            )

    # Sync to all endpoints in parallel
    async with trio.open_nursery() as nursery:

        async def sync_and_collect(endpoint: str) -> None:
            result = await sync_one(endpoint)
            results.append(result)

        for endpoint in sync.endpoints:
            nursery.start_soon(sync_and_collect, endpoint)

    logger.info(f"DiskWeightSync v{version} synced to {len(sync.endpoints)} endpoints")
    return results


def disconnect_disk(sync: DiskWeightSync) -> None:
    """Disconnect from endpoints. Clears state."""
    sync.endpoints = []
    logger.info("DiskWeightSync disconnected")


# ============================================================================
# Functions - ModalVolumeWeightSync
# ============================================================================


def connect_modal_volume(
    sync: ModalVolumeWeightSync, endpoints: list[str], backend: str = "sglang"
) -> None:
    """Connect to inference endpoints for Modal Volume-based sync.

    Creates or gets the Modal volume. Inference sandboxes must mount this
    volume at config.mount_path.
    """
    assert endpoints, "Must provide at least one endpoint"
    assert backend in ("sglang", "vllm"), f"Unknown backend: {backend}"

    try:
        import modal
    except ImportError as e:
        raise ImportError("Modal SDK required. Install with: pip install modal") from e

    sync.endpoints = list(endpoints)
    sync.backend = backend
    sync._volume = modal.Volume.from_name(sync.config.volume_name, create_if_missing=True)

    logger.info(
        f"ModalVolumeWeightSync connected to {len(endpoints)} endpoints, "
        f"volume={sync.config.volume_name}, mount={sync.config.mount_path}"
    )


async def sync_weights_modal_volume(
    sync: ModalVolumeWeightSync,
    checkpoint_path: str | Path,
    version: int,
) -> list[dict[str, Any]]:
    """Sync weights to all connected endpoints via Modal Volume.

    1. Upload checkpoint files to Modal Volume
    2. Tell inference server to reload from mount path

    Args:
        sync: ModalVolumeWeightSync state
        checkpoint_path: Local path to checkpoint directory
        version: Weight version number

    Returns:
        List of responses from each endpoint
    """
    assert sync.endpoints, "Must call connect_modal_volume() first"
    assert sync._volume is not None, "Volume not initialized"

    checkpoint_path = Path(checkpoint_path)
    volume_subpath = f"v{version}"

    # Upload checkpoint files to volume
    # Modal volume.put_directory is sync, run in thread

    def _upload() -> None:
        # Reload volume to ensure we have latest state
        sync._volume.reload()

        # Upload all files from checkpoint directory
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(checkpoint_path)
                volume_path = f"{volume_subpath}/{rel_path}"
                with open(file_path, "rb") as f:
                    sync._volume.write_file(volume_path, f.read())

        # Commit changes
        sync._volume.commit()

    await trio.to_thread.run_sync(_upload)

    # Path that inference server sees (volume mounted at mount_path)
    inference_path = f"{sync.config.mount_path}/{volume_subpath}"

    # Tell all endpoints to reload weights
    results = []

    async def sync_one(endpoint: str) -> dict[str, Any]:
        if sync.backend == "sglang":
            return await _update_sglang_weights_from_disk(
                endpoint, inference_path, sync.config.timeout_seconds
            )
        else:
            return await _update_vllm_weights_from_disk(
                endpoint, inference_path, sync.config.timeout_seconds
            )

    async with trio.open_nursery() as nursery:

        async def sync_and_collect(endpoint: str) -> None:
            result = await sync_one(endpoint)
            results.append(result)

        for endpoint in sync.endpoints:
            nursery.start_soon(sync_and_collect, endpoint)

    logger.info(f"ModalVolumeWeightSync v{version} synced to {len(sync.endpoints)} endpoints")
    return results


def disconnect_modal_volume(sync: ModalVolumeWeightSync) -> None:
    """Disconnect from endpoints."""
    sync.endpoints = []
    sync._volume = None
    logger.info("ModalVolumeWeightSync disconnected")


# ============================================================================
# Functions - R2WeightSync
# ============================================================================


def connect_r2(sync: R2WeightSync, endpoints: list[str], backend: str = "sglang") -> None:
    """Connect to inference endpoints for R2-based sync.

    Initializes boto3 S3 client configured for Cloudflare R2.
    """
    assert endpoints, "Must provide at least one endpoint"
    assert sync.config.account_id, "R2 account_id must be configured"
    assert sync.config.bucket, "R2 bucket must be configured"
    assert sync.config.access_key_id, "R2 access_key_id must be configured"
    assert sync.config.secret_access_key, "R2 secret_access_key must be configured"
    assert backend in ("sglang", "vllm"), f"Unknown backend: {backend}"

    try:
        import boto3
    except ImportError as e:
        raise ImportError("boto3 required for R2. Install with: pip install boto3") from e

    sync.endpoints = list(endpoints)
    sync.backend = backend

    # R2 endpoint URL
    endpoint_url = f"https://{sync.config.account_id}.r2.cloudflarestorage.com"

    sync._s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=sync.config.access_key_id,
        aws_secret_access_key=sync.config.secret_access_key,
        region_name="auto",  # R2 doesn't use regions
    )

    logger.info(
        f"R2WeightSync connected to {len(endpoints)} endpoints, bucket={sync.config.bucket}"
    )


async def sync_weights_r2(
    sync: R2WeightSync,
    checkpoint_path: str | Path,
    version: int,
) -> list[dict[str, Any]]:
    """Sync weights to all connected endpoints via Cloudflare R2.

    1. Upload checkpoint files to R2 bucket
    2. Tell inference server to reload from R2 path
       (requires inference server configured with R2 access)

    Args:
        sync: R2WeightSync state
        checkpoint_path: Local path to checkpoint directory
        version: Weight version number

    Returns:
        List of responses from each endpoint
    """
    assert sync.endpoints, "Must call connect_r2() first"
    assert sync._s3_client is not None, "S3 client not initialized"

    checkpoint_path = Path(checkpoint_path)
    s3_prefix = f"{sync.config.prefix}/v{version}"

    # Upload checkpoint files to R2
    def _upload() -> None:
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(checkpoint_path)
                s3_key = f"{s3_prefix}/{rel_path}"
                sync._s3_client.upload_file(str(file_path), sync.config.bucket, s3_key)

    await trio.to_thread.run_sync(_upload)

    # S3 path for inference server
    # Note: Inference server needs to be configured with R2 credentials
    # SGLang/vLLM support s3:// paths via fsspec/s3fs
    s3_path = f"s3://{sync.config.bucket}/{s3_prefix}"

    # Tell all endpoints to reload weights
    results = []

    async def sync_one(endpoint: str) -> dict[str, Any]:
        if sync.backend == "sglang":
            return await _update_sglang_weights_from_disk(
                endpoint, s3_path, sync.config.timeout_seconds
            )
        else:
            return await _update_vllm_weights_from_disk(
                endpoint, s3_path, sync.config.timeout_seconds
            )

    async with trio.open_nursery() as nursery:

        async def sync_and_collect(endpoint: str) -> None:
            result = await sync_one(endpoint)
            results.append(result)

        for endpoint in sync.endpoints:
            nursery.start_soon(sync_and_collect, endpoint)

    logger.info(f"R2WeightSync v{version} synced to {len(sync.endpoints)} endpoints")
    return results


def disconnect_r2(sync: R2WeightSync) -> None:
    """Disconnect from endpoints."""
    sync.endpoints = []
    sync._s3_client = None
    logger.info("R2WeightSync disconnected")


# ============================================================================
# Functions - NCCLWeightSync (stub)
# ============================================================================


def connect_nccl(sync: NCCLWeightSync, endpoints: list[str], backend: str = "sglang") -> None:
    """Connect to inference endpoints for NCCL-based sync.

    TODO:
    1. Get master address/port (trainer is rank 0)
    2. Call init_weights_update_group on each SGLang endpoint
    3. Initialize local NCCL process group

    See SLIME's connect_rollout_engines_from_distributed() for reference.
    """
    assert endpoints, "Must provide at least one endpoint"
    assert sync.config.master_addr, "NCCL master_addr must be configured"

    sync.endpoints = list(endpoints)
    sync.backend = backend

    # TODO: Initialize NCCL process group
    # world_size = len(endpoints) * gpus_per_endpoint + 1  # +1 for trainer
    #
    # for i, endpoint in enumerate(endpoints):
    #     requests.post(f"{endpoint}/init_weights_update_group", json={
    #         "master_address": sync.config.master_addr,
    #         "master_port": sync.config.master_port,
    #         "rank": i + 1,
    #         "world_size": world_size,
    #         "group_name": "rollouts_weight_sync",
    #         "backend": "nccl",
    #     })
    #
    # sync._process_group = dist.new_group(...)
    # sync._world_size = world_size

    raise NotImplementedError("NCCLWeightSync not yet implemented")


async def sync_weights_nccl(
    sync: NCCLWeightSync,
    model_state_dict: dict[str, Any],
    version: int,
) -> None:
    """Sync weights to all connected endpoints via NCCL broadcast.

    Note: Takes model state dict directly (not checkpoint path).

    TODO:
    1. Send metadata (param names, shapes, dtypes) via HTTP
    2. Broadcast each tensor via NCCL from rank 0
    3. Wait for completion

    See SLIME's update_weights_from_distributed() for reference.
    """
    raise NotImplementedError("NCCLWeightSync not yet implemented")


def disconnect_nccl(sync: NCCLWeightSync) -> None:
    """Disconnect from endpoints. Destroys NCCL process group."""
    # TODO: Destroy process group
    # if sync._process_group is not None:
    #     for endpoint in sync.endpoints:
    #         requests.post(f"{endpoint}/destroy_weights_update_group", ...)
    #     dist.destroy_process_group(sync._process_group)

    sync.endpoints = []
    sync._process_group = None
    sync._world_size = 0


# ============================================================================
# Internal helpers
# ============================================================================


async def _update_sglang_weights_from_disk(
    base_url: str,
    checkpoint_path: str,
    timeout: float,
) -> dict[str, Any]:
    """Call SGLang's /update_weights_from_disk endpoint."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base_url}/update_weights_from_disk",
            json={"model_path": checkpoint_path},
        )
        response.raise_for_status()
        return response.json()


async def _update_vllm_weights_from_disk(
    base_url: str,
    checkpoint_path: str,
    timeout: float,
) -> dict[str, Any]:
    """Call vLLM's collective_rpc endpoint with reload_weights."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base_url}/collective_rpc",
            json={
                "method": "reload_weights",
                "params": {"model_path": checkpoint_path},
            },
        )
        response.raise_for_status()
        return response.json()
