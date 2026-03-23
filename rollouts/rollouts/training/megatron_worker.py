"""Megatron training worker for miniray.

This is the work function that runs on each GPU. It initializes Megatron,
creates the model, and processes training commands from the coordinator.

Usage:
    # Launched by miniray Cluster
    cluster = Cluster(nodes=[NodeConfig("node1", num_workers=8)])
    workers = cluster.start(work_fn="rollouts.training.megatron_worker.train")
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import logging
import multiprocessing
import os
import sys
import time
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miniray import Worker

logger = logging.getLogger(__name__)
_ARGUS_DIAG_EVENT_SENTINEL = "__ARGUS_DIAG__"
_MEGATRON_SGLANG_BUCKET_SIZE_BYTES = 256 * 1024 * 1024
_ISOLATED_REMOTE_INIT_CONNECT_TIMEOUT_SEC = 5.0
# The isolated runtime sender creates a fresh custom process group while the
# inference engine is paused under live training load. The witness path already
# needed a larger coordinator-side timeout; keep the runtime init read timeout
# above the old 10s assumption so bucket init does not fail spuriously before
# the helper itself has a chance to report a real transport error.
_ISOLATED_REMOTE_INIT_READ_TIMEOUT_SEC = 30.0
# This helper wraps the whole isolated runtime publication lifecycle: start the
# subprocess, stand up rank 0's NCCL group, let the receiver join, ship the
# bucket, tear the group down, then unwind. The receiver-side init request
# already advertises a 300s timeout, so a fixed 60s outer wall-clock budget is
# too small and can kill an otherwise in-flight update. Keep this explicit and
# well below the receiver-side ceiling until it becomes a config-owned contract.
_ISOLATED_WEIGHT_SYNC_HELPER_TIMEOUT_SEC = 180.0
_MEGATRON_BATCH_DTYPES: dict[str, str] = {
    "input_ids": "long",
    "labels": "long",
    "position_ids": "long",
    "attention_mask": "float32",
    "loss_mask": "float32",
    "advantages": "float32",
    "old_logprobs": "float32",
    "teacher_logprobs": "float32",
    "group_ids": "long",
    "returns": "float32",
}
_MEGATRON_BATCH_FIELD_ORDER = tuple(_MEGATRON_BATCH_DTYPES)


def _emit_argus_diag(event: str, **data: object) -> None:
    """Best-effort structured diagnostics for remote sandbox runs."""
    try:
        sys.stderr.write(
            f"{_ARGUS_DIAG_EVENT_SENTINEL}{json.dumps({'event': event, **data}, sort_keys=True)}\n"
        )
        sys.stderr.flush()
    except Exception:
        return


def _sequence_hash(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class Command(IntEnum):
    """Command IDs for rank synchronization.

    Using explicit IDs instead of hash() for deterministic behavior.
    """

    SHUTDOWN = 0
    TRAIN_STEP = 1
    SYNC_WEIGHTS = 2
    SAVE_CHECKPOINT = 3
    INIT_NCCL_WEIGHT_SYNC = 4
    SYNC_WEIGHTS_NCCL = 5
    CLEANUP_NCCL_WEIGHT_SYNC = 6
    VALIDATE_INFERENCE_EXPORT = 7


def _read_proc_status_snapshot() -> str:
    """Return a compact /proc/self/status snapshot for import-stage debugging."""
    wanted_keys = {"VmRSS", "VmHWM", "VmSize", "Threads"}
    snapshot: dict[str, str] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                key, _, value = line.partition(":")
                if key in wanted_keys:
                    snapshot[key] = value.strip()
    except OSError as exc:
        return f"proc_status=unavailable error={exc}"

    parts = [f"pid={os.getpid()}"]
    for key in ("VmRSS", "VmHWM", "VmSize", "Threads"):
        value = snapshot.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _log_import_stage(stage: str) -> None:
    logger.info("import_stage=%s %s", stage, _read_proc_status_snapshot())


def _resolve_train_future(future: Any) -> Any:
    """Resolve the local synchronous future shape used by Megatron workers."""
    from rollouts.training.types import ImmediateTrainFuture

    if not isinstance(future, ImmediateTrainFuture):
        raise TypeError(
            "Megatron worker expected ImmediateTrainFuture from local backend; "
            f"got {type(future).__name__}"
        )
    return future._result


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.strip().lower()
    return normalized == "localhost" or normalized.startswith("127.") or normalized == "::1"


def _resolve_local_host_ip() -> tuple[str, str]:
    """Resolve a non-loopback IPv4 address for intra-sandbox rendezvous.

    Weight sync is a separate state machine from the training PG. Reusing an
    ambient loopback MASTER_ADDR here is dishonest because it smuggles the
    training rendezvous denotation into the trainer<->inference update channel.
    """
    import socket
    import subprocess

    env_master_addr = os.environ.get("MASTER_ADDR")
    if env_master_addr and not _is_loopback_host(env_master_addr):
        return env_master_addr.strip(), "env:MASTER_ADDR"

    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5.0,
        )
        for candidate in result.stdout.split():
            if "." not in candidate:
                continue
            if not candidate.startswith("127."):
                return candidate, "hostname -I"
        for candidate in result.stdout.split():
            if "." in candidate:
                return candidate, "hostname -I"
    except Exception:
        pass

    try:
        # UDP connect selects the outward-facing interface without requiring a handshake.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 1))
            candidate = sock.getsockname()[0]
            if candidate and not candidate.startswith("127."):
                return candidate, "udp_connect"
            if candidate:
                return candidate, "udp_connect"
    except OSError:
        pass

    try:
        infos = socket.getaddrinfo(
            socket.gethostname(),
            None,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
        )
        for family, socktype, proto, canonname, sockaddr in infos:
            del family, socktype, proto, canonname
            candidate = sockaddr[0]
            if candidate and not candidate.startswith("127."):
                return candidate, "getaddrinfo"
            if candidate:
                return candidate, "getaddrinfo"
    except OSError:
        pass

    return "127.0.0.1", "fallback:loopback"


def _allocate_tcp_port(preferred_port: int = 29500) -> tuple[int, str]:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("", preferred_port))
            return int(sock.getsockname()[1]), "preferred"
        except OSError:
            sock.bind(("", 0))
            return int(sock.getsockname()[1]), "ephemeral"


def _clear_training_pg_env_for_subprocess() -> None:
    for key in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
    ):
        os.environ.pop(key, None)


def _weight_wire_tensor_nbytes(item: Any) -> int:
    tensor = item.tensor
    return int(tensor.numel() * tensor.element_size())


def _bucket_weight_wire_tensors(
    tensors: tuple[Any, ...],
    *,
    bucket_size_bytes: int,
) -> list[tuple[Any, ...]]:
    if bucket_size_bytes <= 0:
        return [tensors] if tensors else []

    buckets: list[list[Any]] = [[]]
    current_size = 0
    for item in tensors:
        item_size = _weight_wire_tensor_nbytes(item)
        if current_size + item_size > bucket_size_bytes and buckets[-1]:
            buckets.append([])
            current_size = 0
        buckets[-1].append(item)
        current_size += item_size
    return [tuple(bucket) for bucket in buckets if bucket]


def _build_flattened_bucket_update(
    bucket_tensors: tuple[Any, ...],
    *,
    payload_kind: str,
    version: int | None,
    bucket_index: int,
    bucket_count: int,
) -> tuple[Any, dict[str, object]]:
    from rollouts.training.backends.megatron.sglang import FlattenedTensorBucket
    from rollouts.training.weight_sync_protocol import WeightUpdatePayload, WeightWireTensor

    named_tensors = [(item.load_name, item.tensor) for item in bucket_tensors]
    flattened_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    flattened_tensor = flattened_bucket.get_flattened_tensor()
    bucket_name = f"flattened_bucket_{bucket_index:04d}"
    bucket_payload = WeightUpdatePayload(
        tensors=(
            WeightWireTensor(
                wire_name=bucket_name,
                load_name=bucket_name,
                shape=tuple(int(dim) for dim in flattened_tensor.shape),
                dtype=str(flattened_tensor.dtype).replace("torch.", ""),
                tensor=flattened_tensor,
                payload_kind=payload_kind,
                metadata={
                    "source": "megatron_runtime_export",
                    "bucket_index": bucket_index,
                    "bucket_count": bucket_count,
                    "bucket_tensor_count": len(bucket_tensors),
                    "load_format": "flattened_bucket",
                },
            ),
        ),
        payload_kind=payload_kind,
        version=version,
        metadata={
            "source_contract": "megatron_inference_export",
            "bucket_index": bucket_index,
            "bucket_count": bucket_count,
            "load_format": "flattened_bucket",
        },
    )
    request_payload: dict[str, object] = {
        "names": [item.wire_name for item in bucket_tensors],
        "load_names": [item.load_name for item in bucket_tensors],
        "shapes": [list(item.shape) for item in bucket_tensors],
        "dtypes": [item.dtype for item in bucket_tensors],
        "load_format": "flattened_bucket",
    }
    return bucket_payload, request_payload


def _serialize_weight_sync_update(
    payload: Any,
    *,
    version: int,
    request_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    serialized_tensors = []
    for item in payload.tensors:
        serialized_tensors.append({
            "wire_name": item.wire_name,
            "load_name": item.load_name,
            "shape": tuple(item.shape),
            "dtype": item.dtype,
            "payload_kind": item.payload_kind,
            "metadata": dict(item.metadata),
            "cpu_tensor": item.tensor.detach().cpu().clone(),
        })
    return {
        "payload_kind": payload.payload_kind,
        "request_payload": (
            copy.deepcopy(request_payload) if request_payload is not None else None
        ),
        "tensors": serialized_tensors,
        "version": version,
    }


def _isolated_weight_sync_sender_main(
    result_queue: multiprocessing.queues.Queue,
    *,
    master_addr: str,
    master_port: int,
    inference_endpoints: list[str],
    group_name: str,
    updates: list[dict[str, object]],
) -> None:
    def _put_progress(stage: str, **data: object) -> None:
        payload = {"kind": "progress", "stage": stage, **data}
        result_queue.put(payload)
        _emit_argus_diag("weight_sync_megatron_isolated_sender_progress", **payload)

    try:
        import requests
        import torch

        from rollouts.inference.weight_sync import WeightSyncSender
        from rollouts.training.weight_sync_protocol import (
            InitWeightUpdateGroupRequest,
            InitWeightUpdateGroupResponse,
            ReceiveWeightUpdateRequest,
            WeightUpdatePayload,
            WeightWireTensor,
        )

        _clear_training_pg_env_for_subprocess()
        os.environ["NCCL_SHM_DISABLE"] = "1"
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "1")

        logger.info(
            "weight_sync_megatron_isolated_sender_start master=%s:%s group=%s endpoints=%s",
            master_addr,
            master_port,
            group_name,
            inference_endpoints,
        )
        _emit_argus_diag(
            "weight_sync_megatron_isolated_sender_start",
            master_addr=master_addr,
            master_port=master_port,
            group=group_name,
            endpoints=inference_endpoints,
            pid=os.getpid(),
            env={
                "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
                "MASTER_PORT": os.environ.get("MASTER_PORT"),
                "RANK": os.environ.get("RANK"),
                "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
                "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
                "LOCAL_WORLD_SIZE": os.environ.get("LOCAL_WORLD_SIZE"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )
        _put_progress(
            "sender_start",
            master_addr=master_addr,
            master_port=master_port,
            group=group_name,
            endpoints=inference_endpoints,
            pid=os.getpid(),
        )

        world_size = 1 + len(inference_endpoints)
        sender = WeightSyncSender(
            master_addr=master_addr,
            master_port=master_port,
            inference_world_size=len(inference_endpoints),
            group_name=group_name,
        )

        def _init_remote_endpoint(endpoint: str, rank_offset: int) -> dict[str, object]:
            started_at = time.monotonic()
            _put_progress(
                "remote_init_start",
                endpoint=endpoint,
                rank_offset=rank_offset,
                world_size=world_size,
                connect_timeout_sec=_ISOLATED_REMOTE_INIT_CONNECT_TIMEOUT_SEC,
                read_timeout_sec=_ISOLATED_REMOTE_INIT_READ_TIMEOUT_SEC,
            )
            response: requests.Response | None = None
            try:
                request = InitWeightUpdateGroupRequest(
                    master_address=master_addr,
                    master_port=master_port,
                    rank_offset=rank_offset,
                    world_size=world_size,
                    group_name=group_name,
                )
                response = requests.post(
                    f"{endpoint}/init_weights_update_group",
                    json=request.to_dict(),
                    timeout=(
                        _ISOLATED_REMOTE_INIT_CONNECT_TIMEOUT_SEC,
                        _ISOLATED_REMOTE_INIT_READ_TIMEOUT_SEC,
                    ),
                )
                response.raise_for_status()
                InitWeightUpdateGroupResponse.from_dict(response.json())
                elapsed_sec = round(time.monotonic() - started_at, 3)
                _put_progress(
                    "remote_init_ok",
                    endpoint=endpoint,
                    rank_offset=rank_offset,
                    status_code=response.status_code,
                    elapsed_sec=elapsed_sec,
                )
                return {
                    "endpoint": endpoint,
                    "rank_offset": rank_offset,
                    "status_code": response.status_code,
                    "elapsed_sec": elapsed_sec,
                }
            except Exception as exc:
                elapsed_sec = round(time.monotonic() - started_at, 3)
                status_code = None
                response_text = None
                if response is not None:
                    status_code = response.status_code
                    try:
                        response_text = response.text[:400]
                    except Exception:
                        response_text = "<response text unavailable>"
                logger.exception(
                    "weight_sync_megatron_isolated_remote_init_failed endpoint=%s rank_offset=%s",
                    endpoint,
                    rank_offset,
                )
                _emit_argus_diag(
                    "weight_sync_megatron_isolated_remote_init_failed",
                    endpoint=endpoint,
                    rank_offset=rank_offset,
                    elapsed_sec=elapsed_sec,
                    status_code=status_code,
                    response_text=response_text,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                _put_progress(
                    "remote_init_failed",
                    endpoint=endpoint,
                    rank_offset=rank_offset,
                    elapsed_sec=elapsed_sec,
                    status_code=status_code,
                    response_text=response_text,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise

        init_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(inference_endpoints))
        )
        init_futures: list[tuple[str, int, concurrent.futures.Future[dict[str, object]]]] = []
        try:
            for index, endpoint in enumerate(inference_endpoints, start=1):
                init_futures.append((
                    endpoint,
                    index,
                    init_executor.submit(_init_remote_endpoint, endpoint, index),
                ))

            # The receiver-side HTTP handler blocks inside dist.init_process_group(),
            # so rank 0 must join while those requests are still in flight.
            _put_progress("sender_pg_init_start", world_size=world_size)
            sender.init_group()
            _put_progress("sender_pg_init_ok", world_size=world_size)

            for endpoint, rank_offset, future in init_futures:
                _put_progress(
                    "remote_init_wait_start",
                    endpoint=endpoint,
                    rank_offset=rank_offset,
                )
                future.result()
        finally:
            init_executor.shutdown(wait=False, cancel_futures=True)

        update_connect_timeout_sec = 5.0
        update_read_timeout_sec = 300.0

        for update_index, update in enumerate(updates, start=1):
            payload_kind = str(update["payload_kind"])
            version = int(update["version"])
            update_tensors = update["tensors"]
            request_payload = update.get("request_payload")
            if request_payload is not None and not isinstance(request_payload, dict):
                raise TypeError(
                    f"Expected request_payload dict or None, got {type(request_payload).__name__}"
                )

            weight_tensors: list[WeightWireTensor] = []
            for item in update_tensors:
                cpu_tensor = item["cpu_tensor"]
                if not isinstance(cpu_tensor, torch.Tensor):
                    raise TypeError(f"Expected cpu_tensor Tensor, got {type(cpu_tensor).__name__}")
                gpu_tensor = cpu_tensor.to(device="cuda", non_blocking=False)
                weight_tensors.append(
                    WeightWireTensor(
                        wire_name=str(item["wire_name"]),
                        load_name=str(item["load_name"]),
                        shape=tuple(int(dim) for dim in item["shape"]),
                        dtype=str(item["dtype"]),
                        tensor=gpu_tensor,
                        payload_kind=str(item["payload_kind"]),
                        metadata=dict(item.get("metadata", {})),
                    )
                )
            _put_progress(
                "payload_ready",
                update_index=update_index,
                update_count=len(updates),
                tensor_count=len(weight_tensors),
                first_tensor=weight_tensors[0].wire_name if weight_tensors else None,
            )

            if request_payload is None:
                request_payload = ReceiveWeightUpdateRequest(
                    names=tuple(item.wire_name for item in weight_tensors),
                    load_names=tuple(item.load_name for item in weight_tensors),
                    shapes=tuple(item.shape for item in weight_tensors),
                    dtypes=tuple(item.dtype for item in weight_tensors),
                ).to_dict()

            update_load_format = request_payload.get("load_format")
            if update_load_format is None and len(weight_tensors) == 1:
                update_load_format = "flattened_bucket"
                request_payload["load_format"] = update_load_format

            def _update_remote_endpoint(endpoint: str) -> dict[str, object]:
                started_at = time.monotonic()
                request_body: dict[str, object] = {
                    **request_payload,
                    "group_name": group_name,
                    "flush_cache": False,
                    "weight_version": str(version),
                }
                _put_progress(
                    "remote_update_start",
                    endpoint=endpoint,
                    update_index=update_index,
                    update_count=len(updates),
                    tensor_count=len(weight_tensors),
                    request_tensor_count=len(request_body.get("names", [])),
                    group=group_name,
                    version=version,
                    load_format=update_load_format,
                    connect_timeout_sec=update_connect_timeout_sec,
                    read_timeout_sec=update_read_timeout_sec,
                )
                response: requests.Response | None = None
                try:
                    response = requests.post(
                        f"{endpoint}/update_weights_from_distributed",
                        json=request_body,
                        timeout=(update_connect_timeout_sec, update_read_timeout_sec),
                    )
                    elapsed_sec = round(time.monotonic() - started_at, 3)
                    response.raise_for_status()
                    _put_progress(
                        "remote_update_ok",
                        endpoint=endpoint,
                        update_index=update_index,
                        update_count=len(updates),
                        status_code=response.status_code,
                        elapsed_sec=elapsed_sec,
                        load_format=update_load_format,
                    )
                    return {
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "elapsed_sec": elapsed_sec,
                        "load_format": update_load_format,
                    }
                except Exception as exc:
                    elapsed_sec = round(time.monotonic() - started_at, 3)
                    status_code = None
                    response_text = None
                    if response is not None:
                        status_code = response.status_code
                        try:
                            response_text = response.text[:400]
                        except Exception:
                            response_text = "<response text unavailable>"
                    logger.exception(
                        "weight_sync_megatron_isolated_remote_update_failed endpoint=%s",
                        endpoint,
                    )
                    _emit_argus_diag(
                        "weight_sync_megatron_isolated_remote_update_failed",
                        endpoint=endpoint,
                        group=group_name,
                        version=version,
                        update_index=update_index,
                        update_count=len(updates),
                        tensor_count=len(weight_tensors),
                        request_tensor_count=len(request_body.get("names", [])),
                        load_format=update_load_format,
                        elapsed_sec=elapsed_sec,
                        status_code=status_code,
                        response_text=response_text,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    _put_progress(
                        "remote_update_failed",
                        endpoint=endpoint,
                        group=group_name,
                        version=version,
                        update_index=update_index,
                        update_count=len(updates),
                        tensor_count=len(weight_tensors),
                        request_tensor_count=len(request_body.get("names", [])),
                        load_format=update_load_format,
                        elapsed_sec=elapsed_sec,
                        status_code=status_code,
                        response_text=response_text,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise

            futures = []
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(inference_endpoints))
            )
            try:
                for endpoint in inference_endpoints:
                    futures.append((endpoint, executor.submit(_update_remote_endpoint, endpoint)))

                _put_progress(
                    "broadcast_start",
                    update_index=update_index,
                    update_count=len(updates),
                )
                sender.broadcast_payload(
                    WeightUpdatePayload(
                        tensors=tuple(weight_tensors),
                        payload_kind=payload_kind,
                        version=version,
                        metadata={"isolated_sender": True, "update_index": update_index},
                    ),
                    async_op=True,
                    advance_version=False,
                )
                _put_progress(
                    "broadcast_ok",
                    update_index=update_index,
                    update_count=len(updates),
                )

                for endpoint, future in futures:
                    _put_progress(
                        "remote_update_wait_start",
                        endpoint=endpoint,
                        update_index=update_index,
                        update_count=len(updates),
                    )
                    future.result()
            finally:
                executor.shutdown(wait=False)

        destroy_timeout_sec = 10.0
        destroy_futures = []
        destroy_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(inference_endpoints))
        )
        try:
            for endpoint in inference_endpoints:
                _put_progress("remote_destroy_start", endpoint=endpoint, group=group_name)
                destroy_futures.append((
                    endpoint,
                    destroy_executor.submit(
                        requests.post,
                        f"{endpoint}/destroy_weights_update_group",
                        json={"group_name": group_name},
                        timeout=destroy_timeout_sec,
                    ),
                ))
            for endpoint, future in destroy_futures:
                try:
                    response = future.result()
                    response.raise_for_status()
                    _put_progress(
                        "remote_destroy_ok",
                        endpoint=endpoint,
                        group=group_name,
                        status_code=response.status_code,
                    )
                except Exception as exc:
                    logger.exception(
                        "weight_sync_megatron_isolated_remote_destroy_failed endpoint=%s",
                        endpoint,
                    )
                    _put_progress(
                        "remote_destroy_failed",
                        endpoint=endpoint,
                        group=group_name,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
        finally:
            destroy_executor.shutdown(wait=False)
            _put_progress("sender_cleanup_start")
            sender.cleanup()
            _put_progress("sender_cleanup_ok")

        result_queue.put({"kind": "result", "ok": True})
    except Exception as exc:
        logger.exception("weight_sync_megatron_isolated_sender_failed")
        _emit_argus_diag(
            "weight_sync_megatron_isolated_sender_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        result_queue.put({
            "kind": "result",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        })


def _run_isolated_weight_sync_sender(
    *,
    master_addr: str,
    master_port: int,
    inference_endpoints: list[str],
    group_name: str,
    updates: list[dict[str, object]],
    timeout_sec: float = _ISOLATED_WEIGHT_SYNC_HELPER_TIMEOUT_SEC,
) -> None:
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue(maxsize=64)
    helper = ctx.Process(
        target=_isolated_weight_sync_sender_main,
        kwargs={
            "result_queue": result_queue,
            "master_addr": master_addr,
            "master_port": master_port,
            "inference_endpoints": list(inference_endpoints),
            "group_name": group_name,
            "updates": updates,
        },
        daemon=False,
    )
    helper.start()
    helper_result: dict[str, object] | None = None
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        while not result_queue.empty():
            message = result_queue.get_nowait()
            kind = message.get("kind")
            if kind == "progress":
                logger.info(
                    "weight_sync_megatron_isolated_sender_progress stage=%s data=%s",
                    message.get("stage"),
                    {k: v for k, v in message.items() if k not in {"kind", "stage"}},
                )
            elif kind == "result":
                helper_result = message
                break
        if helper_result is not None:
            break
        if not helper.is_alive():
            break
        helper.join(timeout=0.5)
    if helper_result is None and helper.is_alive():
        helper.terminate()
        helper.join(timeout=5.0)
        raise TimeoutError(f"Isolated weight sync sender helper timed out after {timeout_sec:.0f}s")
    if helper_result is None:
        while not result_queue.empty():
            message = result_queue.get_nowait()
            if message.get("kind") == "result":
                helper_result = message
                break
        if helper_result is None:
            raise RuntimeError(
                f"Isolated weight sync sender exited without result exitcode={helper.exitcode}"
            )
    if not helper_result.get("ok"):
        raise RuntimeError(f"Isolated weight sync sender failed: {helper_result.get('error')}")


def _select_native_loss_fn(config: dict[str, Any]) -> Any:
    """Select the backend-native loss function fixed for this worker."""
    from rollouts.training.backends.megatron_backend import (
        megatron_grpo_loss,
        megatron_grpo_loss_clipped,
        megatron_grpo_loss_masked,
        megatron_opd_loss,
    )

    loss_type = config.get("loss_type", "vanilla")
    if loss_type == "vanilla":
        return megatron_grpo_loss
    if loss_type == "clipped":
        return megatron_grpo_loss_clipped
    if loss_type == "masked":
        ratio_low = float(config.get("mask_ratio_low", 0.125))
        ratio_high = float(config.get("mask_ratio_high", 8.0))

        def _masked(logits: Any, batch: dict[str, Any]) -> Any:
            return megatron_grpo_loss_masked(
                logits,
                batch,
                ratio_low=ratio_low,
                ratio_high=ratio_high,
            )

        return _masked
    if loss_type == "opd":
        return megatron_opd_loss
    raise ValueError(
        f"Unknown Megatron worker loss_type={loss_type!r}. "
        "Use 'vanilla', 'clipped', 'masked', or 'opd'."
    )


def _pause_and_flush_inference_endpoints(inference_endpoints: list[str]) -> None:
    """Quiesce SGLang before NCCL update.

    Miles/Slime pause generation and flush request state before distributed
    weight publication. Keep that effect backend-local instead of leaking it
    into the training semantics layer.
    """
    if not inference_endpoints:
        return

    import time

    import requests

    def _pause(endpoint: str) -> None:
        response = requests.post(f"{endpoint}/pause_generation", json={}, timeout=30.0)
        response.raise_for_status()

    def _flush(endpoint: str) -> None:
        last_error: Exception | None = None
        for _ in range(60):
            try:
                response = requests.get(f"{endpoint}/flush_cache", timeout=5.0)
                if response.status_code == 200:
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(1.0)
        if last_error is not None:
            raise RuntimeError(f"Timed out flushing inference cache for {endpoint}") from last_error
        raise RuntimeError(f"Timed out flushing inference cache for {endpoint}")

    logger.info("weight_sync_megatron_quiesce_start endpoints=%s", inference_endpoints)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(inference_endpoints))
    ) as executor:
        pause_futures = [executor.submit(_pause, endpoint) for endpoint in inference_endpoints]
        for future in pause_futures:
            future.result()
        flush_futures = [executor.submit(_flush, endpoint) for endpoint in inference_endpoints]
        for future in flush_futures:
            future.result()
    logger.info("weight_sync_megatron_quiesce_ok endpoints=%s", inference_endpoints)


def _resume_inference_endpoints(inference_endpoints: list[str]) -> None:
    """Resume SGLang generation after NCCL update."""
    if not inference_endpoints:
        return

    import requests

    logger.info("weight_sync_megatron_resume_start endpoints=%s", inference_endpoints)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(inference_endpoints))
    ) as executor:
        futures = [
            executor.submit(
                requests.post,
                f"{endpoint}/continue_generation",
                json={},
                timeout=30.0,
            )
            for endpoint in inference_endpoints
        ]
        for future in futures:
            response = future.result()
            response.raise_for_status()
    logger.info("weight_sync_megatron_resume_ok endpoints=%s", inference_endpoints)


def _emit_command_error_diagnostics(
    handle: Worker,
    *,
    backend: Any,
    rank: int,
    command_name: str,
    exc: BaseException,
) -> None:
    """Best-effort structured command failure for the control/logging boundary."""
    import traceback

    tb = traceback.format_exc()
    error = f"Worker rank {rank} failed during {command_name}: {type(exc).__name__}: {exc}"
    traceback_tail = tb[-8000:] if tb else ""
    _emit_argus_diag(
        "megatron_worker_command_failed",
        rank=rank,
        command_name=command_name,
        error=error,
        traceback_tail=traceback_tail if traceback_tail else None,
        sender_contract=getattr(backend, "_last_weight_sync_sender_contract", None),
    )
    if rank != 0:
        return

    payload = {"status": "error", "error": error}
    if traceback_tail:
        payload["traceback_tail"] = traceback_tail
    sender_contract = getattr(backend, "_last_weight_sync_sender_contract", None)
    if sender_contract is not None:
        payload["sender_contract"] = sender_contract
    try:
        handle.send(payload)
    except Exception:
        pass


def _send_progress_event(
    handle: Worker | None,
    *,
    event: str,
    **data: object,
) -> None:
    """Best-effort non-terminal control-plane event for rank-0 worker progress."""
    _emit_argus_diag(event, **data)
    if handle is None:
        return
    try:
        handle.send({"status": "event", "event": event, **data})
    except Exception:
        pass


def _torch_dtype_from_name(name: str) -> Any:
    import torch

    if name == "long":
        return torch.long
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported Megatron worker batch dtype {name!r}")


def _tensorize_megatron_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Normalize rank-0 IPC batch data into the backend-native tensor batch."""
    import torch

    unsupported_keys = set(batch) - set(_MEGATRON_BATCH_DTYPES)
    if unsupported_keys:
        raise ValueError(
            f"Megatron worker received unsupported batch keys {sorted(unsupported_keys)!r}."
        )

    tensor_batch: dict[str, Any] = {}
    for key in _MEGATRON_BATCH_FIELD_ORDER:
        if key not in batch:
            continue
        value = batch[key]
        if value is None:
            tensor_batch[key] = None
            continue
        tensor_batch[key] = torch.tensor(
            value,
            dtype=_torch_dtype_from_name(_MEGATRON_BATCH_DTYPES[key]),
            device="cuda",
        )
    return tensor_batch


def _broadcast_megatron_batch(
    batch: dict[str, Any] | None,
    *,
    rank: int,
) -> dict[str, Any]:
    """Broadcast the full backend-native Megatron batch contract to all ranks."""
    import torch.distributed as dist

    metadata_list: list[tuple[str, list[int]]] | None = None
    if rank == 0:
        assert batch is not None, "Rank 0 must have batch"
        tensor_batch = _tensorize_megatron_batch(batch)
        metadata_list = []
        for key in _MEGATRON_BATCH_FIELD_ORDER:
            if key not in tensor_batch:
                continue
            value = tensor_batch[key]
            if value is None:
                continue
            metadata_list.append((key, list(value.shape)))
    else:
        import torch

        tensor_batch = {}

    metadata_box = [metadata_list]
    dist.broadcast_object_list(metadata_box, src=0)
    received_metadata = metadata_box[0]
    assert received_metadata is not None, "Megatron worker batch metadata missing"

    if rank != 0:
        import torch

        for key, shape in received_metadata:
            tensor_batch[key] = torch.empty(
                shape,
                dtype=_torch_dtype_from_name(_MEGATRON_BATCH_DTYPES[key]),
                device="cuda",
            )

    for key, _shape in received_metadata:
        dist.broadcast(tensor_batch[key], src=0)

    for key in _MEGATRON_BATCH_FIELD_ORDER:
        tensor_batch.setdefault(key, None)

    return tensor_batch


def train(handle: Worker) -> None:
    """Miniray work function for Megatron distributed training.

    Protocol:
    1. Receive "init" message with rank, world_size, config
    2. Initialize Megatron process groups
    3. Create model/optimizer
    4. Loop: receive commands, execute, respond

    Commands:
    - {"cmd": "init", "rank": int, "world_size": int, "config": dict}
    - {"cmd": "train_step", "batch": dict}
    - {"cmd": "sync_weights"}
    - {"cmd": "save_checkpoint", "path": str}
    - {"cmd": "shutdown"}

    Only rank 0 sends responses back to coordinator.
    """
    import sys
    import traceback

    # Phase 1: Wait for init message
    _emit_argus_diag("megatron_worker_waiting_for_init")
    init_msg = handle.recv(max_size=10 * 1024 * 1024)  # 10MB for config
    assert init_msg["cmd"] == "init", f"Expected init, got {init_msg['cmd']}"

    rank = init_msg["rank"]
    world_size = init_msg["world_size"]
    config = init_msg["config"]
    megatron_overrides = config.get("megatron_overrides") or {}
    _emit_argus_diag(
        "megatron_worker_init_msg_received",
        rank=rank,
        world_size=world_size,
        has_checkpoint=bool(config.get("checkpoint_path")),
    )

    # Set up logging with rank
    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank {rank}] %(levelname)s %(name)s: %(message)s",
        force=True,  # Override any existing config
    )
    logger.info("Worker starting: rank=%d/%d", rank, world_size)

    # Set CUDA device BEFORE importing torch
    local_rank = rank % 8  # Assume max 8 GPUs per node
    cuda_device_ids = config.get("cuda_device_ids")
    if cuda_device_ids and rank < len(cuda_device_ids):
        cuda_device = cuda_device_ids[rank]
    else:
        # Default: use local_rank directly (rank 0 -> GPU 0, rank 1 -> GPU 1, etc.)
        cuda_device = local_rank
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    if megatron_overrides.get("allocator_expandable_segments"):
        current_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
        alloc_parts = [part for part in current_alloc_conf.split(",") if part]
        if "expandable_segments:True" not in alloc_parts:
            alloc_parts.append("expandable_segments:True")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(alloc_parts)
    logger.info("Worker rank %d using CUDA_VISIBLE_DEVICES=%s", rank, cuda_device)

    try:
        # Phase 2: Initialize Megatron
        _log_import_stage("start")
        _log_import_stage("before_initialize_import")
        from rollouts.training.backends.megatron.initialize import (
            MegatronParallelismConfig,
            init_megatron,
        )

        _log_import_stage("after_initialize_import")
        _log_import_stage("before_model_import")
        from rollouts.training.backends.megatron.model import (
            MegatronModelConfig,
            setup_megatron_model,
        )

        _log_import_stage("after_model_import")
        _log_import_stage("before_backend_import")
        from rollouts.training.backends.megatron_backend import (
            MegatronConfig,
            MegatronTrainingBackend,
        )

        _log_import_stage("after_backend_import")
        _log_import_stage("before_lowering_import")
        from rollouts.training.lowering import (
            MegatronLowering,
            MegatronProvisioning,
            RealizationPlan,
        )

        _log_import_stage("after_lowering_import")

        logger.info("Megatron imports successful")
        _emit_argus_diag("megatron_worker_imports_ok", rank=rank)

        lowering_payload = config["lowering"]
        lowering = MegatronLowering.from_realization(
            provisioning=MegatronProvisioning(**lowering_payload["provisioning"]),
            realization=RealizationPlan(**lowering_payload["realization"]),
        )
        provisioning = lowering.provisioning

        parallelism_config = MegatronParallelismConfig(
            tensor_parallel_size=provisioning.tp,
            pipeline_parallel_size=provisioning.pp,
            expert_parallel_size=provisioning.ep,
            sequence_parallel=config.get("sequence_parallel", False),
        )

        logger.info("Calling init_megatron...")
        _emit_argus_diag("megatron_worker_init_megatron_start", rank=rank)
        init_megatron(
            rank=rank,
            world_size=world_size,
            config=parallelism_config,
            master_addr=config.get("master_addr"),
            master_port=config.get("master_port"),
            global_batch_size=config.get("global_batch_size", 8),
            micro_batch_size=config.get("micro_batch_size", 1),
            seq_length=config.get("seq_length", 4096),
        )
        logger.info("init_megatron complete")
        _emit_argus_diag("megatron_worker_init_megatron_ok", rank=rank)

        # Phase 3: Create model
        logger.info("Creating model config...")
        model_config = MegatronModelConfig(
            model_name=config["model_name"],
            lr=config.get("lr", 1e-6),
            bf16=config.get("bf16", True),
            micro_batch_size=config.get("micro_batch_size", 1),
            global_batch_size=config.get("global_batch_size", 8),
            seq_length=config.get("seq_length", 4096),
            sequence_parallel=config.get("sequence_parallel", False),
        )

        checkpoint_path_raw = config.get("checkpoint_path")
        checkpoint_path = Path(checkpoint_path_raw) if checkpoint_path_raw else None
        checkpoint_dir_raw = config.get("checkpoint_dir")
        checkpoint_dir = Path(checkpoint_dir_raw) if checkpoint_dir_raw else Path("./checkpoints")

        logger.info("Setting up Megatron model...")
        _emit_argus_diag(
            "megatron_worker_model_setup_start",
            rank=rank,
            has_checkpoint=checkpoint_path is not None,
        )
        try:
            model, optimizer, scheduler, checkpoint_iteration = setup_megatron_model(
                model_config,
                checkpoint_path=checkpoint_path,
                save_optimizer_state=config.get("save_optimizer_state", True),
            )
        except Exception as exc:
            logger.exception("Megatron model setup failed")
            _emit_argus_diag(
                "megatron_worker_model_setup_failed",
                rank=rank,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        logger.info("Model setup complete")
        _emit_argus_diag(
            "megatron_worker_model_setup_ok",
            rank=rank,
            checkpoint_iteration=int(checkpoint_iteration),
        )

        # Create backend
        backend_config = MegatronConfig(
            tensor_model_parallel_size=parallelism_config.tensor_parallel_size,
            pipeline_model_parallel_size=parallelism_config.pipeline_parallel_size,
            expert_model_parallel_size=parallelism_config.expert_parallel_size,
            sequence_parallel=parallelism_config.sequence_parallel,
            micro_batch_size=model_config.micro_batch_size,
            global_batch_size=model_config.global_batch_size,
            num_microbatches=config.get("num_microbatches", 1),
            seq_length=model_config.seq_length,
            save_optimizer_state=config.get("save_optimizer_state", True),
            clip_grad=config.get("clip_grad", 1.0),
            bf16=model_config.bf16,
        )

        backend = MegatronTrainingBackend(
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            config=backend_config,
            checkpoint_dir=checkpoint_dir,
            lowering=lowering,
            loss_fn=_select_native_loss_fn(config),
        )

        if checkpoint_path is not None:
            backend._step = int(checkpoint_iteration)
            logger.info("Restored backend step from checkpoint: %s", backend._step)

        logger.info("Model initialized, entering training loop")
        _emit_argus_diag(
            "megatron_worker_backend_ready",
            rank=rank,
            step=int(getattr(backend, "_step", 0)),
        )

        # Rank 0 confirms init complete
        if rank == 0:
            _emit_argus_diag("megatron_worker_initialized_send_start", rank=rank)
            handle.send({"status": "initialized", "step": int(getattr(backend, "_step", 0))})
            _emit_argus_diag("megatron_worker_initialized_send_ok", rank=rank)

        # Phase 4: Training loop
        _training_loop(handle, backend, rank, config)

    except Exception as e:
        # Capture full traceback and send to coordinator
        tb = traceback.format_exc()
        error_msg = f"Worker rank {rank} failed: {e}\n{tb}"
        logger.exception(error_msg)
        print(error_msg, file=sys.stderr, flush=True)

        # Keep the control channel small and structured; full tracebacks already
        # go to stderr/logs and can exceed the miniray init message size.
        if rank == 0:
            try:
                handle.send({"status": "error", "error": str(e)})
            except Exception:
                pass  # Socket might be closed

        raise  # Re-raise to trigger worker exit


def _training_loop(
    handle: Worker,
    backend: Any,
    rank: int,
    config: dict[str, Any],
) -> None:
    """Main training loop - receive commands, execute, respond.

    Args:
        handle: Miniray worker handle for IPC
        backend: MegatronTrainingBackend instance
        rank: This worker's rank
        config: Training config dict
    """
    import torch
    import torch.distributed as dist

    # Map command strings to enum values
    CMD_MAP = {
        "shutdown": Command.SHUTDOWN,
        "train_step": Command.TRAIN_STEP,
        "sync_weights": Command.SYNC_WEIGHTS,
        "init_nccl_weight_sync": Command.INIT_NCCL_WEIGHT_SYNC,
        "sync_weights_nccl": Command.SYNC_WEIGHTS_NCCL,
        "cleanup_nccl_weight_sync": Command.CLEANUP_NCCL_WEIGHT_SYNC,
        "validate_inference_export": Command.VALIDATE_INFERENCE_EXPORT,
        "save_checkpoint": Command.SAVE_CHECKPOINT,
    }
    CMD_NAME = {value: key for key, value in CMD_MAP.items()}

    while True:
        # All ranks wait for command from coordinator
        # Rank 0 receives directly, others wait for broadcast
        if rank == 0:
            msg = handle.recv(max_size=100 * 1024 * 1024)  # 100MB for batches
            cmd_str = msg["cmd"]
            cmd_id = CMD_MAP.get(cmd_str, -1)
            if cmd_id == -1:
                error_msg = f"Unknown command: {cmd_str}"
                logger.error(error_msg)
                handle.send({"status": "error", "error": error_msg})
                raise ValueError(error_msg)

            # Broadcast command ID to other ranks
            cmd_tensor = torch.tensor([cmd_id], dtype=torch.long, device="cuda")
            dist.broadcast(cmd_tensor, src=0)
        else:
            # Receive broadcast command ID
            cmd_tensor = torch.zeros(1, dtype=torch.long, device="cuda")
            dist.broadcast(cmd_tensor, src=0)
            cmd_id = int(cmd_tensor.item())
            msg = {}

            if cmd_id == -1:
                raise ValueError(f"Unknown command ID: {cmd_id}")

        checkpoint_step = 0
        if cmd_id == Command.SAVE_CHECKPOINT:
            if rank == 0:
                checkpoint_step = int(msg.get("step", 0))
                checkpoint_step_tensor = torch.tensor(
                    [checkpoint_step], dtype=torch.long, device="cuda"
                )
            else:
                checkpoint_step_tensor = torch.zeros(1, dtype=torch.long, device="cuda")
            dist.broadcast(checkpoint_step_tensor, src=0)
            checkpoint_step = int(checkpoint_step_tensor.item())

        command_name = CMD_NAME.get(Command(cmd_id), f"unknown_{cmd_id}")

        # Handle shutdown
        if cmd_id == Command.SHUTDOWN:
            if rank == 0:
                logger.info("Shutdown requested")
            break

        try:
            logger.info("command_start name=%s rank=%s", command_name, rank)

            # Handle train_step
            if cmd_id == Command.TRAIN_STEP:
                batch = msg.get("batch") if rank == 0 else None
                metrics = _do_train_step(backend, batch, rank)
                if rank == 0:
                    handle.send({"status": "ok", "metrics": metrics})

            # Handle sync_weights
            elif cmd_id == Command.SYNC_WEIGHTS:
                _do_sync_weights(
                    backend, config.get("inference_endpoints", []) if rank == 0 else []
                )
                if rank == 0:
                    handle.send({"status": "synced"})
            elif cmd_id == Command.INIT_NCCL_WEIGHT_SYNC:
                if rank == 0:
                    _init_nccl_weight_sync(
                        backend,
                        inference_endpoints=msg.get(
                            "inference_endpoints",
                            config.get("inference_endpoints", []),
                        ),
                        model_name=msg.get("model_name", config.get("model_name", "")),
                        master_addr=msg.get("master_addr", config.get("master_addr")),
                        master_port=msg.get("master_port", config.get("master_port", 29500)),
                    )
                    handle.send({"status": "nccl_initialized"})
                else:
                    _init_nccl_weight_sync(
                        backend,
                        inference_endpoints=[],
                        model_name=msg.get("model_name", config.get("model_name", "")),
                        master_addr=msg.get("master_addr", config.get("master_addr")),
                        master_port=msg.get("master_port", config.get("master_port", 29500)),
                    )

            elif cmd_id == Command.SYNC_WEIGHTS_NCCL:
                _do_sync_weights_nccl(
                    backend,
                    handle if rank == 0 else None,
                    model_name=config.get("model_name", ""),
                    inference_endpoints=config.get("inference_endpoints", []) if rank == 0 else [],
                    inference_dtype=_inference_dtype_from_worker_config(config),
                    tensor_limit=msg.get("tensor_limit") if rank == 0 else None,
                    witness=bool(msg.get("witness", False)) if rank == 0 else False,
                )
                if rank == 0:
                    handle.send({"status": "nccl_synced"})

            elif cmd_id == Command.CLEANUP_NCCL_WEIGHT_SYNC:
                if rank == 0:
                    _cleanup_nccl_weight_sync(
                        backend,
                        config.get("inference_endpoints", []),
                    )
                    handle.send({"status": "nccl_cleanup"})
                else:
                    _cleanup_nccl_weight_sync(backend, [])

            elif cmd_id == Command.VALIDATE_INFERENCE_EXPORT:
                details = _do_validate_inference_export(
                    backend,
                    model_name=config.get("model_name", ""),
                )
                if rank == 0:
                    handle.send({"status": "validated", "details": details})

            # Handle save_checkpoint
            elif cmd_id == Command.SAVE_CHECKPOINT:
                requested_path = msg.get("path")
                if requested_path:
                    backend.checkpoint_dir = Path(requested_path)
                saved_path = _resolve_train_future(backend.save_checkpoint(checkpoint_step))
                if rank == 0:
                    handle.send({"status": "saved", "path": str(saved_path)})

            logger.info("command_ok name=%s rank=%s", command_name, rank)
        except Exception as exc:
            logger.exception(
                "command_failed name=%s rank=%s error=%s: %s",
                command_name,
                rank,
                type(exc).__name__,
                exc,
            )
            _emit_command_error_diagnostics(
                handle,
                backend=backend,
                rank=rank,
                command_name=command_name,
                exc=exc,
            )
            raise

    logger.info("Worker exiting")


def _do_train_step(
    backend: Any,
    batch: dict[str, Any] | None,
    rank: int,
) -> dict[str, float]:
    """Execute one training step.

    Rank 0 has the batch, broadcasts to other ranks via NCCL.
    All ranks call forward_backward together.

    Args:
        backend: MegatronTrainingBackend
        batch: Training batch (only rank 0 has this, as Python lists from JSON)
        rank: This worker's rank

    Returns:
        Training metrics (only meaningful on rank 0)
    """
    batch = _broadcast_megatron_batch(batch, rank=rank)

    # All ranks call forward_backward
    metrics_future = backend.forward_backward(batch)
    metrics = _resolve_train_future(metrics_future)

    # Optimizer step
    step_future = backend.optim_step()
    step_metrics = _resolve_train_future(step_future)
    metrics.update(step_metrics)

    return metrics


def _do_sync_weights(backend: Any, inference_endpoints: list[str]) -> None:
    """Sync weights to inference engines.

    Gathers weights from all TP/PP ranks and pushes to SGLang.

    Args:
        backend: MegatronTrainingBackend
        inference_endpoints: List of SGLang endpoint URLs
    """
    if not inference_endpoints:
        return

    # Get weights (handles TP/PP gathering internally)
    weights_future = backend.get_weights()
    weights = _resolve_train_future(weights_future)

    if not weights:
        return  # Not rank 0, nothing to send

    # TODO: Convert to HF format and push to SGLang
    # For now, just log
    logger.info(
        "Weight sync: %d parameters to %d endpoints", len(weights), len(inference_endpoints)
    )


def _do_validate_inference_export(
    backend: Any,
    model_name: str,
) -> dict[str, Any]:
    """Validate Megatron runtime export before inference startup.

    All trainer ranks participate in the runtime export collectives. Rank 0
    returns a small summary for the control channel.
    """
    from megatron.core import mpu

    from rollouts.training.backends.megatron.inference_export import (
        build_megatron_inference_export_from_runtime,
    )

    export = build_megatron_inference_export_from_runtime(model_name, backend.model)
    details = {
        "tensor_count": len(export.tensors),
        "dropped_unconverted_keys": len(export.dropped_unconverted_keys),
        "tp_world_size": int(mpu.get_tensor_model_parallel_world_size()),
        "pp_world_size": int(mpu.get_pipeline_model_parallel_world_size()),
        "ep_world_size": int(mpu.get_expert_model_parallel_world_size()),
    }
    logger.info(
        "Validated Megatron runtime inference export: tensors=%s tp=%s pp=%s ep=%s",
        details["tensor_count"],
        details["tp_world_size"],
        details["pp_world_size"],
        details["ep_world_size"],
    )
    return details


def _strip_chunk_prefix(name: str) -> str:
    """Drop `chunk_<N>.` prefix from chunked Megatron parameter names."""
    prefix, separator, remainder = name.partition(".")
    if prefix.startswith("chunk_") and separator and prefix[6:].isdigit():
        return remainder
    return name


def _summarize_tensor_contract(tensors: dict[str, Any], *, limit: int = 3) -> dict[str, object]:
    import torch

    dtype_counts: dict[str, int] = {}
    dtype_total_bytes: dict[str, int] = {}
    first_tensors: list[dict[str, object]] = []
    last_tensors: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    total_bytes = 0

    for index, (name, value) in enumerate(tensors.items()):
        if not isinstance(value, torch.Tensor):
            continue
        dtype_name = str(value.dtype).replace("torch.", "")
        nbytes = int(value.numel() * value.element_size())
        total_bytes += nbytes
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
        dtype_total_bytes[dtype_name] = dtype_total_bytes.get(dtype_name, 0) + nbytes
        tensor_row = {
            "name": name,
            "shape": [int(dim) for dim in value.shape],
            "dtype": dtype_name,
            "device": str(value.device),
            "numel": int(value.numel()),
            "element_size": int(value.element_size()),
            "nbytes": nbytes,
            "is_contiguous": bool(value.is_contiguous()),
            "stride": [int(dim) for dim in value.stride()],
        }
        sequence_rows.append({
            "index": index,
            "name": name,
            "shape": tensor_row["shape"],
            "dtype": dtype_name,
            "nbytes": nbytes,
        })
        if index < limit:
            first_tensors.append(tensor_row)
        if len(last_tensors) == limit:
            last_tensors.pop(0)
        last_tensors.append(tensor_row)

    return {
        "tensor_count": sum(dtype_counts.values()),
        "total_bytes": total_bytes,
        "dtype_counts": dtype_counts,
        "dtype_total_bytes": dtype_total_bytes,
        "first_tensors": first_tensors,
        "last_tensors": last_tensors,
        "sequence_hash": _sequence_hash(sequence_rows),
    }


def _normalize_inference_dtype_name(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = name.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return "bfloat16"
    if normalized in {"fp16", "float16", "half"}:
        return "float16"
    if normalized in {"fp32", "float32"}:
        return "float32"
    return normalized or None


def _inference_dtype_from_worker_config(config: dict[str, Any]) -> str | None:
    dtype = _normalize_inference_dtype_name(config.get("dtype"))
    if dtype is not None:
        return dtype
    if bool(config.get("bf16")):
        return "bfloat16"
    return None


def _convert_megatron_state_dict(
    model_name: str,
    state_dict: dict[str, Any],
    *,
    inference_dtype: str | None = None,
) -> dict[str, Any]:
    """Convert Megatron shard names to HuggingFace names for inference sync.

    The live NCCL publication path can only publish real tensors. Megatron
    export/conversion can surface placeholders like ``None`` for entries that
    are not materialized on this rank, so normalize them out here instead of
    letting the transport layer discover that too late.
    """
    import logging

    import torch
    from transformers import AutoConfig

    from rollouts.training.backends.megatron.weight_conversion import (
        convert_megatron_to_hf,
        remove_padding,
    )

    convert_logger = logging.getLogger(__name__)
    target_dtype_name = _normalize_inference_dtype_name(inference_dtype)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    target_dtype = dtype_map.get(target_dtype_name)
    dropped_metadata_keys: list[str] = []
    casted_keys: list[str] = []

    def _normalize_value(name: str, value: Any) -> Any | None:
        clean_name = _strip_chunk_prefix(name)
        if "_extra_state" in clean_name:
            dropped_metadata_keys.append(clean_name)
            return None
        if not isinstance(value, torch.Tensor):
            return value
        if value.numel() == 0:
            dropped_metadata_keys.append(clean_name)
            return None
        if target_dtype is None or not value.is_floating_point() or value.dtype == target_dtype:
            return value
        casted_keys.append(clean_name)
        return value.to(dtype=target_dtype)

    def _filter_tensor_state_dict(
        values: dict[str, Any], *, log_drops: bool = True
    ) -> dict[str, torch.Tensor]:
        filtered: dict[str, torch.Tensor] = {}
        dropped_non_tensors: list[str] = []
        for name, value in values.items():
            clean_name = _strip_chunk_prefix(name)
            normalized_value = _normalize_value(clean_name, value)
            if isinstance(normalized_value, torch.Tensor):
                filtered[clean_name] = normalized_value
            else:
                dropped_non_tensors.append(clean_name)
        if log_drops and dropped_non_tensors:
            convert_logger.warning(
                "Megatron state dict dropped non-tensor entries count=%s first_keys=%s",
                len(dropped_non_tensors),
                dropped_non_tensors[:8],
            )
        return filtered

    if not model_name:
        return _filter_tensor_state_dict(state_dict)

    try:
        hf_config = AutoConfig.from_pretrained(model_name)
        num_layers = getattr(hf_config, "num_hidden_layers", 0) or getattr(hf_config, "n_layers", 0)
        if not num_layers:
            raise ValueError("Unable to infer num_layers")
        vocab_size = int(getattr(hf_config, "vocab_size", 0))
        num_attention_heads = int(getattr(hf_config, "num_attention_heads", 0))
        hidden_size = int(getattr(hf_config, "hidden_size", 0))
        num_query_groups = getattr(hf_config, "num_query_groups", num_attention_heads)
        kv_channels = getattr(hf_config, "kv_channels", None)
        q_lora_rank = getattr(hf_config, "q_lora_rank", None)
    except Exception as exc:
        convert_logger.warning("Failed to load HF config for Megatron->HF conversion: %s", exc)
        return _filter_tensor_state_dict(state_dict)

    output: dict[str, Any] = {}
    conversion_attempted = False
    dropped_non_tensors: list[str] = []

    def _record_output(name: str, value: Any) -> None:
        filtered = _filter_tensor_state_dict({name: value}, log_drops=False)
        if filtered:
            output.update(filtered)
        else:
            dropped_non_tensors.append(_strip_chunk_prefix(name))

    for name, param in state_dict.items():
        clean_name = _strip_chunk_prefix(name)
        try:
            for hf_name, hf_param in convert_megatron_to_hf(
                model_name=model_name,
                name=clean_name,
                param=param,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                num_query_groups=num_query_groups,
                kv_channels=kv_channels,
                q_lora_rank=q_lora_rank,
            ):
                _record_output(hf_name, remove_padding(hf_name, hf_param, vocab_size))
            conversion_attempted = True
        except Exception:
            _record_output(clean_name, param)

    if not conversion_attempted:
        convert_logger.warning(
            "Megatron->HF conversion not applied for any tensors; using raw keys."
        )
    if dropped_non_tensors:
        convert_logger.warning(
            "Megatron->HF conversion dropped non-tensor entries count=%s first_keys=%s",
            len(dropped_non_tensors),
            dropped_non_tensors[:8],
        )
    if dropped_metadata_keys:
        convert_logger.info(
            "Megatron inference sync dropped metadata tensors count=%s first_keys=%s",
            len(dropped_metadata_keys),
            dropped_metadata_keys[:8],
        )
    if casted_keys:
        convert_logger.info(
            "Megatron inference sync cast tensors to %s count=%s first_keys=%s",
            target_dtype_name,
            len(casted_keys),
            casted_keys[:8],
        )
    return output


def _init_nccl_weight_sync(
    backend: Any,
    inference_endpoints: list[str],
    model_name: str,
    master_addr: str | None = None,
    master_port: int = 29500,
) -> None:
    """Initialize NCCL sender and connect inference engines."""
    if hasattr(backend, "_nccl_weight_sender") and backend._nccl_weight_sender is not None:
        logger.info("weight_sync_megatron_reuse_existing_sender")
        return

    backend._nccl_weight_sender = None
    backend._nccl_weight_version = int(getattr(backend, "_nccl_weight_version", 0))
    backend._nccl_model_name = model_name

    if not inference_endpoints:
        logger.info("NCCL weight sync skipped (no inference endpoints)")
        return

    ambient_master_addr = os.environ.get("MASTER_ADDR")
    resolved_host_ip, resolved_host_ip_source = _resolve_local_host_ip()
    explicit_master_addr = master_addr
    if master_addr is None:
        master_addr = resolved_host_ip

    master_port, master_port_source = _allocate_tcp_port(master_port)

    # Keep the long-lived runtime sync session on its own explicit group name.
    # SGLang already uses distinct custom group names for successful live
    # updates; reusing the generic literal "weight_sync" risks colliding with
    # receiver-side ambient state keyed by that name.
    group_name = f"weight_sync_session_{master_port}"
    world_size = 1 + len(inference_endpoints)

    from rollouts.inference.weight_sync import WeightSyncSender

    sender_holder: dict[str, Any] = {}
    errors: list[tuple[str, str]] = []
    logger.info(
        "weight_sync_megatron_init_start master=%s:%s world_size=%s endpoints=%s group=%s resolved_host_ip=%s resolved_host_ip_source=%s loopback=%s explicit_master_addr=%s hostname=%s master_port_source=%s",
        master_addr,
        master_port,
        world_size,
        inference_endpoints,
        group_name,
        resolved_host_ip,
        resolved_host_ip_source,
        "127.0.0.1",
        explicit_master_addr,
        os.uname().nodename,
        master_port_source,
    )
    _emit_argus_diag(
        "weight_sync_megatron_init_start",
        master_addr=master_addr,
        resolved_host_ip=resolved_host_ip,
        resolved_host_ip_source=resolved_host_ip_source,
        loopback_addr="127.0.0.1",
        ambient_master_addr=ambient_master_addr,
        explicit_master_addr=explicit_master_addr,
        hostname=os.uname().nodename,
        master_port=master_port,
        master_port_source=master_port_source,
        world_size=world_size,
        endpoints=inference_endpoints,
        group=group_name,
    )

    def register_inference_endpoint(endpoint: str, rank: int) -> None:
        import requests

        from rollouts.training.weight_sync_protocol import (
            InitWeightUpdateGroupRequest,
            InitWeightUpdateGroupResponse,
        )

        try:
            init_request = InitWeightUpdateGroupRequest(
                master_address=master_addr,
                master_port=master_port,
                rank_offset=rank,
                world_size=world_size,
                group_name=group_name,
            )
            logger.info(
                "weight_sync_megatron_register_endpoint_start endpoint=%s rank=%s master=%s:%s world_size=%s group=%s",
                endpoint,
                rank,
                master_addr,
                master_port,
                world_size,
                group_name,
            )
            _emit_argus_diag(
                "weight_sync_megatron_register_endpoint_start",
                endpoint=endpoint,
                rank=rank,
                master_addr=master_addr,
                master_port=master_port,
                world_size=world_size,
                group=group_name,
            )
            response = requests.post(
                f"{endpoint}/init_weights_update_group",
                json=init_request.to_dict(),
                timeout=300.0,
            )
            response.raise_for_status()
            InitWeightUpdateGroupResponse.from_dict(response.json())
            logger.info(
                "weight_sync_megatron_register_endpoint_ok endpoint=%s rank=%s status=%s",
                endpoint,
                rank,
                response.status_code,
            )
            _emit_argus_diag(
                "weight_sync_megatron_register_endpoint_ok",
                endpoint=endpoint,
                rank=rank,
                status_code=response.status_code,
            )
        except Exception as e:
            logger.exception(
                "weight_sync_megatron_register_endpoint_failed endpoint=%s rank=%s",
                endpoint,
                rank,
            )
            _emit_argus_diag(
                "weight_sync_megatron_register_endpoint_failed",
                endpoint=endpoint,
                rank=rank,
                error=f"{type(e).__name__}: {e}",
            )
            errors.append((endpoint, str(e)))

    def trainer_join() -> None:
        import os

        os.environ["NCCL_SHM_DISABLE"] = "1"
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "1")
        logger.info(
            "weight_sync_megatron_trainer_join_start master=%s:%s world_size=%s group=%s",
            master_addr,
            master_port,
            world_size,
            group_name,
        )
        _emit_argus_diag(
            "weight_sync_megatron_trainer_join_start",
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            group=group_name,
        )
        sender = WeightSyncSender(
            master_addr=master_addr,
            master_port=master_port,
            inference_world_size=len(inference_endpoints),
            group_name=group_name,
        )
        sender.init_group()
        sender_holder["sender"] = sender
        logger.info(
            "weight_sync_megatron_trainer_join_ok master=%s:%s world_size=%s group=%s",
            master_addr,
            master_port,
            world_size,
            group_name,
        )
        _emit_argus_diag(
            "weight_sync_megatron_trainer_join_ok",
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            group=group_name,
        )

    registration_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(inference_endpoints))
    )
    try:
        registration_futures = []
        for i, endpoint in enumerate(inference_endpoints):
            inference_rank = i + 1
            logger.info(
                "weight_sync_megatron_register_endpoint_submit endpoint=%s rank=%s",
                endpoint,
                inference_rank,
            )
            _emit_argus_diag(
                "weight_sync_megatron_register_endpoint_submit",
                endpoint=endpoint,
                rank=inference_rank,
            )
            registration_futures.append(
                registration_executor.submit(register_inference_endpoint, endpoint, inference_rank)
            )
        logger.info(
            "weight_sync_megatron_registration_submitted total=%s",
            len(registration_futures),
        )
        _emit_argus_diag(
            "weight_sync_megatron_registration_submitted",
            total=len(registration_futures),
        )

        trainer_join_started_at = time.monotonic()
        logger.info("weight_sync_megatron_trainer_join_stage_start")
        _emit_argus_diag("weight_sync_megatron_trainer_join_stage_start")
        trainer_join()
        trainer_join_elapsed = time.monotonic() - trainer_join_started_at
        logger.info(
            "weight_sync_megatron_trainer_join_stage_ok elapsed_sec=%.6f",
            trainer_join_elapsed,
        )
        _emit_argus_diag(
            "weight_sync_megatron_trainer_join_stage_ok",
            elapsed_sec=trainer_join_elapsed,
        )

        for index, future in enumerate(registration_futures, start=1):
            logger.info(
                "weight_sync_megatron_registration_wait_start index=%s total=%s",
                index,
                len(registration_futures),
            )
            _emit_argus_diag(
                "weight_sync_megatron_registration_wait_start",
                index=index,
                total=len(registration_futures),
            )
            future.result()
            logger.info(
                "weight_sync_megatron_registration_wait_ok index=%s total=%s",
                index,
                len(registration_futures),
            )
            _emit_argus_diag(
                "weight_sync_megatron_registration_wait_ok",
                index=index,
                total=len(registration_futures),
            )

        if errors:
            raise RuntimeError(f"Failed to register inference endpoints for NCCL: {errors}")
    finally:
        registration_executor.shutdown(wait=False, cancel_futures=True)

    sender = sender_holder.get("sender")
    if sender is None:
        raise RuntimeError("Failed to initialize NCCL weight sync sender")

    backend._nccl_weight_sender = sender
    backend._nccl_master_addr = master_addr
    backend._nccl_master_port = master_port
    backend._nccl_group_name = group_name
    logger.info(
        "weight_sync_megatron_init_ok master=%s:%s world_size=%s endpoints=%s group=%s",
        master_addr,
        master_port,
        world_size,
        inference_endpoints,
        group_name,
    )
    _emit_argus_diag(
        "weight_sync_megatron_init_ok",
        master_addr=master_addr,
        master_port=master_port,
        world_size=world_size,
        endpoints=inference_endpoints,
        group=group_name,
    )


def _do_sync_weights_nccl(
    backend: Any,
    handle: Worker | None,
    model_name: str,
    inference_endpoints: list[str],
    inference_dtype: str | None = None,
    tensor_limit: int | None = None,
    witness: bool = False,
) -> None:
    """NCCL sync path for inference updates."""
    import concurrent.futures

    import requests
    import torch
    import torch.distributed as dist

    from rollouts.training.backends.megatron.inference_export import (
        build_megatron_inference_export_from_runtime,
    )
    from rollouts.training.weight_sync_protocol import (
        ReceiveWeightUpdateRequest,
        WeightUpdatePayload,
    )

    owns_publication = bool(inference_endpoints)
    sync_error: Exception | None = None

    # Mirror the broader Miles/Slime update state machine:
    # rank 0 quiesces inference, then all training ranks participate in export
    # collectives, then rank 0 performs publication, then all ranks rejoin.
    dist.barrier()
    if owns_publication:
        current_weight_version = int(getattr(backend, "_nccl_weight_version", 0))
        _pause_and_flush_inference_endpoints(inference_endpoints)
        logger.info(
            "weight_sync_megatron_publish_prepare endpoints=%s next_weight_version=%s",
            inference_endpoints,
            current_weight_version if witness else current_weight_version + 1,
        )
    dist.barrier()

    payload: WeightUpdatePayload | None = None
    receive_request: ReceiveWeightUpdateRequest | None = None
    state_dict: dict[str, Any] | None = None
    participant_tensor_count = 0
    current_weight_version = int(getattr(backend, "_nccl_weight_version", 0))

    # All trainer ranks must participate in the same sync preparation path. The
    # witness path uses runtime-export tensors; the live persistent path uses the
    # simpler converted state_dict contract that upstream Slime and our older
    # working implementation used.
    if witness:
        export = build_megatron_inference_export_from_runtime(model_name, backend.model)
        payload = export.to_weight_update_payload(version=current_weight_version + 1)
        if tensor_limit is not None:
            if tensor_limit <= 0:
                raise ValueError(f"tensor_limit must be positive, got {tensor_limit}")
            payload = WeightUpdatePayload(
                tensors=payload.tensors[:tensor_limit],
                payload_kind=payload.payload_kind,
                version=current_weight_version,
                metadata={**payload.metadata, "witness": True, "tensor_limit": tensor_limit},
            )
        if not payload.tensors:
            raise RuntimeError("No weights produced for NCCL sync")
        participant_tensor_count = len(payload.tensors)
        logger.info(
            "weight_sync_megatron_export_ready payload_kind=%s tensors=%s dropped_unconverted=%s witness=%s tensor_limit=%s",
            payload.payload_kind,
            len(payload.tensors),
            len(export.dropped_unconverted_keys),
            witness,
            tensor_limit,
        )
        receive_request = ReceiveWeightUpdateRequest(
            names=tuple(item.wire_name for item in payload.tensors),
            load_names=tuple(item.load_name for item in payload.tensors),
            shapes=tuple(item.shape for item in payload.tensors),
            dtypes=tuple(item.dtype for item in payload.tensors),
        )
        logger.info(
            "weight_sync_megatron_param_info_ready payload_kind=%s tensors=%s first_tensors=%s witness=%s tensor_limit=%s",
            payload.payload_kind,
            len(receive_request.names),
            [
                {
                    "name": name,
                    "load_name": load_name,
                    "shape": list(shape),
                    "dtype": dtype,
                }
                for name, load_name, shape, dtype in zip(
                    receive_request.names[:3],
                    receive_request.load_names[:3],
                    receive_request.shapes[:3],
                    receive_request.dtypes[:3],
                    strict=True,
                )
            ],
            witness,
            tensor_limit,
        )
    else:
        weights_future = backend.get_weights()
        weights = _resolve_train_future(weights_future)
        state_dict = (
            _convert_megatron_state_dict(
                model_name,
                weights,
                inference_dtype=inference_dtype,
            )
            if weights
            else {}
        )
        participant_tensor_count = len(state_dict)
        logger.info(
            "weight_sync_megatron_state_dict_ready tensors=%s witness=%s first_names=%s",
            len(state_dict),
            witness,
            list(state_dict.keys())[:3],
        )

    dist.barrier()
    if not owns_publication:
        logger.info(
            "weight_sync_megatron_export_participant_only tensors=%s witness=%s",
            participant_tensor_count,
            witness,
        )
        dist.barrier()
        return

    request_group_name = str(getattr(backend, "_nccl_group_name", "weight_sync"))
    sender = getattr(backend, "_nccl_weight_sender", None) if owns_publication else None
    if owns_publication and not witness and sender is None:
        raise RuntimeError("NCCL sender not initialized. Call init_nccl_weight_sync first.")

    if witness:
        assert payload is not None
        request_weight_version = payload.version
    else:
        assert sender is not None
        assert state_dict is not None
        if not state_dict:
            raise RuntimeError("No weights produced for NCCL sync")
        request_weight_version = sender.weight_version + 1

    isolated_master_addr: str | None = None
    isolated_master_port: int | None = None
    isolated_master_port_source: str | None = None
    if witness:
        resolved_host_ip, _ = _resolve_local_host_ip()
        isolated_master_addr = resolved_host_ip
        isolated_master_port, isolated_master_port_source = _allocate_tcp_port(
            int(getattr(backend, "_nccl_master_port", 29550))
        )
        request_group_name = f"weight_sync_witness_{isolated_master_port}"

    try:
        try:
            if witness:
                assert payload is not None
                logger.info(
                    "weight_sync_megatron_metadata_requests_sent endpoints=%s count=%s",
                    inference_endpoints,
                    0,
                )
                assert isolated_master_addr is not None
                assert isolated_master_port is not None
                isolated_group_name = request_group_name
                logger.info(
                    "weight_sync_megatron_isolated_witness_start master=%s:%s group=%s tensors=%s port_source=%s",
                    isolated_master_addr,
                    isolated_master_port,
                    isolated_group_name,
                    len(payload.tensors),
                    isolated_master_port_source,
                )
                _emit_argus_diag(
                    "weight_sync_megatron_isolated_witness_start",
                    master_addr=isolated_master_addr,
                    master_port=isolated_master_port,
                    master_port_source=isolated_master_port_source,
                    group=isolated_group_name,
                    tensors=len(payload.tensors),
                )
                _run_isolated_weight_sync_sender(
                    master_addr=isolated_master_addr,
                    master_port=isolated_master_port,
                    inference_endpoints=inference_endpoints,
                    group_name=isolated_group_name,
                    updates=[
                        _serialize_weight_sync_update(
                            payload,
                            version=payload.version,
                        )
                    ],
                )
                logger.info(
                    "weight_sync_megatron_isolated_witness_ok tensors=%s",
                    len(payload.tensors),
                )
                _emit_argus_diag(
                    "weight_sync_megatron_isolated_witness_ok",
                    tensors=len(payload.tensors),
                )
            else:
                assert request_weight_version is not None
                assert sender is not None
                assert state_dict is not None
                update_connect_timeout_sec = 5.0
                update_read_timeout_sec = 300.0
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(1, len(inference_endpoints))
                )
                try:
                    param_info = [
                        {
                            "name": name,
                            "shape": [int(dim) for dim in tensor.shape],
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                        }
                        for name, tensor in state_dict.items()
                    ]
                    request_body = {
                        "names": [item["name"] for item in param_info],
                        "shapes": [item["shape"] for item in param_info],
                        "dtypes": [item["dtype"] for item in param_info],
                        "group_name": request_group_name,
                        "weight_version": str(request_weight_version),
                    }
                    tensor_contract = _summarize_tensor_contract(state_dict)
                    total_bytes = int(tensor_contract["total_bytes"])
                    sender_contract = {
                        "group": request_group_name,
                        "weight_version": request_weight_version,
                        "tensor_count": len(param_info),
                        "total_bytes": total_bytes,
                        "dtype_counts": tensor_contract["dtype_counts"],
                        "dtype_total_bytes": tensor_contract["dtype_total_bytes"],
                        "sequence_hash": tensor_contract["sequence_hash"],
                        "first_tensors": tensor_contract["first_tensors"],
                        "last_tensors": tensor_contract["last_tensors"],
                    }
                    backend._last_weight_sync_sender_contract = sender_contract
                    logger.info(
                        "weight_sync_megatron_persistent_update_start group=%s tensors=%s total_bytes=%s payload_kind=%s weight_version=%s",
                        request_group_name,
                        len(param_info),
                        total_bytes,
                        "inference_load_tensor",
                        request_weight_version,
                    )
                    _emit_argus_diag(
                        "weight_sync_megatron_persistent_update_start",
                        group=request_group_name,
                        tensors=len(param_info),
                        total_bytes=total_bytes,
                        payload_kind="inference_load_tensor",
                        weight_version=request_weight_version,
                    )
                    logger.info(
                        "weight_sync_megatron_persistent_request_ready group=%s first_names=%s first_shapes=%s first_dtypes=%s dtype_counts=%s dtype_total_bytes=%s first_tensors=%s",
                        request_group_name,
                        request_body["names"][:3],
                        request_body["shapes"][:3],
                        request_body["dtypes"][:3],
                        tensor_contract["dtype_counts"],
                        tensor_contract["dtype_total_bytes"],
                        tensor_contract["first_tensors"],
                    )
                    _emit_argus_diag(
                        "weight_sync_megatron_persistent_request_contract",
                        **sender_contract,
                    )
                    _send_progress_event(
                        handle,
                        event="megatron_remote_sender_contract",
                        context="sync_weights_nccl",
                        sender_contract=sender_contract,
                    )

                    def _update_remote_endpoint(endpoint: str) -> dict[str, object]:
                        started_at = time.monotonic()
                        response: requests.Response | None = None
                        try:
                            response = requests.post(
                                f"{endpoint}/update_weights_from_distributed",
                                json=request_body,
                                timeout=(update_connect_timeout_sec, update_read_timeout_sec),
                            )
                            response.raise_for_status()
                            return {
                                "endpoint": endpoint,
                                "status_code": response.status_code,
                                "elapsed_sec": round(time.monotonic() - started_at, 3),
                            }
                        except Exception as exc:
                            status_code = None
                            response_text = None
                            if response is not None:
                                status_code = response.status_code
                                try:
                                    response_text = response.text[:400]
                                except Exception:
                                    response_text = "<response text unavailable>"
                            _emit_argus_diag(
                                "weight_sync_megatron_persistent_remote_update_failed",
                                endpoint=endpoint,
                                group=request_group_name,
                                version=request_weight_version,
                                tensor_count=len(param_info),
                                elapsed_sec=round(time.monotonic() - started_at, 3),
                                status_code=status_code,
                                response_text=response_text,
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )
                            raise RuntimeError(
                                "Megatron persistent weight update request failed "
                                f"endpoint={endpoint} group={request_group_name} "
                                f"tensors={len(param_info)}"
                            ) from exc

                    logger.info(
                        "weight_sync_megatron_metadata_requests_sent endpoints=%s count=%s tensors=%s total_bytes=%s transport=%s",
                        inference_endpoints,
                        len(inference_endpoints),
                        len(param_info),
                        total_bytes,
                        "persistent_session",
                    )
                    futures = [
                        (
                            endpoint,
                            executor.submit(_update_remote_endpoint, endpoint),
                        )
                        for endpoint in inference_endpoints
                    ]
                    try:
                        sender.broadcast_weights(
                            state_dict,
                            async_op=True,
                        )
                        logger.info(
                            "weight_sync_megatron_broadcast_wait_ok tensors=%s transport=%s",
                            len(param_info),
                            "persistent_session",
                        )
                    except Exception:
                        for _endpoint, future in futures:
                            future.cancel()
                        raise

                    for endpoint, future in futures:
                        result = future.result()
                        logger.info(
                            "weight_sync_megatron_metadata_response_ok endpoint=%s status=%s elapsed_sec=%.3f",
                            endpoint,
                            result["status_code"],
                            result["elapsed_sec"],
                        )
                    logger.info(
                        "weight_sync_megatron_metadata_responses_ok endpoints=%s",
                        inference_endpoints,
                    )
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
                backend._nccl_weight_version = sender.weight_version
        except Exception as exc:
            sync_error = exc
    finally:
        try:
            _resume_inference_endpoints(inference_endpoints)
        except Exception:
            logger.exception(
                "weight_sync_megatron_resume_failed endpoints=%s",
                inference_endpoints,
            )
        try:
            dist.barrier()
        except Exception:
            logger.exception("weight_sync_megatron_final_barrier_failed")

    if sync_error is not None:
        raise sync_error

    torch.cuda.empty_cache()


def _cleanup_nccl_weight_sync(
    backend: Any,
    inference_endpoints: list[str],
) -> None:
    """Best-effort NCCL cleanup for Megatron worker."""
    sender = getattr(backend, "_nccl_weight_sender", None)
    if sender is not None:
        try:
            sender.cleanup()
        except Exception:
            pass

    if inference_endpoints:
        import concurrent.futures

        import requests

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(inference_endpoints))
        )
        futures = []
        for endpoint in inference_endpoints:
            futures.append(
                executor.submit(
                    requests.post,
                    f"{endpoint}/destroy_weights_update_group",
                    json={"group_name": str(getattr(backend, "_nccl_group_name", "weight_sync"))},
                    timeout=10.0,
                )
            )
        for future in futures:
            try:
                future.result()
            except Exception:
                pass
        executor.shutdown(wait=False)

    backend._nccl_weight_sender = None
    backend._nccl_weight_version = 0
    backend._nccl_group_name = None
