"""NCCL weight sync for engine_v2.

Implements PipelineRL-style weight sync: training broadcasts weights via NCCL,
inference engines receive and load them.

Two roles:
- Sender (training side): Creates NCCL group, broadcasts state_dict
- Receiver (inference side): Joins NCCL group, receives and loads weights

Usage (sender - training process):
    sender = WeightSyncSender(
        master_addr="10.0.0.1",
        master_port=29500,
        inference_world_size=4,  # Number of inference GPUs
    )
    sender.init_group()

    # In training loop:
    sender.broadcast_weights(model.state_dict())

    sender.cleanup()

Usage (receiver - inference engine):
    receiver = WeightSyncReceiver(
        master_addr="10.0.0.1",
        master_port=29500,
        rank=1,  # This inference GPU's rank (1-indexed, 0 is trainer)
        world_size=5,  # trainer + inference GPUs
    )
    receiver.init_group()

    # When trainer broadcasts:
    state_dict = receiver.receive_weights(param_info)
    engine.reload_weights(state_dict)

    receiver.cleanup()
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from ..training.weight_sync_protocol import WeightUpdatePayload, WeightWireTensor

logger = logging.getLogger(__name__)
_ARGUS_DIAG_EVENT_SENTINEL = "__ARGUS_DIAG__"
_WEIGHT_SYNC_PUBLICATION_LOCK = threading.Lock()


def _emit_argus_diag(event: str, **data: object) -> None:
    """Best-effort structured diagnostics for remote sandbox runs."""
    try:
        sys.stderr.write(
            f"{_ARGUS_DIAG_EVENT_SENTINEL}{json.dumps({'event': event, **data}, sort_keys=True)}\n"
        )
        sys.stderr.flush()
    except Exception:
        return


def _tensor_sync_metadata(name: str, tensor: Tensor) -> dict[str, object]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "stride": list(tensor.stride()),
    }


def _nccl_env_snapshot() -> dict[str, object]:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_ASYNC_ERROR_HANDLING",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "NCCL_SHM_DISABLE",
        "NCCL_CUMEM_ENABLE",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "TORCH_DISABLE_SHARE_RDZV_TCP_STORE",
    )
    return {key: os.environ.get(key) for key in keys}


def _default_group_summary() -> dict[str, object]:
    payload: dict[str, object] = {"is_initialized": bool(dist.is_initialized())}
    if not dist.is_initialized():
        payload.update({
            "backend": None,
            "rank": None,
            "world_size": None,
        })
        return payload
    try:
        payload.update({
            "backend": dist.get_backend(),
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
        })
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _receiver_distributed_state_snapshot() -> dict[str, object]:
    keys = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "CUDA_VISIBLE_DEVICES",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "TORCH_DISABLE_SHARE_RDZV_TCP_STORE",
    )
    return {
        "env": {key: os.environ.get(key) for key in keys},
        "default_group": _default_group_summary(),
    }


def _tcp_state_name(state_hex: str) -> str:
    return {
        "01": "ESTABLISHED",
        "02": "SYN_SENT",
        "03": "SYN_RECV",
        "04": "FIN_WAIT1",
        "05": "FIN_WAIT2",
        "06": "TIME_WAIT",
        "07": "CLOSE",
        "08": "CLOSE_WAIT",
        "09": "LAST_ACK",
        "0A": "LISTEN",
        "0B": "CLOSING",
    }.get(state_hex.upper(), state_hex.upper())


def _decode_proc_ip(hex_ip: str, *, ipv6: bool) -> str:
    try:
        raw = bytes.fromhex(hex_ip)
        if not ipv6:
            return str(ipaddress.IPv4Address(raw[::-1]))
        words = [raw[index : index + 4][::-1] for index in range(0, 16, 4)]
        return str(ipaddress.IPv6Address(b"".join(words)))
    except Exception:
        return hex_ip


def _decode_proc_endpoint(encoded: str, *, ipv6: bool) -> str:
    host_hex, port_hex = encoded.split(":")
    return f"{_decode_proc_ip(host_hex, ipv6=ipv6)}:{int(port_hex, 16)}"


def _collect_socket_inodes(pid: int) -> set[str]:
    inodes: set[str] = set()
    fd_dir = Path(f"/proc/{pid}/fd")
    try:
        for entry in fd_dir.iterdir():
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                inodes.add(target[len("socket:[") : -1])
    except Exception:
        return set()
    return inodes


def _parse_proc_net_tcp(path: Path, *, inodes: set[str], ipv6: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text().splitlines()
    except Exception as exc:
        return [{"path": str(path), "error": f"{type(exc).__name__}: {exc}"}]
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 10:
            continue
        inode = fields[9]
        if inode not in inodes:
            continue
        rows.append({
            "inode": inode,
            "local": _decode_proc_endpoint(fields[1], ipv6=ipv6),
            "remote": _decode_proc_endpoint(fields[2], ipv6=ipv6),
            "state": _tcp_state_name(fields[3]),
            "uid": fields[7],
        })
    return rows


def _proc_socket_snapshot(pid: int | None = None) -> dict[str, object]:
    actual_pid = os.getpid() if pid is None else pid
    inodes = _collect_socket_inodes(actual_pid)
    return {
        "pid": actual_pid,
        "socket_inode_count": len(inodes),
        "tcp": _parse_proc_net_tcp(Path("/proc/net/tcp"), inodes=inodes, ipv6=False)[:64],
        "tcp6": _parse_proc_net_tcp(Path("/proc/net/tcp6"), inodes=inodes, ipv6=True)[:64],
    }


def _filter_socket_snapshot_for_port(
    snapshot: dict[str, object],
    *,
    port: int,
) -> dict[str, object]:
    needle = f":{port}"

    def _filter_rows(rows: object) -> list[dict[str, object]]:
        filtered: list[dict[str, object]] = []
        if not isinstance(rows, list):
            return filtered
        for row in rows:
            if not isinstance(row, dict):
                continue
            local = str(row.get("local", ""))
            remote = str(row.get("remote", ""))
            if needle not in local and needle not in remote:
                continue
            filtered.append({
                "local": local,
                "remote": remote,
                "state": row.get("state"),
            })
        return filtered

    return {
        "pid": snapshot.get("pid"),
        "port": port,
        "tcp": _filter_rows(snapshot.get("tcp")),
        "tcp6": _filter_rows(snapshot.get("tcp6")),
    }


def _store_summary(store: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": type(store).__name__,
        "repr": repr(store)[:240],
    }
    for attr in ("host", "port", "world_size", "timeout", "_underlying_non_prefix_store"):
        try:
            value = getattr(store, attr)
        except Exception:
            continue
        if attr == "_underlying_non_prefix_store":
            payload[attr] = type(value).__name__
        else:
            payload[attr] = repr(value)[:120]
    return payload


def _resolve_socket_ifname() -> tuple[str | None, str]:
    explicit = os.environ.get("NCCL_SOCKET_IFNAME") or os.environ.get("GLOO_SOCKET_IFNAME")
    if explicit:
        return explicit, "env"

    if os.path.exists("/sys/class/net/eth0"):
        return "eth0", "sysfs:eth0"

    try:
        result = subprocess.run(
            ["ip", "route", "get", "1.1.1.1"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5.0,
        )
        fields = result.stdout.split()
        for index, field in enumerate(fields):
            if field == "dev" and index + 1 < len(fields):
                return fields[index + 1], "ip-route"
    except Exception:
        pass

    # Modal sandboxes expose the routable container interface as eth0 in the
    # NCCL logs we are debugging. Prefer an explicit interface pin here over
    # falling back to ambient auto-selection.
    return "eth0", "default:eth0"


def _apply_socket_ifname_defaults() -> tuple[str | None, str]:
    ifname, source = _resolve_socket_ifname()
    if ifname:
        os.environ.setdefault("NCCL_SOCKET_IFNAME", ifname)
        os.environ.setdefault("GLOO_SOCKET_IFNAME", ifname)
    return ifname, source


def _normalize_cuda_device(device: torch.device) -> torch.device:
    """Resolve ambiguous CUDA devices to an explicit local index."""
    if device.type != "cuda":
        return device
    if device.index is not None:
        return device
    return torch.device("cuda", torch.cuda.current_device())


# ═══════════════════════════════════════════════════════════════════════════════
# STATELESS PROCESS GROUP (following vLLM pattern)
# ═══════════════════════════════════════════════════════════════════════════════


def create_stateless_process_group(
    master_addr: str,
    master_port: int,
    rank: int,
    world_size: int,
    group_name: str = "weight_sync",
    backend: str = "nccl",
    timeout_seconds: float = 300.0,
) -> dist.ProcessGroup:
    """Create a process group without touching global torch.distributed state.

    This follows vLLM's StatelessProcessGroup pattern - allows creating NCCL
    groups for weight sync without interfering with training's distributed setup.

    Args:
        master_addr: IP address of the master (trainer rank 0)
        master_port: Port for rendezvous
        rank: This process's rank in the weight sync group
        world_size: Total processes (trainer + inference GPUs)
        group_name: Name for the process group
        backend: "nccl" for GPU-to-GPU, "gloo" for CPU
        timeout_seconds: Timeout for initialization

    Returns:
        Process group for weight sync operations
    """
    from torch.distributed.distributed_c10d import (
        default_pg_timeout,
    )

    init_method = f"tcp://{master_addr}:{master_port}"
    timeout = timedelta(seconds=timeout_seconds)
    logger.info(
        "weight_sync_pg_create_start group=%s backend=%s rank=%s world_size=%s init_method=%s timeout_s=%.1f",
        group_name,
        backend,
        rank,
        world_size,
        init_method,
        timeout_seconds,
    )
    _emit_argus_diag(
        "weight_sync_pg_create_start",
        group=group_name,
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout_s=timeout_seconds,
    )
    return _init_process_group_like_miles(
        backend=backend,
        init_method=init_method,
        timeout=timeout if timeout_seconds > 0.0 else default_pg_timeout,
        world_size=world_size,
        rank=rank,
        group_name=group_name,
    )


def init_extra_process_group(
    *,
    backend: str = "nccl",
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Any | None = None,
    group_name: str | None = None,
) -> dist.ProcessGroup:
    """PipelineRL/QED-style extra process-group creation for trainer-side rank 0."""
    from torch.distributed.distributed_c10d import ProcessGroupNCCL, default_pg_timeout

    if timeout is None:
        timeout = default_pg_timeout

    pg_options = None
    if str(backend) == "nccl":
        pg_options = ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = False

    return _init_process_group_like_miles(
        backend=backend,
        init_method=init_method,
        timeout=timeout,
        world_size=world_size,
        rank=rank,
        store=store,
        group_name=group_name,
        pg_options=pg_options,
    )


def init_process_group_without_pg_options(
    *,
    backend: str = "nccl",
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Any | None = None,
    group_name: str | None = None,
) -> dist.ProcessGroup:
    """Create a process group with the plain Miles/Slime helper surface."""
    from torch.distributed.distributed_c10d import default_pg_timeout

    if timeout is None:
        timeout = default_pg_timeout

    return _init_process_group_like_miles(
        backend=backend,
        init_method=init_method,
        timeout=timeout,
        world_size=world_size,
        rank=rank,
        store=store,
        group_name=group_name,
        pg_options=None,
    )


def _init_process_group_like_miles(
    backend: Any = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Any | None = None,
    group_name: str | None = None,
    pg_options: Any | None = None,
) -> dist.ProcessGroup:
    """Create a non-default process group following Miles' helper shape.

    This keeps our weight-sync PG semantics close to
    `miles.utils.distributed_utils.init_process_group` while preserving the
    local logging/diagnostic surface around the rendezvous effect.
    """
    import inspect

    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    bootstrap_port: int | None = None
    if init_method and init_method.startswith("tcp://"):
        try:
            bootstrap_port = int(init_method.rsplit(":", 1)[1])
        except Exception:
            bootstrap_port = None

    if store is None:
        socket_state = _proc_socket_snapshot()
        logger.info(
            "weight_sync_pg_rendezvous_start group=%s rank=%s world_size=%s",
            group_name,
            rank,
            world_size,
        )
        _emit_argus_diag(
            "weight_sync_pg_rendezvous_start",
            group=group_name,
            rank=rank,
            world_size=world_size,
            init_method=init_method,
            bootstrap_socket_state=(
                _filter_socket_snapshot_for_port(socket_state, port=bootstrap_port)
                if bootstrap_port is not None
                else None
            ),
        )
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        socket_state = _proc_socket_snapshot()
        logger.info(
            "weight_sync_pg_rendezvous_ok group=%s rank=%s world_size=%s",
            group_name,
            rank,
            world_size,
        )
        _emit_argus_diag(
            "weight_sync_pg_rendezvous_ok",
            group=group_name,
            rank=rank,
            world_size=world_size,
            store=_store_summary(store),
            bootstrap_socket_state=(
                _filter_socket_snapshot_for_port(socket_state, port=bootstrap_port)
                if bootstrap_port is not None
                else None
            ),
        )
        prefixed_store = PrefixStore(group_name, store)
        socket_state = _proc_socket_snapshot()
        _emit_argus_diag(
            "weight_sync_pg_prefix_store_created",
            group=group_name,
            rank=rank,
            world_size=world_size,
            base_store=_store_summary(store),
            prefixed_store=_store_summary(prefixed_store),
            bootstrap_socket_state=(
                _filter_socket_snapshot_for_port(socket_state, port=bootstrap_port)
                if bootstrap_port is not None
                else None
            ),
        )
        store = prefixed_store

    pg_helper_sig = inspect.signature(_new_process_group_helper)
    if "backend_options" in pg_helper_sig.parameters:
        pg_kwargs = {"backend_options": pg_options}
    elif "pg_options" in pg_helper_sig.parameters:
        pg_kwargs = {"pg_options": pg_options}
    else:
        raise RuntimeError(
            "Unsupported torch.distributed version: _new_process_group_helper has neither "
            "'backend_options' nor 'pg_options' parameter."
        )

    logger.info(
        "weight_sync_pg_helper_start group=%s backend=%s rank=%s world_size=%s",
        group_name,
        backend,
        rank,
        world_size,
    )
    _emit_argus_diag(
        "weight_sync_pg_helper_start",
        group=group_name,
        backend=backend,
        rank=rank,
        world_size=world_size,
        store=_store_summary(store),
        bootstrap_socket_state=(
            _filter_socket_snapshot_for_port(_proc_socket_snapshot(), port=bootstrap_port)
            if bootstrap_port is not None
            else None
        ),
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        timeout=timeout,
        **pg_kwargs,
    )
    logger.info(
        "weight_sync_pg_helper_ok group=%s backend=%s rank=%s world_size=%s",
        group_name,
        backend,
        rank,
        world_size,
    )
    _emit_argus_diag(
        "weight_sync_pg_helper_ok",
        group=group_name,
        backend=backend,
        rank=rank,
        world_size=world_size,
        store=_store_summary(store),
        bootstrap_socket_state=(
            _filter_socket_snapshot_for_port(_proc_socket_snapshot(), port=bootstrap_port)
            if bootstrap_port is not None
            else None
        ),
    )

    # Register in world for cleanup
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    logger.info(
        "weight_sync_pg_create_ok group=%s backend=%s rank=%s world_size=%s",
        group_name,
        backend,
        rank,
        world_size,
    )
    _emit_argus_diag(
        "weight_sync_pg_create_ok",
        group=group_name,
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    return pg


# ═══════════════════════════════════════════════════════════════════════════════
# SENDER (training side)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class WeightSyncSender:
    """Sends weights from training to inference via NCCL broadcast.

    The sender is rank 0 in the weight sync group. It broadcasts each tensor
    to all inference GPUs.
    """

    master_addr: str
    master_port: int
    inference_world_size: int  # Number of inference GPUs (not including trainer)
    group_name: str = "weight_sync"
    timeout_seconds: float = 300.0
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))

    _process_group: Any = field(default=None, init=False, repr=False)
    _weight_version: int = field(default=0, init=False)

    @property
    def world_size(self) -> int:
        return self.inference_world_size + 1  # +1 for trainer

    @property
    def weight_version(self) -> int:
        return self._weight_version

    def init_group(self) -> None:
        """Initialize NCCL process group. Call once at startup."""
        self.device = _normalize_cuda_device(self.device)
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,COLL")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        # Do not force-disable P2P here. The SGLang receiver side initializes
        # normal NCCL P2P/IPC transport for the custom update group, and
        # asymmetrically disabling it on the trainer creates a dishonest
        # sender/receiver contract.
        socket_ifname, socket_ifname_source = _apply_socket_ifname_defaults()
        logger.info(
            "weight_sync_sender_init_start world_size=%s master=%s:%s device=%s group=%s socket_ifname=%s socket_ifname_source=%s nccl_env=%s",
            self.world_size,
            self.master_addr,
            self.master_port,
            self.device,
            self.group_name,
            socket_ifname,
            socket_ifname_source,
            _nccl_env_snapshot(),
        )
        _emit_argus_diag(
            "weight_sync_sender_init_start",
            world_size=self.world_size,
            master_addr=self.master_addr,
            master_port=self.master_port,
            device=str(self.device),
            group=self.group_name,
            socket_ifname=socket_ifname,
            socket_ifname_source=socket_ifname_source,
            nccl_env=_nccl_env_snapshot(),
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        init_method = f"tcp://{self.master_addr}:{self.master_port}"
        logger.info(
            "weight_sync_sender_pg_create_contract init_method=%s group=%s rank=%s world_size=%s pg_options=%s",
            init_method,
            self.group_name,
            0,
            self.world_size,
            "none",
        )
        _emit_argus_diag(
            "weight_sync_sender_pg_create_contract",
            init_method=init_method,
            group=self.group_name,
            rank=0,
            world_size=self.world_size,
            pg_options="none",
        )
        self._process_group = init_process_group_without_pg_options(
            backend="nccl",
            init_method=init_method,
            rank=0,  # Trainer is always rank 0
            world_size=self.world_size,
            group_name=self.group_name,
            timeout=timedelta(seconds=self.timeout_seconds),
        )
        logger.info(
            "weight_sync_sender_init_ok world_size=%s master=%s:%s device=%s group=%s pg_backend=%s pg_rank=%s pg_world_size=%s socket_ifname=%s socket_ifname_source=%s nccl_env=%s",
            self.world_size,
            self.master_addr,
            self.master_port,
            self.device,
            self.group_name,
            dist.get_backend(self._process_group),
            dist.get_rank(self._process_group),
            dist.get_world_size(self._process_group),
            socket_ifname,
            socket_ifname_source,
            _nccl_env_snapshot(),
        )
        _emit_argus_diag(
            "weight_sync_sender_init_ok",
            world_size=self.world_size,
            master_addr=self.master_addr,
            master_port=self.master_port,
            device=str(self.device),
            group=self.group_name,
            pg_backend=dist.get_backend(self._process_group),
            pg_rank=dist.get_rank(self._process_group),
            pg_world_size=dist.get_world_size(self._process_group),
            socket_ifname=socket_ifname,
            socket_ifname_source=socket_ifname_source,
            nccl_env=_nccl_env_snapshot(),
        )

    def broadcast_payload(
        self,
        payload: WeightUpdatePayload,
        async_op: bool = False,
        *,
        advance_version: bool = True,
    ) -> list[Any] | None:
        """Broadcast explicit wire payload to all inference GPUs.

        Args:
            payload: Concrete payload to broadcast
            async_op: If true, issue async NCCL broadcasts and then wait on all
                returned handles before returning. This matches Slime's
                collective shape without exposing dishonest caller-visible async.

        Returns:
            None for the blocking publication path.
        """
        assert self._process_group is not None, "Call init_group() first"
        self.device = _normalize_cuda_device(self.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            current_device = torch.cuda.current_device()
            assert current_device == self.device.index, (
                "Weight sync sender must broadcast from its configured CUDA device; "
                f"current={current_device} expected={self.device.index}"
            )

        total_tensors = len(payload.tensors)
        total_bytes = sum(
            int(item.tensor.numel() * item.tensor.element_size()) for item in payload.tensors
        )
        first_items = list(payload.tensors[:3])
        logger.info(
            "weight_sync_sender_broadcast_start world_size=%s payload_kind=%s total_tensors=%s total_bytes=%s first_tensors=%s",
            self.world_size,
            payload.payload_kind,
            total_tensors,
            total_bytes,
            [_tensor_sync_metadata(item.wire_name, item.tensor) for item in first_items],
        )

        def _broadcast_all(use_async: bool) -> list[Any]:
            handles = []
            for index, item in enumerate(payload.tensors):
                name = item.wire_name
                param = item.tensor
                data = param.data
                if data.device != self.device:
                    raise RuntimeError(
                        "Weight sync sender expected payload tensor on sender device; "
                        f"name={name} param_device={data.device} sender_device={self.device}"
                    )
                if not data.is_contiguous():
                    raise RuntimeError(
                        "Weight sync sender expected contiguous payload tensor; "
                        f"name={name} stride={list(data.stride())}"
                    )
                tensor_meta = _tensor_sync_metadata(name, data)
                broadcast_meta = {
                    **tensor_meta,
                    "load_name": item.load_name,
                    "payload_kind": item.payload_kind,
                    "param_device": str(param.device),
                    "current_cuda_device": (
                        torch.cuda.current_device() if torch.cuda.is_available() else None
                    ),
                    "data_ptr": int(data.data_ptr()),
                    "storage_offset": int(data.storage_offset()),
                    "element_size": int(data.element_size()),
                    "nbytes": int(data.numel() * data.element_size()),
                    "group_rank": dist.get_rank(self._process_group),
                    "group_world_size": dist.get_world_size(self._process_group),
                }
                logger.info(
                    "weight_sync_sender_broadcast_tensor index=%s total_tensors=%s async_op=%s tensor=%s",
                    index,
                    total_tensors,
                    use_async,
                    broadcast_meta,
                )
                if index == 0:
                    socket_state = _proc_socket_snapshot()
                    _emit_argus_diag(
                        "weight_sync_sender_first_collective_start",
                        total_tensors=total_tensors,
                        async_op=use_async,
                        tensor=broadcast_meta,
                        process={
                            "pid": os.getpid(),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "torch_current_device": (
                                torch.cuda.current_device() if torch.cuda.is_available() else None
                            ),
                        },
                        socket_state=socket_state,
                        bootstrap_socket_state=_filter_socket_snapshot_for_port(
                            socket_state, port=self.master_port
                        ),
                        collective_contract={
                            "master_addr": self.master_addr,
                            "master_port": self.master_port,
                            "group_name": self.group_name,
                            "rank": 0,
                            "world_size": dist.get_world_size(self._process_group),
                            "src": 0,
                        },
                    )
                try:
                    handle = dist.broadcast(
                        data, src=0, group=self._process_group, async_op=use_async
                    )
                except Exception as exc:
                    if index == 0:
                        socket_state = _proc_socket_snapshot()
                        _emit_argus_diag(
                            "weight_sync_sender_first_collective_failed",
                            total_tensors=total_tensors,
                            async_op=use_async,
                            tensor=broadcast_meta,
                            error=f"{type(exc).__name__}: {exc}",
                            process={
                                "pid": os.getpid(),
                                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                "torch_current_device": (
                                    torch.cuda.current_device()
                                    if torch.cuda.is_available()
                                    else None
                                ),
                            },
                            socket_state=socket_state,
                            bootstrap_socket_state=_filter_socket_snapshot_for_port(
                                socket_state, port=self.master_port
                            ),
                            collective_contract={
                                "master_addr": self.master_addr,
                                "master_port": self.master_port,
                                "group_name": self.group_name,
                                "rank": 0,
                                "world_size": dist.get_world_size(self._process_group),
                                "src": 0,
                            },
                        )
                    raise RuntimeError(
                        "Weight sync sender broadcast failed "
                        f"index={index} total_tensors={total_tensors} tensor={broadcast_meta}"
                    ) from exc
                if index == 0:
                    socket_state = _proc_socket_snapshot()
                    _emit_argus_diag(
                        "weight_sync_sender_first_collective_ok",
                        total_tensors=total_tensors,
                        async_op=use_async,
                        tensor=broadcast_meta,
                        process={
                            "pid": os.getpid(),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "torch_current_device": (
                                torch.cuda.current_device() if torch.cuda.is_available() else None
                            ),
                        },
                        socket_state=socket_state,
                        bootstrap_socket_state=_filter_socket_snapshot_for_port(
                            socket_state, port=self.master_port
                        ),
                        collective_contract={
                            "master_addr": self.master_addr,
                            "master_port": self.master_port,
                            "group_name": self.group_name,
                            "rank": 0,
                            "world_size": dist.get_world_size(self._process_group),
                            "src": 0,
                        },
                    )
                if use_async:
                    handles.append(handle)
            return handles

        lock_wait_start = time.monotonic()
        _WEIGHT_SYNC_PUBLICATION_LOCK.acquire()
        lock_wait_sec = time.monotonic() - lock_wait_start
        logger.info(
            "weight_sync_sender_publication_lock_acquired waited_sec=%.6f total_tensors=%s",
            lock_wait_sec,
            total_tensors,
        )
        try:
            issue_started_at = time.monotonic()
            handles = _broadcast_all(use_async=async_op)
            issue_elapsed = time.monotonic() - issue_started_at
            logger.info(
                "weight_sync_sender_collective_issue_done total_tensors=%s async_op=%s elapsed_sec=%.6f handles=%s",
                total_tensors,
                async_op,
                issue_elapsed,
                len(handles),
            )
            _emit_argus_diag(
                "weight_sync_sender_collective_issue_done",
                total_tensors=total_tensors,
                async_op=async_op,
                elapsed_sec=issue_elapsed,
                handles=len(handles),
            )
            if handles:
                wait_started_at = time.monotonic()
                socket_state = _proc_socket_snapshot()
                _emit_argus_diag(
                    "weight_sync_sender_collective_wait_start",
                    total_tensors=total_tensors,
                    async_op=async_op,
                    handles=len(handles),
                    socket_state=socket_state,
                    bootstrap_socket_state=_filter_socket_snapshot_for_port(
                        socket_state, port=self.master_port
                    ),
                    collective_contract={
                        "master_addr": self.master_addr,
                        "master_port": self.master_port,
                        "group_name": self.group_name,
                        "rank": 0,
                        "world_size": dist.get_world_size(self._process_group),
                        "src": 0,
                    },
                )
                try:
                    for handle in handles:
                        handle.wait()
                except Exception as exc:
                    wait_elapsed = time.monotonic() - wait_started_at
                    _emit_argus_diag(
                        "weight_sync_sender_collective_wait_failed",
                        total_tensors=total_tensors,
                        async_op=async_op,
                        handles=len(handles),
                        elapsed_sec=wait_elapsed,
                        error=f"{type(exc).__name__}: {exc}",
                        socket_state=_proc_socket_snapshot(),
                    )
                    raise
                wait_elapsed = time.monotonic() - wait_started_at
                logger.info(
                    "weight_sync_sender_collective_wait_ok total_tensors=%s async_op=%s elapsed_sec=%.6f handles=%s",
                    total_tensors,
                    async_op,
                    wait_elapsed,
                    len(handles),
                )
                _emit_argus_diag(
                    "weight_sync_sender_collective_wait_ok",
                    total_tensors=total_tensors,
                    async_op=async_op,
                    elapsed_sec=wait_elapsed,
                    handles=len(handles),
                    socket_state=_proc_socket_snapshot(),
                )
        finally:
            _WEIGHT_SYNC_PUBLICATION_LOCK.release()
            logger.info(
                "weight_sync_sender_publication_lock_released total_tensors=%s",
                total_tensors,
            )

        if advance_version:
            self._weight_version += 1
        return None

    def broadcast_weights(
        self,
        state_dict: dict[str, Tensor],
        async_op: bool = False,
        *,
        advance_version: bool = True,
    ) -> list[Any] | None:
        """Backward-compatible wrapper for raw tensor dictionaries."""
        payload = WeightUpdatePayload(
            tensors=tuple(
                WeightWireTensor(
                    wire_name=name,
                    load_name=name,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype).replace("torch.", ""),
                    tensor=tensor,
                    payload_kind="inference_load_tensor",
                )
                for name, tensor in state_dict.items()
            ),
            payload_kind="inference_load_tensor",
        )
        return self.broadcast_payload(payload, async_op=async_op, advance_version=advance_version)

    def cleanup(self) -> None:
        """Cleanup process group."""
        if self._process_group is not None:
            dist.destroy_process_group(self._process_group)
            self._process_group = None
            logger.info("Weight sync sender cleaned up")


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIVER (inference side)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ParamInfo:
    """Metadata for a parameter to receive."""

    wire_name: str
    load_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class WeightSyncReceiver:
    """Receives weights from training via NCCL broadcast.

    The receiver joins the weight sync group and receives tensors broadcast
    by the sender (trainer).
    """

    master_addr: str
    master_port: int
    rank: int  # This GPU's rank in weight sync group (1-indexed)
    world_size: int  # Total: trainer + inference GPUs
    group_name: str = "weight_sync"
    timeout_seconds: float = 300.0
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))

    _process_group: Any = field(default=None, init=False, repr=False)
    _stateless_group: Any = field(default=None, init=False, repr=False)
    _communicator: Any = field(default=None, init=False, repr=False)
    _weight_version: int = field(default=0, init=False)

    @property
    def weight_version(self) -> int:
        return self._weight_version

    def init_group(self) -> None:
        """Initialize NCCL process group. Call once at startup."""
        assert self.rank > 0, "Rank 0 is reserved for trainer (sender)"
        self.device = _normalize_cuda_device(self.device)
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,COLL")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        socket_ifname, socket_ifname_source = _apply_socket_ifname_defaults()
        logger.info(
            "weight_sync_receiver_init_start rank=%s world_size=%s master=%s:%s device=%s group=%s socket_ifname=%s socket_ifname_source=%s nccl_env=%s",
            self.rank,
            self.world_size,
            self.master_addr,
            self.master_port,
            self.device,
            self.group_name,
            socket_ifname,
            socket_ifname_source,
            _nccl_env_snapshot(),
        )
        _emit_argus_diag(
            "weight_sync_receiver_init_start",
            rank=self.rank,
            world_size=self.world_size,
            master_addr=self.master_addr,
            master_port=self.master_port,
            device=str(self.device),
            group=self.group_name,
            socket_ifname=socket_ifname,
            socket_ifname_source=socket_ifname_source,
            nccl_env=_nccl_env_snapshot(),
        )
        _emit_argus_diag(
            "weight_sync_receiver_distributed_state",
            phase="init_start",
            rank=self.rank,
            world_size=self.world_size,
            **_receiver_distributed_state_snapshot(),
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        init_method = f"tcp://{self.master_addr}:{self.master_port}"
        logger.info(
            "weight_sync_receiver_pg_create_start rank=%s world_size=%s init_method=%s device=%s group=%s",
            self.rank,
            self.world_size,
            init_method,
            self.device,
            self.group_name,
        )
        _emit_argus_diag(
            "weight_sync_receiver_pg_create_start",
            rank=self.rank,
            world_size=self.world_size,
            init_method=init_method,
            device=str(self.device),
            group=self.group_name,
        )
        self._process_group = init_extra_process_group(
            group_name=self.group_name,
            backend="nccl",
            init_method=init_method,
            rank=self.rank,
            world_size=self.world_size,
        )
        self._stateless_group = None
        self._communicator = None
        logger.info(
            "weight_sync_receiver_pg_create_ok rank=%s world_size=%s init_method=%s device=%s group=%s pg_backend=%s pg_rank=%s pg_world_size=%s",
            self.rank,
            self.world_size,
            init_method,
            self.device,
            self.group_name,
            dist.get_backend(self._process_group),
            dist.get_rank(self._process_group),
            dist.get_world_size(self._process_group),
        )
        _emit_argus_diag(
            "weight_sync_receiver_pg_create_ok",
            rank=self.rank,
            world_size=self.world_size,
            init_method=init_method,
            device=str(self.device),
            group=self.group_name,
            pg_backend=dist.get_backend(self._process_group),
            pg_rank=dist.get_rank(self._process_group),
            pg_world_size=dist.get_world_size(self._process_group),
        )
        logger.info(
            "weight_sync_receiver_init_ok rank=%s world_size=%s master=%s:%s device=%s group=%s pg_backend=%s pg_rank=%s pg_world_size=%s communicator=%s socket_ifname=%s socket_ifname_source=%s nccl_env=%s",
            self.rank,
            self.world_size,
            self.master_addr,
            self.master_port,
            self.device,
            self.group_name,
            dist.get_backend(self._process_group),
            dist.get_rank(self._process_group),
            dist.get_world_size(self._process_group),
            "dist.broadcast",
            socket_ifname,
            socket_ifname_source,
            _nccl_env_snapshot(),
        )
        _emit_argus_diag(
            "weight_sync_receiver_init_ok",
            rank=self.rank,
            world_size=self.world_size,
            master_addr=self.master_addr,
            master_port=self.master_port,
            device=str(self.device),
            group=self.group_name,
            pg_backend=dist.get_backend(self._process_group),
            pg_rank=dist.get_rank(self._process_group),
            pg_world_size=dist.get_world_size(self._process_group),
            communicator="dist.broadcast",
            socket_ifname=socket_ifname,
            socket_ifname_source=socket_ifname_source,
            nccl_env=_nccl_env_snapshot(),
        )
        _emit_argus_diag(
            "weight_sync_receiver_distributed_state",
            phase="init_ok",
            rank=self.rank,
            world_size=self.world_size,
            **_receiver_distributed_state_snapshot(),
        )

    def receive_weights(
        self,
        param_info: list[ParamInfo],
    ) -> dict[str, Tensor]:
        """Receive weights from trainer broadcast.

        Args:
            param_info: List of (name, shape, dtype) for expected params

        Returns:
            State dict with received weights
        """
        assert self._communicator is not None or self._process_group is not None, (
            "Call init_group() first"
        )

        state_dict = {}
        total_tensors = len(param_info)
        logger.info(
            "weight_sync_receiver_receive_start rank=%s world_size=%s total_tensors=%s communicator=%s first_tensors=%s nccl_env=%s",
            self.rank,
            self.world_size,
            total_tensors,
            type(self._communicator).__name__
            if self._communicator is not None
            else "dist.broadcast",
            [
                {
                    "wire_name": info.wire_name,
                    "load_name": info.load_name,
                    "shape": list(info.shape),
                    "dtype": str(info.dtype).replace("torch.", ""),
                }
                for info in param_info[:3]
            ],
            _nccl_env_snapshot(),
        )
        _emit_argus_diag(
            "weight_sync_receiver_receive_start",
            rank=self.rank,
            world_size=self.world_size,
            total_tensors=total_tensors,
            communicator=(
                type(self._communicator).__name__
                if self._communicator is not None
                else "dist.broadcast"
            ),
            first_tensors=[
                {
                    "wire_name": info.wire_name,
                    "load_name": info.load_name,
                    "shape": list(info.shape),
                    "dtype": str(info.dtype).replace("torch.", ""),
                }
                for info in param_info[:3]
            ],
            nccl_env=_nccl_env_snapshot(),
        )
        _emit_argus_diag(
            "weight_sync_receiver_distributed_state",
            phase="receive_start",
            rank=self.rank,
            world_size=self.world_size,
            **_receiver_distributed_state_snapshot(),
        )
        for index, info in enumerate(param_info):
            # Allocate buffer
            buffer = torch.empty(info.shape, dtype=info.dtype, device=self.device)

            # Receive broadcast from rank 0
            tensor_meta = _tensor_sync_metadata(info.wire_name, buffer)
            receive_meta = {
                **tensor_meta,
                "load_name": info.load_name,
                "current_cuda_device": (
                    torch.cuda.current_device() if torch.cuda.is_available() else None
                ),
                "data_ptr": int(buffer.data_ptr()),
                "storage_offset": int(buffer.storage_offset()),
                "element_size": int(buffer.element_size()),
                "nbytes": int(buffer.numel() * buffer.element_size()),
                "group_rank": dist.get_rank(self._process_group)
                if self._process_group is not None
                else self.rank,
                "group_world_size": dist.get_world_size(self._process_group)
                if self._process_group is not None
                else self.world_size,
            }
            logger.info(
                "weight_sync_receiver_receive_tensor index=%s total_tensors=%s tensor=%s",
                index,
                total_tensors,
                receive_meta,
            )
            if index == 0:
                _emit_argus_diag(
                    "weight_sync_receiver_first_collective_start",
                    rank=self.rank,
                    world_size=self.world_size,
                    total_tensors=total_tensors,
                    tensor=receive_meta,
                )
            try:
                if self._communicator is not None:
                    self._communicator.broadcast(buffer, src=0, stream=torch.cuda.current_stream())
                else:
                    dist.broadcast(buffer, src=0, group=self._process_group)
            except Exception as exc:
                raise RuntimeError(
                    "Weight sync receiver broadcast failed "
                    f"index={index} total_tensors={total_tensors} tensor={receive_meta}"
                ) from exc
            if index == 0:
                _emit_argus_diag(
                    "weight_sync_receiver_first_collective_ok",
                    rank=self.rank,
                    world_size=self.world_size,
                    total_tensors=total_tensors,
                    tensor=receive_meta,
                )

            state_dict[info.load_name] = buffer

        self._weight_version += 1
        logger.info(f"Received weights v{self._weight_version}")
        return state_dict

    def receive_weights_into(
        self,
        state_dict: dict[str, Tensor],
    ) -> None:
        """Receive weights directly into existing state_dict tensors (in-place).

        This is more efficient than receive_weights() as it avoids allocation.

        Args:
            state_dict: Existing state dict to receive into
        """
        assert self._communicator is not None or self._process_group is not None, (
            "Call init_group() first"
        )

        for _name, param in state_dict.items():
            # Receive broadcast from rank 0 directly into existing tensor
            if self._communicator is not None:
                self._communicator.broadcast(param.data, src=0, stream=torch.cuda.current_stream())
            else:
                dist.broadcast(param.data, src=0, group=self._process_group)

        self._weight_version += 1
        logger.info(f"Received weights v{self._weight_version} (in-place)")

    def cleanup(self) -> None:
        """Cleanup process group."""
        if self._communicator is not None:
            del self._communicator
            self._communicator = None
        self._stateless_group = None
        if self._process_group is not None:
            dist.destroy_process_group(self._process_group)
            self._process_group = None
            logger.info("Weight sync receiver cleaned up")


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


def add_weight_sync_to_engine(
    engine: Any,
    master_addr: str,
    master_port: int,
    rank: int,
    world_size: int,
    group_name: str = "weight_sync",
) -> WeightSyncReceiver:
    """Add NCCL weight sync capability to an inference engine.

    This creates a WeightSyncReceiver and attaches it to the engine.
    When weights are broadcast, call engine.receive_and_reload_weights().

    Args:
        engine: InferenceEngineV2 instance
        master_addr: IP of trainer
        master_port: Port for NCCL rendezvous
        rank: This engine's rank (1-indexed)
        world_size: Total processes

    Returns:
        WeightSyncReceiver attached to engine
    """
    receiver = WeightSyncReceiver(
        master_addr=master_addr,
        master_port=master_port,
        rank=rank,
        world_size=world_size,
        group_name=group_name,
        device=engine.device,
    )
    receiver.init_group()

    # Store receiver on engine for later use
    engine._weight_sync_receiver = receiver

    return receiver


def receive_and_reload_weights(engine: Any) -> None:
    """Receive broadcasted weights and reload into engine.

    Call this after sender.broadcast_weights().
    """
    receiver = getattr(engine, "_weight_sync_receiver", None)
    assert receiver is not None, "Call add_weight_sync_to_engine() first"

    # Get current model's state dict structure
    if engine._use_functional_model:
        template = engine._functional_weights
    else:
        template = engine.model.state_dict()

    # Receive in-place
    receiver.receive_weights_into(template)

    # Reload (handles cache invalidation)
    engine.reload_weights(template)
