"""SGLang launcher with transformers compatibility patches.

Patches transformers.utils.hub.list_repo_templates to handle 404s gracefully.
See: https://github.com/huggingface/transformers/issues/41813

Usage:
    python -m rollouts.inference.realizations.slime_sglang --model-path ... --port ...
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import ipaddress
import json
import os
import subprocess
import sys
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType

_SIDECAR_LOCK = threading.Lock()
_WEIGHT_UPDATE_CONTEXT = threading.local()


def _sidecar_trace_path() -> Path | None:
    raw = os.environ.get("ROLLOUTS_SGLANG_TRACE_PATH")
    if not raw:
        return None
    try:
        return Path(raw)
    except Exception:
        return None


def _write_sidecar_event(payload: dict[str, object]) -> None:
    path = _sidecar_trace_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _SIDECAR_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        return


def _should_mirror_to_stderr(event: str) -> bool:
    if _sidecar_trace_path() is None:
        return True
    # TODO(child-event-contract): `rollouts` should declare which engine-local
    # milestones belong in the parent execution event stream and `bifrost`
    # should carry them over a dedicated child-event channel. Until then, keep
    # high-volume runtime introspection in the sidecar JSONL and only mirror
    # concise failure-class diagnostics to stderr.
    return event.endswith("_failed")


def _patch_transformers() -> None:
    """Patch list_repo_templates to catch RemoteEntryNotFoundError."""
    try:
        import transformers.utils.hub as hub
    except ImportError:
        return  # transformers not installed

    _original = hub.list_repo_templates

    def _patched_list_repo_templates(*args: object, **kwargs: object) -> object:  # noqa: ANN202
        try:
            yield from _original(*args, **kwargs)
        except Exception:
            # additional_chat_templates directory doesn't exist - that's fine
            return

    hub.list_repo_templates = _patched_list_repo_templates  # type: ignore[invalid-assignment]


def _emit_argus_diag(event: str, **data: object) -> None:
    payload = {
        "event": event,
        "ts": datetime.now(UTC).isoformat(),
        **data,
    }
    _write_sidecar_event(payload)
    if not _should_mirror_to_stderr(event):
        return
    try:
        sys.stderr.write(f"__ARGUS_DIAG__{json.dumps(payload, sort_keys=True)}\n")
        sys.stderr.flush()
    except Exception:
        return


def _module_origin(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        return None
    if spec is None:
        return None
    return spec.origin


def _safe_repr(value: object, *, max_len: int = 240) -> str:
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr failed: {type(exc).__name__}: {exc}>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _jsonable_summary(value: object, *, depth: int = 0) -> object:
    if depth >= 2:
        return _safe_repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        items = [_jsonable_summary(item, depth=depth + 1) for item in value[:5]]
        if len(value) > 5:
            items.append(f"...(+{len(value) - 5} more)")
        return items
    if isinstance(value, dict):
        items = list(value.items())[:8]
        summary = {str(key): _jsonable_summary(item, depth=depth + 1) for key, item in items}
        if len(value) > 8:
            summary["..."] = f"+{len(value) - 8} more"
        return summary
    for method_name in ("model_dump", "dict", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                dumped = method()
            except TypeError:
                try:
                    dumped = method(exclude_none=False)
                except Exception:
                    continue
            except Exception:
                continue
            return _jsonable_summary(dumped, depth=depth + 1)
    if hasattr(value, "__dict__"):
        try:
            payload = {
                key: _jsonable_summary(item, depth=depth + 1)
                for key, item in list(vars(value).items())[:8]
            }
            if len(vars(value)) > 8:
                payload["..."] = f"+{len(vars(value)) - 8} more"
            payload["__class__"] = type(value).__name__
            return payload
        except Exception:
            pass
    return _safe_repr(value)


def _process_context() -> dict[str, object]:
    payload: dict[str, object] = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "thread_id": threading.get_ident(),
        "thread_name": threading.current_thread().name,
        "cwd": os.getcwd(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        payload["torch_cuda_is_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            payload["torch_current_device"] = torch.cuda.current_device()
            payload["torch_device_count"] = torch.cuda.device_count()
    except Exception as exc:
        payload["torch_context_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _distributed_env_summary() -> dict[str, object]:
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
    return {key: os.environ.get(key) for key in keys}


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() not in {"", "0", "false", "no"}


def _method_owner_state(owner: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "class": type(owner).__name__,
        "id": hex(id(owner)),
        "module": type(owner).__module__,
    }
    for attr in (
        "server_args",
        "tp_worker",
        "model_update_lock",
        "send_to_scheduler",
        "scheduler",
    ):
        value = getattr(owner, attr, None)
        if value is not None:
            payload[f"has_{attr}"] = True
            payload[f"{attr}_type"] = type(value).__name__
        else:
            payload[f"has_{attr}"] = False
    writer_lock = getattr(getattr(owner, "model_update_lock", None), "writer_lock", None)
    if writer_lock is not None:
        payload["writer_lock_type"] = type(writer_lock).__name__
    return payload


def _traceback_tail(limit: int = 8) -> list[str]:
    return traceback.format_exc().strip().splitlines()[-limit:]


def _weight_update_stack() -> list[dict[str, object]]:
    stack = getattr(_WEIGHT_UPDATE_CONTEXT, "stack", None)
    if stack is None:
        stack = []
        _WEIGHT_UPDATE_CONTEXT.stack = stack
    return stack


def _push_weight_update_context(*, owner_class: str, method: str) -> None:
    _weight_update_stack().append({
        "owner_class": owner_class,
        "method": method,
    })


def _pop_weight_update_context() -> None:
    stack = _weight_update_stack()
    if stack:
        stack.pop()


def _current_weight_update_context() -> dict[str, object] | None:
    stack = _weight_update_stack()
    if not stack:
        return None
    return {
        "stack": list(stack),
        "depth": len(stack),
        "current": stack[-1],
    }


def _extract_weight_sync_contract(
    owner: object,
    *,
    owner_class: str,
    method_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> dict[str, object] | None:
    def _from_object(value: object) -> dict[str, object]:
        contract: dict[str, object] = {}
        for key in (
            "master_address",
            "master_port",
            "rank_offset",
            "world_size",
            "group_name",
            "backend",
        ):
            attr = getattr(value, key, None)
            if attr is not None:
                contract[key] = attr
        return contract

    contract: dict[str, object] = {}
    for key in (
        "master_address",
        "master_port",
        "rank_offset",
        "world_size",
        "group_name",
        "backend",
    ):
        if key in kwargs and kwargs[key] is not None:
            contract[key] = kwargs[key]
    if args:
        if hasattr(args[0], "__dict__"):
            contract.update({k: v for k, v in _from_object(args[0]).items() if k not in contract})
        if method_name == "init_weights_update_group" and owner_class == "ModelRunner":
            names = (
                "master_address",
                "master_port",
                "rank_offset",
                "world_size",
                "group_name",
                "backend",
            )
            for index, key in enumerate(names):
                if index < len(args) and key not in contract and args[index] is not None:
                    contract[key] = args[index]
    if "group_name" not in contract and method_name.startswith("update_weights"):
        remembered = getattr(owner, "_rollouts_weight_sync_contracts", None)
        if isinstance(remembered, dict) and len(remembered) == 1:
            only_contract = next(iter(remembered.values()))
            if isinstance(only_contract, dict):
                contract.update({k: v for k, v in only_contract.items() if k not in contract})
    if not contract:
        return None
    return contract


def _remember_weight_sync_contract(owner: object, contract: dict[str, object] | None) -> None:
    if not contract:
        return
    group_name = contract.get("group_name")
    if group_name is None:
        return
    contracts = getattr(owner, "_rollouts_weight_sync_contracts", None)
    if not isinstance(contracts, dict):
        contracts = {}
        try:
            owner._rollouts_weight_sync_contracts = contracts
        except Exception:
            return
    contracts[str(group_name)] = dict(contract)


def _filter_socket_snapshot_for_port(
    snapshot: dict[str, object], *, port: int | None
) -> dict[str, object] | None:
    if port is None:
        return None
    proc_net = snapshot.get("proc_net")
    if not isinstance(proc_net, dict):
        return None
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
        "tcp": _filter_rows(proc_net.get("tcp")),
        "tcp6": _filter_rows(proc_net.get("tcp6")),
    }


def _should_trace_weight_update_collectives(owner_cls: type[object], method_name: str) -> bool:
    if method_name not in {
        "init_weights_update_group",
        "update_weights_from_distributed",
        "update_weights_from_ipc",
    }:
        return False
    return owner_cls.__name__ in {
        "TokenizerManager",
        "SchedulerUpdateWeightsMixin",
        "BaseTpWorker",
        "ModelRunner",
    }


def _group_state_summary(group: object) -> dict[str, object]:
    if isinstance(group, dict):
        items: dict[str, object] = {}
        for key, value in group.items():
            key_name = str(key)
            try:
                items[key_name] = _group_state_summary(value)
            except Exception as exc:
                items[key_name] = {
                    "type": type(value).__name__,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return {
            "type": type(group).__name__,
            "is_none": False,
            "size": len(group),
            "items": items,
        }
    payload: dict[str, object] = {
        "type": type(group).__name__ if group is not None else None,
        "is_none": group is None,
    }
    try:
        import torch.distributed as dist

        if group is None:
            if dist.is_initialized():
                payload["backend"] = dist.get_backend()
                payload["rank"] = dist.get_rank()
                payload["world_size"] = dist.get_world_size()
            else:
                payload["backend"] = None
                payload["rank"] = None
                payload["world_size"] = None
        else:
            payload["backend"] = dist.get_backend(group)
            payload["rank"] = dist.get_rank(group)
            payload["world_size"] = dist.get_world_size(group)
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _owner_process_group_summary(owner: object) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    for attr_name in dir(owner):
        if "group" not in attr_name.lower():
            continue
        try:
            value = getattr(owner, attr_name)
        except Exception as exc:
            summaries.append({
                "attr": attr_name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        if value is None:
            continue
        object_id = id(value)
        if object_id in seen_ids:
            continue
        seen_ids.add(object_id)
        summary = _group_state_summary(value)
        if "backend" not in summary and "error" in summary:
            continue
        summary["attr"] = attr_name
        summaries.append(summary)
    return summaries


def _tensor_summary(tensor: object) -> dict[str, object]:
    payload: dict[str, object] = {"type": type(tensor).__name__}
    try:
        import torch

        if isinstance(tensor, torch.Tensor):
            payload.update({
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "device": str(tensor.device),
                "numel": int(tensor.numel()),
                "is_contiguous": bool(tensor.is_contiguous()),
                "stride": list(tensor.stride()),
            })
            if tensor.device.type == "cuda":
                payload["current_cuda_device"] = torch.cuda.current_device()
                payload["data_ptr"] = int(tensor.data_ptr())
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _instrument_torch_distributed_collectives() -> None:
    try:
        import torch.distributed as dist
    except Exception as exc:
        _emit_argus_diag(
            "sglang_runtime_collective_instrumentation_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        return

    def _wrap_collective(name: str) -> None:
        original = getattr(dist, name, None)
        if not callable(original) or getattr(original, "__rollouts_argus_wrapped__", False):
            return

        @functools.wraps(original)
        def wrapped(*args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
            context = _current_weight_update_context()
            if context is None:
                return original(*args, **kwargs)

            payload: dict[str, object] = {
                "collective": name,
                "context": context,
                "process": _process_context(),
            }
            current = context.get("current", {})
            if isinstance(current, dict) and current.get("contract") is not None:
                payload["weight_sync_contract"] = current["contract"]
            group = kwargs.get("group")
            if group is None and name == "barrier" and args:
                group = args[0]
            payload["group"] = _group_state_summary(group)
            payload["default_group"] = _group_state_summary(None)
            if name == "broadcast":
                tensor = args[0] if args else kwargs.get("tensor")
                payload["tensor"] = _tensor_summary(tensor)
                payload["src"] = kwargs.get("src", args[1] if len(args) > 1 else None)
                payload["async_op"] = kwargs.get("async_op", False)
                payload["socket_state"] = _capture_ss_snapshot(pid=os.getpid())
                contract = payload.get("weight_sync_contract")
                master_port = None
                if isinstance(contract, dict):
                    try:
                        raw_port = contract.get("master_port")
                        master_port = int(raw_port) if raw_port is not None else None
                    except Exception:
                        master_port = None
                payload["bootstrap_socket_state"] = _filter_socket_snapshot_for_port(
                    payload["socket_state"], port=master_port
                )
                try:
                    import torch

                    if torch.cuda.is_available():
                        stream = torch.cuda.current_stream()
                        payload["cuda_stream"] = {
                            "device": str(stream.device),
                            "cuda_stream": int(stream.cuda_stream),
                        }
                except Exception as exc:
                    payload["cuda_stream_error"] = f"{type(exc).__name__}: {exc}"
                if _env_flag("ROLLOUTS_SGLANG_FORCE_SYNC_BROADCAST") and kwargs.get(
                    "async_op", False
                ):
                    kwargs = dict(kwargs)
                    kwargs["async_op"] = False
                    payload["async_override"] = {
                        "env": "ROLLOUTS_SGLANG_FORCE_SYNC_BROADCAST",
                        "forced_async_op": False,
                    }
                    _emit_argus_diag("sglang_runtime_collective_async_override", **payload)
                    payload["async_op"] = False

            _emit_argus_diag("sglang_runtime_collective_enter", **payload)
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                _emit_argus_diag(
                    "sglang_runtime_collective_failed",
                    **payload,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback_tail=_traceback_tail(),
                )
                raise

            _emit_argus_diag(
                "sglang_runtime_collective_ok",
                **payload,
                result_type=type(result).__name__,
                result=_jsonable_summary(result),
            )
            return result

        wrapped.__rollouts_argus_wrapped__ = True  # type: ignore[attr-defined]
        setattr(dist, name, wrapped)

    for collective_name in ("broadcast", "barrier", "all_reduce"):
        _wrap_collective(collective_name)
    _emit_argus_diag(
        "sglang_runtime_collective_instrumentation_ready",
        wrapped=["broadcast", "barrier", "all_reduce"],
    )


def _should_snapshot_modelrunner_sockets(owner_cls: type[object], method_name: str) -> bool:
    return owner_cls.__name__ == "ModelRunner" and method_name in {
        "init_weights_update_group",
        "update_weights_from_distributed",
    }


def _truncate_text(value: str, *, max_len: int = 320) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _capture_ss_snapshot(*, pid: int) -> dict[str, object]:
    snapshots: list[dict[str, object]] = []
    ss_missing = False
    for command in (
        ["ss", "-H", "-tnlp"],
        ["ss", "-H", "-tnp", "state", "all"],
    ):
        command_name = " ".join(command)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
        except Exception as exc:
            if isinstance(exc, FileNotFoundError):
                ss_missing = True
            snapshots.append({
                "command": command_name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        matched_lines = [
            _truncate_text(line.strip())
            for line in completed.stdout.splitlines()
            if f"pid={pid}," in line
        ]
        snapshots.append({
            "command": command_name,
            "returncode": completed.returncode,
            "matched_lines": matched_lines[:64],
            "matched_count": len(matched_lines),
            "stderr_tail": [
                _truncate_text(line.strip())
                for line in completed.stderr.splitlines()[-8:]
                if line.strip()
            ],
        })
    proc_snapshot = _capture_proc_socket_snapshot(pid=pid) if ss_missing else None
    return {"pid": pid, "snapshots": snapshots, "proc_net": proc_snapshot}


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
        # /proc/net/tcp6 stores the 128-bit address in 4 little-endian u32 words.
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


def _capture_proc_socket_snapshot(*, pid: int) -> dict[str, object]:
    inodes = _collect_socket_inodes(pid)
    return {
        "pid": pid,
        "socket_inode_count": len(inodes),
        "tcp": _parse_proc_net_tcp(Path("/proc/net/tcp"), inodes=inodes, ipv6=False)[:64],
        "tcp6": _parse_proc_net_tcp(Path("/proc/net/tcp6"), inodes=inodes, ipv6=True)[:64],
    }


def _emit_modelrunner_socket_snapshot(
    *,
    owner_cls: type[object],
    method_name: str,
    phase: str,
    self: object,
) -> None:
    if not _should_snapshot_modelrunner_sockets(owner_cls, method_name):
        return
    _emit_argus_diag(
        "sglang_runtime_socket_snapshot",
        owner_class=owner_cls.__name__,
        method=method_name,
        phase=phase,
        process=_process_context(),
        owner_state=_method_owner_state(self),
        socket_state=_capture_ss_snapshot(pid=os.getpid()),
    )


def _emit_process_group_snapshot(
    *,
    owner_cls: type[object],
    method_name: str,
    phase: str,
    self: object,
) -> None:
    if owner_cls.__name__ not in {"ModelRunner", "BaseTpWorker", "SchedulerUpdateWeightsMixin"}:
        return
    summaries = _owner_process_group_summary(self)
    if not summaries:
        return
    _emit_argus_diag(
        "sglang_runtime_process_group_snapshot",
        owner_class=owner_cls.__name__,
        method=method_name,
        phase=phase,
        process=_process_context(),
        groups=summaries,
    )


def _emit_distributed_state_snapshot(
    *,
    owner_cls: type[object],
    method_name: str,
    phase: str,
) -> None:
    if owner_cls.__name__ not in {
        "TokenizerManager",
        "SchedulerUpdateWeightsMixin",
        "BaseTpWorker",
        "ModelRunner",
    }:
        return
    _emit_argus_diag(
        "sglang_runtime_distributed_state",
        owner_class=owner_cls.__name__,
        method=method_name,
        phase=phase,
        process=_process_context(),
        env=_distributed_env_summary(),
        default_group=_group_state_summary(None),
    )


def _instrument_async_context_type(lock_type: type[object], *, label: str) -> None:
    if getattr(lock_type, "__rollouts_argus_wrapped__", False):
        return
    enter = getattr(lock_type, "__aenter__", None)
    exit_ = getattr(lock_type, "__aexit__", None)
    if not callable(enter) or not callable(exit_):
        return

    @functools.wraps(enter)
    async def wrapped_enter(self: object, *args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        _emit_argus_diag(
            "sglang_runtime_lock_enter",
            label=label,
            lock_type=lock_type.__name__,
            process=_process_context(),
        )
        try:
            result = await enter(self, *args, **kwargs)
        except Exception as exc:
            _emit_argus_diag(
                "sglang_runtime_lock_enter_failed",
                label=label,
                lock_type=lock_type.__name__,
                process=_process_context(),
                error=f"{type(exc).__name__}: {exc}",
                traceback_tail=_traceback_tail(),
            )
            raise
        _emit_argus_diag(
            "sglang_runtime_lock_enter_ok",
            label=label,
            lock_type=lock_type.__name__,
            process=_process_context(),
        )
        return result

    @functools.wraps(exit_)
    async def wrapped_exit(self: object, *args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        _emit_argus_diag(
            "sglang_runtime_lock_exit",
            label=label,
            lock_type=lock_type.__name__,
            process=_process_context(),
        )
        try:
            result = await exit_(self, *args, **kwargs)
        except Exception as exc:
            _emit_argus_diag(
                "sglang_runtime_lock_exit_failed",
                label=label,
                lock_type=lock_type.__name__,
                process=_process_context(),
                error=f"{type(exc).__name__}: {exc}",
                traceback_tail=_traceback_tail(),
            )
            raise
        _emit_argus_diag(
            "sglang_runtime_lock_exit_ok",
            label=label,
            lock_type=lock_type.__name__,
            process=_process_context(),
        )
        return result

    lock_type.__aenter__ = wrapped_enter  # type: ignore[assignment]
    lock_type.__aexit__ = wrapped_exit  # type: ignore[assignment]
    lock_type.__rollouts_argus_wrapped__ = True  # type: ignore[attr-defined]


def _instrument_bound_method_instance(
    instance: object,
    method_name: str,
    *,
    event_prefix: str,
    label: str,
) -> None:
    method = getattr(instance, method_name, None)
    if not callable(method) or getattr(method, "__rollouts_argus_wrapped__", False):
        return

    def _normalize_call_args(args: tuple[object, ...]) -> tuple[object, ...]:
        # `getattr(instance, method_name)` returns a bound method. After rebinding the
        # wrapper onto the instance with `MethodType`, Python passes `instance` again.
        # Drop that extra leading self so instrumentation does not mutate call semantics.
        if args and args[0] is instance:
            return args[1:]
        return args

    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def wrapped(*args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
            call_args = _normalize_call_args(args)
            _emit_argus_diag(
                f"{event_prefix}_enter",
                label=label,
                method=method_name,
                process=_process_context(),
                owner_type=type(instance).__name__,
                args=_jsonable_summary(call_args),
                kwargs=_jsonable_summary(kwargs),
            )
            try:
                result = await method(*call_args, **kwargs)
            except Exception as exc:
                _emit_argus_diag(
                    f"{event_prefix}_failed",
                    label=label,
                    method=method_name,
                    process=_process_context(),
                    owner_type=type(instance).__name__,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback_tail=_traceback_tail(),
                )
                raise
            _emit_argus_diag(
                f"{event_prefix}_ok",
                label=label,
                method=method_name,
                process=_process_context(),
                owner_type=type(instance).__name__,
                result=_jsonable_summary(result),
            )
            return result

    else:

        @functools.wraps(method)
        def wrapped(*args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
            call_args = _normalize_call_args(args)
            _emit_argus_diag(
                f"{event_prefix}_enter",
                label=label,
                method=method_name,
                process=_process_context(),
                owner_type=type(instance).__name__,
                args=_jsonable_summary(call_args),
                kwargs=_jsonable_summary(kwargs),
            )
            try:
                result = method(*call_args, **kwargs)
            except Exception as exc:
                _emit_argus_diag(
                    f"{event_prefix}_failed",
                    label=label,
                    method=method_name,
                    process=_process_context(),
                    owner_type=type(instance).__name__,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback_tail=_traceback_tail(),
                )
                raise
            _emit_argus_diag(
                f"{event_prefix}_ok",
                label=label,
                method=method_name,
                process=_process_context(),
                owner_type=type(instance).__name__,
                result=_jsonable_summary(result),
            )
            return result

    wrapped.__rollouts_argus_wrapped__ = True  # type: ignore[attr-defined]
    setattr(instance, method_name, MethodType(wrapped, instance))


def _instrument_communicator_instance(label: str, communicator: object) -> None:
    try:
        communicator._rollouts_argus_label = label  # type: ignore[attr-defined]
    except Exception:
        pass
    communicator_type = type(communicator)
    if not getattr(communicator_type, "__rollouts_argus_call_wrapped__", False):
        call = communicator_type.__call__ if callable(communicator) else None
        if call is not None:
            if inspect.iscoroutinefunction(call):

                @functools.wraps(call)
                async def wrapped_call(self: object, *args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
                    call_label = getattr(self, "_rollouts_argus_label", communicator_type.__name__)
                    _emit_argus_diag(
                        "sglang_runtime_communicator_call_enter",
                        label=call_label,
                        process=_process_context(),
                        owner_type=communicator_type.__name__,
                        args=_jsonable_summary(args),
                        kwargs=_jsonable_summary(kwargs),
                    )
                    try:
                        result = await call(self, *args, **kwargs)
                    except Exception as exc:
                        _emit_argus_diag(
                            "sglang_runtime_communicator_call_failed",
                            label=call_label,
                            process=_process_context(),
                            owner_type=communicator_type.__name__,
                            error=f"{type(exc).__name__}: {exc}",
                            traceback_tail=_traceback_tail(),
                        )
                        raise
                    _emit_argus_diag(
                        "sglang_runtime_communicator_call_ok",
                        label=call_label,
                        process=_process_context(),
                        owner_type=communicator_type.__name__,
                        result=_jsonable_summary(result),
                    )
                    return result

            else:

                @functools.wraps(call)
                def wrapped_call(self: object, *args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
                    call_label = getattr(self, "_rollouts_argus_label", communicator_type.__name__)
                    _emit_argus_diag(
                        "sglang_runtime_communicator_call_enter",
                        label=call_label,
                        process=_process_context(),
                        owner_type=communicator_type.__name__,
                        args=_jsonable_summary(args),
                        kwargs=_jsonable_summary(kwargs),
                    )
                    try:
                        result = call(self, *args, **kwargs)
                    except Exception as exc:
                        _emit_argus_diag(
                            "sglang_runtime_communicator_call_failed",
                            label=call_label,
                            process=_process_context(),
                            owner_type=communicator_type.__name__,
                            error=f"{type(exc).__name__}: {exc}",
                            traceback_tail=_traceback_tail(),
                        )
                        raise
                    _emit_argus_diag(
                        "sglang_runtime_communicator_call_ok",
                        label=call_label,
                        process=_process_context(),
                        owner_type=communicator_type.__name__,
                        result=_jsonable_summary(result),
                    )
                    return result

            communicator_type.__call__ = wrapped_call  # type: ignore[assignment]
            communicator_type.__rollouts_argus_call_wrapped__ = True  # type: ignore[attr-defined]
    if getattr(communicator, "__rollouts_argus_wrapped__", False):
        return
    for method_name in ("handle_recv", "merge_results"):
        _instrument_bound_method_instance(
            communicator,
            method_name,
            event_prefix="sglang_runtime_communicator",
            label=label,
        )
    communicator.__rollouts_argus_wrapped__ = True  # type: ignore[attr-defined]


def _instrument_socket_instance(label: str, sock: object) -> None:
    if getattr(sock, "__rollouts_argus_wrapped__", False):
        return
    for method_name in ("send_pyobj", "recv_pyobj", "recv", "send"):
        try:
            _instrument_bound_method_instance(
                sock,
                method_name,
                event_prefix="sglang_runtime_scheduler_socket",
                label=label,
            )
        except Exception:
            continue
    sock.__rollouts_argus_wrapped__ = True  # type: ignore[attr-defined]


def _ensure_owner_instrumented(owner: object) -> None:
    writer_lock = getattr(getattr(owner, "model_update_lock", None), "writer_lock", None)
    if writer_lock is not None:
        _instrument_async_context_type(type(writer_lock), label="model_update_writer_lock")
    send_to_scheduler = getattr(owner, "send_to_scheduler", None)
    if send_to_scheduler is not None:
        try:
            _instrument_socket_instance("send_to_scheduler", send_to_scheduler)
        except Exception:
            pass
    for attr_name in dir(owner):
        if not attr_name.endswith("_communicator"):
            continue
        try:
            communicator = getattr(owner, attr_name)
        except Exception:
            continue
        if communicator is None:
            continue
        try:
            _instrument_communicator_instance(attr_name, communicator)
        except Exception:
            continue


def _wrap_runtime_method(owner_cls: type[object], method_name: str) -> None:
    method = getattr(owner_cls, method_name, None)
    if not callable(method) or getattr(method, "__rollouts_argus_wrapped__", False):
        return

    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def wrapped(self: object, *args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
            _ensure_owner_instrumented(self)
            trace_collectives = _should_trace_weight_update_collectives(owner_cls, method_name)
            contract = _extract_weight_sync_contract(
                self,
                owner_class=owner_cls.__name__,
                method_name=method_name,
                args=args,
                kwargs=kwargs,
            )
            if trace_collectives:
                _push_weight_update_context(
                    owner_class=owner_cls.__name__,
                    method=method_name,
                )
                if contract is not None:
                    _weight_update_stack()[-1]["contract"] = contract
            _emit_argus_diag(
                "sglang_runtime_method_enter",
                owner_class=owner_cls.__name__,
                method=method_name,
                process=_process_context(),
                owner_state=_method_owner_state(self),
                weight_sync_contract=contract,
                args=_jsonable_summary(args),
                kwargs=_jsonable_summary(kwargs),
            )
            _emit_modelrunner_socket_snapshot(
                owner_cls=owner_cls,
                method_name=method_name,
                phase="enter",
                self=self,
            )
            _emit_distributed_state_snapshot(
                owner_cls=owner_cls,
                method_name=method_name,
                phase="enter",
            )
            _emit_process_group_snapshot(
                owner_cls=owner_cls,
                method_name=method_name,
                phase="enter",
                self=self,
            )
            try:
                result = await method(self, *args, **kwargs)
            except Exception as exc:
                _emit_argus_diag(
                    "sglang_runtime_method_failed",
                    owner_class=owner_cls.__name__,
                    method=method_name,
                    process=_process_context(),
                    owner_state=_method_owner_state(self),
                    error=f"{type(exc).__name__}: {exc}",
                    traceback_tail=_traceback_tail(),
                )
                _emit_modelrunner_socket_snapshot(
                    owner_cls=owner_cls,
                    method_name=method_name,
                    phase="failed",
                    self=self,
                )
                _emit_distributed_state_snapshot(
                    owner_cls=owner_cls,
                    method_name=method_name,
                    phase="failed",
                )
                _emit_process_group_snapshot(
                    owner_cls=owner_cls,
                    method_name=method_name,
                    phase="failed",
                    self=self,
                )
                raise
            finally:
                if trace_collectives:
                    _pop_weight_update_context()
            _remember_weight_sync_contract(self, contract)
            _emit_argus_diag(
                "sglang_runtime_method_ok",
                owner_class=owner_cls.__name__,
                method=method_name,
                process=_process_context(),
                owner_state=_method_owner_state(self),
                weight_sync_contract=contract,
                result=_jsonable_summary(result),
            )
            _emit_modelrunner_socket_snapshot(
                owner_cls=owner_cls,
                method_name=method_name,
                phase="ok",
                self=self,
            )
            _emit_distributed_state_snapshot(
                owner_cls=owner_cls,
                method_name=method_name,
                phase="ok",
            )
            _emit_process_group_snapshot(
                owner_cls=owner_cls,
                method_name=method_name,
                phase="ok",
                self=self,
            )
            return result

    else:

        @functools.wraps(method)
        def wrapped(self: object, *args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
            _ensure_owner_instrumented(self)
            trace_collectives = _should_trace_weight_update_collectives(owner_cls, method_name)
            contract = _extract_weight_sync_contract(
                self,
                owner_class=owner_cls.__name__,
                method_name=method_name,
                args=args,
                kwargs=kwargs,
            )
            if trace_collectives:
                _push_weight_update_context(
                    owner_class=owner_cls.__name__,
                    method=method_name,
                )
                if contract is not None:
                    _weight_update_stack()[-1]["contract"] = contract
            _emit_argus_diag(
                "sglang_runtime_method_enter",
                owner_class=owner_cls.__name__,
                method=method_name,
                process=_process_context(),
                owner_state=_method_owner_state(self),
                weight_sync_contract=contract,
                args=_jsonable_summary(args),
                kwargs=_jsonable_summary(kwargs),
            )
            _emit_modelrunner_socket_snapshot(
                owner_cls=owner_cls,
                method_name=method_name,
                phase="enter",
                self=self,
            )
            _emit_distributed_state_snapshot(
                owner_cls=owner_cls,
                method_name=method_name,
                phase="enter",
            )
            _emit_process_group_snapshot(
                owner_cls=owner_cls,
                method_name=method_name,
                phase="enter",
                self=self,
            )
            try:
                result = method(self, *args, **kwargs)
            except Exception as exc:
                _emit_argus_diag(
                    "sglang_runtime_method_failed",
                    owner_class=owner_cls.__name__,
                    method=method_name,
                    process=_process_context(),
                    owner_state=_method_owner_state(self),
                    error=f"{type(exc).__name__}: {exc}",
                    traceback_tail=_traceback_tail(),
                )
                _emit_modelrunner_socket_snapshot(
                    owner_cls=owner_cls,
                    method_name=method_name,
                    phase="failed",
                    self=self,
                )
                _emit_distributed_state_snapshot(
                    owner_cls=owner_cls,
                    method_name=method_name,
                    phase="failed",
                )
                _emit_process_group_snapshot(
                    owner_cls=owner_cls,
                    method_name=method_name,
                    phase="failed",
                    self=self,
                )
                raise
            finally:
                if trace_collectives:
                    _pop_weight_update_context()
            _remember_weight_sync_contract(self, contract)
            _emit_argus_diag(
                "sglang_runtime_method_ok",
                owner_class=owner_cls.__name__,
                method=method_name,
                process=_process_context(),
                owner_state=_method_owner_state(self),
                weight_sync_contract=contract,
                result=_jsonable_summary(result),
            )
            _emit_modelrunner_socket_snapshot(
                owner_cls=owner_cls,
                method_name=method_name,
                phase="ok",
                self=self,
            )
            _emit_distributed_state_snapshot(
                owner_cls=owner_cls,
                method_name=method_name,
                phase="ok",
            )
            _emit_process_group_snapshot(
                owner_cls=owner_cls,
                method_name=method_name,
                phase="ok",
                self=self,
            )
            return result

    wrapped.__rollouts_argus_wrapped__ = True  # type: ignore[attr-defined]
    setattr(owner_cls, method_name, wrapped)


def _instrument_sglang_runtime_methods() -> None:
    _instrument_torch_distributed_collectives()
    try:
        from sglang.srt.managers import tokenizer_manager
        from sglang.srt.managers.scheduler_update_weights_mixin import (
            SchedulerUpdateWeightsMixin,
        )
        from sglang.srt.managers.tp_worker import BaseTpWorker
    except Exception as exc:
        _emit_argus_diag(
            "sglang_runtime_method_instrumentation_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        return

    optional_targets: list[tuple[type[object], tuple[str, ...]]] = []
    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except Exception:
        ModelRunner = None  # type: ignore[assignment]
    if ModelRunner is not None:
        optional_targets.append((
            ModelRunner,
            (
                "update_weights_from_distributed",
                "update_weights_from_ipc",
                "init_weights_update_group",
                "post_process_weights",
            ),
        ))

    wrapped: list[dict[str, str]] = []
    targets: list[tuple[type[object], tuple[str, ...]]] = [
        (
            tokenizer_manager.TokenizerManager,
            (
                "init_weights_update_group",
                "update_weights_from_distributed",
                "pause_generation",
                "continue_generation",
                "post_process_weights",
                "init_weights_update_group_communicator",
                "update_weights_from_distributed_communicator",
                "post_process_weights_communicator",
            ),
        ),
        (
            SchedulerUpdateWeightsMixin,
            (
                "init_weights_update_group",
                "update_weights_from_distributed",
                "update_weights_from_ipc",
                "post_process_weights",
            ),
        ),
        (
            BaseTpWorker,
            (
                "init_weights_update_group",
                "update_weights_from_distributed",
                "update_weights_from_ipc",
                "post_process_weights",
            ),
        ),
    ] + optional_targets
    for owner_cls, method_names in targets:
        for method_name in method_names:
            method = getattr(owner_cls, method_name, None)
            if callable(method):
                _wrap_runtime_method(owner_cls, method_name)
                wrapped.append({"class": owner_cls.__name__, "method": method_name})
    _emit_argus_diag("sglang_runtime_method_instrumentation_ready", wrapped=wrapped)


def _log_sglang_runtime_fingerprint() -> None:
    payload: dict[str, object] = {
        "sglang_origin": _module_origin("sglang"),
        "tokenizer_manager_origin": _module_origin("sglang.srt.managers.tokenizer_manager"),
        "io_struct_origin": _module_origin("sglang.srt.managers.io_struct"),
        "pynccl_origin": _module_origin("sglang.srt.distributed.device_communicators.pynccl"),
        "env": {
            key: os.environ.get(key)
            for key in (
                "AMEM_ENABLE",
                "CUDA_VISIBLE_DEVICES",
                "NCCL_DEBUG",
                "NCCL_DEBUG_SUBSYS",
                "NCCL_SOCKET_IFNAME",
                "GLOO_SOCKET_IFNAME",
                "NCCL_CUMEM_ENABLE",
                "NCCL_SHM_DISABLE",
            )
        },
    }

    try:
        import sglang

        payload["sglang_version"] = getattr(sglang, "__version__", None)
    except Exception as exc:
        payload["sglang_import_error"] = f"{type(exc).__name__}: {exc}"
        _emit_argus_diag("sglang_runtime_fingerprint", **payload)
        return

    try:
        from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
        from sglang.srt.managers import io_struct, tokenizer_manager

        tokenizer_cls = tokenizer_manager.TokenizerManager
        update_fn = getattr(tokenizer_cls, "update_weights_from_distributed", None)
        init_fn = getattr(tokenizer_cls, "init_weights_update_group", None)
        payload["tokenizer_manager_has_update_weights_from_distributed"] = callable(update_fn)
        payload["tokenizer_manager_has_init_weights_update_group"] = callable(init_fn)
        payload["pynccl_has_nccl_pause"] = hasattr(PyNcclCommunicator, "nccl_pause")
        payload["pynccl_has_nccl_resume"] = hasattr(PyNcclCommunicator, "nccl_resume")
        payload["io_struct_has_update_from_distributed_req"] = hasattr(
            io_struct,
            "UpdateWeightFromDistributedReqInput",
        )
        payload["io_struct_has_release_memory_req"] = hasattr(
            io_struct,
            "ReleaseMemoryOccupationReqInput",
        )
        payload["io_struct_has_resume_memory_req"] = hasattr(
            io_struct,
            "ResumeMemoryOccupationReqInput",
        )
        if callable(update_fn):
            payload["update_weights_from_distributed_signature"] = str(inspect.signature(update_fn))
            try:
                source = inspect.getsource(update_fn)
                payload["update_weights_from_distributed_mentions_model_update_lock"] = (
                    "model_update_lock" in source
                )
                payload["update_weights_from_distributed_mentions_communicator"] = (
                    "update_weights_from_distributed_communicator" in source
                )
            except Exception as exc:
                payload["update_weights_from_distributed_source_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        if callable(init_fn):
            payload["init_weights_update_group_signature"] = str(inspect.signature(init_fn))
            try:
                source = inspect.getsource(init_fn)
                payload["init_weights_update_group_mentions_communicator"] = (
                    "init_weights_update_group_communicator" in source
                )
            except Exception as exc:
                payload["init_weights_update_group_source_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        payload["sglang_runtime_introspection_error"] = f"{type(exc).__name__}: {exc}"

    _emit_argus_diag("sglang_runtime_fingerprint", **payload)


# Apply patch before importing sglang
_patch_transformers()

# Forward to sglang.launch_server
if __name__ == "__main__":
    import sys

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args

    _log_sglang_runtime_fingerprint()
    _instrument_sglang_runtime_methods()
    server_args = prepare_server_args(sys.argv[1:])
    run_server(server_args)
