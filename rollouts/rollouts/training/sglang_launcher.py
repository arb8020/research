"""SGLang launcher with transformers compatibility patches.

Patches transformers.utils.hub.list_repo_templates to handle 404s gracefully.
See: https://github.com/huggingface/transformers/issues/41813

Usage:
    python -m rollouts.training.sglang_launcher --model-path ... --port ...
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import os
import sys
import threading
import traceback


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
    try:
        sys.stderr.write(f"__ARGUS_DIAG__{json.dumps({'event': event, **data}, sort_keys=True)}\n")
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


def _ensure_owner_instrumented(owner: object) -> None:
    writer_lock = getattr(getattr(owner, "model_update_lock", None), "writer_lock", None)
    if writer_lock is not None:
        _instrument_async_context_type(type(writer_lock), label="model_update_writer_lock")


def _wrap_runtime_method(owner_cls: type[object], method_name: str) -> None:
    method = getattr(owner_cls, method_name, None)
    if not callable(method) or getattr(method, "__rollouts_argus_wrapped__", False):
        return

    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def wrapped(self: object, *args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
            _ensure_owner_instrumented(self)
            _emit_argus_diag(
                "sglang_runtime_method_enter",
                owner_class=owner_cls.__name__,
                method=method_name,
                process=_process_context(),
                owner_state=_method_owner_state(self),
                args=_jsonable_summary(args),
                kwargs=_jsonable_summary(kwargs),
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
                raise
            _emit_argus_diag(
                "sglang_runtime_method_ok",
                owner_class=owner_cls.__name__,
                method=method_name,
                process=_process_context(),
                owner_state=_method_owner_state(self),
                result=_jsonable_summary(result),
            )
            return result

    else:

        @functools.wraps(method)
        def wrapped(self: object, *args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
            _ensure_owner_instrumented(self)
            _emit_argus_diag(
                "sglang_runtime_method_enter",
                owner_class=owner_cls.__name__,
                method=method_name,
                process=_process_context(),
                owner_state=_method_owner_state(self),
                args=_jsonable_summary(args),
                kwargs=_jsonable_summary(kwargs),
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
                raise
            _emit_argus_diag(
                "sglang_runtime_method_ok",
                owner_class=owner_cls.__name__,
                method=method_name,
                process=_process_context(),
                owner_state=_method_owner_state(self),
                result=_jsonable_summary(result),
            )
            return result

    wrapped.__rollouts_argus_wrapped__ = True  # type: ignore[attr-defined]
    setattr(owner_cls, method_name, wrapped)


def _instrument_sglang_runtime_methods() -> None:
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
                "post_process_weights",
            ),
        ),
        (
            BaseTpWorker,
            (
                "init_weights_update_group",
                "update_weights_from_distributed",
                "post_process_weights",
            ),
        ),
    ]
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
