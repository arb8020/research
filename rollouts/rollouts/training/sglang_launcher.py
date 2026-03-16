"""SGLang launcher with transformers compatibility patches.

Patches transformers.utils.hub.list_repo_templates to handle 404s gracefully.
See: https://github.com/huggingface/transformers/issues/41813

Usage:
    python -m rollouts.training.sglang_launcher --model-path ... --port ...
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys


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
    server_args = prepare_server_args(sys.argv[1:])
    run_server(server_args)
