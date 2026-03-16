"""Smoke for one real TorchTitan sync tensor -> patched vLLM NCCL publish."""

from __future__ import annotations

import logging
import socket
import threading
from pathlib import Path
from typing import Any

import httpx
import trio

from rollouts.training.grpo import _run_training_preflight
from rollouts.training.weight_sync import VLLMEngine
from rollouts.training.weight_sync_protocol import (
    VLLM_CUSTOM_NCCL_BROADCAST,
    WeightUpdatePayload,
    WeightWireTensor,
)

logger = logging.getLogger(__name__)


async def run_torchtitan_vllm_real_tensor_smoke(
    config: Any,
    *,
    run_logger: Any | None = None,
    **_: Any,
) -> None:
    def emit(event: str, **data: Any) -> None:
        if run_logger is not None:
            run_logger.event(event, **data)

    output_root = Path(getattr(getattr(config, "output", None), "output_dir", "results"))
    experiment_name = getattr(
        getattr(config, "output", None),
        "experiment_name",
        "torchtitan_real_tensor_nccl_smoke",
    )
    output_dir = output_root / f"{experiment_name}_torchtitan_real_tensor_nccl_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = {
        "output_dir": str(output_dir),
        "model_name": config.model.name,
        "trainer_backend": config.trainer.backend,
        "inference_backend": config.inference.backend,
        "hostname": socket.gethostname(),
    }

    emit("torchtitan_real_tensor_smoke_start", **run_context)

    backend = None
    backend_cleanup = None
    engine = None
    try:
        emit("torchtitan_real_tensor_preflight_start")
        backend, backend_cleanup = await _run_training_preflight(
            config,
            output_dir,
            logger,
            run_context=run_context,
        )
        assert backend is not None
        emit("torchtitan_real_tensor_backend_ready")

        emit("torchtitan_real_tensor_engine_construct_start")
        engine = VLLMEngine(
            model_name=config.model.name,
            port=config.inference.ports[0],
            cuda_device_ids=config.inference.gpu_assignments[0],
            output_dir=output_dir,
            dtype=config.model.dtype,
            gpu_memory_utilization=config.inference.mem_fraction,
            available_sync_realizations=(VLLM_CUSTOM_NCCL_BROADCAST.name,),
            default_sync_realization=VLLM_CUSTOM_NCCL_BROADCAST.name,
        )
        emit("torchtitan_real_tensor_engine_constructed", port=engine.port)
        emit("torchtitan_real_tensor_engine_launch_start", port=engine.port)
        engine.launch()
        engine.start_log_tailer()
        emit("torchtitan_real_tensor_engine_launched", port=engine.port)
        emit("torchtitan_real_tensor_engine_wait_ready_start", port=engine.port)
        await engine.wait_until_ready(max_wait=600.0)
        emit("torchtitan_real_tensor_engine_ready", base_url=engine.base_url)

        init_fn = getattr(backend, "init_nccl_weight_sync", None)
        assert init_fn is not None, "TorchTitan backend does not implement init_nccl_weight_sync()"
        emit("torchtitan_real_tensor_sender_init_start", endpoint=engine.base_url)
        await init_fn(
            [engine.base_url],
            master_addr="127.0.0.1",
            master_port=config.checkpoint.nccl_master_port,
        )
        emit("torchtitan_real_tensor_sender_ready", endpoint=engine.base_url)

        payload_fn = getattr(backend, "_build_inference_weight_update_payload", None)
        assert payload_fn is not None, (
            "TorchTitan backend does not expose inference weight update payload builder"
        )
        payload = payload_fn()
        first_item = next(
            (item for item in payload.tensors if item.wire_name != item.load_name),
            None,
        )
        assert first_item is not None, (
            "TorchTitan real-tensor smoke requires a non-identity wire_name -> load_name mapping"
        )
        first_name = first_item.wire_name
        first_load_name = first_item.load_name
        first_tensor = first_item.tensor
        emit(
            "torchtitan_real_tensor_selected",
            parameter_name=first_name,
            load_name=first_load_name,
            parameter_shape=list(first_tensor.shape),
            parameter_dtype=str(first_tensor.dtype).replace("torch.", ""),
            parameter_device=str(first_tensor.device),
            parameter_stride=list(first_tensor.stride()),
            parameter_is_contiguous=bool(first_tensor.is_contiguous()),
        )
        cloned_tensor = first_tensor.clone()
        materialized_tensor = cloned_tensor.new_empty(cloned_tensor.shape)
        materialized_tensor.copy_(cloned_tensor)
        emit(
            "torchtitan_real_tensor_variants_ready",
            parameter_name=first_name,
            original_data_ptr=int(first_tensor.data_ptr()),
            cloned_data_ptr=int(cloned_tensor.data_ptr()),
            materialized_data_ptr=int(materialized_tensor.data_ptr()),
        )

        sender = getattr(backend, "_nccl_weight_sender", None)
        assert sender is not None, "TorchTitan backend did not initialize NCCL sender"

        async def _attempt_broadcast(label: str, tensor: Any) -> None:
            variant_payload = WeightUpdatePayload(
                tensors=(
                    WeightWireTensor(
                        wire_name=first_name,
                        load_name=first_load_name,
                        shape=tuple(tensor.shape),
                        dtype=str(tensor.dtype).replace("torch.", ""),
                        tensor=tensor,
                        payload_kind=first_item.payload_kind,
                        metadata=dict(first_item.metadata),
                    ),
                ),
                payload_kind=payload.payload_kind,
                version=payload.version,
                metadata=dict(payload.metadata),
            )
            receive_result: dict[str, Any] = {}
            receive_error: dict[str, BaseException] = {}
            receive_done = threading.Event()

            def request_receive_sync() -> None:
                try:
                    with httpx.Client(timeout=httpx.Timeout(300.0, connect=5.0)) as client:
                        response = client.post(
                            f"{engine.base_url}/receive_weight_update",
                            json={
                                "names": [first_name],
                                "load_names": [first_load_name],
                                "shapes": [list(tensor.shape)],
                                "dtypes": [str(tensor.dtype).replace("torch.", "")],
                            },
                        )
                        response.raise_for_status()
                        receive_result["response"] = response.json()
                except BaseException as exc:
                    receive_error["error"] = exc
                finally:
                    receive_done.set()

            emit(
                "torchtitan_real_tensor_receive_request_start",
                parameter_name=first_name,
                tensor_variant=label,
            )
            receive_thread = threading.Thread(
                target=request_receive_sync,
                name=f"torchtitan-real-tensor-receive-{label}",
                daemon=True,
            )
            receive_thread.start()
            await trio.sleep(0.2)
            emit(
                "torchtitan_real_tensor_broadcast_start",
                parameter_name=first_name,
                load_name=first_load_name,
                tensor_variant=label,
                data_ptr=int(tensor.data_ptr()),
            )
            sender.broadcast_payload(variant_payload)
            emit(
                "torchtitan_real_tensor_broadcast_finished",
                parameter_name=first_name,
                load_name=first_load_name,
                tensor_variant=label,
            )
            with trio.move_on_after(300):
                while not receive_done.is_set():
                    await trio.sleep(0.1)
            if not receive_done.is_set():
                raise TimeoutError(
                    f"receive_weight_update did not finish after broadcast for {label}"
                )
            if "error" in receive_error:
                raise receive_error["error"]
            emit(
                "torchtitan_real_tensor_receive_request_finished",
                parameter_name=first_name,
                tensor_variant=label,
                response=receive_result["response"],
            )

        try:
            await _attempt_broadcast("original", first_tensor)
        except Exception as exc:
            emit(
                "torchtitan_real_tensor_original_failed",
                parameter_name=first_name,
                error=repr(exc),
            )
            try:
                await _attempt_broadcast("clone", cloned_tensor)
            except Exception as clone_exc:
                emit(
                    "torchtitan_real_tensor_clone_failed",
                    parameter_name=first_name,
                    error=repr(clone_exc),
                )
                await _attempt_broadcast("materialized_copy", materialized_tensor)

        emit("torchtitan_real_tensor_smoke_finished", status="ok")
    finally:
        if backend is not None:
            cleanup_fn = getattr(backend, "cleanup_nccl_weight_sync", None)
            if cleanup_fn is not None:
                try:
                    await cleanup_fn()
                except Exception:
                    pass
        if backend_cleanup is not None:
            backend_cleanup()
        if engine is not None:
            engine.shutdown()
