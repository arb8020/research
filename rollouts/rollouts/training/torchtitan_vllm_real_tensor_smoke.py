"""Smoke for one real TorchTitan sync tensor -> patched vLLM NCCL publish."""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Any

import httpx
import trio

from rollouts.training.grpo import _run_training_preflight
from rollouts.training.weight_sync import VLLMEngine
from rollouts.training.weight_sync_protocol import VLLM_CUSTOM_NCCL_BROADCAST

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
        backend, backend_cleanup = await _run_training_preflight(
            config,
            output_dir,
            logger,
            run_context=run_context,
        )
        assert backend is not None
        emit("torchtitan_real_tensor_backend_ready")

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
        engine.launch()
        engine.start_log_tailer()
        emit("torchtitan_real_tensor_engine_launch", port=engine.port)
        await engine.wait_until_ready(max_wait=600.0)
        emit("torchtitan_real_tensor_engine_ready", base_url=engine.base_url)

        init_fn = getattr(backend, "init_nccl_weight_sync", None)
        assert init_fn is not None, "TorchTitan backend does not implement init_nccl_weight_sync()"
        await init_fn(
            [engine.base_url],
            master_addr="127.0.0.1",
            master_port=config.checkpoint.nccl_master_port,
        )
        emit("torchtitan_real_tensor_sender_ready", endpoint=engine.base_url)

        state_dict_fn = getattr(backend, "_build_hf_state_dict_for_inference_sync", None)
        assert state_dict_fn is not None, (
            "TorchTitan backend does not expose HF sync state dict builder"
        )
        hf_state_dict = state_dict_fn()
        first_name, first_tensor = next(iter(hf_state_dict.items()))
        emit(
            "torchtitan_real_tensor_selected",
            parameter_name=first_name,
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

        async with httpx.AsyncClient(timeout=300.0) as client:

            async def _attempt_broadcast(label: str, tensor: Any) -> None:
                async with trio.open_nursery() as nursery:

                    async def request_receive() -> None:
                        emit(
                            "torchtitan_real_tensor_receive_request_start",
                            parameter_name=first_name,
                            tensor_variant=label,
                        )
                        response = await client.post(
                            f"{engine.base_url}/receive_weight_update",
                            json={
                                "names": [first_name],
                                "shapes": [list(tensor.shape)],
                                "dtypes": [str(tensor.dtype).replace("torch.", "")],
                            },
                        )
                        response.raise_for_status()
                        emit(
                            "torchtitan_real_tensor_receive_request_finished",
                            parameter_name=first_name,
                            tensor_variant=label,
                            response=response.json(),
                        )

                    nursery.start_soon(request_receive)
                    await trio.sleep(0.2)
                    emit(
                        "torchtitan_real_tensor_broadcast_start",
                        parameter_name=first_name,
                        tensor_variant=label,
                        data_ptr=int(tensor.data_ptr()),
                    )
                    await trio.to_thread.run_sync(sender.broadcast_weights, {first_name: tensor})
                    emit(
                        "torchtitan_real_tensor_broadcast_finished",
                        parameter_name=first_name,
                        tensor_variant=label,
                    )

            try:
                await _attempt_broadcast("original", first_tensor)
            except Exception as exc:
                emit(
                    "torchtitan_real_tensor_original_failed",
                    parameter_name=first_name,
                    error=repr(exc),
                )
                await _attempt_broadcast("clone", cloned_tensor)
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
