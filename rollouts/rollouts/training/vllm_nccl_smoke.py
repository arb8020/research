"""Standalone smoke for patched vLLM NCCL weight update plumbing."""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Any

import httpx
import torch
import trio

from rollouts.event_log import emit_run_event
from rollouts.inference.weight_sync import WeightSyncSender
from rollouts.training.weight_sync import VLLMEngine
from rollouts.training.weight_sync_protocol import VLLM_CUSTOM_NCCL_BROADCAST

logger = logging.getLogger(__name__)


def _pick_free_port(preferred: int = 29500) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", preferred))
        return int(sock.getsockname()[1])


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


async def _exercise_generation_traffic(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model_name: str,
    num_requests: int = 8,
) -> None:
    async def _one_request(index: int) -> None:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Reverse this text exactly: smoke-{index}",
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 16,
            },
        )
        response.raise_for_status()

    async with trio.open_nursery() as nursery:
        for index in range(num_requests):
            nursery.start_soon(_one_request, index)


async def run_vllm_nccl_smoke(
    config: Any,
    *,
    exercise_generation_before_sync: bool = False,
    **kwargs: Any,
) -> None:
    run_logger = kwargs.get("run_logger")

    def emit(event: str, **data: Any) -> None:
        if run_logger is not None:
            emit_run_event(run_logger, event, **data)

    output_root = Path(getattr(getattr(config, "output", None), "output_dir", "results"))
    experiment_name = getattr(getattr(config, "output", None), "experiment_name", "vllm_nccl")
    output_dir = output_root / f"{experiment_name}_vllm_nccl_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_assignments = config.inference.gpu_assignments
    ports = config.inference.ports
    trainer_gpu_ids = tuple(getattr(config.trainer, "cuda_device_ids", ()) or ())
    assert gpu_assignments, "inference.gpu_assignments cannot be empty"
    assert ports, "inference.ports cannot be empty"
    assert trainer_gpu_ids, "trainer.cuda_device_ids cannot be empty for NCCL smoke"
    gpus = gpu_assignments[0]
    port = ports[0]
    trainer_device = torch.device(f"cuda:{trainer_gpu_ids[0]}")

    engine = VLLMEngine(
        model_name=config.model.name,
        port=port,
        cuda_device_ids=gpus,
        output_dir=output_dir,
        dtype=config.model.dtype,
        gpu_memory_utilization=config.inference.mem_fraction,
        available_sync_realizations=(VLLM_CUSTOM_NCCL_BROADCAST.name,),
        default_sync_realization=VLLM_CUSTOM_NCCL_BROADCAST.name,
    )

    emit(
        "vllm_nccl_smoke_start",
        model=config.model.name,
        trainer_device=str(trainer_device),
        inference_cuda_device_ids=list(gpus),
        exercise_generation_before_sync=exercise_generation_before_sync,
    )
    logger.info("Launching patched vLLM NCCL smoke server on port %s", port)
    engine.launch()
    engine.start_log_tailer()
    try:
        emit("vllm_nccl_smoke_wait_ready_start", base_url=engine.base_url)
        await engine.wait_until_ready(max_wait=600.0)
        emit("vllm_nccl_smoke_wait_ready_finished", base_url=engine.base_url)
        logger.info("Patched vLLM server is ready")

        async with httpx.AsyncClient(timeout=300.0) as client:
            emit("vllm_nccl_smoke_schema_request_start", base_url=engine.base_url)
            schema_resp = await client.get(
                f"{engine.base_url}/weight_update_schema", params={"limit": 1}
            )
            schema_resp.raise_for_status()
            schema = schema_resp.json()["parameters"]
            assert schema, "Patched vLLM server returned empty weight schema"
            param = schema[0]
            emit(
                "vllm_nccl_smoke_schema_request_finished",
                parameter_name=param["name"],
                parameter_shape=param["shape"],
                parameter_dtype=param["dtype"],
            )
            logger.info(
                "Using smoke tensor target: %s shape=%s dtype=%s",
                param["name"],
                param["shape"],
                param["dtype"],
            )

            master_port = _pick_free_port()
            sender = WeightSyncSender(
                master_addr="127.0.0.1",
                master_port=master_port,
                inference_world_size=1,
                group_name="weight_sync",
                device=trainer_device,
            )

            init_result: dict[str, Any] = {}

            async def _init_group() -> None:
                emit(
                    "vllm_nccl_smoke_group_init_start",
                    master_addr="127.0.0.1",
                    master_port=master_port,
                    world_size=2,
                )
                async with trio.open_nursery() as nursery:

                    async def init_receiver() -> None:
                        resp = await client.post(
                            f"{engine.base_url}/init_weights_update_group",
                            json={
                                "master_address": "127.0.0.1",
                                "master_port": master_port,
                                "rank_offset": 1,
                                "world_size": 2,
                                "group_name": "weight_sync",
                                "timeout_seconds": 300.0,
                            },
                        )
                        resp.raise_for_status()
                        init_result.update(resp.json())

                    nursery.start_soon(init_receiver)
                    await trio.sleep(0.2)
                    await trio.to_thread.run_sync(sender.init_group)

            await _init_group()

            emit("vllm_nccl_smoke_group_init_finished", results=init_result)
            logger.info("Initialized NCCL group: %s", init_result)

            if exercise_generation_before_sync:
                emit("vllm_nccl_smoke_generation_start", requests=8)
                logger.info("Exercising vLLM generation traffic before NCCL sync")
                await _exercise_generation_traffic(
                    client,
                    base_url=engine.base_url,
                    model_name=config.model.name,
                )
                emit("vllm_nccl_smoke_generation_finished", requests=8)

            receive_result: dict[str, Any] = {}

            async with trio.open_nursery() as nursery:

                async def request_receive() -> None:
                    emit(
                        "vllm_nccl_smoke_receive_request_start",
                        parameter_name=param["name"],
                        parameter_shape=param["shape"],
                        parameter_dtype=param["dtype"],
                    )
                    resp = await client.post(
                        f"{engine.base_url}/receive_weight_update",
                        json={
                            "names": [param["name"]],
                            "shapes": [param["shape"]],
                            "dtypes": [param["dtype"]],
                        },
                    )
                    resp.raise_for_status()
                    receive_result.update(resp.json())
                    emit("vllm_nccl_smoke_receive_request_finished", result=receive_result)

                nursery.start_soon(request_receive)
                await trio.sleep(0.2)
                tensor = torch.zeros(
                    tuple(param["shape"]),
                    dtype=_dtype_from_name(param["dtype"]),
                )
                emit(
                    "vllm_nccl_smoke_sender_broadcast_start",
                    parameter_name=param["name"],
                    parameter_shape=param["shape"],
                    parameter_dtype=param["dtype"],
                    trainer_device=str(trainer_device),
                )
                await trio.to_thread.run_sync(sender.broadcast_weights, {param["name"]: tensor})
                emit("vllm_nccl_smoke_sender_broadcast_finished", parameter_name=param["name"])

            emit("vllm_nccl_smoke_sync_finished", result=receive_result)
            logger.info("Received weight update result: %s", receive_result)
            destroy_resp = await client.post(f"{engine.base_url}/destroy_weights_update_group")
            destroy_resp.raise_for_status()
            sender.cleanup()
            emit("vllm_nccl_smoke_finished", status="ok")

    finally:
        engine.shutdown()
