"""Standalone smoke for patched vLLM NCCL weight update plumbing."""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Any

import httpx
import torch
import trio

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


async def run_vllm_nccl_smoke(config: Any, **kwargs: Any) -> None:
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

    logger.info("Launching patched vLLM NCCL smoke server on port %s", port)
    engine.launch()
    engine.start_log_tailer()
    try:
        await engine.wait_until_ready(max_wait=600.0)
        logger.info("Patched vLLM server is ready")

        async with httpx.AsyncClient(timeout=300.0) as client:
            schema_resp = await client.get(
                f"{engine.base_url}/weight_update_schema", params={"limit": 1}
            )
            schema_resp.raise_for_status()
            schema = schema_resp.json()["parameters"]
            assert schema, "Patched vLLM server returned empty weight schema"
            param = schema[0]
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

            async with trio.open_nursery() as nursery:
                init_result: dict[str, Any] = {}

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

            logger.info("Initialized NCCL group: %s", init_result)

            receive_result: dict[str, Any] = {}

            async with trio.open_nursery() as nursery:

                async def request_receive() -> None:
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

                nursery.start_soon(request_receive)
                await trio.sleep(0.2)
                tensor = torch.zeros(
                    tuple(param["shape"]),
                    dtype=_dtype_from_name(param["dtype"]),
                )
                await trio.to_thread.run_sync(sender.broadcast_weights, {param["name"]: tensor})

            logger.info("Received weight update result: %s", receive_result)
            destroy_resp = await client.post(f"{engine.base_url}/destroy_weights_update_group")
            destroy_resp.raise_for_status()
            sender.cleanup()

    finally:
        engine.shutdown()
