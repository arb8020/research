from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rollouts.eval.configs import EndpointConfig, InferenceServerConfig
from rollouts.training.configs import HardwareConfig, InferenceConfig, InferenceWorkerConfig
from rollouts.training.inference_realizations import get_inference_engine_spec
from rollouts.training.weight_sync import InferenceBackend, SGLangEngine, VLLMEngine


@dataclass(frozen=True)
class RealizedEvalEndpoint:
    endpoint_config: EndpointConfig
    engine: InferenceBackend | None = None


def _inference_spec_from_endpoint_provider(provider: str) -> str:
    if provider == "sglang":
        return "slime-sglang"
    if provider == "vllm":
        return "vllm"
    raise ValueError(f"Unsupported worker-backed eval provider {provider!r}")


def _legacy_worker_from_eval_surface(
    *,
    endpoint_config: EndpointConfig,
    server_config: InferenceServerConfig,
) -> InferenceWorkerConfig:
    spec = _inference_spec_from_endpoint_provider(endpoint_config.provider)
    return InferenceWorkerConfig(
        worker_id="eval-endpoint",
        model=endpoint_config.model,
        inference=InferenceConfig(
            spec=spec,
            port=server_config.port,
            cuda_device_ids=tuple(range(server_config.tensor_parallel_size)),
            mem_fraction=server_config.mem_fraction,
            tensor_parallel_size=server_config.tensor_parallel_size,
            startup_timeout=float(server_config.startup_timeout),
        ),
        provider=endpoint_config.provider,
        base_url=endpoint_config.base_url,
    )


def _build_engine(
    *,
    worker: InferenceWorkerConfig,
    output_dir: Path,
) -> InferenceBackend:
    spec = get_inference_engine_spec(worker.inference.spec)
    inference = worker.inference
    if spec.api_format == "sglang":
        return SGLangEngine(
            model_name=worker.model,
            port=inference.port,
            cuda_device_ids=inference.cuda_device_ids,
            output_dir=output_dir,
            mem_fraction=inference.mem_fraction,
            dtype="bfloat16",
            disable_cuda_graph=inference.disable_cuda_graph,
            max_total_tokens=inference.max_total_tokens,
            max_prefill_tokens=inference.max_prefill_tokens,
            max_running_requests=inference.max_running_requests,
            chunked_prefill_size=inference.chunked_prefill_size,
            realization_name=spec.name,
            launch_module=spec.launch_module,
            capability_notes=spec.capability_notes,
            available_sync_realizations=spec.supported_sync_realizations,
            default_sync_realization=spec.default_sync_realization,
        )
    if spec.api_format == "vllm":
        return VLLMEngine(
            model_name=worker.model,
            port=inference.port,
            cuda_device_ids=inference.cuda_device_ids,
            output_dir=output_dir,
            dtype="bfloat16",
            gpu_memory_utilization=inference.mem_fraction,
            realization_name=spec.name,
            launch_module=spec.launch_module,
            capability_notes=spec.capability_notes,
            available_sync_realizations=spec.supported_sync_realizations,
            default_sync_realization=spec.default_sync_realization,
        )
    raise ValueError(
        f"Eval worker-backed endpoint does not support engine spec {spec.name!r} "
        f"(api_format={spec.api_format!r})"
    )


@asynccontextmanager
async def realize_worker_backed_endpoint(
    *,
    endpoint_config: EndpointConfig,
    output_dir: Path,
    hardware_config: HardwareConfig | None,
    server_config: InferenceServerConfig | None,
    worker: InferenceWorkerConfig | None = None,
) -> Any:
    if endpoint_config.base_url is not None or not endpoint_config.requires_server:
        yield RealizedEvalEndpoint(endpoint_config=endpoint_config)
        return

    if hardware_config is None:
        raise ValueError(
            "Auto-realized eval endpoint requires hardware config when base_url is omitted."
        )
    if hardware_config.provider != "local":
        raise NotImplementedError(
            "Worker-backed eval endpoint auto-realization currently supports only "
            "hardware.provider='local'."
        )

    realized_worker = worker
    if realized_worker is None:
        if server_config is None:
            raise ValueError(
                "Auto-realized eval endpoint requires either a worker topology binding or "
                "server config."
            )
        realized_worker = _legacy_worker_from_eval_surface(
            endpoint_config=endpoint_config,
            server_config=server_config,
        )

    engine = _build_engine(worker=realized_worker, output_dir=output_dir)
    engine.launch()
    engine.start_log_tailer()
    try:
        await engine.wait_until_ready(realized_worker.inference.startup_timeout)
        yield RealizedEvalEndpoint(
            endpoint_config=replace(endpoint_config, base_url=engine.api_base),
            engine=engine,
        )
    finally:
        engine.shutdown()
