from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rollouts.eval.configs import EndpointConfig, InferenceServerConfig
from rollouts.remote_runtime import (
    SourceSyncPolicy,
    materialization_plan_from_runtime,
    runtime_contract_from_hardware,
)
from rollouts.training.configs import HardwareConfig, InferenceConfig, InferenceWorkerConfig
from rollouts.training.inference_realizations import get_inference_engine_spec
from rollouts.training.weight_sync import InferenceBackend, SGLangEngine, VLLMEngine

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RealizedEvalEndpoint:
    endpoint_config: EndpointConfig
    engine: InferenceBackend | None = None
    metadata: dict[str, Any] | None = None


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


def _remote_service_spec(
    *,
    worker: InferenceWorkerConfig,
    output_dir: Path,
) -> tuple[str, str]:
    engine = _build_engine(worker=worker, output_dir=output_dir)
    return engine.build_launch_cmd(), engine.health_url.removeprefix("http://localhost")


async def _wait_for_modal_tunnel(
    *,
    sandbox: Any,
    port: int,
    timeout_s: float,
) -> Any:
    import trio

    with trio.fail_after(timeout_s):
        while True:
            tunnels = await trio.to_thread.run_sync(lambda: sandbox.tunnels(timeout=5))
            tunnel = tunnels.get(port)
            if tunnel is not None:
                return tunnel
            await trio.sleep(1.0)


@asynccontextmanager
async def _realize_modal_endpoint(
    *,
    endpoint_config: EndpointConfig,
    output_dir: Path,
    hardware_config: HardwareConfig,
    worker: InferenceWorkerConfig,
    run_name: str,
    force_deploy_committed: bool,
) -> Any:
    from bifrost.modal_backend import (
        ModalExecutionRequest,
        ModalExecutionSession,
        create_modal_sandbox,
        terminate_modal_sandbox,
    )
    from bifrost.types import ProcessSpec, ReadinessProbe, ServiceSpec, WorkspaceMaterializationSpec
    import trio_asyncio

    runtime = runtime_contract_from_hardware(hardware_config)
    request = ModalExecutionRequest(
        config_path="eval-worker-endpoint",
        runtime=runtime,
        materialization=materialization_plan_from_runtime(
            runtime,
            workspace_root="~/.bifrost/workspaces/rollouts-eval",
        ),
        source_sync_policy=SourceSyncPolicy.committed_only(
            dirty_action="warn" if force_deploy_committed else "fail"
        ),
        keep_alive=False,
        cleanup_scope="run",
        run_name=run_name,
        tags={
            "control_plane": "argus",
            "launcher_id": run_name,
            "config_basename": run_name,
            "provider": "modal",
        },
    )
    async with trio_asyncio.open_loop():
        sandbox_handle = await create_modal_sandbox(request)
        session = ModalExecutionSession(sandbox_handle=sandbox_handle, local_root=REPO_ROOT)
        try:
            workspace = await session.materialize(
                WorkspaceMaterializationSpec(
                    requested_root=getattr(request.materialization, "workspace_root", None)
                )
            )
            remote_output_dir = Path(workspace.root) / "results" / "eval" / run_name
            launch_cmd, readiness_target = _remote_service_spec(
                worker=worker,
                output_dir=remote_output_dir,
            )
            service = await session.serve_service(
                ServiceSpec(
                    process=ProcessSpec(
                        command="bash",
                        args=("-lc", launch_cmd),
                        cwd=workspace.root,
                    ),
                    port=worker.inference.port,
                    readiness_probe=ReadinessProbe(kind="http", target=readiness_target),
                ),
                name=f"eval-endpoint-{run_name}",
                workspace=workspace,
                log_file=f"{remote_output_dir}/endpoint_service",
            )
            healthy = await service.wait_until_healthy(timeout=worker.inference.startup_timeout)
            if not healthy:
                logs = await service.logs(tail=80)
                raise RuntimeError(
                    "Modal eval endpoint failed to become healthy.\n"
                    f"Recent service logs:\n{logs}"
                )
            await session.start_process(
                ProcessSpec(
                    command="python",
                    args=(
                        "-m",
                        "rollouts.eval.modal_forwarder",
                        "--port",
                        str(worker.inference.port),
                    ),
                    cwd=workspace.root,
                ),
                name=f"eval-forwarder-{run_name}",
                timeout=86400,
                start_timeout_s=30.0,
            )
            tunnel = await _wait_for_modal_tunnel(
                sandbox=sandbox_handle.sandbox,
                port=worker.inference.port,
                timeout_s=60.0,
            )
            yield RealizedEvalEndpoint(
                endpoint_config=replace(endpoint_config, base_url=f"{tunnel.url.rstrip('/')}/v1"),
                metadata={
                    "provider": "modal",
                    "sandbox_id": sandbox_handle.sandbox_id,
                },
            )
        finally:
            await terminate_modal_sandbox(sandbox_handle)


@asynccontextmanager
async def realize_worker_backed_endpoint(
    *,
    endpoint_config: EndpointConfig,
    output_dir: Path,
    hardware_config: HardwareConfig | None,
    server_config: InferenceServerConfig | None,
    worker: InferenceWorkerConfig | None = None,
    run_name: str = "eval-endpoint",
    force_deploy_committed: bool = False,
) -> Any:
    if endpoint_config.base_url is not None or not endpoint_config.requires_server:
        yield RealizedEvalEndpoint(endpoint_config=endpoint_config)
        return

    if hardware_config is None:
        raise ValueError(
            "Auto-realized eval endpoint requires hardware config when base_url is omitted."
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

    if hardware_config.provider == "modal":
        async with _realize_modal_endpoint(
            endpoint_config=endpoint_config,
            output_dir=output_dir,
            hardware_config=hardware_config,
            worker=realized_worker,
            run_name=run_name,
            force_deploy_committed=force_deploy_committed,
        ) as realized:
            yield realized
        return
    if hardware_config.provider != "local":
        raise NotImplementedError(
            "Worker-backed eval endpoint auto-realization currently supports only "
            "hardware.provider in {'local', 'modal'}."
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
