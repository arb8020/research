from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import trio

from rollouts.eval.configs import EndpointConfig, InferenceServerConfig
from rollouts.remote_runtime import (
    SourceSyncPolicy,
    materialization_plan_from_runtime,
    runtime_contract_from_hardware,
)
from rollouts.training.configs import HardwareConfig, InferenceConfig, InferenceWorkerConfig
from rollouts.training.inference_realizations import get_inference_engine_spec
from rollouts.training.weight_sync import (
    InferenceBackend,
    SGLangEngine,
    VLLMEngine,
    _classify_sglang_startup_phase,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REMOTE_VENV_PYTHON = "/opt/venvs/rollouts/bin/python"
STARTUP_STALL_DIAGNOSTIC_INTERVAL = 15


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
    remote_python: str,
) -> tuple[str, str]:
    original_python = os.environ.get("ROLLOUTS_INFERENCE_PYTHON")
    os.environ["ROLLOUTS_INFERENCE_PYTHON"] = remote_python
    try:
        engine = _build_engine(worker=worker, output_dir=output_dir)
        parsed_health_url = urlsplit(engine.health_url)
        readiness_target = parsed_health_url.path or "/"
        if parsed_health_url.query:
            readiness_target = f"{readiness_target}?{parsed_health_url.query}"
        if parsed_health_url.fragment:
            readiness_target = f"{readiness_target}#{parsed_health_url.fragment}"
        return engine.build_launch_cmd(), readiness_target
    finally:
        if original_python is None:
            os.environ.pop("ROLLOUTS_INFERENCE_PYTHON", None)
        else:
            os.environ["ROLLOUTS_INFERENCE_PYTHON"] = original_python


def _remote_inference_python(hardware_config: HardwareConfig) -> str:
    deps = hardware_config.deps
    if deps is None:
        return REMOTE_VENV_PYTHON
    image = deps.resolved_image(hardware_config.gpu_type)
    if image.python_runtime == "image_owned":
        return image.python_executable
    return REMOTE_VENV_PYTHON


def _startup_log_context(
    *,
    worker: InferenceWorkerConfig,
    sandbox_id: str,
    service_name: str,
    remote_output_dir: Path,
) -> dict[str, Any]:
    return {
        "provider": "modal",
        "sandbox_id": sandbox_id,
        "service_name": service_name,
        "engine_name": worker.inference.spec,
        "engine_port": worker.inference.port,
        "engine_cuda_device_ids": list(worker.inference.cuda_device_ids),
        "model_name": worker.model,
        "engine_log_path": f"{remote_output_dir}/endpoint_service",
        "engine_trace_path": str(remote_output_dir / f"sglang_{worker.inference.port}_trace.jsonl"),
    }


def _parse_service_logs(log_blob: str) -> dict[str, list[str]]:
    streams: dict[str, list[str]] = {"stdout": [], "stderr": []}
    current: str | None = None
    for raw_line in log_blob.splitlines():
        line = raw_line.rstrip()
        if line == "== stdout ==":
            current = "stdout"
            continue
        if line == "== stderr ==":
            current = "stderr"
            continue
        if current is not None and line:
            streams[current].append(line)
    return streams


def _tail_lines(path: Path, max_lines: int = 40) -> str:
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


async def _tail_remote_trace(
    *,
    session: Any,
    trace_path: Path,
    max_lines: int = 40,
) -> str:
    try:
        result = await session.exec(
            f"tail -n {max_lines} {trace_path} 2>/dev/null || true",
        )
    except Exception as exc:
        return f"<failed to read remote trace: {type(exc).__name__}: {exc}>"
    return result.stdout.strip()


def _emit_log_lines(
    *,
    run_logger: Any | None,
    log_blob: str,
    seen_lines: dict[str, set[str]],
    emitted_startup_phases: set[str],
    startup_context: dict[str, Any],
) -> None:
    if run_logger is None:
        return
    parsed = _parse_service_logs(log_blob)
    for stream_name, lines in parsed.items():
        stream_seen = seen_lines.setdefault(stream_name, set())
        for line in lines:
            line_key = f"{stream_name}:{line}"
            if line_key in stream_seen:
                continue
            stream_seen.add(line_key)
            run_logger.event(
                "eval_inference_service_log",
                log_stream=stream_name,
                line=line,
                **startup_context,
            )
            phase = _classify_sglang_startup_phase(line)
            if phase is None:
                continue
            phase_name, _phase_fields = phase
            if phase_name in emitted_startup_phases:
                continue
            emitted_startup_phases.add(phase_name)
            run_logger.event(
                "inference_startup_phase",
                phase=phase_name,
                phase_source=f"service_{stream_name}",
                phase_line=line,
                **startup_context,
            )


async def _service_logs_best_effort(service: Any, *, tail: int = 120) -> str:
    try:
        return await service.logs(tail=tail)
    except Exception as exc:
        return f"== stdout ==\n== stderr ==\n<failed to read service logs: {type(exc).__name__}: {exc}>"


async def _wait_for_modal_service_ready(
    *,
    service: Any,
    session: Any,
    sandbox: Any,
    worker: InferenceWorkerConfig,
    startup_timeout: float,
    run_logger: Any | None,
    startup_context: dict[str, Any],
    remote_output_dir: Path,
) -> bool:
    started = time.monotonic()
    last_health_state: str | None = None
    last_health_detail: str | None = None
    emitted_startup_phases: set[str] = set()
    seen_lines: dict[str, set[str]] = {"stdout": set(), "stderr": set()}

    if run_logger is not None:
        run_logger.event(
            "inference_healthcheck_start",
            startup_timeout=startup_timeout,
            **startup_context,
        )

    async def _emit_health_state(
        state: str,
        *,
        attempt: int,
        detail: str | None = None,
        error: str | None = None,
    ) -> None:
        nonlocal last_health_state, last_health_detail
        last_health_detail = detail
        if last_health_state == state:
            return
        last_health_state = state
        if run_logger is not None:
            run_logger.event(
                "inference_health_state",
                health_state=state,
                health_attempt=attempt,
                health_error=error,
                health_detail=detail,
                **startup_context,
            )

    attempt = 0
    while time.monotonic() - started < startup_timeout:
        log_blob = await _service_logs_best_effort(service, tail=120)
        _emit_log_lines(
            run_logger=run_logger,
            log_blob=log_blob,
            seen_lines=seen_lines,
            emitted_startup_phases=emitted_startup_phases,
            startup_context=startup_context,
        )

        try:
            healthy = await service.is_healthy()
        except Exception as exc:
            healthy = False
            await _emit_health_state(
                "healthcheck_exception",
                attempt=attempt,
                error=type(exc).__name__,
                detail=str(exc),
            )
            try:
                sandbox_state = await trio.to_thread.run_sync(lambda: sandbox.poll())
            except Exception as sandbox_exc:
                trace_tail = await _tail_remote_trace(
                    session=session,
                    trace_path=remote_output_dir / f"sglang_{worker.inference.port}_trace.jsonl",
                )
                if run_logger is not None:
                    run_logger.event(
                        "inference_startup_failed",
                        failure_kind="sandbox_unavailable",
                        health_attempt=attempt,
                        last_health_state=last_health_state,
                        last_health_detail=last_health_detail,
                        log_tail=log_blob,
                        trace_tail=trace_tail,
                        sandbox_error=f"{type(sandbox_exc).__name__}: {sandbox_exc}",
                        **startup_context,
                    )
                raise RuntimeError(
                    "Modal eval endpoint sandbox disappeared during startup.\n"
                    f"Recent service logs:\n{log_blob}\n"
                    f"Sandbox error: {type(sandbox_exc).__name__}: {sandbox_exc}"
                ) from exc
            if sandbox_state is not None:
                trace_tail = await _tail_remote_trace(
                    session=session,
                    trace_path=remote_output_dir / f"sglang_{worker.inference.port}_trace.jsonl",
                )
                if run_logger is not None:
                    run_logger.event(
                        "inference_startup_failed",
                        failure_kind="sandbox_exited",
                        health_attempt=attempt,
                        last_health_state=last_health_state,
                        last_health_detail=last_health_detail,
                        log_tail=log_blob,
                        trace_tail=trace_tail,
                        sandbox_returncode=sandbox_state,
                        **startup_context,
                    )
                raise RuntimeError(
                    "Modal eval endpoint sandbox exited during startup.\n"
                    f"Recent service logs:\n{log_blob}\n"
                    f"Sandbox return code: {sandbox_state}"
                ) from exc
        else:
            if healthy:
                await _emit_health_state("healthy", attempt=attempt)
                if run_logger is not None:
                    run_logger.event("inference_ready", **startup_context)
                return True

            try:
                running = await service.is_running()
            except Exception as exc:
                running = False
                await _emit_health_state(
                    "running_check_exception",
                    attempt=attempt,
                    error=type(exc).__name__,
                    detail=str(exc),
                )
            else:
                if not running:
                    await _emit_health_state("service_exited_before_ready", attempt=attempt)
                    trace_tail = await _tail_remote_trace(
                        session=session,
                        trace_path=remote_output_dir
                        / f"sglang_{worker.inference.port}_trace.jsonl",
                    )
                    if run_logger is not None:
                        run_logger.event(
                            "inference_startup_failed",
                            failure_kind="service_exited_before_ready",
                            health_attempt=attempt,
                            last_health_state=last_health_state,
                            last_health_detail=last_health_detail,
                            log_tail=log_blob,
                            trace_tail=trace_tail,
                            **startup_context,
                        )
                    raise RuntimeError(
                        "Modal eval endpoint service exited before becoming healthy.\n"
                        f"Recent service logs:\n{log_blob}"
                    )
                await _emit_health_state("transport_pending", attempt=attempt)

        if (
            run_logger is not None
            and attempt > 0
            and attempt % STARTUP_STALL_DIAGNOSTIC_INTERVAL == 0
        ):
            trace_tail = await _tail_remote_trace(
                session=session,
                trace_path=remote_output_dir / f"sglang_{worker.inference.port}_trace.jsonl",
                max_lines=20,
            )
            run_logger.event(
                "inference_health_stall",
                health_attempt=attempt,
                last_health_state=last_health_state,
                last_health_detail=last_health_detail,
                log_tail=log_blob,
                trace_tail=trace_tail,
                **startup_context,
            )
        attempt += 1
        await trio.sleep(1.0)

    log_blob = await _service_logs_best_effort(service, tail=120)
    trace_tail = await _tail_remote_trace(
        session=session,
        trace_path=remote_output_dir / f"sglang_{worker.inference.port}_trace.jsonl",
    )
    if run_logger is not None:
        run_logger.event(
            "inference_startup_failed",
            failure_kind="timeout",
            health_attempt=attempt,
            last_health_state=last_health_state,
            last_health_detail=last_health_detail,
            log_tail=log_blob,
            trace_tail=trace_tail,
            **startup_context,
        )
    raise RuntimeError(
        f"Modal eval endpoint failed to become healthy within {startup_timeout}s.\n"
        f"Recent service logs:\n{log_blob}\n"
        f"Recent trace logs:\n{trace_tail}"
    )


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
    run_logger: Any | None,
) -> Any:
    import modal
    import trio_asyncio

    from bifrost.modal_backend import (
        ModalExecutionRequest,
        ModalExecutionSession,
        create_modal_sandbox,
        terminate_modal_sandbox,
    )
    from bifrost.types import ProcessSpec, ReadinessProbe, ServiceSpec, WorkspaceMaterializationSpec

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
        run_logger=run_logger,
    )
    with modal.enable_output():
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
                remote_python = _remote_inference_python(hardware_config)
                launch_cmd, readiness_target = _remote_service_spec(
                    worker=worker,
                    output_dir=remote_output_dir,
                    remote_python=remote_python,
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
                startup_context = _startup_log_context(
                    worker=worker,
                    sandbox_id=sandbox_handle.sandbox_id,
                    service_name=f"eval-endpoint-{run_name}",
                    remote_output_dir=remote_output_dir,
                )
                if run_logger is not None:
                    run_logger.event(
                        "inference_engine_launch",
                        engine_launch_cmd=launch_cmd,
                        readiness_target=readiness_target,
                        **startup_context,
                    )
                await _wait_for_modal_service_ready(
                    service=service,
                    session=session,
                    sandbox=sandbox_handle.sandbox,
                    worker=worker,
                    startup_timeout=worker.inference.startup_timeout,
                    run_logger=run_logger,
                    startup_context=startup_context,
                    remote_output_dir=remote_output_dir,
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
                    endpoint_config=replace(
                        endpoint_config, base_url=f"{tunnel.url.rstrip('/')}/v1"
                    ),
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
    run_logger: Any | None = None,
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
            run_logger=run_logger,
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
