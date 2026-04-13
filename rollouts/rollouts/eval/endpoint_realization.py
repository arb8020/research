from __future__ import annotations

import logging
import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_logger = logging.getLogger(__name__)

import trio

from rollouts.eval.configs import (
    EndpointCapabilities,
    EndpointConfig,
    ExternalEndpoint,
    InferenceEndpoint,
    InferenceServerConfig,
    OwnedEndpoint,
)
from rollouts.remote_runtime import (
    SourceSyncPolicy,
    materialization_plan_from_runtime,
    runtime_contract_from_hardware,
)
from rollouts.training.configs import HardwareConfig, InferenceConfig, InferenceWorkerConfig
from rollouts.training.inference_realizations import get_inference_engine_spec
from rollouts.training.inference_runtime_factory import build_owned_endpoint
from rollouts.training.weight_sync import InferenceBackend, _classify_sglang_startup_phase

REPO_ROOT = Path(__file__).resolve().parents[3]
REMOTE_VENV_PYTHON = "/opt/venvs/rollouts/bin/python"
STARTUP_STALL_DIAGNOSTIC_INTERVAL = 15
MODAL_SANDBOX_CLEANUP_TIMEOUT_S = 15.0
MODAL_SANDBOX_CLEANUP_POLL_INTERVAL_S = 5.0
MODAL_SANDBOX_FORCE_TERMINATE_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class RealizedEvalEndpoint:
    endpoint_config: ExternalEndpoint | EndpointConfig
    engine: InferenceBackend | OwnedEndpoint | None = None
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
) -> OwnedEndpoint:
    spec = get_inference_engine_spec(worker.inference.spec)
    inference = worker.inference
    if spec.api_format not in ("sglang", "vllm"):
        raise ValueError(
            f"Eval worker-backed endpoint does not support engine spec {spec.name!r} "
            f"(api_format={spec.api_format!r})"
        )
    return build_owned_endpoint(
        spec=spec,
        model=worker.model,
        cuda_device_ids=inference.cuda_device_ids,
        port=inference.port,
        output_dir=output_dir,
        capabilities=EndpointCapabilities(weight_sync=None),
        dtype="bfloat16",
        mem_fraction=inference.mem_fraction,
        disable_cuda_graph=inference.disable_cuda_graph,
        max_total_tokens=inference.max_total_tokens,
        max_prefill_tokens=inference.max_prefill_tokens,
        max_running_requests=inference.max_running_requests,
        chunked_prefill_size=inference.chunked_prefill_size,
        startup_timeout=inference.startup_timeout,
    )


def _worker_from_owned_endpoint(endpoint_config: OwnedEndpoint) -> InferenceWorkerConfig:
    return InferenceWorkerConfig(
        worker_id="eval-endpoint",
        model=endpoint_config.model,
        inference=InferenceConfig(
            spec=endpoint_config.spec,
            port=endpoint_config.port,
            cuda_device_ids=endpoint_config.cuda_device_ids,
            mem_fraction=endpoint_config.mem_fraction,
            tensor_parallel_size=len(endpoint_config.cuda_device_ids),
            startup_timeout=endpoint_config.startup_timeout,
        ),
        provider=endpoint_config.provider,
    )


def _externalize_owned_endpoint(endpoint_config: OwnedEndpoint, url: str) -> ExternalEndpoint:
    return ExternalEndpoint(
        url=url,
        model=endpoint_config.model,
        provider=endpoint_config.provider,
        temperature=endpoint_config.temperature,
        max_tokens=endpoint_config.max_tokens,
        extra_params=endpoint_config.extra_params,
    )


def _remote_service_spec(
    *,
    worker: InferenceWorkerConfig,
    output_dir: Path,
    remote_python: str,
    owned_endpoint: OwnedEndpoint | None = None,
) -> tuple[str, str]:
    # OwnedEndpoint with launch_module bypasses _build_engine entirely -
    # the launch cmd is derived from the module path, which works remotely
    # because the repo is synced to the Modal sandbox by bifrost.
    if owned_endpoint is not None and owned_endpoint.launch_module is not None:
        # Prepend workspace to PYTHONPATH so the synced rollouts source tree
        # takes precedence over any installed package version in the venv.
        # Must export inside the bash -lc so it survives the login shell env reset.
        launch_cmd = (
            f"export PYTHONPATH=/workspace/research/rollouts:${{PYTHONPATH:-}}; "
            f"{remote_python} -m {owned_endpoint.launch_module} "
            f"--model {owned_endpoint.model} "
            f"--port {owned_endpoint.port}"
        )
        return launch_cmd, owned_endpoint.readiness_path

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


async def _list_modal_sandbox_ids() -> set[str]:
    import modal

    return await trio.to_thread.run_sync(
        lambda: {sandbox.object_id for sandbox in modal.Sandbox.list()},
    )


async def _terminate_modal_sandbox_ids(sandbox_ids: list[str]) -> list[str]:
    import modal

    def _terminate_sync() -> list[str]:
        terminated: list[str] = []
        for sandbox_id in sandbox_ids:
            try:
                modal.Sandbox.from_id(sandbox_id).terminate()
            except Exception:
                continue
            terminated.append(sandbox_id)
        return terminated

    return await trio.to_thread.run_sync(_terminate_sync)


async def _wait_for_modal_sandbox_baseline(
    *,
    baseline_ids: set[str],
    run_logger: Any | None,
    run_name: str,
    timeout_s: float = MODAL_SANDBOX_CLEANUP_TIMEOUT_S,
    force_terminate_timeout_s: float = MODAL_SANDBOX_FORCE_TERMINATE_TIMEOUT_S,
) -> None:
    deadline = trio.current_time() + timeout_s
    while True:
        current_ids = await _list_modal_sandbox_ids()
        residual_ids = sorted(current_ids - baseline_ids)
        if not residual_ids:
            if run_logger is not None:
                run_logger.event(
                    "modal_sandbox_cleanup_converged",
                    provider="modal",
                    run_name=run_name,
                )
            return
        if trio.current_time() >= deadline:
            if run_logger is not None:
                run_logger.event(
                    "modal_sandbox_cleanup_force_terminate_start",
                    provider="modal",
                    run_name=run_name,
                    residual_sandbox_ids=residual_ids,
                )
            terminated_ids = await _terminate_modal_sandbox_ids(residual_ids)
            force_deadline = trio.current_time() + force_terminate_timeout_s
            while True:
                current_ids = await _list_modal_sandbox_ids()
                residual_ids = sorted(current_ids - baseline_ids)
                if not residual_ids:
                    if run_logger is not None:
                        run_logger.event(
                            "modal_sandbox_cleanup_force_terminate_finished",
                            provider="modal",
                            run_name=run_name,
                            terminated_sandbox_ids=terminated_ids,
                        )
                        run_logger.event(
                            "modal_sandbox_cleanup_converged",
                            provider="modal",
                            run_name=run_name,
                            cleanup_mode="force_terminate",
                        )
                    return
                if trio.current_time() >= force_deadline:
                    break
                await trio.sleep(MODAL_SANDBOX_CLEANUP_POLL_INTERVAL_S)
            if run_logger is not None:
                run_logger.event(
                    "modal_sandbox_cleanup_incomplete",
                    provider="modal",
                    run_name=run_name,
                    residual_sandbox_ids=residual_ids,
                    timeout_s=timeout_s,
                    terminated_sandbox_ids=terminated_ids,
                )
            return
        await trio.sleep(MODAL_SANDBOX_CLEANUP_POLL_INTERVAL_S)


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


def _pick_free_local_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


async def _wait_for_local_port_ready(*, port: int, timeout_s: float) -> None:
    deadline = trio.current_time() + timeout_s
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError as exc:
            if trio.current_time() >= deadline:
                raise RuntimeError(
                    f"SSH tunnel local port {port} did not become ready"
                ) from exc
            await trio.sleep(0.1)


async def _wait_for_forwarded_health(
    *,
    local_port: int,
    readiness_target: str,
    timeout_s: float,
) -> None:
    import urllib.error
    import urllib.request

    target = readiness_target if readiness_target.startswith("/") else "/health"
    url = f"http://127.0.0.1:{local_port}{target}"
    deadline = trio.current_time() + timeout_s
    while True:
        try:
            await trio.to_thread.run_sync(
                lambda: urllib.request.urlopen(url, timeout=1.0).read(),
            )
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if trio.current_time() >= deadline:
                raise RuntimeError(
                    f"Forwarded SSH endpoint did not become healthy at {url} within {timeout_s}s"
                ) from exc
            await trio.sleep(0.2)


@asynccontextmanager
async def _forward_ssh_port(
    *,
    ssh_target: str,
    ssh_key_path: str,
    remote_port: int,
    local_port: int | None = None,
) -> Any:
    import getpass

    import paramiko

    ssh_key_path = os.path.expanduser(ssh_key_path)
    local_port = _pick_free_local_port() if local_port is None else local_port

    ssh_target_parts = ssh_target.split("@", 1)
    if len(ssh_target_parts) == 2:
        username, host_port = ssh_target_parts
    else:
        username, host_port = getpass.getuser(), ssh_target_parts[0]
    host, _, port_text = host_port.partition(":")
    ssh_port = int(port_text) if port_text else 22

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", local_port))
    server.listen(4)
    server.settimeout(0.5)

    stop_event = threading.Event()

    def _forward(src: socket.socket, dst: socket.socket) -> None:
        try:
            while not stop_event.is_set():
                data = src.recv(4096)
                if not data:
                    break
                dst.sendall(data)
        except (OSError, EOFError):
            pass
        finally:
            try:
                src.close()
            except (OSError, EOFError):
                pass
            try:
                dst.close()
            except (OSError, EOFError):
                pass

    def _accept_loop() -> None:
        transport = ssh_client.get_transport()
        assert transport is not None, "SSH transport unavailable for local port forward"
        while not stop_event.is_set():
            try:
                client_sock, addr = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            try:
                channel = transport.open_channel(
                    "direct-tcpip",
                    ("127.0.0.1", remote_port),
                    addr,
                )
            except Exception:
                client_sock.close()
                if stop_event.is_set():
                    break
                continue
            threading.Thread(target=_forward, args=(client_sock, channel), daemon=True).start()
            threading.Thread(target=_forward, args=(channel, client_sock), daemon=True).start()

    await trio.to_thread.run_sync(
        lambda: ssh_client.connect(
            hostname=host,
            port=ssh_port,
            username=username,
            key_filename=ssh_key_path,
            timeout=30,
        )
    )
    tunnel_thread = threading.Thread(target=_accept_loop, daemon=True)
    tunnel_thread.start()
    try:
        await _wait_for_local_port_ready(port=local_port, timeout_s=5.0)
        yield local_port
    finally:
        stop_event.set()
        try:
            server.close()
        except OSError:
            pass
        await trio.to_thread.run_sync(ssh_client.close)
        tunnel_thread.join(timeout=1.0)


async def _wait_for_ssh_service_ready(
    *,
    service: Any,
    session: Any,
    worker: InferenceWorkerConfig,
    startup_timeout: float,
    run_logger: Any | None,
    startup_context: dict[str, Any],
    remote_output_dir: Path,
) -> None:
    started = time.monotonic()
    last_health_state: str | None = None
    last_health_detail: str | None = None
    emitted_startup_phases: set[str] = set()
    seen_lines: dict[str, set[str]] = {"stdout": set(), "stderr": set()}
    attempt = 0

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
                health_detail=detail,
                error=error,
                **startup_context,
            )

    while time.monotonic() - started < startup_timeout:
        is_running = await service.is_running()
        if not is_running:
            await _emit_health_state("exited", attempt=attempt)
            break

        is_healthy = await service.is_healthy()
        log_blob = await _service_logs_best_effort(service, tail=120)
        _emit_log_lines(
            run_logger=run_logger,
            log_blob=log_blob,
            seen_lines=seen_lines,
            emitted_startup_phases=emitted_startup_phases,
            startup_context=startup_context,
        )

        if is_healthy:
            await _emit_health_state("healthy", attempt=attempt)
            return

        await _emit_health_state("waiting", attempt=attempt, detail="remote /health not ready yet")

        if (
            attempt > 0
            and attempt % STARTUP_STALL_DIAGNOSTIC_INTERVAL == 0
            and run_logger is not None
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
            failure_kind="timeout" if last_health_state != "exited" else "exited",
            health_attempt=attempt,
            last_health_state=last_health_state,
            last_health_detail=last_health_detail,
            log_tail=log_blob,
            trace_tail=trace_tail,
            **startup_context,
        )
    raise RuntimeError(
        f"SSH eval endpoint failed to become healthy within {startup_timeout}s.\n"
        f"Recent service logs:\n{log_blob}\n"
        f"Recent trace logs:\n{trace_tail}"
    )


@asynccontextmanager
async def _realize_ssh_endpoint(
    *,
    endpoint_config: EndpointConfig | OwnedEndpoint,
    output_dir: Path,
    hardware_config: HardwareConfig,
    worker: InferenceWorkerConfig,
    run_name: str,
    run_logger: Any | None,
) -> Any:
    from bifrost import AsyncBifrostClient
    from bifrost.types import ProcessSpec, ReadinessProbe, ServiceSpec, WorkspaceMaterializationSpec

    assert hardware_config.ssh is not None, "ssh provider requires hardware_config.ssh"
    assert hardware_config.ssh_key_path is not None, (
        "ssh provider requires hardware_config.ssh_key_path"
    )

    service = None
    startup_context: dict[str, Any] | None = None
    remote_output_dir: Path | None = None

    async with AsyncBifrostClient(
        hardware_config.ssh,
        ssh_key_path=hardware_config.ssh_key_path,
    ) as session:
        workspace = await session.materialize(
            WorkspaceMaterializationSpec(
                requested_root="~/.bifrost/workspaces/rollouts-eval",
                bootstrap_commands=hardware_config.deps.bootstrap_commands
                if hardware_config.deps is not None
                else (),
            )
        )
        remote_output_dir = Path(workspace.root) / "results" / "eval" / run_name
        remote_python = _remote_inference_python(hardware_config)
        launch_cmd, readiness_target = _remote_service_spec(
            worker=worker,
            output_dir=remote_output_dir,
            remote_python=remote_python,
            owned_endpoint=endpoint_config if isinstance(endpoint_config, OwnedEndpoint) else None,
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
        startup_context = {
            "provider": "ssh",
            "ssh_target": hardware_config.ssh,
            "service_name": f"eval-endpoint-{run_name}",
            "engine_name": worker.inference.spec,
            "engine_port": worker.inference.port,
            "engine_cuda_device_ids": list(worker.inference.cuda_device_ids),
            "model_name": worker.model,
            "engine_log_path": f"{remote_output_dir}/endpoint_service",
            "engine_trace_path": str(
                remote_output_dir / f"sglang_{worker.inference.port}_trace.jsonl"
            ),
        }
        if run_logger is not None:
            run_logger.event(
                "inference_engine_launch",
                engine_launch_cmd=launch_cmd,
                readiness_target=readiness_target,
                **startup_context,
            )
        try:
            await _wait_for_ssh_service_ready(
                service=service,
                session=session,
                worker=worker,
                startup_timeout=worker.inference.startup_timeout,
                run_logger=run_logger,
                startup_context=startup_context,
                remote_output_dir=remote_output_dir,
            )
            async with _forward_ssh_port(
                ssh_target=hardware_config.ssh,
                ssh_key_path=hardware_config.ssh_key_path,
                remote_port=worker.inference.port,
            ) as local_port:
                await _wait_for_forwarded_health(
                    local_port=local_port,
                    readiness_target=readiness_target,
                    timeout_s=10.0,
                )
                base_url = f"http://127.0.0.1:{local_port}/v1"
                yield RealizedEvalEndpoint(
                    endpoint_config=(
                        _externalize_owned_endpoint(endpoint_config, base_url)
                        if isinstance(endpoint_config, OwnedEndpoint)
                        else replace(endpoint_config, base_url=base_url)
                    ),
                    metadata={
                        "provider": "ssh",
                        "ssh_target": hardware_config.ssh,
                        "remote_port": worker.inference.port,
                        "local_port": local_port,
                    },
                )
        finally:
            final_log = (
                await _service_logs_best_effort(service, tail=200) if service is not None else ""
            )
            if run_logger is not None and startup_context is not None:
                run_logger.event(
                    "inference_service_final_log",
                    log_blob=final_log,
                    **startup_context,
                )
            elif final_log:
                _logger.info(
                    "inference service final log\n%s",
                    final_log,
                    extra={"event": "inference_service_final_log"},
                )
            if service is not None:
                await service.stop()


@asynccontextmanager
async def _realize_modal_endpoint(
    *,
    endpoint_config: EndpointConfig | OwnedEndpoint,
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
        MODAL_PARENT_LEASE_PATH,
        MODAL_PARENT_LEASE_TTL_S,
        ModalExecutionRequest,
        ModalExecutionSession,
        _maintain_modal_parent_lease,
        _refresh_modal_parent_lease,
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
        keep_alive=hardware_config.keep_alive,
        sandbox_id=hardware_config.sandbox_id,
        cleanup_scope="none" if hardware_config.sandbox_id else "run",
        run_name=run_name,
        tags={
            "control_plane": "argus",
            "launcher_id": run_name,
            "config_basename": run_name,
            "provider": "modal",
        },
        run_logger=run_logger,
        encrypted_ports=(worker.inference.port,),
    )

    def emit_modal_event(event: str, **data: Any) -> None:
        if run_logger is not None:
            run_logger.event(
                event,
                provider="modal",
                run_name=run_name,
                **data,
            )

    with modal.enable_output():
        async with trio_asyncio.open_loop():
            baseline_sandbox_ids = await _list_modal_sandbox_ids()
            sandbox_handle = await create_modal_sandbox(request)
            session = ModalExecutionSession(sandbox_handle=sandbox_handle, local_root=REPO_ROOT)
            service = None
            startup_context: dict[str, Any] | None = None
            try:
                await _refresh_modal_parent_lease(sandbox_handle.sandbox)
                emit_modal_event(
                    "modal_parent_lease_initialized",
                    sandbox_id=sandbox_handle.sandbox_id,
                    lease_path=MODAL_PARENT_LEASE_PATH,
                    ttl_s=MODAL_PARENT_LEASE_TTL_S,
                )
                workspace = await session.materialize(
                    WorkspaceMaterializationSpec(
                        requested_root=getattr(request.materialization, "workspace_root", None)
                    )
                )
                remote_output_dir = Path(workspace.root) / "results" / "eval" / run_name
                remote_python = _remote_inference_python(hardware_config)

                # On sandbox reuse, kill any server still running on the inference port
                # so the new (updated) code takes effect when serve_service starts it.
                if hardware_config.sandbox_id:
                    port = worker.inference.port
                    await session.exec(f"fuser -k {port}/tcp 2>/dev/null || true")
                    _logger.info("Killed existing server on port %d (sandbox reuse)", port)

                launch_cmd, readiness_target = _remote_service_spec(
                    worker=worker,
                    output_dir=remote_output_dir,
                    remote_python=remote_python,
                    owned_endpoint=endpoint_config
                    if isinstance(endpoint_config, OwnedEndpoint)
                    else None,
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
                async with trio.open_nursery() as nursery:
                    nursery.start_soon(
                        _maintain_modal_parent_lease,
                        sandbox_handle.sandbox,
                        emit_modal_event,
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
                    tunnel = await _wait_for_modal_tunnel(
                        sandbox=sandbox_handle.sandbox,
                        port=worker.inference.port,
                        timeout_s=60.0,
                    )
                    try:
                        yield RealizedEvalEndpoint(
                            endpoint_config=(
                                _externalize_owned_endpoint(
                                    endpoint_config,
                                    f"{tunnel.url.rstrip('/')}/v1",
                                )
                                if isinstance(endpoint_config, OwnedEndpoint)
                                else replace(
                                    endpoint_config, base_url=f"{tunnel.url.rstrip('/')}/v1"
                                )
                            ),
                            metadata={
                                "provider": "modal",
                                "sandbox_id": sandbox_handle.sandbox_id,
                            },
                        )
                    finally:
                        nursery.cancel_scope.cancel()
            finally:
                # Capture final service logs before sandbox teardown.
                # This is the only window to read stdout/stderr from the
                # inference server process (e.g. gold_server.py diagnostics).
                final_log = (
                    await _service_logs_best_effort(service, tail=200)
                    if service is not None
                    else ""
                )
                if run_logger is not None and startup_context is not None:
                    run_logger.event(
                        "inference_service_final_log",
                        log_blob=final_log,
                        **startup_context,
                    )
                elif final_log:
                    _logger.info(
                        "inference service final log\n%s",
                        final_log,
                        extra={"event": "inference_service_final_log"},
                    )
                await terminate_modal_sandbox(sandbox_handle)
                if not sandbox_handle.keep_alive:
                    await _wait_for_modal_sandbox_baseline(
                        baseline_ids=baseline_sandbox_ids,
                        run_logger=run_logger,
                        run_name=run_name,
                    )


@asynccontextmanager
async def realize_worker_backed_endpoint(
    *,
    endpoint_config: InferenceEndpoint | EndpointConfig,
    output_dir: Path,
    hardware_config: HardwareConfig | None,
    server_config: InferenceServerConfig | None,
    worker: InferenceWorkerConfig | None = None,
    run_name: str = "eval-endpoint",
    force_deploy_committed: bool = False,
    run_logger: Any | None = None,
) -> Any:
    if isinstance(endpoint_config, ExternalEndpoint):
        yield RealizedEvalEndpoint(endpoint_config=endpoint_config)
        return
    if isinstance(endpoint_config, EndpointConfig):
        if endpoint_config.base_url is not None or not endpoint_config.requires_server:
            yield RealizedEvalEndpoint(endpoint_config=endpoint_config)
            return

    if hardware_config is None:
        raise ValueError(
            "Auto-realized eval endpoint requires hardware config for OwnedEndpoint or "
            "when base_url is omitted."
        )

    realized_worker = worker
    if realized_worker is None:
        if isinstance(endpoint_config, OwnedEndpoint):
            realized_worker = _worker_from_owned_endpoint(endpoint_config)
        else:
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
    if hardware_config.provider == "ssh":
        async with _realize_ssh_endpoint(
            endpoint_config=endpoint_config,
            output_dir=output_dir,
            hardware_config=hardware_config,
            worker=realized_worker,
            run_name=run_name,
            run_logger=run_logger,
        ) as realized:
            yield realized
        return
    if hardware_config.provider != "local":
        raise NotImplementedError(
            "Worker-backed eval endpoint auto-realization currently supports only "
            "hardware.provider in {'local', 'modal', 'ssh'}."
        )

    engine = _build_engine(worker=realized_worker, output_dir=output_dir)
    engine.launch()
    engine.start_log_tailer()
    try:
        await engine.wait_until_ready(realized_worker.inference.startup_timeout)
        yield RealizedEvalEndpoint(
            endpoint_config=(
                _externalize_owned_endpoint(endpoint_config, engine.base_url)
                if isinstance(endpoint_config, OwnedEndpoint)
                else replace(endpoint_config, base_url=engine.api_base)
            ),
            engine=engine,
        )
    finally:
        engine.shutdown()
