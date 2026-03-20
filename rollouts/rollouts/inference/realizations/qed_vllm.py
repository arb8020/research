"""Patched vLLM OpenAI server with QED-Nano-style NCCL weight update hooks.

This is intentionally a standalone inference entrypoint. It lets us verify the vLLM-side
worker extension and HTTP control surface in isolation before wiring it into a
training backend.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import time
from collections.abc import Awaitable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from rollouts.inference.weight_sync import (
    ParamInfo,
    WeightSyncReceiver,
)
from rollouts.training.weight_sync_protocol import (
    InitWeightUpdateGroupRequest,
    InitWeightUpdateGroupResponse,
    ReceiveWeightUpdateRequest,
)

logger = logging.getLogger(__name__)
_ARGUS_DIAG_EVENT_SENTINEL = "__ARGUS_DIAG__"
_WEIGHT_SYNC_TRACE_PATH = Path("/tmp/rollouts_vllm_weight_sync_trace.jsonl")


def _emit_argus_diag(event: str, **data: object) -> None:
    try:
        sys.stderr.write(
            f"{_ARGUS_DIAG_EVENT_SENTINEL}{json.dumps({'event': event, **data}, sort_keys=True)}\n"
        )
        sys.stderr.flush()
    except Exception:
        return


def _append_weight_sync_trace(event: str, **data: object) -> None:
    try:
        record = {"ts_unix": time.time(), "event": event, **data}
        with _WEIGHT_SYNC_TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
    except Exception:
        return


def _route_paths(app: Any) -> list[str]:
    return sorted(
        path
        for route in getattr(app, "routes", ())
        if isinstance(path := getattr(route, "path", None), str)
    )


class _LikeWorker(Protocol):
    rank: int
    local_rank: int
    device: torch.device
    model_runner: Any


def _maybe_await(value: Any) -> Awaitable[Any] | None:
    if inspect.isawaitable(value):
        return cast(Awaitable[Any], value)
    return None


def _worker_rank(worker: _LikeWorker) -> int:
    return int(getattr(worker, "rank", getattr(worker, "local_rank", 0)))


def _weight_sync_receiver(worker: _LikeWorker) -> WeightSyncReceiver:
    receiver = getattr(worker, "_rollouts_weight_sync_receiver", None)
    if receiver is None:
        raise RuntimeError("Weight sync receiver not initialized")
    return cast(WeightSyncReceiver, receiver)


def _model(worker: _LikeWorker) -> Any:
    return worker.model_runner.model


async def _await_collective_rpc(
    engine_client: Any,
    *,
    method: str,
    args: tuple[Any, ...],
    timeout_seconds: float | None = None,
) -> Any:
    result = engine_client.collective_rpc(method, args=args)
    maybe = _maybe_await(result)
    if maybe is None:
        return result
    if timeout_seconds is None:
        return await maybe
    return await asyncio.wait_for(maybe, timeout=timeout_seconds)


async def _dispatch_init_weight_update_group(
    engine_client: Any,
    request_payload: Mapping[str, Any],
) -> dict[str, Any]:
    request = InitWeightUpdateGroupRequest.from_dict(request_payload)
    _append_weight_sync_trace(
        "vllm_route_init_weight_update_group_start",
        request=request.to_dict(),
    )
    _emit_argus_diag(
        "vllm_route_init_weight_update_group_start",
        request=request.to_dict(),
    )
    logger.info("init_weights_update_group request start request=%s", request)
    payload = await _await_collective_rpc(
        engine_client,
        method="init_weight_update_group",
        args=(
            request.master_address,
            request.master_port,
            request.rank_offset,
            request.world_size,
            request.group_name,
            request.timeout_seconds,
        ),
        timeout_seconds=request.timeout_seconds,
    )
    _append_weight_sync_trace(
        "vllm_route_init_weight_update_group_collective_rpc_finished",
        request=request.to_dict(),
        result=payload,
    )
    response = InitWeightUpdateGroupResponse(results=payload)
    return {"status": "ok", "results": response.results}


async def _dispatch_receive_weight_update(
    engine_client: Any,
    request_payload: Mapping[str, Any],
) -> dict[str, Any]:
    request = ReceiveWeightUpdateRequest.from_dict(request_payload)
    request_dict = request.to_dict()
    first_tensors = [
        {
            "wire_name": name,
            "load_name": load_name,
            "shape": list(shape),
            "dtype": dtype,
        }
        for name, load_name, shape, dtype in zip(
            request.names,
            request.load_names,
            request.shapes,
            request.dtypes,
            strict=True,
        )
    ][:3]
    _append_weight_sync_trace(
        "vllm_route_receive_weight_update_start",
        request=request_dict,
        total_tensors=len(request.names),
        first_tensors=first_tensors,
    )
    _emit_argus_diag(
        "vllm_route_receive_weight_update_start",
        request=request_dict,
        total_tensors=len(request.names),
        first_tensors=first_tensors,
    )
    logger.info(
        "receive_weight_update request start total_tensors=%s first_tensors=%s",
        len(request.names),
        first_tensors,
    )
    payload = await _await_collective_rpc(
        engine_client,
        method="receive_weight_update",
        args=(
            list(request.names),
            [list(shape) for shape in request.shapes],
            list(request.dtypes),
            list(request.load_names),
        ),
    )
    _append_weight_sync_trace(
        "vllm_route_receive_weight_update_collective_rpc_finished",
        request=request_dict,
        total_tensors=len(request.names),
        results=payload,
    )
    _emit_argus_diag(
        "vllm_route_receive_weight_update_collective_rpc_finished",
        request=request_dict,
        total_tensors=len(request.names),
    )
    return {"status": "ok", "results": payload}


class WorkerExtension:
    """Methods invoked on vLLM workers via collective RPC."""

    def init_weight_update_group(
        self: _LikeWorker,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str = "weight_sync",
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        worker_rank = _worker_rank(self)
        rank = int(rank_offset) + worker_rank
        existing_receiver = getattr(self, "_rollouts_weight_sync_receiver", None)
        if existing_receiver is not None:
            existing_group_name = getattr(existing_receiver, "group_name", None)
            if existing_group_name == group_name:
                _emit_argus_diag(
                    "vllm_worker_init_weight_update_group_already_initialized",
                    worker_rank=worker_rank,
                    rank=rank,
                    world_size=int(world_size),
                    group_name=group_name,
                    device=str(self.device),
                )
                _append_weight_sync_trace(
                    "vllm_worker_init_weight_update_group_already_initialized",
                    worker_rank=worker_rank,
                    rank=rank,
                    world_size=int(world_size),
                    group_name=group_name,
                    device=str(self.device),
                )
                logger.info(
                    "vllm worker init_weight_update_group already initialized worker_rank=%s rank=%s world_size=%s group=%s device=%s",
                    worker_rank,
                    rank,
                    world_size,
                    group_name,
                    self.device,
                )
                return {
                    "rank": rank,
                    "world_size": int(world_size),
                    "group_name": group_name,
                    "already_initialized": True,
                }
            raise RuntimeError(
                "Weight sync receiver already initialized with different group "
                f"{existing_group_name!r}, cannot reinitialize with {group_name!r}"
            )
        _emit_argus_diag(
            "vllm_worker_init_weight_update_group_start",
            worker_rank=worker_rank,
            rank=rank,
            world_size=int(world_size),
            group_name=group_name,
            device=str(self.device),
            master_address=master_address,
            master_port=int(master_port),
            timeout_seconds=float(timeout_seconds),
        )
        _append_weight_sync_trace(
            "vllm_worker_init_weight_update_group_start",
            worker_rank=worker_rank,
            rank=rank,
            world_size=int(world_size),
            group_name=group_name,
            device=str(self.device),
            master_address=master_address,
            master_port=int(master_port),
            timeout_seconds=float(timeout_seconds),
        )
        logger.info(
            "vllm worker init_weight_update_group start worker_rank=%s rank=%s world_size=%s group=%s device=%s master=%s:%s timeout=%s",
            worker_rank,
            rank,
            world_size,
            group_name,
            self.device,
            master_address,
            master_port,
            timeout_seconds,
        )
        receiver = WeightSyncReceiver(
            master_addr=master_address,
            master_port=int(master_port),
            rank=rank,
            world_size=int(world_size),
            group_name=group_name,
            timeout_seconds=timeout_seconds,
            device=self.device,
        )
        _emit_argus_diag(
            "vllm_worker_init_weight_update_group_before_receiver_init",
            rank=rank,
            group_name=group_name,
            device=str(self.device),
        )
        _append_weight_sync_trace(
            "vllm_worker_init_weight_update_group_before_receiver_init",
            rank=rank,
            group_name=group_name,
            device=str(self.device),
        )
        logger.info(
            "vllm worker init_weight_update_group before receiver.init_group rank=%s group=%s",
            rank,
            group_name,
        )
        receiver.init_group()
        _emit_argus_diag(
            "vllm_worker_init_weight_update_group_after_receiver_init",
            rank=rank,
            group_name=group_name,
            device=str(self.device),
        )
        _append_weight_sync_trace(
            "vllm_worker_init_weight_update_group_after_receiver_init",
            rank=rank,
            group_name=group_name,
            device=str(self.device),
        )
        logger.info(
            "vllm worker init_weight_update_group after receiver.init_group rank=%s group=%s",
            rank,
            group_name,
        )
        self._rollouts_weight_sync_receiver = receiver
        logger.info(
            "Initialized vLLM NCCL receiver: rank=%s world_size=%s group=%s device=%s",
            rank,
            world_size,
            group_name,
            self.device,
        )
        return {"rank": rank, "world_size": int(world_size), "group_name": group_name}

    def get_weight_update_schema(
        self: _LikeWorker,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params = list(_model(self).named_parameters())
        if limit is not None:
            params = params[: int(limit)]
        return [
            {
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype).replace("torch.", ""),
            }
            for name, param in params
        ]

    def receive_weight_update(
        self: _LikeWorker,
        names: list[str],
        shapes: list[list[int]],
        dtypes: list[str],
        load_names: list[str] | None = None,
    ) -> dict[str, Any]:
        worker_rank = _worker_rank(self)
        receiver = _weight_sync_receiver(self)
        model = _model(self)
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        resolved_load_names = load_names or names
        param_info = [
            ParamInfo(
                wire_name=name,
                load_name=load_name,
                shape=tuple(shape),
                dtype=dtype_map.get(dtype_str, torch.bfloat16),
            )
            for name, load_name, shape, dtype_str in zip(
                names,
                resolved_load_names,
                shapes,
                dtypes,
                strict=True,
            )
        ]
        first_tensors = [
            {
                "wire_name": info.wire_name,
                "load_name": info.load_name,
                "shape": list(info.shape),
                "dtype": str(info.dtype).replace("torch.", ""),
            }
            for info in param_info[:3]
        ]
        _append_weight_sync_trace(
            "vllm_worker_receive_weight_update_start",
            worker_rank=worker_rank,
            total_tensors=len(param_info),
            first_tensors=first_tensors,
            device=str(self.device),
        )
        _emit_argus_diag(
            "vllm_worker_receive_weight_update_start",
            worker_rank=worker_rank,
            total_tensors=len(param_info),
            first_tensors=first_tensors,
            device=str(self.device),
        )
        logger.info(
            "vllm worker receive_weight_update start worker_rank=%s total_tensors=%s first_tensors=%s device=%s",
            worker_rank,
            len(param_info),
            first_tensors,
            self.device,
        )
        torch.cuda.synchronize(self.device)
        for index, info in enumerate(param_info):
            if index == 0:
                _append_weight_sync_trace(
                    "vllm_worker_receive_weight_update_first_tensor_receive_start",
                    worker_rank=worker_rank,
                    tensor=first_tensors[0],
                    device=str(self.device),
                )
                _emit_argus_diag(
                    "vllm_worker_receive_weight_update_first_tensor_receive_start",
                    worker_rank=worker_rank,
                    tensor=first_tensors[0],
                    device=str(self.device),
                )
            buffer = receiver.receive_weights([info])[info.load_name]
            if index == 0:
                _append_weight_sync_trace(
                    "vllm_worker_receive_weight_update_first_tensor_receive_ok",
                    worker_rank=worker_rank,
                    tensor=first_tensors[0],
                    device=str(self.device),
                )
                _emit_argus_diag(
                    "vllm_worker_receive_weight_update_first_tensor_receive_ok",
                    worker_rank=worker_rank,
                    tensor=first_tensors[0],
                    device=str(self.device),
                )
            loaded = model.load_weights(weights=[(info.load_name, buffer)])
            if len(loaded) != 1:
                raise RuntimeError(
                    f"Failed to load weight {info.load_name!r} into vLLM worker model"
                )
            if index == 0:
                _append_weight_sync_trace(
                    "vllm_worker_receive_weight_update_first_tensor_load_ok",
                    worker_rank=worker_rank,
                    tensor=first_tensors[0],
                    device=str(self.device),
                )
                _emit_argus_diag(
                    "vllm_worker_receive_weight_update_first_tensor_load_ok",
                    worker_rank=worker_rank,
                    tensor=first_tensors[0],
                    device=str(self.device),
                )
        torch.cuda.synchronize(self.device)
        _append_weight_sync_trace(
            "vllm_worker_receive_weight_update_finished",
            worker_rank=worker_rank,
            total_tensors=len(param_info),
            device=str(self.device),
        )
        _emit_argus_diag(
            "vllm_worker_receive_weight_update_finished",
            worker_rank=worker_rank,
            total_tensors=len(param_info),
            device=str(self.device),
        )
        logger.info("Applied vLLM NCCL weight update for %d tensors", len(param_info))
        return {"status": "ok", "num_tensors": len(param_info)}

    def destroy_weight_update_group(self: _LikeWorker) -> dict[str, Any]:
        receiver = getattr(self, "_rollouts_weight_sync_receiver", None)
        if receiver is not None:
            cast(WeightSyncReceiver, receiver).cleanup()
            delattr(self, "_rollouts_weight_sync_receiver")
        return {"status": "ok"}


async def run_server(args: Any, **uvicorn_kwargs: Any) -> None:
    from fastapi import HTTPException
    from vllm.entrypoints.launcher import serve_http
    from vllm.entrypoints.openai.api_server import (
        build_app,
        build_async_engine_client,
        init_app_state,
        setup_server,
    )

    args.worker_extension_cls = f"{__name__}.WorkerExtension"
    logger.info(
        "qed_vllm.run_server start worker_extension_cls=%s host=%s port=%s model=%s",
        args.worker_extension_cls,
        getattr(args, "host", None),
        getattr(args, "port", None),
        getattr(args, "model", None),
    )
    _emit_argus_diag(
        "qed_vllm_run_server_start",
        worker_extension_cls=args.worker_extension_cls,
        host=getattr(args, "host", None),
        port=getattr(args, "port", None),
        model=getattr(args, "model", None),
    )

    _listen_address, sock = setup_server(args)

    async with build_async_engine_client(args) as engine_client:
        logger.info("qed_vllm engine_client ready")
        startup_master_address = getattr(args, "rollouts_weight_sync_master_address", None)
        if startup_master_address:
            startup_request = InitWeightUpdateGroupRequest(
                master_address=str(startup_master_address),
                master_port=int(args.rollouts_weight_sync_master_port),
                rank_offset=int(args.rollouts_weight_sync_rank_offset),
                world_size=int(args.rollouts_weight_sync_world_size),
                group_name=str(args.rollouts_weight_sync_group_name),
                timeout_seconds=float(args.rollouts_weight_sync_timeout_seconds),
            )
            _append_weight_sync_trace(
                "vllm_startup_init_weight_update_group_start",
                request=startup_request.to_dict(),
            )
            _emit_argus_diag(
                "vllm_startup_init_weight_update_group_start",
                request=startup_request.to_dict(),
            )
            payload = await _await_collective_rpc(
                engine_client,
                method="init_weight_update_group",
                args=(
                    startup_request.master_address,
                    startup_request.master_port,
                    startup_request.rank_offset,
                    startup_request.world_size,
                    startup_request.group_name,
                    startup_request.timeout_seconds,
                ),
                timeout_seconds=startup_request.timeout_seconds,
            )
            _append_weight_sync_trace(
                "vllm_startup_init_weight_update_group_finished",
                request=startup_request.to_dict(),
                result=payload,
            )
            _emit_argus_diag(
                "vllm_startup_init_weight_update_group_finished",
                request=startup_request.to_dict(),
            )
        # Keep this explicit warmup call even though we do not consume the value.
        # vLLM's engine client path is not purely query-like here; forcing task
        # discovery exercises initialization that the patched server depends on.
        _ = await engine_client.get_supported_tasks()
        logger.info("qed_vllm supported_tasks warmup finished")
        app = build_app(args)
        logger.info("qed_vllm build_app finished routes=%s", _route_paths(app))
        # Initialize upstream app state before attaching custom control-plane routes.
        # Newer vLLM startup may mutate app wiring here, and we need the patched
        # NCCL endpoints to survive whatever initialization the base server does.
        await init_app_state(engine_client, app.state, args)
        logger.info("qed_vllm init_app_state finished routes=%s", _route_paths(app))

        @app.get("/weight_update_schema")
        async def weight_update_schema(limit: int = 1) -> dict[str, Any]:
            result = engine_client.collective_rpc("get_weight_update_schema", args=(limit,))
            maybe = _maybe_await(result)
            payload = await maybe if maybe is not None else result
            if isinstance(payload, list) and payload:
                payload = payload[0]
            return {"status": "ok", "parameters": payload}

        @app.get("/init_weights_update_group_probe")
        async def init_weights_update_group_probe() -> dict[str, Any]:
            _emit_argus_diag("vllm_route_init_weight_update_group_probe")
            _append_weight_sync_trace("vllm_route_init_weight_update_group_probe")
            return {
                "status": "ok",
                "route": "init_weights_update_group_probe",
                "worker_extension_cls": getattr(args, "worker_extension_cls", None),
            }

        @app.get("/weight_sync_trace")
        async def weight_sync_trace(limit: int = 50) -> dict[str, Any]:
            entries: list[dict[str, Any]] = []
            try:
                if _WEIGHT_SYNC_TRACE_PATH.exists():
                    lines = _WEIGHT_SYNC_TRACE_PATH.read_text(encoding="utf-8").splitlines()
                    for line in lines[-max(1, int(limit)) :]:
                        entries.append(json.loads(line))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=repr(exc)) from exc
            return {"status": "ok", "path": str(_WEIGHT_SYNC_TRACE_PATH), "entries": entries}

        @app.post("/init_weights_update_group")
        async def init_weights_update_group(request: dict[str, Any]) -> dict[str, Any]:
            try:
                return await _dispatch_init_weight_update_group(engine_client, request)
            except (AssertionError, TypeError, ValueError) as exc:
                logger.exception(
                    "init_weights_update_group_invalid_request request=%s error_type=%s error=%r",
                    request,
                    type(exc).__name__,
                    exc,
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                _append_weight_sync_trace(
                    "vllm_route_init_weight_update_group_failed",
                    request=request,
                    error_type=type(exc).__name__,
                    error=repr(exc),
                )
                _emit_argus_diag(
                    "vllm_route_init_weight_update_group_failed",
                    request=request,
                    error_type=type(exc).__name__,
                    error=repr(exc),
                )
                logger.exception(
                    "init_weights_update_group_failed request=%s error_type=%s error=%r",
                    request,
                    type(exc).__name__,
                    exc,
                )
                raise HTTPException(status_code=500, detail=repr(exc)) from exc

        @app.post("/receive_weight_update")
        async def receive_weight_update(request: dict[str, Any]) -> dict[str, Any]:
            try:
                return await _dispatch_receive_weight_update(engine_client, request)
            except (AssertionError, TypeError, ValueError) as exc:
                _append_weight_sync_trace(
                    "vllm_route_receive_weight_update_invalid_request",
                    request=request,
                    error_type=type(exc).__name__,
                    error=repr(exc),
                )
                _emit_argus_diag(
                    "vllm_route_receive_weight_update_invalid_request",
                    request=request,
                    error_type=type(exc).__name__,
                    error=repr(exc),
                )
                logger.exception(
                    "receive_weight_update_invalid_request request=%s error_type=%s error=%r",
                    request,
                    type(exc).__name__,
                    exc,
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                _append_weight_sync_trace(
                    "vllm_route_receive_weight_update_failed",
                    request=request,
                    error_type=type(exc).__name__,
                    error=repr(exc),
                )
                _emit_argus_diag(
                    "vllm_route_receive_weight_update_failed",
                    request=request,
                    error_type=type(exc).__name__,
                    error=repr(exc),
                )
                logger.exception(
                    "receive_weight_update_failed request=%s error_type=%s error=%r",
                    request,
                    type(exc).__name__,
                    exc,
                )
                raise HTTPException(status_code=500, detail=repr(exc)) from exc

        @app.post("/destroy_weights_update_group")
        async def destroy_weights_update_group() -> dict[str, Any]:
            result = engine_client.collective_rpc("destroy_weight_update_group")
            maybe = _maybe_await(result)
            payload = await maybe if maybe is not None else result
            return {"status": "ok", "results": payload}

        routes = _route_paths(app)
        logger.info("qed_vllm patched routes registered routes=%s", routes)
        _emit_argus_diag("qed_vllm_routes_registered", routes=routes)
        required_routes = {
            "/weight_update_schema",
            "/init_weights_update_group",
            "/receive_weight_update",
            "/destroy_weights_update_group",
        }
        missing_routes = sorted(required_routes.difference(routes))
        if missing_routes:
            raise RuntimeError(
                "qed_vllm patched routes missing after registration: "
                f"{missing_routes}; available routes={routes}"
            )

        logger.info("qed_vllm serving patched app")

        shutdown_task = await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            **uvicorn_kwargs,
        )
    try:
        await shutdown_task
    finally:
        sock.close()


def main() -> None:
    import argparse

    import uvloop
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

    original_boolean_optional_init = argparse.BooleanOptionalAction.__init__

    def _patched_boolean_optional_init(
        self: Any,
        option_strings: Any,
        dest: Any,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("deprecated", None)
        original_boolean_optional_init(self, option_strings, dest, **kwargs)

    argparse.BooleanOptionalAction.__init__ = _patched_boolean_optional_init

    parser = argparse.ArgumentParser(description="rollouts patched vLLM OpenAI server")
    parser = make_arg_parser(parser)
    parser.add_argument("--rollouts-weight-sync-master-address", type=str, default=None)
    parser.add_argument("--rollouts-weight-sync-master-port", type=int, default=None)
    parser.add_argument("--rollouts-weight-sync-rank-offset", type=int, default=None)
    parser.add_argument("--rollouts-weight-sync-world-size", type=int, default=None)
    parser.add_argument("--rollouts-weight-sync-group-name", type=str, default=None)
    parser.add_argument("--rollouts-weight-sync-timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
