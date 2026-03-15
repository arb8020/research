"""Patched vLLM OpenAI server with QED-Nano-style NCCL weight update hooks.

This is intentionally a standalone entrypoint. It lets us verify the vLLM-side
worker extension and HTTP control surface in isolation before wiring it into a
training backend.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable
from typing import Any, Protocol, cast

import torch

from rollouts.inference.weight_sync import ParamInfo, WeightSyncReceiver

logger = logging.getLogger(__name__)


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
        receiver = WeightSyncReceiver(
            master_addr=master_address,
            master_port=int(master_port),
            rank=rank,
            world_size=int(world_size),
            group_name=group_name,
            timeout_seconds=timeout_seconds,
            device=self.device,
        )
        receiver.init_group()
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
    ) -> dict[str, Any]:
        receiver = _weight_sync_receiver(self)
        model = _model(self)
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        param_info = [
            ParamInfo(
                name=name,
                shape=tuple(shape),
                dtype=dtype_map.get(dtype_str, torch.bfloat16),
            )
            for name, shape, dtype_str in zip(names, shapes, dtypes, strict=True)
        ]
        torch.cuda.synchronize(self.device)
        for info in param_info:
            buffer = receiver.receive_weights([info])[info.name]
            loaded = model.load_weights(weights=[(info.name, buffer)])
            if len(loaded) != 1:
                raise RuntimeError(f"Failed to load weight {info.name!r} into vLLM worker model")
        torch.cuda.synchronize(self.device)
        logger.info("Applied vLLM NCCL weight update for %d tensors", len(param_info))
        return {"status": "ok", "num_tensors": len(param_info)}

    def destroy_weight_update_group(self: _LikeWorker) -> dict[str, Any]:
        receiver = getattr(self, "_rollouts_weight_sync_receiver", None)
        if receiver is not None:
            cast(WeightSyncReceiver, receiver).cleanup()
            delattr(self, "_rollouts_weight_sync_receiver")
        return {"status": "ok"}


async def run_server(args: Any, **uvicorn_kwargs: Any) -> None:
    from vllm.entrypoints.launcher import serve_http
    from vllm.entrypoints.openai.api_server import (
        build_app,
        build_async_engine_client,
        init_app_state,
        setup_server,
    )

    args.worker_extension_cls = f"{__name__}.WorkerExtension"

    _listen_address, sock = setup_server(args)

    async with build_async_engine_client(args) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        app = build_app(args, supported_tasks)

        @app.get("/weight_update_schema")
        async def weight_update_schema(limit: int = 1) -> dict[str, Any]:
            result = engine_client.collective_rpc("get_weight_update_schema", args=(limit,))
            maybe = _maybe_await(result)
            payload = await maybe if maybe is not None else result
            if isinstance(payload, list) and payload:
                payload = payload[0]
            return {"status": "ok", "parameters": payload}

        @app.post("/init_weights_update_group")
        async def init_weights_update_group(request: dict[str, Any]) -> dict[str, Any]:
            result = engine_client.collective_rpc(
                "init_weight_update_group",
                args=(
                    request.get("master_address", "127.0.0.1"),
                    int(request.get("master_port", 29500)),
                    int(request.get("rank_offset", 1)),
                    int(request.get("world_size", 2)),
                    request.get("group_name", "weight_sync"),
                    float(request.get("timeout_seconds", 300.0)),
                ),
            )
            maybe = _maybe_await(result)
            payload = await maybe if maybe is not None else result
            return {"status": "ok", "results": payload}

        @app.post("/receive_weight_update")
        async def receive_weight_update(request: dict[str, Any]) -> dict[str, Any]:
            result = engine_client.collective_rpc(
                "receive_weight_update",
                args=(
                    request.get("names", []),
                    request.get("shapes", []),
                    request.get("dtypes", []),
                ),
            )
            maybe = _maybe_await(result)
            payload = await maybe if maybe is not None else result
            return {"status": "ok", "results": payload}

        @app.post("/destroy_weights_update_group")
        async def destroy_weights_update_group() -> dict[str, Any]:
            result = engine_client.collective_rpc("destroy_weight_update_group")
            maybe = _maybe_await(result)
            payload = await maybe if maybe is not None else result
            return {"status": "ok", "results": payload}

        await init_app_state(engine_client, app.state, args, supported_tasks)
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
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
