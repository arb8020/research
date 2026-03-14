"""Cheap runtime smoke paths for remote training environments.

These are intentionally narrower than full training runs. They exercise one
real backend stage at a time so we can fail before expensive startup like
inference engine boot or RL orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import trio


def _pick(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


async def _torchtitan_backend_init_smoke_async(config: Any) -> dict[str, Any]:
    from .backends.torchtitan_factory import create_torchtitan_backend
    from .preflight import preflight_torchtitan_runtime

    assert getattr(config.trainer, "backend", None) == "torchtitan", (
        "torchtitan backend smoke requires trainer.backend == 'torchtitan'"
    )

    runtime_preflight = preflight_torchtitan_runtime()
    runtime_preflight.require_ok()

    checkpoint_dir = Path(getattr(config, "output_dir", "results")) / "smoke_torchtitan_backend"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = config.trainer
    gpu_ids = tuple(_pick(trainer, "cuda_device_ids", default=(0,)))
    gpu_rank = int(gpu_ids[0]) if gpu_ids else 0

    backend, cleanup = create_torchtitan_backend(
        checkpoint_dir=checkpoint_dir,
        hf_checkpoint=None,
        torchtitan_model=_pick(trainer, "torchtitan_model", default="qwen3"),
        torchtitan_model_size=_pick(trainer, "torchtitan_model_size", default="0.6B"),
        gpu_rank=gpu_rank,
        seq_len=int(_pick(trainer, "seq_len", default=4096)),
        learning_rate=float(_pick(trainer, "learning_rate", "lr", default=1e-5)),
        weight_decay=float(_pick(trainer, "weight_decay", default=0.0)),
        max_grad_norm=float(_pick(trainer, "max_grad_norm", default=1.0)),
        tp=int(_pick(trainer, "tensor_parallel_size", default=1)),
        cp=int(_pick(trainer, "context_parallel_size", default=1)),
        pp=int(_pick(trainer, "pipeline_parallel_size", default=1)),
        enable_loss_parallel=bool(_pick(trainer, "enable_loss_parallel", default=True)),
        packed_sequences=bool(_pick(trainer, "packed_sequences", default=True)),
        mode="rl",
    )

    try:
        return {
            "smoke": "torchtitan_backend_init",
            "backend": getattr(trainer, "backend", None),
            "model": _pick(trainer, "torchtitan_model", default="qwen3"),
            "model_size": _pick(trainer, "torchtitan_model_size", default="0.6B"),
            "torch_version": runtime_preflight.details.get("torch_version"),
            "parallel_dims_type": type(getattr(backend, "_parallel_dims", None)).__name__,
            "optimizer_type": type(getattr(backend, "_optimizer", None)).__name__,
        }
    finally:
        if cleanup is not None:
            cleanup()


def run_torchtitan_backend_init_smoke(config: Any, **_: Any) -> dict[str, Any]:
    """Run the cheapest real TorchTitan backend-init stage in the target runtime.

    This is the intended first remote check for backend API/runtime issues.
    It should fail before inference startup, RL orchestration, or checkpoint
    loading if the backend integration is broken.
    """

    return trio.run(_torchtitan_backend_init_smoke_async, config)


async def _inference_startup_smoke_async(config: Any) -> dict[str, Any]:
    from .grpo import _create_inference_engines, _create_teacher_engine

    output_root = Path(getattr(getattr(config, "output", None), "output_dir", "results"))
    checkpoint_dir = output_root / "smoke_inference_startup"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    engines = _create_inference_engines(config, checkpoint_dir)
    teacher_engine = _create_teacher_engine(config, checkpoint_dir)

    launched: list[dict[str, Any]] = []
    try:
        for idx, engine in enumerate(engines):
            engine.launch()
            engine.start_log_tailer()
            launched.append(
                {
                    "engine_index": idx,
                    "engine_name": getattr(engine, "name", "unknown"),
                    "port": getattr(engine, "port", None),
                    "cuda_device_ids": list(getattr(engine, "cuda_device_ids", ())),
                }
            )

        if teacher_engine is not None:
            teacher_engine.launch()
            teacher_engine.start_log_tailer()

        startup_timeout = float(getattr(config.inference, "startup_timeout", 300.0))
        async with trio.open_nursery() as startup_nursery:
            for engine in engines:
                startup_nursery.start_soon(engine.wait_until_ready, startup_timeout)
            if teacher_engine is not None:
                startup_nursery.start_soon(teacher_engine.wait_until_ready, startup_timeout)

        return {
            "smoke": "inference_startup",
            "backend": getattr(config.inference, "backend", None),
            "model": getattr(config.model, "name", None),
            "num_engines": len(engines),
            "engines": launched,
            "teacher_engine": teacher_engine is not None,
        }
    finally:
        for engine in engines:
            try:
                engine.shutdown()
            except Exception:
                pass
        if teacher_engine is not None:
            try:
                teacher_engine.shutdown()
            except Exception:
                pass


def run_inference_startup_smoke(config: Any, **_: Any) -> dict[str, Any]:
    """Run the cheapest real inference startup stage in the target runtime.

    This starts the configured inference engine(s), waits for health, and exits.
    It is intended to isolate inference bring-up failures from training and RL
    orchestration.
    """

    return trio.run(_inference_startup_smoke_async, config)


async def _training_and_inference_startup_smoke_async(config: Any, run_logger: Any | None = None) -> dict[str, Any]:
    import logging
    import os
    import socket

    from .grpo import (
        _build_grpo_run_context,
        _create_inference_engines,
        _create_teacher_engine,
        _run_training_preflight,
    )

    output_root = Path(getattr(getattr(config, "output", None), "output_dir", "results"))
    checkpoint_dir = output_root / "smoke_training_and_inference_startup"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rollouts.training.smoke")
    logger.setLevel(logging.INFO)

    def emit(event: str, **data: Any) -> None:
        if run_logger is not None:
            run_logger.event(event, **data)

    run_context = _build_grpo_run_context(
        config=config,
        run_name=checkpoint_dir.name,
        output_dir=checkpoint_dir,
        node_id=os.environ.get("ROLLOUTS_NODE_ID"),
        num_inference_engines=config.inference.num_engines,
    )
    emit(
        "combined_startup_smoke_start",
        **run_context,
        teacher_model=getattr(config.trainer, "teacher_model", None),
        hostname=socket.gethostname(),
    )

    backend = None
    backend_cleanup = None
    inference_engines = []
    teacher_engine = None
    try:
        emit(
            "combined_smoke_training_preflight_start",
            **run_context,
            trainer_backend=getattr(config.trainer, "backend", None),
        )
        backend, backend_cleanup = await _run_training_preflight(
            config,
            checkpoint_dir,
            logger,
            node_id=os.environ.get("ROLLOUTS_NODE_ID"),
            run_context=run_context,
        )
        emit(
            "combined_smoke_training_preflight_finished",
            **run_context,
            backend_type=type(backend).__name__ if backend is not None else None,
            optimizer_type=type(getattr(backend, "_optimizer", None)).__name__ if backend is not None else None,
        )

        emit(
            "combined_smoke_inference_setup_start",
            **run_context,
            inference_backend=getattr(config.inference, "backend", None),
        )
        inference_engines = _create_inference_engines(config, checkpoint_dir)
        teacher_engine = _create_teacher_engine(config, checkpoint_dir)
        emit(
            "combined_smoke_inference_setup_finished",
            **run_context,
            num_engines=len(inference_engines),
            teacher_engine=teacher_engine is not None,
        )

        for idx, engine in enumerate(inference_engines):
            emit(
                "combined_smoke_inference_engine_launch",
                **run_context,
                engine_index=idx,
                engine_name=getattr(engine, "name", "unknown"),
                engine_port=getattr(engine, "port", None),
                engine_cuda_device_ids=list(getattr(engine, "cuda_device_ids", ())),
            )
            engine.launch()
            engine.start_log_tailer()
            emit(
                "combined_smoke_inference_engine_launched",
                **run_context,
                engine_index=idx,
                engine_name=getattr(engine, "name", "unknown"),
            )

        if teacher_engine is not None:
            emit("combined_smoke_teacher_engine_launch", **run_context)
            teacher_engine.launch()
            teacher_engine.start_log_tailer()
            emit("combined_smoke_teacher_engine_launched", **run_context)

        startup_timeout = float(getattr(config.inference, "startup_timeout", 300.0))
        emit(
            "combined_smoke_inference_healthcheck_start",
            **run_context,
            startup_timeout=startup_timeout,
            teacher_engine=teacher_engine is not None,
        )
        async with trio.open_nursery() as startup_nursery:
            for engine in inference_engines:
                startup_nursery.start_soon(engine.wait_until_ready, startup_timeout)
            if teacher_engine is not None:
                startup_nursery.start_soon(teacher_engine.wait_until_ready, startup_timeout)

        emit(
            "combined_smoke_inference_healthcheck_finished",
            **run_context,
            num_engines=len(inference_engines),
            teacher_engine=teacher_engine is not None,
        )

        return {
            "smoke": "training_and_inference_startup",
            "trainer_backend": getattr(config.trainer, "backend", None),
            "inference_backend": getattr(config.inference, "backend", None),
            "num_engines": len(inference_engines),
            "teacher_engine": teacher_engine is not None,
            "trainer_cuda_device_ids": list(config.trainer.cuda_device_ids),
            "inference_gpu_assignments": [list(gpus) for gpus in config.inference.gpu_assignments],
        }
    finally:
        if backend_cleanup is not None:
            try:
                backend_cleanup()
            except Exception:
                pass
        for engine in inference_engines:
            try:
                engine.shutdown()
            except Exception:
                pass
        if teacher_engine is not None:
            try:
                teacher_engine.shutdown()
            except Exception:
                pass


def run_training_and_inference_startup_smoke(config: Any, **kwargs: Any) -> dict[str, Any]:
    """Run training preflight plus inference startup, but no RL loop.

    This isolates the boundary where both services are alive together, which is
    the next stage after backend-only and inference-only smokes.
    """

    return trio.run(_training_and_inference_startup_smoke_async, config, kwargs.get("run_logger"))
