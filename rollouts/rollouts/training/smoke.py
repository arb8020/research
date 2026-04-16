"""Cheap runtime smoke paths for remote training environments.

These are intentionally narrower than full training runs. They exercise one
real backend stage at a time so we can fail before expensive startup like
inference engine boot or RL orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import trio

from ..event_log import emit_run_event


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
    from .grpo import _create_teacher_engine
    from .inference_runtime_factory import create_inference_backend_runtime

    output_root = Path(getattr(getattr(config, "output", None), "output_dir", "results"))
    checkpoint_dir = output_root / "smoke_inference_startup"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    inference_runtime = create_inference_backend_runtime(
        model=config.model,
        inference=config.inference,
        rollout=config.rollout,
        checkpoint=config.checkpoint,
        output_dir=checkpoint_dir,
    )
    engines = list(inference_runtime.engines)
    teacher_engine = _create_teacher_engine(config, checkpoint_dir)

    launched: list[dict[str, Any]] = []
    try:
        for idx, engine in enumerate(engines):
            engine.launch()
            engine.start_log_tailer()
            launched.append({
                "engine_index": idx,
                "engine_name": getattr(engine, "name", "unknown"),
                "port": getattr(engine, "port", None),
                "cuda_device_ids": list(getattr(engine, "cuda_device_ids", ())),
            })

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
            "realization": inference_runtime.realization.name,
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


async def _training_and_inference_startup_smoke_async(
    config: Any, run_logger: Any | None = None
) -> dict[str, Any]:
    import logging
    import os
    import subprocess

    from .grpo import (
        _build_grpo_run_context,
        _create_teacher_engine,
        _run_training_preflight,
    )
    from .inference_runtime_factory import create_inference_backend_runtime

    output_root = Path(getattr(getattr(config, "output", None), "output_dir", "results"))
    checkpoint_dir = output_root / "smoke_training_and_inference_startup"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rollouts.training.smoke")
    logger.setLevel(logging.INFO)

    def emit(event: str, **data: Any) -> None:
        if run_logger is not None:
            emit_run_event(run_logger, event, **data)

    def _compute_app_snapshot() -> str | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return None
            snapshot = result.stdout.strip()
            return snapshot or "<none>"
        except Exception:
            return None

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
    )

    backend = None
    backend_cleanup = None
    inference_engines = []
    teacher_engine = None
    try:
        emit(
            "combined_smoke_training_preflight_start",
            **run_context,
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
            optimizer_type=type(getattr(backend, "_optimizer", None)).__name__
            if backend is not None
            else None,
        )

        emit(
            "combined_smoke_inference_setup_start",
            **run_context,
        )
        inference_runtime = create_inference_backend_runtime(
            model=config.model,
            inference=config.inference,
            rollout=config.rollout,
            checkpoint=config.checkpoint,
            output_dir=checkpoint_dir,
        )
        inference_engines = list(inference_runtime.engines)
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
                session_name=getattr(engine, "session_name", None),
                log_path=str(getattr(engine, "log_path", "")) or None,
                launch_cmd=engine.build_launch_cmd()
                if hasattr(engine, "build_launch_cmd")
                else None,
                gpu_memory_utilization=getattr(engine, "gpu_memory_utilization", None),
                dtype=getattr(engine, "dtype", None),
                prelaunch_compute_apps=_compute_app_snapshot(),
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
            "inference_realization": inference_runtime.realization.name,
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


async def _megatron_checkpoint_resume_smoke_async(
    config: Any,
    run_logger: Any | None = None,
) -> dict[str, Any]:
    import logging
    import math

    import torch

    from .backends.megatron.remote_backend import (
        MegatronRemoteBackend,
        MegatronRemoteConfig,
        spawn_megatron_workers,
    )
    from .runtime_factory import build_megatron_lowering

    assert getattr(config.trainer, "backend", None) == "megatron", (
        "Megatron checkpoint resume smoke requires trainer.backend == 'megatron'"
    )
    if not getattr(config.checkpoint, "save_optimizer_state", True):
        raise ValueError(
            "Megatron checkpoint resume smoke requires checkpoint.save_optimizer_state=True"
        )

    logger = logging.getLogger("rollouts.training.smoke")
    logger.setLevel(logging.INFO)

    def emit(event: str, **data: Any) -> None:
        logger.info("%s %s", event, data)
        if run_logger is not None:
            emit_run_event(run_logger, event, **data)

    output = getattr(config, "output", None)
    output_root = Path(getattr(output, "output_dir", "results"))
    experiment_name = str(getattr(output, "experiment_name", "smoke_megatron_checkpoint_resume"))
    base_dir = output_root / experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)

    trainer = config.trainer
    checkpoint = config.checkpoint
    trainer_cuda_device_ids = tuple(_pick(trainer, "cuda_device_ids", default=(0, 1)))
    if len(trainer_cuda_device_ids) < 2:
        raise ValueError(
            "Megatron checkpoint resume smoke requires at least 2 trainer GPUs; "
            f"got {trainer_cuda_device_ids!r}"
        )

    training_mode = "rl"
    lowering = build_megatron_lowering(trainer, training_mode=training_mode)
    micro_batch_size = int(_pick(trainer, "micro_batch_size", default=1) or 1)
    global_batch_size = int(
        _pick(getattr(config, "rollout", None), "batch_size", default=micro_batch_size)
    )
    seq_len = int(_pick(trainer, "seq_length", default=128))
    seq_len = max(8, min(seq_len, 32))
    loss_type = str(_pick(trainer, "loss_type", default="vanilla"))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260318)

    def build_batches(num_steps: int) -> list[dict[str, torch.Tensor]]:
        batches: list[dict[str, torch.Tensor]] = []
        for _ in range(num_steps):
            input_ids = torch.randint(
                0,
                1024,
                (micro_batch_size, seq_len),
                generator=generator,
                dtype=torch.long,
            )
            batch: dict[str, torch.Tensor] = {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "loss_mask": torch.ones(micro_batch_size, seq_len, dtype=torch.float32),
                "advantages": torch.randn(
                    micro_batch_size, generator=generator, dtype=torch.float32
                ),
            }
            if loss_type in {"clipped", "masked"}:
                batch["old_logprobs"] = torch.randn(
                    micro_batch_size,
                    generator=generator,
                    dtype=torch.float32,
                )
            if loss_type == "opd":
                batch["teacher_logprobs"] = torch.randn(
                    micro_batch_size,
                    seq_len,
                    generator=generator,
                    dtype=torch.float32,
                )
            batches.append(batch)
        return batches

    def create_backend(
        *, checkpoint_dir: Path, checkpoint_path: str | None
    ) -> MegatronRemoteBackend:
        from .runtime_factory import resolve_megatron_batch_realization

        megatron_overrides = getattr(trainer, "megatron_overrides", None)
        sequence_parallel = bool(_pick(trainer, "sequence_parallel", default=False))
        if megatron_overrides is not None and megatron_overrides.sequence_parallel is not None:
            sequence_parallel = megatron_overrides.sequence_parallel
        micro_batch_size, num_microbatches = resolve_megatron_batch_realization(
            trainer,
            global_batch_size=global_batch_size,
        )
        remote_config = MegatronRemoteConfig(
            model_name=config.model.name,
            dtype=config.model.dtype,
            checkpoint_path=checkpoint_path,
            lowering=lowering,
            sequence_parallel=sequence_parallel,
            megatron_overrides=megatron_overrides,
            lr=float(_pick(trainer, "lr", default=1e-6)),
            weight_decay=float(_pick(trainer, "weight_decay", default=0.0)),
            max_grad_norm=float(_pick(trainer, "max_grad_norm", default=1.0)),
            loss_type=loss_type,
            mask_ratio_low=float(_pick(trainer, "mask_ratio_low", default=0.125)),
            mask_ratio_high=float(_pick(trainer, "mask_ratio_high", default=8.0)),
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            num_microbatches=num_microbatches,
            seq_length=seq_len,
            save_optimizer_state=bool(getattr(checkpoint, "save_optimizer_state", True)),
            master_port=int(getattr(checkpoint, "nccl_master_port", 29500)),
            inference_endpoints=[],
            cuda_device_ids=trainer_cuda_device_ids,
        )
        workers = spawn_megatron_workers(
            num_gpus=len(trainer_cuda_device_ids),
            config=remote_config,
        )
        backend = MegatronRemoteBackend(
            workers=workers,
            config=remote_config,
            checkpoint_dir=checkpoint_dir,
        )
        try:
            backend.initialize()
        except Exception:
            backend.shutdown()
            raise
        return backend

    async def run_steps(
        backend: MegatronRemoteBackend,
        batches: list[dict[str, torch.Tensor]],
        *,
        save_at_step: int | None = None,
    ) -> tuple[list[dict[str, float]], str | None]:
        metrics_history: list[dict[str, float]] = []
        checkpoint_path: str | None = None
        for step_index, batch in enumerate(batches, start=1):
            forward_metrics = await backend.forward_backward(batch).result()
            step_metrics = await backend.optim_step().result()
            merged_metrics = {
                key: float(value) if isinstance(value, (int, float)) else value
                for key, value in {**forward_metrics, **step_metrics}.items()
            }
            metrics_history.append(merged_metrics)
            emit(
                "megatron_checkpoint_resume_step",
                step=step_index,
                backend_step=int(merged_metrics.get("step", -1)),
                checkpoint_dir=str(backend.checkpoint_dir),
                loss=float(merged_metrics.get("loss", 0.0)),
                grad_norm=float(merged_metrics.get("grad_norm", 0.0)),
                lr=float(merged_metrics.get("lr", 0.0)),
            )
            if save_at_step == step_index:
                numeric_metrics = {
                    key: float(value)
                    for key, value in merged_metrics.items()
                    if isinstance(value, (int, float))
                }
                checkpoint_path = str(
                    await backend.save_checkpoint(step_index, numeric_metrics).result()
                )
                emit(
                    "megatron_checkpoint_resume_saved",
                    step=step_index,
                    checkpoint_path=checkpoint_path,
                )
        return metrics_history, checkpoint_path

    async def shutdown_backend(backend: MegatronRemoteBackend | None) -> None:
        if backend is None:
            return
        backend.shutdown()
        await trio.sleep(1.0)
        for worker in getattr(backend, "workers", ()):
            try:
                worker.close()
            except Exception:
                pass

    def assert_metrics_close(
        *,
        label: str,
        control_metrics: dict[str, float],
        resumed_metrics: dict[str, float],
        keys: tuple[str, ...],
    ) -> dict[str, float]:
        deltas: dict[str, float] = {}
        for key in keys:
            if key not in control_metrics or key not in resumed_metrics:
                raise AssertionError(
                    f"{label}: metric {key!r} missing from comparison "
                    f"(control keys={sorted(control_metrics)}, resumed keys={sorted(resumed_metrics)})"
                )
            control_value = float(control_metrics[key])
            resumed_value = float(resumed_metrics[key])
            delta = abs(control_value - resumed_value)
            deltas[key] = delta
            if key == "step":
                if int(control_value) != int(resumed_value):
                    raise AssertionError(
                        f"{label}: step mismatch control={control_value} resumed={resumed_value}"
                    )
                continue
            if not math.isclose(control_value, resumed_value, rel_tol=1e-4, abs_tol=1e-5):
                raise AssertionError(
                    f"{label}: metric {key} diverged control={control_value} resumed={resumed_value} "
                    f"(delta={delta})"
                )
        return deltas

    control_dir = base_dir / "control"
    resume_source_dir = base_dir / "resume_source"
    resume_check_dir = base_dir / "resume_check"
    for directory in (control_dir, resume_source_dir, resume_check_dir):
        directory.mkdir(parents=True, exist_ok=True)

    batches = build_batches(num_steps=4)
    compared_keys = (
        "step",
        "loss",
        "pg_loss",
        "entropy",
        "avg_logprob",
        "avg_advantage",
        "grad_norm",
        "lr",
    )

    emit(
        "megatron_checkpoint_resume_smoke_start",
        output_dir=str(base_dir),
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        loss_type=loss_type,
        trainer_cuda_device_ids=list(trainer_cuda_device_ids),
    )

    control_backend: MegatronRemoteBackend | None = None
    resume_source_backend: MegatronRemoteBackend | None = None
    resumed_backend: MegatronRemoteBackend | None = None
    try:
        control_backend = create_backend(checkpoint_dir=control_dir, checkpoint_path=None)
        control_history, _ = await run_steps(control_backend, batches)
        await shutdown_backend(control_backend)
        control_backend = None

        resume_source_backend = create_backend(
            checkpoint_dir=resume_source_dir, checkpoint_path=None
        )
        partial_history, saved_checkpoint_path = await run_steps(
            resume_source_backend,
            batches[:2],
            save_at_step=2,
        )
        assert saved_checkpoint_path is not None, "Checkpoint save did not return a path"
        await shutdown_backend(resume_source_backend)
        resume_source_backend = None

        resumed_backend = create_backend(
            checkpoint_dir=resume_check_dir,
            checkpoint_path=saved_checkpoint_path,
        )
        resumed_history, _ = await run_steps(resumed_backend, batches[2:])

        step3_deltas = assert_metrics_close(
            label="step3_after_resume",
            control_metrics=control_history[2],
            resumed_metrics=resumed_history[0],
            keys=compared_keys,
        )
        step4_deltas = assert_metrics_close(
            label="step4_after_resume",
            control_metrics=control_history[3],
            resumed_metrics=resumed_history[1],
            keys=compared_keys,
        )

        emit(
            "megatron_checkpoint_resume_smoke_ok",
            checkpoint_path=saved_checkpoint_path,
            step3_deltas=step3_deltas,
            step4_deltas=step4_deltas,
        )
        return {
            "smoke": "megatron_checkpoint_resume",
            "checkpoint_path": saved_checkpoint_path,
            "control_history": control_history,
            "partial_history": partial_history,
            "resumed_history": resumed_history,
            "step3_deltas": step3_deltas,
            "step4_deltas": step4_deltas,
        }
    finally:
        for backend in (resumed_backend, resume_source_backend, control_backend):
            await shutdown_backend(backend)


def run_megatron_checkpoint_resume_smoke(config: Any, **kwargs: Any) -> dict[str, Any]:
    """Run a Megatron save/resume witness inside one runtime.

    This proves more than "save did not crash": it requires the resumed backend
    to match uninterrupted training on the next two steps.
    """

    return trio.run(_megatron_checkpoint_resume_smoke_async, config, kwargs.get("run_logger"))
