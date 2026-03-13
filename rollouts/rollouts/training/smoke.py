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
