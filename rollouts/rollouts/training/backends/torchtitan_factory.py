"""Convenience factory for TorchTitan backends."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ...training.contract_witnesses import rl_contract_loss, supervised_contract_loss
from ...training.lowering import (
    ParallelIntent,
    RealizationPlan,
    TorchTitanLowering,
    dense_rl_realization,
    dense_supervised_realization,
)
from .torchtitan_backend import TorchTitanBackend, TorchTitanConfig


def _find_free_port(start_port: int, max_attempts: int = 100) -> int:
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts}")


def ensure_single_rank_torchtitan_dist(
    *,
    gpu_rank: int,
    port_start: int = 29600,
) -> callable | None:
    """Initialize a single-rank NCCL process group if needed."""
    torch.cuda.set_device(gpu_rank)

    if dist.is_initialized():
        return None

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = _find_free_port(port_start)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=0,
        world_size=1,
    )

    def _cleanup_dist() -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    return _cleanup_dist


def create_torchtitan_backend(
    *,
    checkpoint_dir: Path,
    hf_checkpoint: str,
    torchtitan_model: str,
    torchtitan_model_size: str,
    gpu_rank: int = 0,
    seq_len: int = 4096,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    tp: int = 1,
    cp: int = 1,
    pp: int = 1,
    enable_loss_parallel: bool = True,
    packed_sequences: bool = True,
    mode: str = "supervised",
    realization: RealizationPlan | None = None,
    lowering: TorchTitanLowering | None = None,
) -> tuple[TorchTitanBackend, callable | None]:
    """Create a TorchTitan backend plus optional dist cleanup."""
    # Import GLM to register with torchtitan when requested.
    if torchtitan_model == "glm":
        from ..models import glm  # noqa: F401

    cleanup = ensure_single_rank_torchtitan_dist(gpu_rank=gpu_rank)

    config = TorchTitanConfig(
        tp_degree=tp,
        cp_degree=cp,
        pp_degree=pp,
        seq_len=seq_len,
        lr=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
    )

    if mode == "supervised":
        loss_fn: Any = supervised_contract_loss
        default_realization = realization or dense_supervised_realization(
            tp=tp,
            cp=cp,
            pp=pp,
            packed_sequences=packed_sequences,
        )
    elif mode == "rl":
        loss_fn = rl_contract_loss
        default_realization = realization or dense_rl_realization(
            tp=tp,
            cp=cp,
            pp=pp,
            packed_sequences=packed_sequences,
        )
    else:
        raise ValueError(f"Unknown TorchTitan factory mode: {mode!r}")

    if lowering is None:
        lowering = TorchTitanLowering.from_realization(
            parallel=ParallelIntent(
                dp=1,
                tp=tp,
                cp=cp,
                pp=pp,
                ep=1,
                enable_loss_parallel=enable_loss_parallel,
                packed_sequences=packed_sequences,
            ),
            realization=default_realization,
        )

    backend = TorchTitanBackend(
        model_name=torchtitan_model,
        model_size=torchtitan_model_size,
        checkpoint_dir=checkpoint_dir,
        loss_fn=loss_fn,
        config=config,
        lowering=lowering,
        hf_checkpoint=hf_checkpoint,
    )
    return backend, cleanup
