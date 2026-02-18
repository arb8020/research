"""Runtime initialization for distributed training.

Handles device assignment, distributed setup, and seeds.
Seamlessly supports single GPU, single-node multi-GPU, and multi-node training.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init(seed: int = 42) -> tuple[int, int, torch.device]:
    """Initialize runtime for training. Returns (rank, world_size, device).

    Handles:
    - Device assignment (LOCAL_RANK env var from torchrun)
    - Distributed init (automatic when WORLD_SIZE > 1)
    - Seeds and TF32

    Works seamlessly for:
    - Single GPU: rank=0, world=1
    - Single-node multi-GPU: torchrun sets LOCAL_RANK, init NCCL
    - Multi-node: same as single-node, world > local_world

    Usage:
        rank, world, device = runtime.init(seed=42)
        # rank: global rank (0 to world-1)
        # world: total number of processes
        # device: torch.device for this process
    """
    if not torch.cuda.is_available():
        # CPU fallback for local testing
        torch.manual_seed(seed)
        return 0, 1, torch.device("cpu")

    # TF32 for speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Device assignment (torchrun sets LOCAL_RANK; single-process defaults to 0)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Distributed init (only when launched under torchrun with WORLD_SIZE > 1)
    world_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_env > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")

    # Get rank and world (or default to single GPU)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1

    # Seeds (offset by rank for data parallelism diversity)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)

    return rank, world, device


def is_main() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def all_reduce_grads(weights: dict[str, torch.Tensor]) -> None:
    """All-reduce gradients across processes.

    For functional weights (dict of tensors), we manually all-reduce
    instead of using DDP wrapper.
    """
    if not dist.is_initialized():
        return

    world = dist.get_world_size()
    for param in weights.values():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world)


def finalize() -> None:
    """Cleanup distributed state."""
    if dist.is_initialized():
        dist.destroy_process_group()
