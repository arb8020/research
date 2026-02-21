"""Tensor Parallelism support for inference.

Implements TP following mini-sglang patterns:
- Column-parallel: QKV, gate/up projections (shard output dim, no sync)
- Row-parallel: O, down projections (shard input dim, all_reduce after)

Usage:
    # Single GPU (default)
    tp = TPConfig()

    # Multi-GPU (call from each rank)
    tp = TPConfig(rank=local_rank, world_size=num_gpus)
    tp.init_process_group()

    # Use in layers
    linear = RowParallelLinear(in_features, out_features, tp=tp)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch.distributed as dist
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class TPConfig:
    """Tensor parallelism configuration.

    For single-GPU, use defaults (rank=0, world_size=1).
    For multi-GPU, each process creates with its rank.
    """

    rank: int = 0
    world_size: int = 1
    backend: str = "nccl"

    # Process group for TP communication (created by init_process_group)
    _process_group: Any = field(default=None, init=False, repr=False)

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def init_process_group(
        self,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
    ) -> None:
        """Initialize NCCL process group for TP.

        Call this once per process before using TP layers.
        For single-node multi-GPU, use default address.
        """
        if not self.is_distributed:
            return

        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", str(master_port))

        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                rank=self.rank,
                world_size=self.world_size,
            )
            logger.info(f"TP process group initialized: rank={self.rank}/{self.world_size}")

        self._process_group = dist.distributed_c10d._get_default_group()

    def all_reduce(self, tensor: Tensor) -> Tensor:
        """All-reduce tensor across TP ranks (sum).

        For single-GPU, returns tensor unchanged.
        For multi-GPU, synchronizes and sums across all ranks.
        """
        if not self.is_distributed:
            return tensor

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self._process_group)
        return tensor

    def cleanup(self) -> None:
        """Cleanup process group."""
        if self._process_group is not None and dist.is_initialized():
            dist.destroy_process_group()
            self._process_group = None


def div_up(a: int, b: int) -> int:
    """Divide and round up."""
    return (a + b - 1) // b


def shard_dim(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Get shard range for a dimension.

    Returns (start_idx, local_size) for this rank's shard.
    Handles non-even division by giving extra to earlier ranks.
    """
    base_size = total // world_size
    remainder = total % world_size

    if rank < remainder:
        local_size = base_size + 1
        start_idx = rank * (base_size + 1)
    else:
        local_size = base_size
        start_idx = remainder * (base_size + 1) + (rank - remainder) * base_size

    return start_idx, local_size


def get_local_size(total: int, world_size: int) -> int:
    """Get local shard size (assumes even division)."""
    assert total % world_size == 0, f"{total} not divisible by {world_size}"
    return total // world_size
