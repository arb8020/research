"""Megatron process group initialization.

Sets up the distributed process groups for tensor/pipeline/expert parallelism.
Must be called once per process before any Megatron operations.

Based on SLIME's initialize.py but simplified - we just call Megatron's APIs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MegatronParallelismConfig:
    """Parallelism configuration for Megatron.

    Megatron supports multiple parallelism dimensions:
    - Tensor Parallel (TP): Split attention/MLP across GPUs
    - Pipeline Parallel (PP): Split layers across GPUs
    - Expert Parallel (EP): Distribute MoE experts
    - Data Parallel (DP): Replicate model (automatic from world_size / TP / PP)

    Example for 16 GPUs with TP=4, PP=2:
        - 4 GPUs share each transformer layer (TP)
        - 2 pipeline stages (PP)
        - DP = 16 / (4 * 2) = 2 data parallel replicas
    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    sequence_parallel: bool = False

    # Virtual pipeline parallelism (interleaved schedule)
    virtual_pipeline_model_parallel_size: int | None = None

    def __post_init__(self) -> None:
        assert self.tensor_parallel_size >= 1
        assert self.pipeline_parallel_size >= 1
        assert self.expert_parallel_size >= 1


def init_megatron(
    rank: int,
    world_size: int,
    config: MegatronParallelismConfig,
    *,
    master_addr: str | None = None,
    master_port: int | None = None,
) -> None:
    """Initialize Megatron distributed process groups.

    Must be called once per process before any Megatron model operations.
    Sets up NCCL process groups for TP, PP, EP, and DP.

    Args:
        rank: Global rank of this process (0 to world_size-1)
        world_size: Total number of processes
        config: Parallelism configuration
        master_addr: Master address (defaults to env MASTER_ADDR)
        master_port: Master port (defaults to env MASTER_PORT)

    Raises:
        ImportError: If Megatron-Core is not installed
        AssertionError: If world_size is not divisible by parallelism config
    """
    # Validate parallelism config fits world size
    model_parallel_size = config.tensor_parallel_size * config.pipeline_parallel_size
    assert world_size % model_parallel_size == 0, (
        f"world_size ({world_size}) must be divisible by "
        f"TP ({config.tensor_parallel_size}) * PP ({config.pipeline_parallel_size})"
    )

    # Set environment variables if provided
    if master_addr is not None:
        os.environ["MASTER_ADDR"] = master_addr
    if master_port is not None:
        os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    logger.info(
        "Initializing Megatron: rank=%d/%d, TP=%d, PP=%d, EP=%d",
        rank,
        world_size,
        config.tensor_parallel_size,
        config.pipeline_parallel_size,
        config.expert_parallel_size,
    )

    try:
        import torch.distributed as dist
        from megatron.core import mpu
    except ImportError as e:
        raise ImportError(
            "Megatron-Core is required. Install from: "
            "pip install megatron-core or "
            "pip install git+https://github.com/NVIDIA/Megatron-LM.git"
        ) from e

    # Initialize torch.distributed first
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    # Initialize Megatron's model parallel groups
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=config.tensor_parallel_size,
        pipeline_model_parallel_size=config.pipeline_parallel_size,
        virtual_pipeline_model_parallel_size=config.virtual_pipeline_model_parallel_size,
        expert_model_parallel_size=config.expert_parallel_size,
    )

    # Log the resulting ranks
    logger.info(
        "Megatron initialized: TP rank=%d, PP rank=%d, DP rank=%d",
        mpu.get_tensor_model_parallel_rank(),
        mpu.get_pipeline_model_parallel_rank(),
        mpu.get_data_parallel_rank(),
    )


def destroy_megatron() -> None:
    """Clean up Megatron process groups.

    Call this before process exit for clean shutdown.
    """
    try:
        import torch.distributed as dist
        from megatron.core import mpu

        mpu.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
    except ImportError:
        pass  # Megatron not installed, nothing to clean up
