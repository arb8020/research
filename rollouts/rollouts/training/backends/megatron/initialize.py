"""Megatron process group initialization.

Sets up the distributed process groups for tensor/pipeline/expert parallelism.
Must be called once per process before any Megatron operations.

Based on SLIME's initialize.py but simplified - we just call Megatron's APIs.
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
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


def _build_megatron_args(
    rank: int,
    world_size: int,
    config: MegatronParallelismConfig,
    *,
    master_addr: str,
    master_port: int,
    global_batch_size: int,
    micro_batch_size: int,
    seq_length: int,
) -> Namespace:
    """Build a Megatron argument namespace before model construction."""
    try:
        from megatron.training.arguments import parse_args

        try:
            args = parse_args()
        except TypeError:
            # Some Megatron versions use a callback hook for parser extension.
            args = parse_args(lambda parser: parser)  # type: ignore[call-arg]
    except Exception:
        # Conservative fallback if parser integration is unavailable.
        args = Namespace()

    model_parallel_size = (
        config.tensor_parallel_size * config.pipeline_parallel_size * config.expert_parallel_size
    )
    data_parallel_size = max(1, world_size // max(1, model_parallel_size))

    args.rank = rank
    args.local_rank = rank
    args.world_size = world_size
    args.tensor_model_parallel_size = config.tensor_parallel_size
    args.pipeline_model_parallel_size = config.pipeline_parallel_size
    args.expert_model_parallel_size = config.expert_parallel_size
    args.virtual_pipeline_model_parallel_size = config.virtual_pipeline_model_parallel_size
    args.sequence_parallel = config.sequence_parallel
    args.data_parallel_size = data_parallel_size

    args.master_addr = master_addr
    args.master_port = master_port
    args.distributed_backend = getattr(args, "distributed_backend", "nccl")

    args.micro_batch_size = micro_batch_size
    args.global_batch_size = global_batch_size
    args.seq_length = seq_length
    args.max_position_embeddings = seq_length

    args.fp16 = getattr(args, "fp16", False)
    args.bf16 = getattr(args, "bf16", True)
    args.use_distributed_optimizer = getattr(args, "use_distributed_optimizer", True)
    args.overlap_grad_reduce = getattr(args, "overlap_grad_reduce", False)
    args.overlap_param_gather = getattr(args, "overlap_param_gather", False)
    args.accumulate_allreduce_grads_in_fp32 = getattr(
        args, "accumulate_allreduce_grads_in_fp32", False
    )

    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = 0
    if getattr(args, "vocab_size", None) is None:
        args.vocab_size = 0
    if getattr(args, "num_layers", None) is None:
        args.num_layers = 1
    if getattr(args, "seed", None) is None:
        args.seed = 0

    return args


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
        from megatron.training.global_vars import set_args
    except ImportError as e:
        raise ImportError(
            "Megatron-Core is required. Install from: "
            "pip install megatron-core or "
            "pip install git+https://github.com/NVIDIA/Megatron-LM.git"
        ) from e

    master_addr = os.environ.get("MASTER_ADDR", master_addr or "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", str(master_port) if master_port else "29500"))

    # Set environment vars first for dist init and to keep worker behavior consistent.
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    # Register Megatron args in global state before calling get_model.
    megatron_args = _build_megatron_args(
        rank=rank,
        world_size=world_size,
        config=config,
        master_addr=master_addr,
        master_port=master_port,
        global_batch_size=max(1, world_size),
        micro_batch_size=1,
        seq_length=4096,
    )
    set_args(megatron_args)

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
