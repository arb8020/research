"""Distributed training utilities (SLIME-inspired).

Provides process group management and distributed communication utilities
for multi-GPU training with FSDP and Megatron backends.

Tiger Style: Explicit state management, clear error messages.
Casey Muratori: Minimal coupling, explicit operations.
"""

from datetime import timedelta

import torch
import torch.distributed as dist

try:
    from torch.distributed.distributed_c10d import (
        Store,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
    )
except ImportError:
    # Fallback for older PyTorch versions
    from torch.distributed import (
        Store,
        default_pg_timeout,
    )

    _new_process_group_helper = None
    _world = None


# Global Gloo group for CPU communication (when needed)
GLOO_GROUP = None


def init_process_group(
    backend: str = "nccl",
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str | None = None,
) -> None:
    """Initialize distributed process group.

    Args:
        backend: Backend to use ("nccl" for GPU, "gloo" for CPU)
        init_method: Initialization method (e.g., "env://", "tcp://...")
        timeout: Timeout for operations (default: 30 minutes)
        world_size: Total number of processes (-1 to read from env)
        rank: Rank of this process (-1 to read from env)
        store: Optional store for rendezvous
        group_name: Optional group name

    Side effects:
        - Initializes torch.distributed
        - Sets up process group for multi-GPU training

    Example:
        >>> # Single-node multi-GPU (reads from torchrun env vars)
        >>> init_process_group(backend="nccl")
        >>>
        >>> # Multi-node with explicit parameters
        >>> init_process_group(
        ...     backend="nccl",
        ...     init_method="tcp://10.0.0.1:23456",
        ...     world_size=8,
        ...     rank=0,
        ... )
    """
    # Tiger Style: Assert preconditions
    if init_method is None:
        init_method = "env://"

    if timeout is None:
        timeout = default_pg_timeout

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        timeout=timeout,
        world_size=world_size,
        rank=rank,
    )

    # Tiger Style: Assert postconditions
    assert dist.is_initialized(), "Failed to initialize distributed process group"
    assert dist.get_world_size() > 0, "World size must be > 0"
    assert dist.get_rank() >= 0, "Rank must be >= 0"


def init_gloo_group() -> dist.ProcessGroup:
    """Initialize Gloo group for CPU-based distributed communication.

    Useful for operations that can't run on GPU or need CPU fallback.

    Returns:
        Gloo process group

    Side effects:
        - Creates global Gloo process group
        - Stores in GLOO_GROUP module variable

    Example:
        >>> # After initializing main NCCL process group
        >>> init_process_group(backend="nccl")
        >>> gloo_group = init_gloo_group()
        >>> # Use for CPU operations
        >>> dist.all_reduce(cpu_tensor, group=gloo_group)
    """
    global GLOO_GROUP

    # Tiger Style: Check if already initialized
    if GLOO_GROUP is not None:
        return GLOO_GROUP

    # Must have main process group initialized first
    assert dist.is_initialized(), "Must call init_process_group() before init_gloo_group()"

    # Create Gloo group with all ranks
    GLOO_GROUP = dist.new_group(backend="gloo")

    return GLOO_GROUP


def get_gloo_group() -> dist.ProcessGroup:
    """Get the Gloo group for CPU communication.

    Returns:
        Gloo process group

    Raises:
        RuntimeError: If Gloo group not initialized

    Example:
        >>> gloo_group = get_gloo_group()
        >>> dist.barrier(group=gloo_group)
    """
    global GLOO_GROUP

    if GLOO_GROUP is None:
        raise RuntimeError("Gloo group has not been initialized. Call init_gloo_group() first.")

    return GLOO_GROUP


def get_rank() -> int:
    """Get rank of current process.

    Returns:
        Rank (0 to world_size-1), or 0 if not using distributed

    Example:
        >>> rank = get_rank()
        >>> if rank == 0:
        ...     print("I am the main process")
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes.

    Returns:
        World size, or 1 if not using distributed

    Example:
        >>> world_size = get_world_size()
        >>> local_batch_size = global_batch_size // world_size
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0).

    Returns:
        True if rank 0 or not distributed, False otherwise

    Example:
        >>> if is_main_process():
        ...     print("Logging metrics...")
        ...     wandb.log(metrics)
    """
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes.

    All processes wait at this point until everyone arrives.

    Example:
        >>> # Process 0 saves checkpoint
        >>> if is_main_process():
        ...     torch.save(model.state_dict(), "checkpoint.pt")
        >>> # Wait for checkpoint to be saved
        >>> barrier()
        >>> # Now all processes can safely load it
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """All-reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce (modified in-place)
        op: Reduction operation (SUM, AVG, MIN, MAX, etc.)
        group: Process group (None = default group)

    Returns:
        The same tensor (modified in-place)

    Example:
        >>> # Compute global average loss
        >>> local_loss = torch.tensor([loss_value], device="cuda")
        >>> all_reduce(local_loss, op=dist.ReduceOp.AVG)
        >>> global_loss = local_loss.item()
    """
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op, group=group)
    return tensor


def all_gather(
    tensor_list: list[torch.Tensor],
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> None:
    """Gather tensors from all processes into a list.

    Args:
        tensor_list: Output list (must have world_size elements)
        tensor: Input tensor to gather
        group: Process group (None = default group)

    Side effects:
        - Modifies tensor_list in-place

    Example:
        >>> # Gather batch sizes from all ranks
        >>> local_batch_size = torch.tensor([len(batch)], device="cuda")
        >>> world_size = get_world_size()
        >>> all_batch_sizes = [torch.zeros_like(local_batch_size) for _ in range(world_size)]
        >>> all_gather(all_batch_sizes, local_batch_size)
    """
    if dist.is_available() and dist.is_initialized():
        dist.all_gather(tensor_list, tensor, group=group)
    else:
        # Single process case
        tensor_list[0] = tensor


def broadcast(
    tensor: torch.Tensor,
    src: int = 0,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Broadcast tensor from src rank to all other ranks.

    Args:
        tensor: Tensor to broadcast (modified in-place for non-src ranks)
        src: Source rank (default: 0)
        group: Process group (None = default group)

    Returns:
        The same tensor (possibly modified)

    Example:
        >>> # Rank 0 decides on learning rate
        >>> if is_main_process():
        ...     lr_tensor = torch.tensor([new_lr], device="cuda")
        >>> else:
        ...     lr_tensor = torch.zeros(1, device="cuda")
        >>> broadcast(lr_tensor, src=0)
        >>> new_lr = lr_tensor.item()
    """
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src=src, group=group)
    return tensor


def distributed_masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    process_group: dist.ProcessGroup | None = None,
    shift_mean: bool = True,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Whiten values using global statistics across all GPUs (SLIME-inspired).

    Useful for advantage normalization in RL training.

    Args:
        values: Local tensor of values to whiten [...]
        mask: Local mask for valid values [...] (same shape as values)
        process_group: Process group for all_reduce (None = default)
        shift_mean: If True, output is zero-mean (default: True)
        epsilon: Small value for numerical stability

    Returns:
        Whitened values (same shape as input)

    Example:
        >>> # Normalize advantages across all GPUs
        >>> advantages = torch.randn(32, 128, device="cuda")
        >>> loss_mask = torch.ones_like(advantages)
        >>> whitened_advantages = distributed_masked_whiten(
        ...     advantages, loss_mask
        ... )
    """
    # Calculate local statistics
    local_sum = (values * mask).sum()
    local_sum_sq = ((values**2) * mask).sum()
    local_mask_sum = mask.sum()

    # Aggregate statistics across all GPUs
    stats_tensor = torch.tensor(
        [local_sum, local_sum_sq, local_mask_sum],
        device=values.device,
        dtype=torch.float32,
    )

    all_reduce(stats_tensor, op=dist.ReduceOp.SUM, group=process_group)

    # Compute global statistics
    global_sum, global_sum_sq, global_mask_sum = stats_tensor

    # Tiger Style: Assert valid statistics
    assert global_mask_sum.item() > 0, "Global mask sum is zero (no valid values)"

    global_mean = global_sum / global_mask_sum
    global_mean_sq = global_sum_sq / global_mask_sum
    global_var = global_mean_sq - global_mean**2

    # Bessel's correction for unbiased variance estimate
    if global_mask_sum.item() >= 2:
        bessel_correction = global_mask_sum / (global_mask_sum - 1)
        global_var = global_var * bessel_correction

    # Whiten using global statistics
    whitened_values = (values - global_mean) * torch.rsqrt(global_var + epsilon)

    if not shift_mean:
        whitened_values += global_mean

    return whitened_values


def cleanup() -> None:
    """Clean up distributed process group.

    Call this at the end of training to properly shut down.

    Example:
        >>> try:
        ...     run_training()
        ... finally:
        ...     cleanup()
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# Convenience functions for logging
def print_rank_0(message: str) -> None:
    """Print message only from rank 0.

    Args:
        message: Message to print

    Example:
        >>> print_rank_0("Starting training...")
    """
    if is_main_process():
        print(message)
