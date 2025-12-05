"""GPU NUMA affinity utilities for optimal memory bandwidth.

Sets CPU affinity to match GPU NUMA node for better memory performance.
Based on implementations from Slime and VERL.

Tiger Style: Graceful fallback, clear error messages.
"""

import os


def set_gpu_affinity(local_rank: int = None, verbose: bool = True) -> bool:
    """Set CPU affinity to GPU's NUMA node for optimal memory bandwidth.

    This improves performance by ensuring CPU cores are on the same NUMA
    node as the GPU, reducing memory latency.

    Args:
        local_rank: GPU rank on this node (0-7). If None, reads from LOCAL_RANK env var.
        verbose: Print status messages

    Returns:
        True if affinity was set, False if skipped/failed

    Example:
        >>> # In your work function
        >>> def train_worker(handle, rank, world_size):
        ...     # Set GPU affinity before torch.cuda operations
        ...     set_gpu_affinity(local_rank=rank)
        ...
        ...     import torch
        ...     torch.cuda.set_device(rank)
        ...     # ... training ...

    Note:
        - Requires pynvml library: pip install nvidia-ml-py
        - Silently skips on ROCm/HIP (AMD GPUs)
        - Gracefully falls back if pynvml unavailable
    """
    # Get local rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Tiger Style: Assert preconditions
    assert isinstance(local_rank, int), f"local_rank must be int, got {type(local_rank)}"
    assert local_rank >= 0, f"local_rank must be >= 0, got {local_rank}"
    assert local_rank < 16, f"local_rank too large: {local_rank} (max 16)"

    try:
        # Check for ROCm/HIP (AMD GPUs)
        import torch
        if torch.version.hip is not None:
            if verbose:
                print("[MiniRay] ROCm/HIP detected, skipping NUMA affinity")
            return False

        # Import pynvml
        import pynvml

        # Initialize NVML
        pynvml.nvmlInit()

        try:
            # Get GPU handle
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

            # Set CPU affinity to GPU's NUMA node
            pynvml.nvmlDeviceSetCpuAffinity(handle)

            if verbose:
                print(f"[MiniRay] Set NUMA affinity for GPU {local_rank}")

            return True

        finally:
            # Always shutdown NVML
            pynvml.nvmlShutdown()

    except ImportError as e:
        if verbose:
            print(f"[MiniRay] pynvml not available, skipping NUMA affinity: {e}")
            print("[MiniRay] Install with: pip install nvidia-ml-py")
        return False

    except Exception as e:
        if verbose:
            print(f"[MiniRay] Failed to set NUMA affinity: {e}")
        return False


def get_gpu_numa_node(local_rank: int = 0) -> int:
    """Get NUMA node for a GPU (for debugging).

    Args:
        local_rank: GPU rank on this node

    Returns:
        NUMA node ID (0-7)

    Example:
        >>> numa_node = get_gpu_numa_node(local_rank=0)
        >>> print(f"GPU 0 is on NUMA node {numa_node}")
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            # Get CPU affinity mask
            affinity = pynvml.nvmlDeviceGetCpuAffinity(handle, 1)
            return affinity[0]
        finally:
            pynvml.nvmlShutdown()

    except Exception as e:
        print(f"Failed to get NUMA node: {e}")
        return -1


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    """Test GPU affinity"""
    import sys

    if len(sys.argv) > 1:
        local_rank = int(sys.argv[1])
    else:
        local_rank = 0

    print(f"Testing GPU affinity for local_rank={local_rank}")
    success = set_gpu_affinity(local_rank=local_rank, verbose=True)

    if success:
        print("✅ Successfully set NUMA affinity")
        numa_node = get_gpu_numa_node(local_rank)
        print(f"GPU {local_rank} NUMA node: {numa_node}")
    else:
        print("❌ Failed to set NUMA affinity (see above)")
