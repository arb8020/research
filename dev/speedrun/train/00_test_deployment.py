#!/usr/bin/env python3
"""
Quick smoke test to validate deployment infrastructure.
Tests: distributed setup, CUDA availability, GPU communication.
Should complete in <1 minute.
"""

# Check dependencies are installed
try:
    import huggingface_hub
    import numpy as np
    import torch
    import torch.distributed as dist
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nInstalled packages:")
    import subprocess

    subprocess.run(["pip", "list"])
    raise

import torch
import torch.distributed as dist


def main():
    # Initialize distributed training
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device based on local rank to avoid duplicate GPU assignment
    # torchrun sets CUDA_VISIBLE_DEVICES, so local_rank is the correct device index
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    print(f"[Rank {rank}/{world_size}] Initialized successfully")
    print(f"[Rank {rank}] Device: {device} - {torch.cuda.get_device_name(device)}")
    print(f"[Rank {rank}] CUDA version: {torch.version.cuda}")
    print(f"[Rank {rank}] PyTorch version: {torch.__version__}")

    # Test CUDA operations
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x.T
    result = y.sum().item()
    print(f"[Rank {rank}] Matrix multiply test passed: sum={result:.2f}")

    # Test distributed all-reduce
    tensor = torch.tensor([rank], dtype=torch.float32, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(world_size))

    print(f"[Rank {rank}] All-reduce test: {tensor.item():.0f} (expected: {expected_sum})")

    if rank == 0:
        if abs(tensor.item() - expected_sum) < 0.01:
            print("\n" + "=" * 60)
            print("✓ All tests passed! Deployment infrastructure is working.")
            print("=" * 60)
        else:
            print("\n✗ All-reduce test failed!")
            raise RuntimeError("Distributed communication test failed")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
