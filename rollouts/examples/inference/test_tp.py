"""Test tensor parallelism for engine_v2.

Run remotely with 2 GPUs:
    /Users/chiraagbalu/research/.venv/bin/python examples/inference/test_tp.py --provision --provider runpod

Run locally (if you have 2 GPUs):
    torchrun --standalone --nproc_per_node=2 examples/inference/test_tp.py --local

The test:
1. Launches 2 processes (one per GPU) via torchrun
2. Each creates engine with tp_rank=0/1, tp_size=2
3. Generates text and verifies outputs match across ranks
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from examples.rl.base_config import default_remote_training_deps
from rollouts.training.configs import HardwareConfig

# =============================================================================
# Hardware Configuration (for remote provisioning)
# =============================================================================

hardware = HardwareConfig(
    gpu_type="RTX 4090",  # Cheap 2x GPU option
    gpu_count=2,
    provider="runpod",
    deps=default_remote_training_deps(),
    hf_cache_dir="/workspace/.cache/huggingface",
)


@dataclass
class TPTestConfig:
    """Minimal config for TP test."""

    model_path: str = "Qwen/Qwen2.5-0.5B"
    prompt: str = "The capital of France is"
    max_tokens: int = 20


# Export config (required by rollouts.run)
config = TPTestConfig()

# =============================================================================
# Test Implementation
# =============================================================================


def _run_tp_test() -> None:
    """Actual TP test - runs inside torchrun."""
    import torch
    import torch.distributed as dist

    def init_distributed() -> tuple[int, int, torch.device]:
        """Initialize distributed environment from torchrun env vars."""
        if not torch.cuda.is_available():
            return 0, 1, torch.device("cpu")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")

        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1

        return rank, world, device

    rank, world_size, device = init_distributed()
    print(f"[Rank {rank}/{world_size}] Device: {device}")

    from rollouts.inference.engine_v2 import EngineConfig, InferenceEngineV2

    # Use a small model for testing
    model_path = "Qwen/Qwen2.5-0.5B"

    config = EngineConfig(
        model_path=model_path,
        tp_rank=rank,
        tp_size=world_size,
        max_batch_size=4,
        max_seq_len=256,
        enable_cuda_graphs=False,
        enable_overlap=False,
        enable_radix_cache=False,
    )

    print(f"[Rank {rank}] Creating engine...")
    engine = InferenceEngineV2(config)

    # Use torchrun's process group for TP
    if world_size > 1 and engine.tp is not None:
        engine.tp._process_group = dist.distributed_c10d._get_default_group()
        print(f"[Rank {rank}] TP group initialized")

    # Test prompt
    prompt = "The capital of France is"
    print(f"[Rank {rank}] Generating for prompt: {prompt!r}")

    from rollouts.inference.core import SamplingParams

    results = engine.generate(
        [prompt],
        sampling_params=SamplingParams(max_tokens=20),
    )

    output_tokens = results[0].input_ids.tolist()
    output_text = engine.tokenizer.decode(output_tokens)

    print(f"[Rank {rank}] Output: {output_text!r}")
    print(f"[Rank {rank}] Tokens: {output_tokens}")

    # Verify all ranks produce identical output
    if world_size > 1:
        output_tensor = torch.tensor(output_tokens, device=device, dtype=torch.long)
        max_len = 256
        padded = torch.zeros(max_len, device=device, dtype=torch.long)
        padded[: len(output_tokens)] = output_tensor

        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)

        if rank == 0:
            for i, g in enumerate(gathered):
                tokens = g[g != 0].tolist()
                print(f"[Rank 0] Rank {i} produced: {tokens}")

            all_same = all(torch.equal(gathered[0], gathered[i]) for i in range(1, world_size))
            if all_same:
                print("[Rank 0] ✓ All ranks produced identical output!")
            else:
                print("[Rank 0] ✗ Ranks produced different outputs!")
                sys.exit(1)

    engine.shutdown()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    print(f"[Rank {rank}] Done")


def train(num_samples: int = 1, **kwargs: object) -> dict:
    """Entry point called by rollouts.run on remote GPU.

    This launches torchrun to spawn 2 processes for TP test.
    """
    # Get number of GPUs
    import torch

    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPUs")

    if gpu_count < 2:
        print("ERROR: Need at least 2 GPUs for TP test")
        return {"success": False, "error": "need 2+ GPUs"}

    # Launch with torchrun
    script_path = Path(__file__).resolve()
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        str(script_path),
        "--run-test",  # Flag to run actual test
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    success = result.returncode == 0
    return {"success": success, "returncode": result.returncode}


if __name__ == "__main__":
    if "--run-test" in sys.argv:
        # Called by torchrun - run the actual test
        _run_tp_test()
    elif "--local" in sys.argv:
        # Run locally with torchrun
        train()
    else:
        # Use rollouts.run infrastructure
        from rollouts.run import main

        sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
        main()
