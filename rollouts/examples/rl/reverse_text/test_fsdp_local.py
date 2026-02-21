#!/usr/bin/env python
"""Test FSDP training locally on a single node.

This script tests the FSDP training pipeline without multi-node provisioning.
It simulates a 2-GPU setup: 1 inference + 1 trainer.

Run with:
    # Requires 2+ GPUs
    python examples/rl/reverse_text/test_fsdp_local.py

    # Dry run (no GPU required)
    python examples/rl/reverse_text/test_fsdp_local.py --dry-run
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def check_gpus() -> int:
    """Check available GPUs."""
    try:
        import torch

        count = torch.cuda.device_count()
        logger.info(f"Found {count} GPUs")
        return count
    except Exception as e:
        logger.warning(f"Could not check GPUs: {e}")
        return 0


def launch_inference_engine(
    model_name: str,
    port: int,
    gpu_id: int,
) -> subprocess.Popen:
    """Launch SGLang inference engine."""
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_name,
        "--port",
        str(port),
        "--mem-fraction-static",
        "0.8",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info(f"Launching inference engine on GPU {gpu_id}, port {port}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


async def wait_for_inference(port: int, timeout: float = 120.0) -> bool:
    """Wait for inference engine to be ready."""
    import httpx

    url = f"http://localhost:{port}/health"
    start = time.time()

    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() - start < timeout:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    logger.info(f"Inference engine ready on port {port}")
                    return True
            except Exception:
                pass
            await asyncio.sleep(1.0)

    logger.error(f"Inference engine failed to start on port {port}")
    return False


def launch_fsdp_trainer(
    config_path: str,
    gpu_id: int,
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: int = 29500,
    inference_endpoints: str = "",
) -> subprocess.Popen:
    """Launch FSDP trainer process."""
    cmd = [
        sys.executable,
        "-m",
        "rollouts.training.fsdp_worker",
        "--config",
        config_path,
        "--is-rank-0",
        "1" if rank == 0 else "0",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = str(master_port)
    env["WORLD_SIZE"] = str(world_size)
    env["RANK"] = str(rank)
    env["LOCAL_RANK"] = "0"
    env["INFERENCE_ENDPOINTS"] = inference_endpoints

    logger.info(f"Launching FSDP rank {rank} on GPU {gpu_id}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


async def run_local_fsdp_test(
    model_name: str = "Qwen/Qwen3-0.6B",
    num_steps: int = 5,
    inference_gpu: int = 0,
    trainer_gpu: int = 1,
) -> dict:
    """Run a local FSDP training test.

    Args:
        model_name: Model to use
        num_steps: Training steps
        inference_gpu: GPU for inference
        trainer_gpu: GPU for training

    Returns:
        Test results
    """
    from rollouts.training.grpo import (
        CheckpointConfig,
        GRPOConfig,
        GRPOOutputConfig,
        InferenceConfig,
        ModelConfig,
        RolloutConfig,
        TrainerConfig,
    )

    logger.info("=" * 60)
    logger.info("Local FSDP Test")
    logger.info("=" * 60)

    # Create config
    config = GRPOConfig(
        output=GRPOOutputConfig(experiment_name="fsdp_local_test"),
        model=ModelConfig(name=model_name),
        checkpoint=CheckpointConfig(
            num_steps=num_steps,
            checkpoint_every=num_steps,
            sync_weights_every=1,
            pipeline_mode="true_pipeline",
            weight_sync_mode="nccl",
        ),
        rollout=RolloutConfig(
            batch_size=4,
            n_samples_per_prompt=4,
            max_tokens=64,
        ),
        trainer=TrainerConfig(
            cuda_device_ids=(trainer_gpu,),
            lr=1e-6,
            num_minibatches=4,
        ),
        inference=InferenceConfig(
            cuda_device_ids=(inference_gpu,),
            port=30000,
        ),
    )

    # Save config
    output_dir = Path("results/fsdp_local_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    config.save(config_path)
    logger.info(f"Saved config to {config_path}")

    # Launch inference engine
    inference_proc = launch_inference_engine(
        model_name=model_name,
        port=30000,
        gpu_id=inference_gpu,
    )

    try:
        # Wait for inference to be ready
        if not await wait_for_inference(30000):
            raise RuntimeError("Inference engine failed to start")

        # Launch trainer (single-GPU FSDP for testing)
        inference_endpoint = "http://localhost:30000"
        trainer_proc = launch_fsdp_trainer(
            config_path=str(config_path),
            gpu_id=trainer_gpu,
            rank=0,
            world_size=1,
            inference_endpoints=inference_endpoint,
        )

        # Wait for trainer to complete
        logger.info("Waiting for trainer to complete...")
        stdout, _ = trainer_proc.communicate(timeout=300)
        logger.info(f"Trainer output:\n{stdout.decode()}")

        return_code = trainer_proc.returncode
        if return_code != 0:
            logger.error(f"Trainer failed with exit code {return_code}")
            return {"success": False, "error": f"exit code {return_code}"}

        logger.info("Training completed successfully!")
        return {"success": True}

    finally:
        # Cleanup
        logger.info("Cleaning up...")
        inference_proc.terminate()
        try:
            inference_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            inference_proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local FSDP test")
    parser.add_argument("--dry-run", action="store_true", help="Just print config, don't run")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--steps", type=int, default=5, help="Training steps")
    args = parser.parse_args()

    num_gpus = check_gpus()

    if args.dry_run:
        logger.info("Dry run - would test FSDP training with:")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Steps: {args.steps}")
        logger.info(f"  GPUs available: {num_gpus}")
        return

    if num_gpus < 2:
        logger.error(f"Need at least 2 GPUs, found {num_gpus}")
        sys.exit(1)

    result = asyncio.run(
        run_local_fsdp_test(
            model_name=args.model,
            num_steps=args.steps,
        )
    )

    if not result.get("success"):
        logger.error(f"Test failed: {result.get('error')}")
        sys.exit(1)

    logger.info("Test passed!")


if __name__ == "__main__":
    main()
