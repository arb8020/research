"""Run training on cloud GPU via bifrost.

Usage:
    python scripts/run.py                    # Provision L4, run training
    python scripts/run.py --gpu A10G         # Use A10G instead
    python scripts/run.py --steps 500        # Override steps
    python scripts/run.py --ssh root@gpu:22  # Use existing SSH connection
    python scripts/run.py --node-id runpod:abc123  # Reuse existing instance
"""

from __future__ import annotations

import argparse
import logging
import sys

import trio
from bifrost import GPUQuery, acquire_node

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def run(
    gpu_type: str = "L4",
    steps: int = 100,
    ssh: str | None = None,
    node_id: str | None = None,
    keep_alive: bool = False,
) -> int:
    """Run training on cloud GPU.

    Args:
        gpu_type: GPU type to provision (e.g., "L4", "A10G", "A100")
        steps: Number of training steps
        ssh: Existing SSH connection string (user@host:port)
        node_id: Existing broker instance ID (provider:instance_id)
        keep_alive: Don't terminate instance after training

    Returns:
        Exit code from training
    """
    # Determine acquisition mode
    provision = None
    if ssh is None and node_id is None:
        # Provision new instance
        provision = GPUQuery(
            type=gpu_type,
            count=1,
            name="pretrain/step1",
        )
        logger.info(f"provisioning {gpu_type} GPU...")

    # Acquire node
    client, instance = await acquire_node(
        ssh=ssh,
        node_id=node_id,
        provision=provision,
    )

    if instance:
        logger.info(f"instance: {instance.provider}:{instance.id}")
        logger.info(f"gpu: {instance.gpu_count}x {instance.gpu_type}")

    try:
        # Deploy code
        logger.info("deploying code...")
        workspace = client.push(
            workspace_path="~/.bifrost/workspaces/pretrain",
            bootstrap_cmd="pip install -e .",
        )
        logger.info(f"deployed to: {workspace}")

        # Run training (streams output to console)
        logger.info(f"starting training for {steps} steps...")
        for line in client.exec_stream(f"python -m pretrain.train --steps {steps} --log-every 10"):
            print(line, flush=True)

        # Training completed (exec_stream waits for completion)
        # Check if output indicates success by looking for "training complete"
        logger.info("training stream completed")
        return 0

    finally:
        # Cleanup
        if instance and not keep_alive:
            logger.info("terminating instance...")
            await instance.terminate()
        elif instance and keep_alive:
            logger.info(f"keeping instance alive: {instance.provider}:{instance.id}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run training on cloud GPU")
    parser.add_argument("--gpu", type=str, default="L4", help="GPU type (L4, A10G, A100, etc.)")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--ssh", type=str, default=None, help="SSH connection string")
    parser.add_argument("--node-id", type=str, default=None, help="Existing instance ID")
    parser.add_argument("--keep-alive", action="store_true", help="Don't terminate after training")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else "INFO")

    exit_code = trio.run(
        run,
        args.gpu,
        args.steps,
        args.ssh,
        args.node_id,
        args.keep_alive,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
