"""Run training on cloud GPU via bifrost.

Usage:
    uv run python pretrain/scripts/run.py                    # Provision L4, run training
    uv run python pretrain/scripts/run.py --gpu A10G         # Use A10G instead
    uv run python pretrain/scripts/run.py --provider runpod  # Force RunPod (faster boot)
    uv run python pretrain/scripts/run.py --steps 500        # Override steps
    uv run python pretrain/scripts/run.py --ssh root@gpu:22  # Use existing SSH connection
    uv run python pretrain/scripts/run.py --node-id runpod:abc123  # Reuse existing instance
"""

from __future__ import annotations

import argparse
import logging
import sys

import trio

from bifrost import GPUQuery, acquire_node
from broker import api as broker_api
from broker.client import GPUClient
from broker.credentials import get_credentials

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
    provider: str | None = None,
    real_data: bool = False,
) -> int:
    """Run training on cloud GPU.

    Args:
        gpu_type: GPU type to provision (e.g., "L4", "A10G", "A100")
        steps: Number of training steps
        ssh: Existing SSH connection string (user@host:port)
        node_id: Existing broker instance ID (provider:instance_id)
        keep_alive: Don't terminate instance after training
        provider: Force specific provider (e.g., "runpod", "primeintellect")
        real_data: Use fineweb data instead of random

    Returns:
        Exit code from training
    """
    instance = None

    # Determine acquisition mode
    if ssh is None and node_id is None:
        if provider:
            # Manual search + create with provider filter
            logger.info(f"searching for {gpu_type} on {provider}...")
            credentials = get_credentials()
            broker = GPUClient(credentials=credentials)
            offers = await broker_api.search(
                gpu_type=gpu_type,
                provider=provider,
                credentials=credentials,
            )
            if not offers:
                logger.error(f"no {gpu_type} offers found on {provider}")
                return 1

            # Sort by price and pick cheapest
            offers.sort(key=lambda x: x.price_per_hour)
            offer = offers[0]
            logger.info(
                f"found: {offer.gpu_type} @ ${offer.price_per_hour:.2f}/hr on {offer.provider}"
            )

            # Create instance
            instance = await broker.create(
                offer,
                name="pretrain/step1",
                container_disk_gb=100,
            )
            logger.info(f"created instance: {instance.provider}:{instance.id}")

            # Wait for SSH
            logger.info("waiting for instance to boot...")
            ssh_info = await instance.wait_for_ssh(timeout=600)
            logger.info(f"instance ready: {ssh_info.host}:{ssh_info.port}")

            # Connect via SSH
            from bifrost.ssh import SSHClient

            key_path = broker.get_ssh_key_path(provider)
            client = SSHClient(
                host=ssh_info.host,
                port=ssh_info.port,
                user=ssh_info.user,
                key_path=key_path,
            )
        else:
            # Use bifrost's default provisioning (cheapest across all providers)
            provision = GPUQuery(
                type=gpu_type,
                count=1,
                name="pretrain/step1",
            )
            logger.info(f"provisioning {gpu_type} GPU...")
            client, instance = await acquire_node(provision=provision)
    else:
        # Use existing SSH or node_id
        client, instance = await acquire_node(
            ssh=ssh,
            node_id=node_id,
        )

    if instance:
        logger.info(f"instance: {instance.provider}:{instance.id}")
        logger.info(f"gpu: {instance.gpu_count}x {instance.gpu_type}")

    try:
        # Deploy code (git sync only, no bootstrap)
        logger.info("deploying code...")
        workspace = client.push("~/.bifrost/workspaces/rollouts-pretrain")
        logger.info(f"deployed to: {workspace}")

        # Bootstrap steps (following rollouts/run.py pattern)
        bootstrap_steps = [
            ("Installing uv", "curl -LsSf https://astral.sh/uv/install.sh | sh"),
            (
                "Syncing Python deps",
                "~/.local/bin/uv python install 3.12 && ~/.local/bin/uv sync --python 3.12 --package rollouts",
            ),
            (
                "Installing torch + huggingface_hub",
                "~/.local/bin/uv pip install torch huggingface_hub",
            ),
        ]

        for label, cmd in bootstrap_steps:
            logger.info(f"{label}...")
            client.exec(cmd, working_dir=workspace)

        # Run training (streams output to console)
        logger.info(f"starting training for {steps} steps...")
        real_data_flag = " --real-data" if real_data else ""
        train_cmd = f"~/.local/bin/uv run python -m rollouts.pretrain.train --steps {steps} --log-every 10{real_data_flag}"
        for line in client.exec_stream(train_cmd, working_dir=f"{workspace}/rollouts"):
            print(line, flush=True)

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
    parser.add_argument(
        "--provider", type=str, default=None, help="Force provider (runpod, primeintellect, etc.)"
    )
    parser.add_argument(
        "--real-data", action="store_true", help="Use fineweb data instead of random"
    )
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
        args.provider,
        args.real_data,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
