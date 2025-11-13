#!/usr/bin/env python3
"""Deploy kernel_utils smoke test to remote GPU via bifrost.

Usage:
    python smoke_deploy.py --ssh root@host:port
"""
import argparse
import logging
import sys
from pathlib import Path

from bifrost.client import BifrostClient
from kerbal import DependencyConfig, setup_script_deps

# Import shared logging
from shared.logging_config import setup_logging

logger = logging.getLogger(__name__)


def deploy_and_test(ssh_target: str, ssh_key: str, gpu_id: int) -> tuple[bool, str]:
    """Deploy kernel utils and run smoke test on remote.

    Args:
        ssh_target: SSH connection string (user@host:port)
        ssh_key: Path to SSH private key
        gpu_id: GPU device ID to use (e.g., 0, 1, 2, ...)

    Returns:
        (success, message)
    """
    logger.info("🚀 Deploying kernel_utils smoke test")
    logger.info(f"   Target: {ssh_target}")
    logger.info(f"   GPU: {gpu_id}")
    logger.info("=" * 60)

    # Connect
    client = BifrostClient(ssh_target, ssh_key)

    # Deploy code
    logger.info("\n📦 Deploying code...")
    workspace = client.push(workspace_path="~/.bifrost/workspaces/kernel_smoke")
    logger.info(f"   Deployed to: {workspace}")

    # Expand path
    result = client.exec(f"echo {workspace}")
    if result.exit_code != 0:
        return False, "Failed to expand workspace path"
    workspace_expanded = result.stdout.strip()

    project_dir = f"{workspace_expanded}/dev/integration-evaluation"

    # Check directory exists
    result = client.exec(f"test -d {project_dir} && echo OK || echo MISSING")
    if result.stdout.strip() != "OK":
        return False, f"Project directory not found: {project_dir}"

    logger.info(f"   Project dir: {project_dir}")

    # Setup dependencies
    logger.info("\n📦 Setting up dependencies...")
    deps = DependencyConfig(
        project_name="kernel-smoke",
        dependencies=[
            "torch>=2.4.0",
        ],
        python_version=">=3.10",
    )

    setup_script_deps(client, project_dir, deps, install_extras=None)

    # Run smoke test
    logger.info(f"\n🔥 Running smoke test on GPU {gpu_id}...")
    cmd = f"cd {project_dir} && CUDA_VISIBLE_DEVICES={gpu_id} python -m kernel_utils.smoke_test"
    result = client.exec(cmd)  # No timeout parameter

    logger.info(result.stdout)
    if result.stderr:
        logger.warning(f"stderr: {result.stderr}")

    success = (result.exit_code == 0)
    message = "Smoke tests passed" if success else f"Smoke tests failed (exit {result.exit_code})"

    return success, message


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy kernel smoke test to remote GPU")
    parser.add_argument("--ssh", required=True, help="SSH connection (user@host:port)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH private key path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use (default: 0)")
    args = parser.parse_args()

    # Setup logging
    setup_logging(
        level="INFO",
        logger_levels={
            "httpx": "WARNING",
            "urllib3": "WARNING",
            "paramiko": "WARNING",
        }
    )

    success, message = deploy_and_test(args.ssh, args.ssh_key, args.gpu)

    if success:
        logger.info(f"\n✅ {message}")
        return 0
    else:
        logger.error(f"\n❌ {message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
