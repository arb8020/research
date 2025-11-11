#!/usr/bin/env python3
"""Example: Simple deployment using the deploy API.

This shows how to use the new deploy package instead of writing
400+ line deploy.py files.

Usage:
    python deploy/examples/simple_deployment.py \\
        --ssh root@host:22 \\
        --project dev/speedrun \\
        --extra dev-speedrun \\
        --command "python train.py"
"""

import argparse
import logging
from pathlib import Path

from bifrost import BifrostClient
from deploy import deploy_and_run, deploy_project, run_in_project
from deploy.backends import UvBackend

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def example_high_level(ssh_connection: str, ssh_key: str):
    """Example: High-level API (simplest - one function call).

    This is the "I don't want to think about it" approach.
    Best for simple cases where you just want to run something.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: High-level API (deploy_and_run)")
    print("=" * 60)

    bifrost = BifrostClient(ssh_connection, ssh_key)

    # Deploy and run in one call
    result = deploy_and_run(
        bifrost,
        local_path="dev/speedrun",
        extra="dev-speedrun",
        command="python -c 'print(\"Hello from remote!\")'",
        detached=False,  # Wait for completion
    )

    if result and result.success:
        print("\n✅ Command completed successfully!")
        print(f"Output: {result.stdout}")
    else:
        print("\n❌ Command failed")
        if result:
            print(f"Error: {result.stderr}")


def example_mid_level(ssh_connection: str, ssh_key: str):
    """Example: Mid-level API (deploy once, run multiple commands).

    This is better when you want to run multiple commands in the same
    environment without re-deploying each time.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Mid-level API (deploy_project + run_in_project)")
    print("=" * 60)

    bifrost = BifrostClient(ssh_connection, ssh_key)

    # Deploy once
    print("\n📦 Deploying project...")
    workspace = deploy_project(
        bifrost,
        local_path="dev/speedrun",
        extra="dev-speedrun",
        backend=UvBackend(),  # Explicit backend (optional)
    )
    print(f"✅ Project deployed to: {workspace}")

    # Run multiple commands
    print("\n🚀 Running commands...")

    commands = [
        "python --version",
        "python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'",
        "ls -la",
    ]

    for cmd in commands:
        print(f"\n  Running: {cmd}")
        result = run_in_project(bifrost, workspace, cmd)
        if result.success:
            print(f"  ✅ {result.stdout.strip()}")
        else:
            print(f"  ❌ Failed: {result.stderr.strip()}")


def example_low_level(ssh_connection: str, ssh_key: str):
    """Example: Low-level API (full control over each step).

    This gives you maximum control. Useful when you need to customize
    the deployment process or handle special cases.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Low-level API (manual orchestration)")
    print("=" * 60)

    from deploy.api import push_code, bootstrap_env, start_tmux_session, sync_results

    bifrost = BifrostClient(ssh_connection, ssh_key)

    # Step 1: Push code
    print("\n📦 Step 1: Pushing code...")
    workspace = push_code(bifrost, local_path="dev/speedrun")
    print(f"✅ Code pushed to: {workspace}")

    # Step 2: Bootstrap environment
    print("\n🔧 Step 2: Bootstrapping environment...")
    bootstrap_env(bifrost, workspace, extra="dev-speedrun")
    print("✅ Environment ready")

    # Step 3: Start tmux session (detached)
    print("\n🚀 Step 3: Starting training in tmux...")
    start_tmux_session(
        bifrost,
        session_name="example_training",
        command="python -c 'import time; print(\"Training...\"); time.sleep(5); print(\"Done!\")'",
        workspace=workspace,
        log_file="training.log",
    )
    print("✅ Training started in tmux (session: example_training)")
    print("   Attach: tmux attach -t example_training")

    # Step 4: Sync results (would normally wait for training to complete)
    print("\n💾 Step 4: Syncing results...")
    # In real usage, you'd wait for training to complete first
    # For this example, we'll just sync the log file
    local_results = Path("./example_results")
    sync_results(bifrost, f"{workspace}/training.log", str(local_results))
    print(f"✅ Results synced to: {local_results}")


def example_custom_backend(ssh_connection: str, ssh_key: str):
    """Example: Using a custom backend (future).

    This shows how you would use NixBackend or DockerBackend
    once they're implemented.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Backend (future)")
    print("=" * 60)

    from deploy.backends import NixBackend, DockerBackend

    bifrost = BifrostClient(ssh_connection, ssh_key)

    # Example with NixBackend (stub - will error)
    print("\nNixBackend (not yet implemented):")
    try:
        nix_backend = NixBackend(flake_ref=".#speedrun")
        workspace = deploy_project(
            bifrost,
            local_path="dev/speedrun",
            extra="dev-speedrun",
            backend=nix_backend,
        )
    except NotImplementedError as e:
        print(f"  ⚠️  {e}")

    # Example with DockerBackend (stub - will error)
    print("\nDockerBackend (not yet implemented):")
    try:
        docker_backend = DockerBackend(image="speedrun:latest", gpu=True)
        workspace = deploy_project(
            bifrost,
            local_path="dev/speedrun",
            extra="dev-speedrun",
            backend=docker_backend,
        )
    except NotImplementedError as e:
        print(f"  ⚠️  {e}")

    print("\n💡 These backends are stubs showing design intent.")
    print("   Implement them when you need Nix or Docker!")


def main():
    parser = argparse.ArgumentParser(
        description="Example deployment using deploy API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Run all examples
  python deploy/examples/simple_deployment.py --ssh root@host:22

  # Run specific example
  python deploy/examples/simple_deployment.py --ssh root@host:22 --example high

Available examples:
  high   - High-level API (deploy_and_run)
  mid    - Mid-level API (deploy_project + run_in_project)
  low    - Low-level API (manual orchestration)
  custom - Custom backends (Nix/Docker - stubs)
  all    - Run all examples (default)
        """
    )
    parser.add_argument(
        "--ssh",
        required=True,
        help="SSH connection string (e.g., root@host:22)"
    )
    parser.add_argument(
        "--ssh-key",
        default="~/.ssh/id_ed25519",
        help="Path to SSH private key"
    )
    parser.add_argument(
        "--example",
        choices=["high", "mid", "low", "custom", "all"],
        default="all",
        help="Which example to run"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Deploy API Examples")
    print("=" * 60)
    print(f"SSH: {args.ssh}")
    print(f"Key: {args.ssh_key}")

    # Run requested examples
    if args.example in ["high", "all"]:
        example_high_level(args.ssh, args.ssh_key)

    if args.example in ["mid", "all"]:
        example_mid_level(args.ssh, args.ssh_key)

    if args.example in ["low", "all"]:
        example_low_level(args.ssh, args.ssh_key)

    if args.example in ["custom", "all"]:
        example_custom_backend(args.ssh, args.ssh_key)

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
