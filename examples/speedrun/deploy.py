#!/usr/bin/env python3
"""Deploy GPT-2 training to remote GPU cluster using broker and bifrost."""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import broker and bifrost
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from broker import GPUClient
from bifrost import BifrostClient


def main():
    parser = argparse.ArgumentParser(
        description="Deploy GPT-2 training to remote GPU cluster"
    )

    # Training configuration
    parser.add_argument(
        "--script",
        default="single-file.py",
        help="Training script to run (e.g., single-file.py, train/01_train_basic.py)",
    )

    # GPU configuration
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=8,
        help="Number of GPUs to use (default: 8)",
    )
    parser.add_argument(
        "--gpu-type",
        default="H100",
        help="GPU type to request (default: H100)",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=3.5,
        help="Maximum price per GPU per hour in USD (default: 3.5)",
    )

    # Disk configuration
    parser.add_argument(
        "--container-disk-gb",
        type=int,
        default=250,
        help="Container disk size in GB (default: 250)",
    )
    parser.add_argument(
        "--volume-disk-gb",
        type=int,
        default=100,
        help="Volume disk size in GB for data (default: 100)",
    )

    # Deployment configuration
    parser.add_argument(
        "--use-existing",
        type=str,
        help="Use existing GPU instance by ID (skips provisioning)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Provider name when using --use-existing (e.g., runpod, vast)",
    )
    parser.add_argument(
        "--ssh-key-path",
        default="~/.ssh/id_ed25519",
        help="Path to SSH private key (default: ~/.ssh/id_ed25519)",
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Run training in detached mode (tmux session)",
    )
    parser.add_argument(
        "--no-terminate",
        action="store_true",
        help="Don't terminate instance after deployment (for debugging)",
    )

    args = parser.parse_args()

    # Load .env file from repo root (searches parent directories)
    load_dotenv()

    # Load credentials from environment
    credentials = {}
    if runpod_key := os.getenv("RUNPOD_API_KEY"):
        credentials["runpod"] = runpod_key
    if vast_key := os.getenv("VAST_API_KEY"):
        credentials["vast"] = vast_key

    if not credentials:
        print("✗ No GPU provider credentials found")
        print("Set RUNPOD_API_KEY or VAST_API_KEY environment variables")
        print("\nExample:")
        print("  export RUNPOD_API_KEY=your-key-here")
        print("  export VAST_API_KEY=your-key-here")
        return 1

    print("=" * 60)
    print("GPT-2 Training Deployment")
    print("=" * 60)
    print(f"Training script: {args.script}")
    if not args.use_existing:
        print(f"GPU configuration: {args.gpu_count}x {args.gpu_type}")
        print(f"Max price: ${args.max_price}/GPU/hour")
        print(f"Disk: {args.container_disk_gb}GB container + {args.volume_disk_gb}GB volume")
    else:
        print(f"Using existing instance: {args.use_existing}")
    print("=" * 60)

    # Step 1: Get or create GPU instance
    gpu_client = GPUClient(credentials=credentials, ssh_key_path=args.ssh_key_path)

    if args.use_existing:
        print(f"\n[1/4] Connecting to existing GPU instance...")

        if args.provider:
            # Use explicit provider (faster)
            gpu_instance = gpu_client.get_instance(args.use_existing, args.provider)
            if not gpu_instance:
                print(f"✗ Instance {args.use_existing} not found in {args.provider}")
                return 1
        else:
            # Auto-detect provider
            instances = gpu_client.list_instances()
            matches = [i for i in instances if i.id == args.use_existing]

            if not matches:
                print(f"✗ Instance {args.use_existing} not found in any provider")
                print("Available instances:")
                for i in instances:
                    print(f"  - {i.id} ({i.provider})")
                return 1

            gpu_instance = matches[0]

        print(f"✓ Connected to instance: {gpu_instance.id}")
        print(f"  Provider: {gpu_instance.provider}")
        print(f"  GPU: {gpu_instance.gpu_count}x {gpu_instance.gpu_type}")
        print(f"  SSH: {gpu_instance.ssh_connection_string()}")
    else:
        print("\n[1/4] Provisioning GPU instance...")

        # Build query using the query builder API
        query = (
            (gpu_client.gpu_type.contains(args.gpu_type)) &
            (gpu_client.price_per_hour <= args.max_price)
        )
        print(f"Query: GPU type contains '{args.gpu_type}' and price <= ${args.max_price}/hr")

        gpu_instance = gpu_client.create(
            query=query,
            gpu_count=args.gpu_count,
            container_disk_gb=args.container_disk_gb,
            volume_disk_gb=args.volume_disk_gb,
            sort=lambda x: x.price_per_hour,  # Sort by price (cheapest first)
            reverse=False,
        )

        print(f"✓ Instance created: {gpu_instance.id}")
        print(f"  Provider: {gpu_instance.provider}")
        print(f"  GPU: {gpu_instance.gpu_count}x {gpu_instance.gpu_type}")

        # Wait for SSH to be ready (RunPod can take up to 15 minutes)
        print(f"  Waiting for SSH to be ready (this can take up to 15 minutes)...")
        ssh_ready = gpu_instance.wait_until_ssh_ready(timeout=900)
        if not ssh_ready:
            print(f"✗ SSH failed to become ready on {gpu_instance.id}, terminating...")
            gpu_client.terminate_instance(gpu_instance.id, gpu_instance.provider)
            return 1

        print(f"  ✓ SSH ready: {gpu_instance.ssh_connection_string()}")

    try:
        # Step 2: Deploy code with bifrost
        print("\n[2/4] Deploying code with bifrost...")
        bifrost_client = BifrostClient(
            gpu_instance.ssh_connection_string(),
            ssh_key_path=args.ssh_key_path,
        )

        # Bootstrap: install dependencies
        workspace_path = bifrost_client.push(bootstrap_cmd="uv sync --extra example-speedrun")
        print(f"✓ Code deployed to: {workspace_path}")

        # Step 3: Execute training
        print("\n[3/4] Starting training...")

        command = f"bash examples/speedrun/run.sh --script {args.script} --gpu-count {args.gpu_count}"
        print(f"Command: {command}")

        if args.detached:
            # Use run_detached for background execution
            job_info = bifrost_client.run_detached(
                command=command,
                no_deploy=True  # Already deployed in step 2
            )
            print(f"✓ Training launched in detached mode")
            print(f"  Job ID: {job_info.job_id}")
            print(f"\nTo monitor the job:")
            print(f"  bifrost logs {job_info.job_id}")
            print(f"  bifrost status {job_info.job_id}")
            print(f"\nTo download logs later:")
            print(f"  bifrost download {job_info.job_id} logs/ ./results/")
        else:
            # Use exec for synchronous execution
            result = bifrost_client.exec(
                command,
                working_dir=workspace_path,
            )
            print(f"✓ Training completed")
            print(f"  Exit code: {result.exit_code}")
            if result.exit_code != 0:
                print(f"  ⚠ Training failed!")
                print(result.stderr)
            else:
                print(result.stdout)

        # Step 4: Cleanup (optional)
        # Don't terminate if using existing instance
        if not args.no_terminate and not args.use_existing:
            print("\n[4/4] Terminating instance...")
            gpu_client.terminate_instance(gpu_instance.id, gpu_instance.provider)
            print("✓ Instance terminated")
        else:
            reason = "using existing instance" if args.use_existing else "--no-terminate flag set"
            print(f"\n[4/4] Keeping instance alive ({reason})")
            print(f"  To terminate manually: broker terminate {gpu_instance.id}")

    except Exception as e:
        print(f"\n✗ Error during deployment: {e}")

        # Cleanup on error (but not if using existing instance)
        if not args.no_terminate and not args.use_existing:
            print("Terminating instance due to error...")
            gpu_client.terminate_instance(gpu_instance.id, gpu_instance.provider)

        return 1

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
