#!/usr/bin/env python3
"""Deploy GPT-2 training to remote GPU cluster using broker and bifrost."""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add parent directory to path to import broker and bifrost
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

from bifrost import BifrostClient
from broker import GPUClient


@dataclass
class PricingConfig:
    """Pricing constraints for GPU provisioning."""

    max_price_per_gpu: float | None
    max_total_price: float | None


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for deployment script."""
    parser = argparse.ArgumentParser(description="Deploy GPT-2 training to remote GPU cluster")

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
        "--max-price-per-gpu",
        type=float,
        default=None,
        help="Maximum price per GPU per hour in USD (e.g., 12.0 for $12/GPU/hr)",
    )
    parser.add_argument(
        "--max-total-price",
        type=float,
        default=None,
        help="Maximum total price per hour in USD (e.g., 100.0 for $100/hr total)",
    )
    # Deprecated but keep for backward compatibility
    parser.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="(Deprecated) Use --max-price-per-gpu instead. Maximum price per GPU per hour.",
    )
    parser.add_argument(
        "--cloud-type",
        type=str,
        default="secure",
        choices=["secure", "community"],
        help="Cloud type: 'secure' (default, guaranteed) or 'community' (spot, ~50%% cheaper but can be interrupted)",
    )

    # Disk configuration
    parser.add_argument(
        "--container-disk-gb",
        type=int,
        default=250,
        help="Container disk size in GB (default: 250 - all storage in container, matching outlier-features)",
    )
    parser.add_argument(
        "--volume-disk-gb",
        type=int,
        default=0,
        help="Volume disk size in GB for data (default: 0 - avoid RunPod mount errors, use container disk only)",
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
        help="Provider name to use (e.g., runpod, primeintellect, vast)",
    )
    parser.add_argument(
        "--template-id",
        type=str,
        help="RunPod template ID (e.g., runpod-torch-v280) - use instead of image for multi-GPU",
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

    return parser


def resolve_pricing_constraints(
    args: argparse.Namespace,
) -> PricingConfig | None:
    """Resolve and validate pricing constraints from arguments.

    Returns None if validation fails (with error messages printed).
    """
    max_price_per_gpu = None
    max_total_price = None

    # Priority: new flags > deprecated flag > defaults
    if args.max_price_per_gpu is not None:
        max_price_per_gpu = args.max_price_per_gpu
    elif args.max_price is not None:
        print("⚠ Warning: --max-price is deprecated, use --max-price-per-gpu instead")
        max_price_per_gpu = args.max_price

    if args.max_total_price is not None:
        max_total_price = args.max_total_price

    # Set sensible defaults if neither is specified
    if max_price_per_gpu is None and max_total_price is None:
        # Default: reasonable per-GPU price based on GPU type
        if "H100" in args.gpu_type:
            max_price_per_gpu = 15.0  # H100s are expensive
        elif "A100" in args.gpu_type:
            max_price_per_gpu = 3.0
        else:
            max_price_per_gpu = 2.0
        print(f"ℹ Using default max price: ${max_price_per_gpu}/GPU/hour")

    # Validate GPU count for training script constraints
    if "single-file-smoke.py" in args.script or "single-file.py" in args.script:
        if args.gpu_count not in [1, 2, 4, 8]:
            print("✗ GPU count validation failed")
            print(f"  Script: {args.script}")
            print(f"  Requested: {args.gpu_count} GPUs")
            print("  Supported: 1, 2, 4, or 8 GPUs")
            print(
                "\n  Reason: Training script uses gradient accumulation with hardcoded constraint:"
            )
            print("    assert 8 % world_size == 0")
            print("    grad_accum_steps = 8 // world_size")
            print("\n  Choose a GPU count that divides 8 evenly: --gpu-count [1|2|4|8]")
            return None

    # Validate constraints make sense
    if max_price_per_gpu is not None and max_total_price is not None:
        implied_total = max_price_per_gpu * args.gpu_count
        if implied_total > max_total_price:
            print("✗ Conflicting constraints:")
            print(
                f"  --max-price-per-gpu ${max_price_per_gpu} × {args.gpu_count} GPUs = ${implied_total:.2f}/hr"
            )
            print(f"  --max-total-price ${max_total_price:.2f}/hr")
            print("  These constraints cannot both be satisfied.")
            return None

    return PricingConfig(max_price_per_gpu=max_price_per_gpu, max_total_price=max_total_price)


def load_provider_credentials() -> dict[str, str] | None:
    """Load GPU provider credentials from environment.

    Returns None if no credentials found (with error messages printed).
    """
    load_dotenv()

    credentials = {}
    if runpod_key := os.getenv("RUNPOD_API_KEY"):
        credentials["runpod"] = runpod_key
    if vast_key := os.getenv("VAST_API_KEY"):
        credentials["vast"] = vast_key
    if prime_key := os.getenv("PRIME_API_KEY"):
        credentials["primeintellect"] = prime_key

    if not credentials:
        print("✗ No GPU provider credentials found")
        print("Set RUNPOD_API_KEY, VAST_API_KEY, or PRIME_API_KEY environment variables")
        print("\nExample:")
        print("  export RUNPOD_API_KEY=your-key-here")
        print("  export VAST_API_KEY=your-key-here")
        print("  export PRIME_API_KEY=your-key-here")
        return None

    return credentials


def print_deployment_header(args: argparse.Namespace, pricing: PricingConfig) -> None:
    """Print deployment configuration header."""
    print("=" * 60)
    print("GPT-2 Training Deployment")
    print("=" * 60)
    print(f"Training script: {args.script}")
    if not args.use_existing:
        print(f"GPU configuration: {args.gpu_count}x {args.gpu_type}")
        if pricing.max_price_per_gpu:
            print(
                f"Max price per GPU: ${pricing.max_price_per_gpu}/GPU/hour (${pricing.max_price_per_gpu * args.gpu_count:.2f}/hr total)"
            )
        if pricing.max_total_price:
            print(
                f"Max total price: ${pricing.max_total_price}/hour (${pricing.max_total_price / args.gpu_count:.2f}/GPU/hr)"
            )
        print(f"Disk: {args.container_disk_gb}GB container + {args.volume_disk_gb}GB volume")
    else:
        print(f"Using existing instance: {args.use_existing}")
    print("=" * 60)


def connect_to_existing_instance(
    gpu_client: GPUClient, instance_id: str, provider: str | None
) -> Any | None:
    """Connect to an existing GPU instance.

    Returns None if instance not found (with error messages printed).
    """
    print("\n[1/4] Connecting to existing GPU instance...")

    if provider:
        # Use explicit provider (faster)
        gpu_instance = gpu_client.get_instance(instance_id, provider)
        if not gpu_instance:
            print(f"✗ Instance {instance_id} not found in {provider}")
            return None
    else:
        # Auto-detect provider
        instances = gpu_client.list_instances()
        matches = [i for i in instances if i.id == instance_id]

        if not matches:
            print(f"✗ Instance {instance_id} not found in any provider")
            print("Available instances:")
            for i in instances:
                print(f"  - {i.id} ({i.provider})")
            return None

        gpu_instance = matches[0]

    print(f"✓ Connected to instance: {gpu_instance.id}")
    print(f"  Provider: {gpu_instance.provider}")
    print(f"  GPU: {gpu_instance.gpu_count}x {gpu_instance.gpu_type}")
    print(f"  SSH: {gpu_instance.ssh_connection_string()}")
    return gpu_instance


def provision_new_instance(
    gpu_client: GPUClient, args: argparse.Namespace, pricing: PricingConfig
) -> Any | None:
    """Provision a new GPU instance.

    Returns None if provisioning fails (with error messages printed).
    """
    print("\n[1/4] Provisioning GPU instance...")

    # Build query using the query builder API
    query = gpu_client.gpu_type.contains(args.gpu_type)

    # Apply provider filter if specified
    query_description = f"GPU type contains '{args.gpu_type}'"
    if args.provider:
        query = query & (gpu_client.provider == args.provider)
        query_description += f" and provider={args.provider}"

    if pricing.max_price_per_gpu is not None:
        query = query & (gpu_client.price_per_hour <= pricing.max_price_per_gpu)
        query_description += f" and price <= ${pricing.max_price_per_gpu}/GPU/hr"

    if pricing.max_total_price is not None:
        # Total price filter: price_per_hour * gpu_count <= max_total_price
        # So: price_per_hour <= max_total_price / gpu_count
        max_per_gpu_from_total = pricing.max_total_price / args.gpu_count
        query = query & (gpu_client.price_per_hour <= max_per_gpu_from_total)
        if pricing.max_price_per_gpu is None:
            query_description += f" and total <= ${pricing.max_total_price}/hr"
        else:
            query_description += f" (total <= ${pricing.max_total_price}/hr)"

    # Apply cloud type filter
    if args.cloud_type:
        from broker.types import CloudType

        if args.cloud_type == "secure":
            query = query & (gpu_client.cloud_type == CloudType.SECURE)
            query_description += " and cloud=secure"
        elif args.cloud_type == "community":
            query = query & (gpu_client.cloud_type == CloudType.COMMUNITY)
            query_description += " and cloud=community"

    print(f"Query: {query_description}")

    # First, search to see what's available (for better error messages)
    available_offers = gpu_client.search(query, sort=lambda x: x.price_per_hour, reverse=False)

    gpu_instance = gpu_client.create(
        query=query,
        gpu_count=args.gpu_count,
        container_disk_gb=args.container_disk_gb,
        volume_disk_gb=args.volume_disk_gb,
        template_id=args.template_id,
        sort=lambda x: x.price_per_hour,  # Sort by price (cheapest first)
        reverse=False,
    )

    if not gpu_instance:
        print_provisioning_failure(args, pricing, available_offers)
        return None

    print(f"✓ Instance created: {gpu_instance.id}")
    print(f"  Provider: {gpu_instance.provider}")
    print(f"  GPU: {gpu_instance.gpu_count}x {gpu_instance.gpu_type}")

    # Wait for SSH to be ready (RunPod can take up to 15 minutes)
    print("  Waiting for SSH to be ready (this can take up to 15 minutes)...")
    ssh_ready = gpu_instance.wait_until_ssh_ready(timeout=900)
    if not ssh_ready:
        print(f"✗ SSH failed to become ready on {gpu_instance.id}, terminating...")
        gpu_client.terminate_instance(gpu_instance.id, gpu_instance.provider)
        return None

    print(f"  ✓ SSH ready: {gpu_instance.ssh_connection_string()}")
    return gpu_instance


def print_provisioning_failure(
    args: argparse.Namespace, pricing: PricingConfig, available_offers: list[Any]
) -> None:
    """Print detailed error message when provisioning fails."""
    print("\n✗ Failed to provision GPU instance")
    print(f"  No {args.gpu_count}x {args.gpu_type} available matching your constraints")

    # Show the constraints that were applied
    print("\n  Your search criteria:")
    print(f"  - GPU Type: {args.gpu_type}")
    print(f"  - GPU Count: {args.gpu_count}")
    if args.provider:
        print(f"  - Provider: {args.provider}")
    if args.cloud_type:
        print(f"  - Cloud Type: {args.cloud_type}")
    if pricing.max_price_per_gpu:
        print(f"  - Max Price per GPU: ${pricing.max_price_per_gpu}/GPU/hr")
    if pricing.max_total_price:
        print(
            f"  - Max Total Price: ${pricing.max_total_price}/hr (${pricing.max_total_price / args.gpu_count:.2f}/GPU/hr)"
        )

    # Show availability information if we have it
    if available_offers:
        print(
            f"\n  Found {len(available_offers)} offer(s) matching some criteria, but none available for {args.gpu_count} GPUs:"
        )
        for i, offer in enumerate(available_offers[:3], 1):
            total_price = offer.price_per_hour * offer.gpu_count
            print(f"\n  {i}. {offer.provider} - {offer.gpu_type} ({offer.cloud_type.value})")
            print(
                f"     Price: ${offer.price_per_hour:.2f}/GPU/hr (${total_price:.2f}/hr for {offer.gpu_count} GPU)"
            )

            max_gpus = offer.max_gpu_count
            avail_counts = offer.available_gpu_counts
            if max_gpus is not None:
                print(f"     Max GPUs Available: {max_gpus}")
            if avail_counts:
                print(f"     Available GPU Counts: {avail_counts}")

            # Explain why this doesn't work
            if max_gpus is not None and max_gpus < args.gpu_count:
                print(f"     ✗ Cannot provision {args.gpu_count} GPUs (max available: {max_gpus})")
            elif avail_counts and args.gpu_count not in avail_counts:
                print(f"     ✗ {args.gpu_count} GPUs not in available counts")
    else:
        print("\n  No offers found matching your criteria at all.")

    print("\n  Suggestions:")
    # Suggest a valid GPU count that's close to what was requested
    valid_counts = [1, 2, 4, 8]
    suggested_count = max([c for c in valid_counts if c < args.gpu_count], default=4)
    print(f"  1. Use {suggested_count} GPUs instead: --gpu-count {suggested_count}")
    if args.cloud_type == "secure":
        print("  2. Try community cloud (cheaper, but spot instances): --cloud-type community")
    if pricing.max_total_price:
        print(f"  3. Increase budget: --max-total-price {int(pricing.max_total_price * 1.5)}")
    if args.provider:
        print(f"  4. Try different provider: remove --provider {args.provider}")
    else:
        print("  4. Try specific provider: --provider primeintellect")
    print(
        f"  5. Search for available GPUs first: broker search --gpu-type {args.gpu_type} --gpu-count {suggested_count}"
    )


def deploy_code_and_dependencies(gpu_instance: Any, ssh_key_path: str) -> str | None:
    """Deploy code and install dependencies on GPU instance.

    Returns workspace_path on success, None on failure.
    """
    print("\n[2/4] Deploying code with bifrost...")
    bifrost_client = BifrostClient(
        gpu_instance.ssh_connection_string(),
        ssh_key_path=ssh_key_path,
    )

    # Deploy code (without bootstrap first)
    workspace_path = bifrost_client.push()
    print(f"✓ Code deployed to: {workspace_path}")

    # Bootstrap: install dependencies (with streaming output)
    print("\n[2.5/4] Installing dependencies...")
    print("Running: uv sync --extra dev-speedrun")
    for line in bifrost_client.exec_stream(
        "uv sync --extra dev-speedrun", working_dir=workspace_path
    ):
        print(f"  {line}")
    print("✓ Dependencies installed")

    return workspace_path


def run_training(
    bifrost_client: BifrostClient,
    workspace_path: str,
    script: str,
    gpu_count: int,
    detached: bool,
) -> int:
    """Execute training job on GPU instance.

    Returns 0 on success, 1 on failure.
    """
    print("\n[3/4] Starting training...")

    command = f"bash examples/speedrun/run.sh --script {script} --gpu-count {gpu_count}"
    print(f"Command: {command}")

    if detached:
        # Use run_detached for background execution
        job_info = bifrost_client.run_detached(
            command=command,
            no_deploy=True,  # Already deployed in step 2
        )
        print("✓ Training launched in detached mode")
        print(f"  Job ID: {job_info.job_id}")
        print("\nTo monitor the job:")
        print(f"  bifrost logs {job_info.job_id}")
        print(f"  bifrost status {job_info.job_id}")
        print("\nTo download logs later:")
        print(f"  bifrost download {job_info.job_id} logs/ ./results/")
        return 0

    # Use exec_stream for real-time output
    exit_code = 0
    line_count = 0
    try:
        for line in bifrost_client.exec_stream(
            command,
            working_dir=workspace_path,
        ):
            print(line, flush=True)
            line_count += 1
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        exit_code = 1

    print(f"✓ Training completed (received {line_count} lines)")
    if exit_code != 0:
        print("  ⚠ Training failed!")

    return exit_code


def cleanup_instance(
    gpu_client: GPUClient,
    gpu_instance: Any,
    should_terminate: bool,
    use_existing: bool,
) -> None:
    """Terminate or keep GPU instance based on configuration."""
    if should_terminate and not use_existing:
        print("\n[4/4] Terminating instance...")
        gpu_client.terminate_instance(gpu_instance.id, gpu_instance.provider)
        print("✓ Instance terminated")
    else:
        reason = "using existing instance" if use_existing else "--no-terminate flag set"
        print(f"\n[4/4] Keeping instance alive ({reason})")
        print(f"  To terminate manually: broker terminate {gpu_instance.id}")


def main() -> int:
    """Main entry point for deployment script."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Resolve pricing constraints
    pricing = resolve_pricing_constraints(args)
    if pricing is None:
        return 1

    # Load credentials
    credentials = load_provider_credentials()
    if credentials is None:
        return 1

    # Print deployment header
    print_deployment_header(args, pricing)

    # Get or create GPU instance
    gpu_client = GPUClient(credentials=credentials, ssh_key_path=args.ssh_key_path)

    if args.use_existing:
        gpu_instance = connect_to_existing_instance(gpu_client, args.use_existing, args.provider)
    else:
        gpu_instance = provision_new_instance(gpu_client, args, pricing)

    if gpu_instance is None:
        return 1

    try:
        # Deploy code and dependencies
        workspace_path = deploy_code_and_dependencies(gpu_instance, args.ssh_key_path)
        if workspace_path is None:
            return 1

        # Create bifrost client for training execution
        bifrost_client = BifrostClient(
            gpu_instance.ssh_connection_string(),
            ssh_key_path=args.ssh_key_path,
        )

        # Execute training
        exit_code = run_training(
            bifrost_client, workspace_path, args.script, args.gpu_count, args.detached
        )

        # Cleanup
        cleanup_instance(gpu_client, gpu_instance, not args.no_terminate, args.use_existing)

        if exit_code != 0:
            return 1

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
