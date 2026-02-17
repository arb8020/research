#!/usr/bin/env python3
"""
Integration test for Prime Intellect GPU provider.

Tests the full lifecycle:
1. Search for GPU offers
2. Provision an instance
3. Wait for SSH
4. Run nvidia-smi
5. Terminate instance

Usage:
    # Dry run (search only, no provisioning)
    python tests/test_primeintellect_integration.py --dry-run

    # Full integration test (will cost money!)
    python tests/test_primeintellect_integration.py -y

    # Keep instance alive after test (for debugging)
    python tests/test_primeintellect_integration.py -y --keep-alive

Requirements:
    - PRIME_API_KEY in environment or .env
    - SSH key registered with Prime Intellect
    - SSH_KEY_PATH in environment or .env
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import trio
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Setup basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def log(msg: str) -> None:
    """Print and log a message."""
    print(msg, flush=True)
    logger.info(msg)


def get_credentials() -> dict[str, str]:
    """Load Prime Intellect credentials from environment."""
    api_key = os.getenv("PRIME_API_KEY")
    if not api_key:
        log("ERROR: PRIME_API_KEY not set")
        log("Set it in .env or environment")
        sys.exit(1)
    return {"primeintellect": api_key}


def get_ssh_key_path() -> str:
    """Get SSH key path from environment."""
    ssh_key = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")
    expanded = os.path.expanduser(ssh_key)
    if not os.path.exists(expanded):
        log(f"ERROR: SSH key not found at {expanded}")
        sys.exit(1)
    return ssh_key


async def test_search() -> bool:
    """Test searching for GPU offers."""
    log("\n" + "=" * 60)
    log("TEST: Search GPU Offers")
    log("=" * 60)

    from broker.providers import primeintellect

    api_key = os.getenv("PRIME_API_KEY")

    log("\n1. Searching for all GPU offers...")
    offers = await primeintellect.search_gpu_offers(api_key=api_key)

    if not offers:
        log("   No GPU offers found")
        return False

    log(f"   Found {len(offers)} GPU offers:")
    for offer in offers[:10]:
        log(
            f"     - {offer.gpu_type} ({offer.gpu_count}x) - {offer.vram_gb}GB VRAM - ${offer.price_per_hour:.2f}/hr - {offer.availability_zone}"
        )

    # Test filtering by GPU count
    log("\n2. Testing gpu_count filter (1 GPU)...")
    single_gpu_offers = await primeintellect.search_gpu_offers(gpu_count=1, api_key=api_key)
    log(f"   Found {len(single_gpu_offers)} single-GPU offers")

    # Show cheapest offer
    if single_gpu_offers:
        cheapest = min(single_gpu_offers, key=lambda x: x.price_per_hour)
        log("\n3. Cheapest single-GPU offer:")
        log(
            f"   {cheapest.gpu_type} - ${cheapest.price_per_hour:.2f}/hr - {cheapest.availability_zone}"
        )

    log("\n Search test PASSED")
    return True


async def test_provision_and_ssh(keep_alive: bool = False) -> bool:
    """Test provisioning an instance and running nvidia-smi."""
    log("\n" + "=" * 60)
    log("TEST: Provision Instance and Run nvidia-smi")
    log("=" * 60)

    from broker import GPUClient

    credentials = get_credentials()
    ssh_key_path = get_ssh_key_path()

    client = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)

    # Find cheapest available GPU
    log("\n1. Searching for cheapest GPU...")
    offers = await client.search(
        client.provider == "primeintellect",
        sort=lambda x: x.price_per_hour,
    )

    if not offers:
        log("   No GPU offers available")
        return False

    cheapest = offers[0]
    log(
        f"   Found: {cheapest.gpu_type} - ${cheapest.price_per_hour:.2f}/hr - {cheapest.availability_zone}"
    )

    # Provision instance
    log(f"\n2. Provisioning {cheapest.gpu_type} in {cheapest.availability_zone}...")
    log("   (This may take 5-10 minutes for PrimeIntellect cold starts)")

    instance = None
    try:
        instance = await client.create(
            cheapest,
            name=f"broker-test-{int(time.time())}",
        )

        if not instance:
            log("   Failed to provision instance")
            return False

        log(f"   Instance created: {instance.id}")
        log(f"     Provider: {instance.provider}")
        log(f"     Status: {instance.status}")

        # Wait for SSH
        log("\n3. Waiting for SSH to be ready...")
        log("   (This may take 5-10 minutes)")

        if not await instance.wait_until_ssh_ready(timeout=600):
            log("   SSH not ready after 10 minutes")
            return False

        log(f"   SSH ready: {instance.ssh_connection_string()}")

        # Run nvidia-smi
        log("\n4. Running nvidia-smi...")
        result = await instance.aexec("nvidia-smi", ssh_key_path=ssh_key_path, timeout=60)

        if not result.success:
            log(f"   nvidia-smi failed: {result.stderr}")
            return False

        log("   nvidia-smi output:")
        for line in result.stdout.split("\n")[:15]:
            log(f"     {line}")

        # Quick GPU info extraction
        if "NVIDIA" in result.stdout:
            log("\n   GPU detected successfully!")
        else:
            log("\n   GPU info not found in output")

        log("\n Provision and SSH test PASSED")

    except Exception as e:
        log(f"\n   Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False

    else:
        return True

    finally:
        # Cleanup
        if instance:
            if keep_alive:
                log(f"\n Instance kept alive: {instance.id}")
                log(f"   SSH: {instance.ssh_connection_string()}")
                log(
                    f"   Terminate manually: broker terminate {instance.id} --provider primeintellect"
                )
            else:
                log(f"\n5. Terminating instance {instance.id}...")
                try:
                    if await instance.terminate():
                        log("   Instance terminated")
                    else:
                        log("   Terminate returned False - check manually")
                except Exception as e:
                    log(f"   Failed to terminate: {e}")
                    log(
                        f"   Terminate manually: broker terminate {instance.id} --provider primeintellect"
                    )


def main():
    parser = argparse.ArgumentParser(description="Prime Intellect GPU provider integration test")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only test search (no provisioning, no cost)",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Don't terminate instance after test",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    log("=" * 60)
    log("Prime Intellect GPU Provider Integration Test")
    log("=" * 60)

    results = {}

    # Always run search test
    results["search"] = trio.run(test_search)

    # Run provision test unless dry-run
    if not args.dry_run:
        if not args.yes:
            print("\n  WARNING: This will provision a real GPU instance!", flush=True)
            print("   Estimated cost: varies by GPU type", flush=True)
            response = input("\n   Continue? (y/n): ")
            if response.lower() != "y":
                log("   Aborted.")
                sys.exit(0)

        results["provision"] = trio.run(test_provision_and_ssh, args.keep_alive)
    else:
        log("\n Dry run mode - skipping provisioning test")
        results["provision"] = None

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    all_passed = True
    for name, passed in results.items():
        if passed is None:
            status = " SKIPPED"
        elif passed:
            status = " PASSED"
        else:
            status = " FAILED"
            all_passed = False
        log(f"  {name}: {status}")

    print(flush=True)
    if all_passed:
        log("All tests passed!")
        return 0
    else:
        log("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
