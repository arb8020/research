#!/usr/bin/env python3
"""
Integration test for DigitalOcean GPU provider.

Tests the full lifecycle:
1. Search for GPU offers
2. Provision an instance
3. Wait for SSH
4. Run nvidia-smi
5. Terminate instance

Usage:
    # Dry run (search only, no provisioning)
    python tests/test_digitalocean_integration.py --dry-run

    # Full integration test (will cost money!)
    python tests/test_digitalocean_integration.py

    # Keep instance alive after test (for debugging)
    python tests/test_digitalocean_integration.py --keep-alive

Requirements:
    - DIGITALOCEAN_API_KEY in environment or .env
    - SSH key registered in DigitalOcean account
    - SSH_KEY_PATH in environment or .env
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Load .env file
from dotenv import load_dotenv

load_dotenv()


def get_credentials() -> dict[str, str]:
    """Load DigitalOcean credentials from environment."""
    api_key = os.getenv("DIGITALOCEAN_API_KEY")
    if not api_key:
        print("ERROR: DIGITALOCEAN_API_KEY not set")
        print("Set it in .env or environment")
        sys.exit(1)
    return {"digitalocean": api_key}


def get_ssh_key_path() -> str:
    """Get SSH key path from environment."""
    ssh_key = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")
    expanded = os.path.expanduser(ssh_key)
    if not os.path.exists(expanded):
        print(f"ERROR: SSH key not found at {expanded}")
        sys.exit(1)
    return ssh_key


def test_search() -> bool:
    """Test searching for GPU offers."""
    print("\n" + "=" * 60)
    print("TEST: Search GPU Offers")
    print("=" * 60)

    from broker.providers import digitalocean

    api_key = os.getenv("DIGITALOCEAN_API_KEY")

    print("\n1. Searching for all GPU offers...")
    offers = digitalocean.search_gpu_offers(api_key=api_key)

    if not offers:
        print("   ❌ No GPU offers found")
        return False

    print(f"   ✓ Found {len(offers)} GPU offers:")
    for offer in offers[:10]:
        print(
            f"     - {offer.gpu_type} ({offer.gpu_count}x) - {offer.vram_gb}GB VRAM - ${offer.price_per_hour:.2f}/hr - {offer.availability_zone}"
        )

    # Test filtering by manufacturer
    print("\n2. Testing manufacturer filter (nvidia)...")
    nvidia_offers = digitalocean.search_gpu_offers(manufacturer="nvidia", api_key=api_key)
    print(f"   ✓ Found {len(nvidia_offers)} NVIDIA offers")

    # Test filtering by GPU count
    print("\n3. Testing gpu_count filter (1 GPU)...")
    single_gpu_offers = digitalocean.search_gpu_offers(gpu_count=1, api_key=api_key)
    print(f"   ✓ Found {len(single_gpu_offers)} single-GPU offers")

    print("\n✓ Search test PASSED")
    return True


def test_provision_and_ssh(keep_alive: bool = False) -> bool:
    """Test provisioning an instance and running nvidia-smi."""
    print("\n" + "=" * 60)
    print("TEST: Provision Instance and Run nvidia-smi")
    print("=" * 60)

    from broker import GPUClient

    credentials = get_credentials()
    ssh_key_path = get_ssh_key_path()

    client = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)

    # Find cheapest available GPU
    print("\n1. Searching for cheapest GPU...")
    offers = client.search(
        client.provider == "digitalocean",
        sort=lambda x: x.price_per_hour,
    )

    if not offers:
        print("   ❌ No GPU offers available")
        return False

    cheapest = offers[0]
    print(
        f"   ✓ Found: {cheapest.gpu_type} - ${cheapest.price_per_hour:.2f}/hr - {cheapest.availability_zone}"
    )

    # Provision instance
    print(f"\n2. Provisioning {cheapest.gpu_type} in {cheapest.availability_zone}...")
    print("   (This may take a few minutes)")

    instance = None
    try:
        instance = client.create(
            cheapest,
            name=f"broker-test-{int(time.time())}",
        )

        if not instance:
            print("   ❌ Failed to provision instance")
            return False

        print(f"   ✓ Instance created: {instance.id}")
        print(f"     Provider: {instance.provider}")
        print(f"     Status: {instance.status}")

        # Wait for SSH
        print("\n3. Waiting for SSH to be ready...")
        print("   (This may take 2-5 minutes)")

        if not instance.wait_until_ssh_ready(timeout=600):
            print("   ❌ SSH not ready after 10 minutes")
            return False

        print(f"   ✓ SSH ready: {instance.ssh_connection_string()}")

        # Run nvidia-smi
        print("\n4. Running nvidia-smi...")
        result = instance.exec("nvidia-smi", ssh_key_path=ssh_key_path, timeout=60)

        if not result.success:
            print(f"   ❌ nvidia-smi failed: {result.stderr}")
            return False

        print("   ✓ nvidia-smi output:")
        for line in result.stdout.split("\n")[:15]:
            print(f"     {line}")

        # Quick GPU info extraction
        if "NVIDIA" in result.stdout:
            print("\n   ✓ GPU detected successfully!")
        else:
            print("\n   ⚠ GPU info not found in output")

        print("\n✓ Provision and SSH test PASSED")
        return True

    except Exception as e:
        print(f"\n   ❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if instance:
            if keep_alive:
                print(f"\n💡 Instance kept alive: {instance.id}")
                print(f"   SSH: {instance.ssh_connection_string()}")
                print(
                    f"   Terminate manually: broker terminate {instance.id} --provider digitalocean"
                )
            else:
                print(f"\n5. Terminating instance {instance.id}...")
                try:
                    if instance.terminate():
                        print("   ✓ Instance terminated")
                    else:
                        print("   ⚠ Terminate returned False - check manually")
                except Exception as e:
                    print(f"   ⚠ Failed to terminate: {e}")
                    print(
                        f"   Terminate manually: broker terminate {instance.id} --provider digitalocean"
                    )


def main():
    parser = argparse.ArgumentParser(description="DigitalOcean GPU provider integration test")
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

    args = parser.parse_args()

    print("=" * 60)
    print("DigitalOcean GPU Provider Integration Test")
    print("=" * 60)

    results = {}

    # Always run search test
    results["search"] = test_search()

    # Run provision test unless dry-run
    if not args.dry_run:
        print("\n⚠️  WARNING: This will provision a real GPU instance!")
        print("   Estimated cost: ~$0.76-$3.39/hr (depending on availability)")
        response = input("\n   Continue? (y/n): ")
        if response.lower() != "y":
            print("   Aborted.")
            sys.exit(0)

        results["provision"] = test_provision_and_ssh(keep_alive=args.keep_alive)
    else:
        print("\n📋 Dry run mode - skipping provisioning test")
        results["provision"] = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        if passed is None:
            status = "⏭ SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
            all_passed = False
        print(f"  {name}: {status}")

    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
