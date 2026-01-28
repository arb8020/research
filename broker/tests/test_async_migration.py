"""Integration tests for the async migration.

No mocks. Hits real provider APIs to verify the full async stack:
  trio.run → GPUClient → api → provider → async_retry → httpx.AsyncClient → real API

Two tiers:
  - Default: search-only (free, read-only, fast)
  - --slow:  full lifecycle (provision → ssh → nvidia-smi → terminate, costs money)

Run:
  .venv/bin/pytest tests/test_async_migration.py -v
  .venv/bin/pytest tests/test_async_migration.py -v --slow   # WARNING: provisions real GPU
"""

from __future__ import annotations

import os

import pytest
import trio
from dotenv import load_dotenv

from broker.client import GPUClient
from broker.types import GPUOffer, InstanceStatus

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
load_dotenv(os.path.expanduser("~/wafer/.env"))


def _get_credentials() -> dict[str, str]:
    """Load credentials from environment. Skips test if none available."""
    creds = {}
    key_map = {
        "runpod": "RUNPOD_API_KEY",
        "lambdalabs": "LAMBDA_API_KEY",
        "vast": "VAST_API_KEY",
        "primeintellect": "PRIME_API_KEY",
        "digitalocean": "DIGITALOCEAN_API_KEY",
        "digitalocean_amd": "AMD_DIGITALOCEAN_API_KEY",
    }
    # Override env vars: prefer funded keys (e.g. wafer RunPod key)
    overrides = {
        "runpod": "WAFER_RUNPOD_API_KEY",
    }
    for provider, env_var in key_map.items():
        # Check override first, then default
        override = overrides.get(provider)
        val = os.getenv(override) if override else None
        if not val:
            val = os.getenv(env_var)
        if val:
            creds[provider] = val
    return creds


def _make_client() -> GPUClient:
    creds = _get_credentials()
    if not creds:
        pytest.skip("No provider API keys in environment")
    return GPUClient(
        credentials=creds,
        ssh_key_path=os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519"),
    )


# --slow flag handling lives in conftest.py


# ---------------------------------------------------------------------------
# Path 1: Search (free, read-only)
# Every caller starts here. Exercises the full async chain.
# ---------------------------------------------------------------------------


def test_search_runpod():
    """Search RunPod for GPU offers — the most common caller path."""

    async def _run():
        client = _make_client()
        offers = await client.search(
            client.provider == "runpod",
            sort=lambda x: x.price_per_hour,
        )

        # Positive space: offers exist with expected structure
        assert len(offers) > 0, "RunPod should have at least one GPU offer"
        for offer in offers:
            assert isinstance(offer, GPUOffer)
            assert offer.provider == "runpod"
            assert offer.gpu_type, "offer must have a gpu_type"
            assert offer.price_per_hour > 0, "price must be positive"
            assert offer.vram_gb > 0, "vram must be positive"

        # Negative space: sorted correctly
        prices = [o.price_per_hour for o in offers]
        assert prices == sorted(prices), "offers should be sorted by price ascending"

    trio.run(_run)


def test_search_all_providers():
    """Search all configured providers at once — exercises concurrent async calls."""

    async def _run():
        client = _make_client()
        offers = await client.search(sort=lambda x: x.price_per_hour)

        assert len(offers) > 0, "At least one provider should return offers"

        # Verify we got offers from multiple providers (if multiple keys configured)
        providers_seen = {o.provider for o in offers}
        creds = _get_credentials()
        # We should see at least one provider we have creds for
        assert len(providers_seen) >= 1
        # Every offer provider must be one we have credentials for
        for p in providers_seen:
            assert p in creds, f"Got offer from {p} but no credentials configured"

    trio.run(_run)


def test_search_no_results():
    """Search with impossible filter returns empty list, not an error."""

    async def _run():
        client = _make_client()
        # Search for a GPU that doesn't exist
        offers = await client.search(
            client.gpu_type.contains("NONEXISTENT_GPU_THAT_WILL_NEVER_EXIST_12345"),
        )
        # Negative space: empty list, not None, not an exception
        assert isinstance(offers, list)
        assert len(offers) == 0

    trio.run(_run)


# ---------------------------------------------------------------------------
# Path 4 (partial): List instances (free, read-only)
# ---------------------------------------------------------------------------


def test_list_instances():
    """List instances — read-only, exercises list path."""

    async def _run():
        client = _make_client()
        instances = await client.list_instances("runpod")

        # Can be empty (no running instances), but must be a list
        assert isinstance(instances, list)
        for inst in instances:
            assert inst.provider == "runpod"
            assert inst.id, "instance must have an id"

    trio.run(_run)


# ---------------------------------------------------------------------------
# Full lifecycle (--slow): provision → ssh → nvidia-smi → terminate
# This is the critical path from rollouts/remote.py and bifrost/provision.py:
#   broker.create(query) → instance.wait_until_ssh_ready() → instance.exec() → instance.terminate()
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_full_lifecycle_runpod():
    """Provision a real RunPod GPU, SSH in, run nvidia-smi, terminate.

    WARNING: This costs real money (~$0.50-$1.00 for the few minutes it runs).
    Only runs with --slow flag.
    """

    async def _run():
        client = _make_client()
        creds = _get_credentials()
        assert "runpod" in creds, "RUNPOD_API_KEY required for lifecycle test"

        # 1. Search for cheapest community GPU
        offers = await client.search(
            (client.provider == "runpod") & (client.cloud_type == "community"),
            sort=lambda x: x.price_per_hour,
        )
        assert len(offers) > 0, "No RunPod community offers available"

        print(f"\nFound {len(offers)} offers, cheapest: {offers[0].gpu_type} @ ${offers[0].price_per_hour:.2f}/hr")

        # 2. Provision — pass full offer list so broker can fallback through them
        instance = None
        try:
            instance = await client.create(
                offers,
                name="broker-async-test",
                n_offers=5,
            )

            assert instance is not None, "create() returned None"
            assert instance.id, "instance missing ID"
            assert instance.provider == "runpod"
            print(f"Instance: {instance.id}")

            # 3. Wait for SSH
            print("Waiting for SSH...")
            ssh_ready = await instance.wait_until_ssh_ready(timeout=600)
            assert ssh_ready, "SSH not ready after 600s"

            # Tiger style: assert positive space (SSH details populated)
            assert instance.public_ip, "SSH ready but no public_ip"
            assert instance.ssh_port, "SSH ready but no ssh_port"
            assert instance.ssh_username, "SSH ready but no ssh_username"
            print(f"SSH: {instance.ssh_connection_string()}")

            # 4. Run nvidia-smi
            result = instance.exec(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                ssh_key_path=client.get_ssh_key_path("runpod"),
                timeout=30,
            )
            assert result.success, f"nvidia-smi failed: {result.stderr}"
            assert "NVIDIA" in result.stdout or "GPU" in result.stdout or "MiB" in result.stdout, (
                f"nvidia-smi output doesn't look like GPU info: {result.stdout[:200]}"
            )
            print(f"GPU: {result.stdout.strip()}")

        finally:
            # 5. Always terminate
            if instance:
                print(f"Terminating {instance.id}...")
                terminated = await instance.terminate()
                assert terminated, f"Failed to terminate {instance.id}"
                print("Terminated.")

    trio.run(_run)
