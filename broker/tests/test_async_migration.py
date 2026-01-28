"""Integration tests for async migration.

Tests the 4 caller paths through GPUClient (the stable cut point):
1. Search — returns offers
2. Create lifecycle — provision + get instance back
3. Get + reconnect — look up existing instance
4. List + terminate — enumerate + destroy

Mock at the provider HTTP boundary (_make_api_request / _make_graphql_request).
No real API calls, no mocking internal broker logic.

Run: python -m pytest tests/test_async_migration.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import trio

from broker.client import GPUClient
from broker.types import GPUOffer

# ---------------------------------------------------------------------------
# Fixtures: fake provider responses at the HTTP boundary
# ---------------------------------------------------------------------------


def _fake_runpod_search_response() -> dict:
    """What RunPod's GraphQL API returns for gpuTypes query."""
    return {
        "gpuTypes": [
            {
                "id": "NVIDIA GeForce RTX 4090",
                "displayName": "RTX 4090",
                "memoryInGb": 24,
                "manufacturer": "Nvidia",
                "secureCloud": True,
                "communityCloud": True,
                "lowestPrice": {
                    "minimumBidPrice": 0.34,
                    "uninterruptablePrice": 0.69,
                    "stockStatus": "available",
                    "maxUnreservedGpuCount": 8,
                    "availableGpuCounts": [1, 2, 4],
                },
            }
        ]
    }


def _fake_runpod_provision_response() -> dict:
    """What RunPod returns for podFindAndDeployOnDemand."""
    return {
        "podFindAndDeployOnDemand": {
            "id": "abc123",
            "machineId": "machine-1",
            "machine": {"podHostId": "host-1"},
        }
    }


def _fake_runpod_pod_details() -> dict:
    """What RunPod returns for pod query (instance details)."""
    return {
        "pod": {
            "id": "abc123",
            "name": "test-pod",
            "machineId": "machine-1",
            "desiredStatus": "RUNNING",
            "gpuCount": 1,
            "costPerHr": 0.69,
            "machine": {
                "podHostId": "host-1",
                "gpuType": {
                    "displayName": "RTX 4090",
                    "manufacturer": "Nvidia",
                    "memoryInGb": 24,
                },
            },
            "runtime": {
                "uptimeInSeconds": 120,
                "ports": [
                    {
                        "ip": "1.2.3.4",
                        "isIpPublic": True,
                        "privatePort": 22,
                        "publicPort": 22222,
                        "type": "tcp",
                    }
                ],
                "gpus": [],
            },
        }
    }


def _fake_runpod_list_response() -> dict:
    """What RunPod returns for myself.pods query."""
    return {"myself": {"pods": [_fake_runpod_pod_details()["pod"]]}}


def _fake_runpod_terminate_response() -> dict:
    """RunPod returns null on successful terminate."""
    return {"podTerminate": None}


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_client() -> GPUClient:
    return GPUClient(
        credentials={"runpod": "fake-api-key"},
        ssh_key_path="~/.ssh/id_ed25519",
    )


# ---------------------------------------------------------------------------
# Path 1: Search
# ---------------------------------------------------------------------------


def test_search_returns_offers():
    """GPUClient.search() returns GPUOffer list — the path every caller uses first."""

    async def _run():
        client = _make_client()

        with patch(
            "broker.providers.runpod._make_graphql_request",
            new_callable=AsyncMock,
            return_value=_fake_runpod_search_response(),
        ):
            offers = await client.search(
                client.provider == "runpod",
                sort=lambda x: x.price_per_hour,
            )

        assert len(offers) > 0, "Expected at least one offer"
        offer = offers[0]
        assert isinstance(offer, GPUOffer)
        assert offer.provider == "runpod"
        assert offer.gpu_type == "RTX 4090"
        assert offer.price_per_hour > 0

    trio.run(_run)


# ---------------------------------------------------------------------------
# Path 2: Create lifecycle (search → provision → get instance)
# ---------------------------------------------------------------------------


def test_create_returns_instance():
    """GPUClient.create() provisions and returns a ClientGPUInstance.

    This is the critical path for rollouts/remote.py and bifrost/provision.py.
    """

    async def _run():
        client = _make_client()

        # Mock both the search call and provision call
        with patch(
            "broker.providers.runpod._make_graphql_request",
            new_callable=AsyncMock,
        ) as mock_gql:
            # First call: search (secure cloud)
            # Second call: search (community cloud)
            # Third call: provision
            mock_gql.side_effect = [
                _fake_runpod_search_response(),  # secure cloud search
                _fake_runpod_search_response(),  # community cloud search
                _fake_runpod_provision_response(),  # provision
            ]

            result = await client.create(
                client.provider == "runpod",
                gpu_count=1,
                sort=lambda x: x.price_per_hour,
            )

        assert result is not None, "create() should return an instance"
        assert result.id == "abc123"
        assert result.provider == "runpod"

    trio.run(_run)


# ---------------------------------------------------------------------------
# Path 3: Get existing instance (reconnect path)
# ---------------------------------------------------------------------------


def test_get_instance_reconnect():
    """GPUClient.get_instance() looks up an existing instance by ID.

    Used by rollouts/remote.py and bifrost/provision.py when reconnecting
    to an already-provisioned node.
    """

    async def _run():
        client = _make_client()

        with patch(
            "broker.providers.runpod._make_graphql_request",
            new_callable=AsyncMock,
            return_value=_fake_runpod_pod_details(),
        ):
            instance = await client.get_instance("abc123", "runpod")

        assert instance is not None
        assert instance.id == "abc123"
        assert instance.provider == "runpod"
        assert instance.gpu_type == "RTX 4090"
        assert instance.status.value == "running"
        # SSH details should be populated (direct SSH path)
        assert instance.public_ip == "1.2.3.4"
        assert instance.ssh_port == 22222

    trio.run(_run)


# ---------------------------------------------------------------------------
# Path 4: List + Terminate (cleanup path)
# ---------------------------------------------------------------------------


def test_list_instances():
    """GPUClient.list_instances() enumerates running instances."""

    async def _run():
        client = _make_client()

        with patch(
            "broker.providers.runpod._make_graphql_request",
            new_callable=AsyncMock,
            return_value=_fake_runpod_list_response(),
        ):
            instances = await client.list_instances("runpod")

        assert len(instances) == 1
        assert instances[0].id == "abc123"

    trio.run(_run)


def test_terminate_instance():
    """GPUClient.terminate_instance() destroys an instance."""

    async def _run():
        client = _make_client()

        with patch(
            "broker.providers.runpod._make_graphql_request",
            new_callable=AsyncMock,
            return_value=_fake_runpod_terminate_response(),
        ):
            result = await client.terminate_instance("abc123", "runpod")

        assert result is True

    trio.run(_run)
