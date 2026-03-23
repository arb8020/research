from __future__ import annotations

import pytest

from broker.providers import runpod
from broker.types import ProvisionRequest


@pytest.mark.trio
async def test_runpod_provision_uses_docker_args_for_custom_image_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_make_graphql_request(
        query: str,
        variables: dict | None = None,
        api_key: str | None = None,
    ) -> dict[str, object]:
        captured["query"] = query
        captured["variables"] = variables
        captured["api_key"] = api_key
        return {
            "podFindAndDeployOnDemand": {
                "id": "pod-123",
                "machineId": "machine-123",
                "machine": {"podHostId": "host-123"},
            }
        }

    monkeypatch.setattr(runpod, "_make_graphql_request", _fake_make_graphql_request)

    request = ProvisionRequest(
        gpu_type="NVIDIA A100-SXM4-80GB",
        image="slimerl/slime:v0.2.3",
        docker_args="bash -c 'service ssh start; sleep infinity'",
    )

    instance = await runpod.provision_instance(request, api_key="token")

    assert instance is not None
    variables = captured["variables"]
    assert isinstance(variables, dict)
    pod_input = variables["input"]
    assert isinstance(pod_input, dict)
    assert pod_input["dockerArgs"] == "bash -c 'service ssh start; sleep infinity'"
    assert pod_input["ports"] == "22/tcp"
