from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import trio

from broker.providers import runpod


@dataclass
class _FakeStatus:
    value: str


@dataclass
class _FakeInstance:
    id: str = "pod-123"
    provider: str = "runpod"
    api_key: str = "token"
    public_ip: str | None = "1.2.3.4"
    ssh_port: int | None = 2200
    ssh_username: str | None = "root"
    status: _FakeStatus = field(default_factory=lambda: _FakeStatus("running"))
    raw_data: dict | None = None


@pytest.mark.trio
async def test_wait_for_direct_ssh_connectivity_retries_until_probe_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _FakeInstance()
    probe_attempts = {"count": 0}

    async def _fake_get_instance_details(instance_id: str, api_key: str) -> _FakeInstance:
        assert instance_id == "pod-123"
        assert api_key == "token"
        return instance

    async def _fake_test_ssh_connectivity(current_instance: _FakeInstance) -> bool:
        assert current_instance is instance
        probe_attempts["count"] += 1
        return probe_attempts["count"] >= 3

    async def _fast_sleep(seconds: float) -> None:
        await trio.lowlevel.checkpoint()

    monkeypatch.setattr(runpod, "get_instance_details", _fake_get_instance_details)
    monkeypatch.setattr(runpod, "_test_ssh_connectivity", _fake_test_ssh_connectivity)
    monkeypatch.setattr(runpod.trio, "sleep", _fast_sleep)

    ready = await runpod._wait_for_direct_ssh_connectivity(instance, timeout=60)

    assert ready is True
    assert probe_attempts["count"] == 3
