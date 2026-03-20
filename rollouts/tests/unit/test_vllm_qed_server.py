from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

import pytest

import rollouts.inference.realizations.qed_vllm as vllm_qed_server
from rollouts.inference.realizations.qed_vllm import (
    _dispatch_init_weight_update_group,
    _dispatch_receive_weight_update,
)
from rollouts.training.weight_sync_protocol import InitWeightUpdateGroupResponse


class FakeEngineClient:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def collective_rpc(self, method: str, args: tuple[Any, ...]) -> Any | Awaitable[Any]:
        self.calls.append((method, args))
        return self._result


class AsyncFakeEngineClient(FakeEngineClient):
    def collective_rpc(self, method: str, args: tuple[Any, ...]) -> Any | Awaitable[Any]:
        self.calls.append((method, args))

        async def _result() -> Any:
            return self._result

        return _result()


@pytest.mark.trio
async def test_dispatch_init_weight_update_group_normalizes_request() -> None:
    client = FakeEngineClient([{"rank": 1, "world_size": 2, "group_name": "weight_sync"}])

    response = await _dispatch_init_weight_update_group(
        client,
        {
            "master_address": "127.0.0.1",
            "master_port": 29517,
            "rank_offset": 1,
            "world_size": 2,
            "group_name": "weight_sync",
            "backend": "nccl",
        },
    )

    assert response == {"status": "ok", "results": client._result}
    assert client.calls == [
        (
            "init_weight_update_group",
            ("127.0.0.1", 29517, 1, 2, "weight_sync", 300.0),
        )
    ]


@pytest.mark.trio
async def test_dispatch_init_weight_update_group_uses_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncFakeEngineClient({"rank": 1})
    observed: list[float] = []

    async def _fake_wait_for(awaitable: Awaitable[Any], timeout: float) -> Any:
        observed.append(timeout)
        return await awaitable

    monkeypatch.setattr(vllm_qed_server.asyncio, "wait_for", _fake_wait_for)

    await _dispatch_init_weight_update_group(
        client,
        {
            "master_address": "127.0.0.1",
            "master_port": 29517,
            "rank_offset": 1,
            "world_size": 2,
            "group_name": "weight_sync",
            "timeout_seconds": 123.0,
        },
    )

    assert observed == [123.0]


@pytest.mark.trio
async def test_dispatch_receive_weight_update_forwards_load_names() -> None:
    client = FakeEngineClient({"status": "ok", "num_tensors": 1})

    response = await _dispatch_receive_weight_update(
        client,
        {
            "names": ["wire.weight"],
            "load_names": ["load.weight"],
            "shapes": [[2, 3]],
            "dtypes": ["float16"],
        },
    )

    assert response == {"status": "ok", "results": {"status": "ok", "num_tensors": 1}}
    assert client.calls == [
        (
            "receive_weight_update",
            (["wire.weight"], [[2, 3]], ["float16"], ["load.weight"]),
        )
    ]


def test_init_weight_update_group_response_rejects_in_band_error() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        InitWeightUpdateGroupResponse.from_dict({
            "status": "error",
            "error_type": "RuntimeError",
            "error": "boom",
        })
