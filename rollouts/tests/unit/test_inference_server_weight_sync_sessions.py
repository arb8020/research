from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from rollouts.inference import server as inference_server


@dataclass
class _FakeReq:
    uid: int
    input_ids: torch.Tensor


class _FakeEngine:
    def __init__(self) -> None:
        self.tokenizer = object()
        self.config = SimpleNamespace(model_path="fake-model", dtype=torch.bfloat16)
        self.scheduler_config = SimpleNamespace(max_batch_size=1)
        self.device = torch.device("cpu")
        self.state = SimpleNamespace(decode_set=[])
        self._next_uid = 0

    def add_request(self, input_ids: list[int], sampling_params: object) -> int:
        self._next_uid += 1
        return self._next_uid

    def has_pending(self) -> bool:
        return False

    def step(self) -> list[_FakeReq]:
        return []

    def reload_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        del state_dict


class _FakeReceiver:
    instances: list[_FakeReceiver] = []

    def __init__(
        self,
        *,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        group_name: str,
        device: torch.device,
    ) -> None:
        self.master_addr = master_addr
        self.master_port = master_port
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        self.device = device
        self.init_calls = 0
        self.cleanup_calls = 0
        _FakeReceiver.instances.append(self)

    def init_group(self) -> None:
        self.init_calls += 1

    def cleanup(self) -> None:
        self.cleanup_calls += 1

    def receive_weights(self, param_infos: list[object]) -> dict[str, torch.Tensor]:
        del param_infos
        return {}


def _make_client(monkeypatch: MonkeyPatch) -> tuple[TestClient, _FakeEngine]:
    engine = _FakeEngine()
    monkeypatch.setattr("rollouts.inference.weight_sync.WeightSyncReceiver", _FakeReceiver)
    inference_server._clear_nccl_state()
    _FakeReceiver.instances.clear()
    app = inference_server.create_app(engine)
    client = TestClient(app)
    return client, engine


def _close_client(client: TestClient) -> None:
    client.app.state.inference_server._engine_thread.stop()
    client.close()
    inference_server._clear_nccl_state()


def test_reuses_matching_weight_sync_receiver(monkeypatch: MonkeyPatch) -> None:
    client, _engine = _make_client(monkeypatch)
    try:
        request = {
            "master_address": "127.0.0.1",
            "master_port": 29500,
            "rank_offset": 1,
            "world_size": 2,
            "group_name": "weight_sync_session",
            "backend": "nccl",
        }
        first = client.post("/init_weights_update_group", json=request)
        second = client.post("/init_weights_update_group", json=request)

        assert first.status_code == 200
        assert first.json()["reused"] is False
        assert second.status_code == 200
        assert second.json()["reused"] is True
        assert len(_FakeReceiver.instances) == 1
        assert _FakeReceiver.instances[0].init_calls == 1
        assert _FakeReceiver.instances[0].cleanup_calls == 0
    finally:
        _close_client(client)


def test_replaces_mismatched_weight_sync_receiver(monkeypatch: MonkeyPatch) -> None:
    client, _engine = _make_client(monkeypatch)
    try:
        first = client.post(
            "/init_weights_update_group",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29500,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "weight_sync_session_a",
                "backend": "nccl",
            },
        )
        second = client.post(
            "/init_weights_update_group",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29501,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "weight_sync_session_b",
                "backend": "nccl",
            },
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["reused"] is False
        assert len(_FakeReceiver.instances) == 2
        assert _FakeReceiver.instances[0].cleanup_calls == 1
        assert _FakeReceiver.instances[1].init_calls == 1
    finally:
        _close_client(client)


def test_rejects_weight_update_for_wrong_group(monkeypatch: MonkeyPatch) -> None:
    client, _engine = _make_client(monkeypatch)
    try:
        init_response = client.post(
            "/init_weights_update_group",
            json={
                "master_address": "127.0.0.1",
                "master_port": 29500,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "weight_sync_session",
                "backend": "nccl",
            },
        )
        update_response = client.post(
            "/update_weights_from_distributed",
            json={
                "group_name": "different_group",
                "weight_version": "v1",
                "names": [],
                "shapes": [],
                "dtypes": [],
            },
        )

        assert init_response.status_code == 200
        assert update_response.status_code == 400
        assert "NCCL update group mismatch" in update_response.json()["detail"]
    finally:
        _close_client(client)
