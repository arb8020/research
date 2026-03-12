import logging
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from rollouts.training.configs import CheckpointConfig
from rollouts.training.train import train


class FakeFuture:
    def __init__(self, result: dict[str, float]) -> None:
        self._result = result

    async def result(self) -> dict[str, float]:
        return self._result


class FakeBackend:
    def forward_backward(self, _batch: object) -> FakeFuture:
        return FakeFuture({"loss": 0.0})

    def optim_step(self) -> FakeFuture:
        return FakeFuture({"grad_norm": 0.0})


class FakeWeightSyncer:
    def __init__(self) -> None:
        self.sync_calls = 0
        self.closed = False

    async def sync(self) -> None:
        self.sync_calls += 1

    async def close(self) -> None:
        self.closed = True


@pytest.mark.trio
async def test_train_calls_sync_hooks_around_weight_sync() -> None:
    events: list[str] = []
    syncer = FakeWeightSyncer()

    async def batch_iterator() -> AsyncIterator[dict[str, int]]:
        yield {"batch": 1}

    async def process_batch(_step: int, _batch: object, _backend: object) -> dict[str, float]:
        events.append("process")
        return {"reward": 1.0}

    async def before_weight_sync() -> None:
        events.append("before_sync")

    async def after_weight_sync() -> None:
        events.append("after_sync")

    result = await train(
        config=CheckpointConfig(
            num_steps=1, checkpoint_every=100, sync_weights_every=1, log_every=1
        ),
        backend=cast(Any, FakeBackend()),
        batch_iterator=batch_iterator(),
        process_batch=process_batch,
        weight_syncer=cast(Any, syncer),
        save_checkpoint=None,
        metrics_logger=None,
        logger=logging.getLogger("test_train_sync_hooks"),
        before_weight_sync=before_weight_sync,
        after_weight_sync=after_weight_sync,
    )

    assert [entry["step"] for entry in result.metrics_history] == [1]
    assert events == ["process", "before_sync", "after_sync"]
    assert syncer.sync_calls == 1
    assert syncer.closed is True
