import logging

import pytest

from rollouts.training.grpo import (
    _maybe_start_environment_factory,
    _maybe_stop_environment_factory,
)


class _FakeEnvironmentFactory:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1


@pytest.mark.trio
async def test_grpo_environment_factory_lifecycle_hooks() -> None:
    factory = _FakeEnvironmentFactory()
    logger = logging.getLogger("test.grpo.environment_factory")

    await _maybe_start_environment_factory(factory, logger)
    await _maybe_stop_environment_factory(factory, logger)

    assert factory.started == 1
    assert factory.stopped == 1


@pytest.mark.trio
async def test_grpo_environment_factory_lifecycle_hooks_ignore_none() -> None:
    logger = logging.getLogger("test.grpo.environment_factory.none")

    await _maybe_start_environment_factory(None, logger)
    await _maybe_stop_environment_factory(None, logger)
