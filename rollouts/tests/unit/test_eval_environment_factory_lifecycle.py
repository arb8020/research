from __future__ import annotations

import pytest
import trio

from rollouts.core import Endpoint, EvalConfig, Metric, Score
from rollouts.eval.native import evaluate
from rollouts.training.types import AttemptResult


class _FactoryWithLifecycle:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    def __call__(self, sample_data: dict[str, object]) -> object:
        del sample_data
        raise AssertionError("factory should not be called for an empty dataset")


def _prepare_messages(sample: dict[str, object]) -> list[object]:
    del sample
    return []


class _StaticScorer:
    async def score(self, result: object, context: object) -> Score:
        del result, context
        return Score(metrics=(Metric("reward", 0.0, weight=1.0),))


class _FactoryWithSlowStop:
    def __init__(self) -> None:
        self.started = trio.Event()
        self.stopped = 0
        self.stop_completed = trio.Event()

    async def start(self) -> None:
        self.started.set()

    async def stop(self) -> None:
        self.stopped += 1
        await trio.sleep(0.05)
        self.stop_completed.set()

    def __call__(self, sample_data: dict[str, object]) -> object:
        return object()


@pytest.mark.trio
async def test_eval_environment_factory_lifecycle_hooks() -> None:
    factory = _FactoryWithLifecycle()
    config = EvalConfig(
        endpoint=Endpoint(
            model="anthropic/test-model",
            base_url="https://api.anthropic.com/v1",
            api_format="anthropic-messages",
        ),
        prepare_messages=_prepare_messages,
        scorer=_StaticScorer(),
        environment_factory=factory,
        verbose=False,
        show_progress=False,
    )

    report = await evaluate(iter(()), config)

    assert report.total_samples == 0
    assert factory.started == 1
    assert factory.stopped == 1


@pytest.mark.trio
async def test_eval_environment_factory_stop_is_shielded_from_cancellation() -> None:
    factory = _FactoryWithSlowStop()

    async def _hanging_attempt(
        sample_data: dict[str, object],
        sample_id: str,
        environment: object,
        run_config: object,
    ) -> AttemptResult:
        del sample_data, sample_id, environment, run_config
        await trio.sleep_forever()

    config = EvalConfig(
        endpoint=None,
        prepare_messages=None,
        attempt_executor=_hanging_attempt,
        scorer=_StaticScorer(),
        environment_factory=factory,
        verbose=False,
        show_progress=False,
    )

    async def _run_eval() -> None:
        await evaluate(iter(({"text": "hello"},)), config)

    with trio.move_on_after(1):
        async with trio.open_nursery() as nursery:
            nursery.start_soon(_run_eval)
            await factory.started.wait()
            nursery.cancel_scope.cancel()

    assert factory.stopped == 1
    assert factory.stop_completed.is_set()
