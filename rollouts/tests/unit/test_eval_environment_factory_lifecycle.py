from __future__ import annotations

import pytest

from rollouts.core import Endpoint, EvalConfig
from rollouts.eval.native import evaluate


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


def _score_fn(sample: object) -> object:
    return sample


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
        score_fn=_score_fn,
        environment_factory=factory,
        verbose=False,
        show_progress=False,
    )

    report = await evaluate(iter(()), config)

    assert report.total_samples == 0
    assert factory.started == 1
    assert factory.stopped == 1
