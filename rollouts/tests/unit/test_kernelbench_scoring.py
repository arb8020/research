from __future__ import annotations

from typing import Any, cast

import pytest

from examples.rl.kernelbench.scoring import (
    KEVIN_MULTI_TURN_REWARD_WEIGHTS,
    KernelBenchSampleScorer,
)
from rollouts.core import Metric, Score
from rollouts.eval.native import _compute_score
from rollouts.training.types import Sample


class FakeBatchEvaluator:
    def __init__(self) -> None:
        self.requests: list[dict[str, str]] = []

    async def start(self) -> None:
        return None

    async def score_one(self, kernel_code: str, ref_code: str, timeout: float) -> dict:
        return {
            "compiled": 1.0,
            "correct": 1.0,
            "speedup": 2.5,
            "pass_rate": 1.0,
            "runtime_provenance": {"backend": "fake"},
        }

    async def score_batch(self, requests: list[dict[str, str]], timeout: float) -> list[dict]:
        self.requests.extend(requests)
        return [
            {
                "compiled": 1.0,
                "correct": 1.0,
                "speedup": 2.5,
                "pass_rate": 1.0,
                "runtime_provenance": {"backend": "fake"},
            }
            for _ in requests
        ]

    def stats(self) -> dict[str, float]:
        return {"num_requests": float(len(self.requests))}


class ErrorBatchEvaluator(FakeBatchEvaluator):
    async def score_batch(self, requests: list[dict[str, str]], timeout: float) -> list[dict]:
        self.requests.extend(requests)
        return [
            {
                "compiled": 0.0,
                "correct": 0.0,
                "speedup": 0.0,
                "pass_rate": 0.0,
                "error": "compile failed",
                "runtime_provenance": {"backend": "fake"},
            }
            for _ in requests
        ]


class FakeSampleScorer:
    async def score_samples(self, samples: list[Sample]) -> list[Sample]:
        for sample in samples:
            sample.score = Score(metrics=(Metric("reward", 3.0, weight=1.0),))
            sample.reward = 3.0
        return samples


@pytest.mark.trio
async def test_kernelbench_scorer_uses_environment_metadata_without_evaluator() -> None:
    sample = Sample(
        metadata={
            "best_speedup": 1.75,
            "has_correct_kernel": True,
            "turn_history": [{"turn": 1, "compiled": True, "correct": True}],
        },
        trajectory=None,
    )
    scorer = KernelBenchSampleScorer(
        reward_weights=KEVIN_MULTI_TURN_REWARD_WEIGHTS,
    )

    await scorer.score_samples([sample])

    assert sample.score is not None
    assert sample.reward == pytest.approx(2.05)
    assert sample.score.reward == pytest.approx(2.05)


@pytest.mark.trio
async def test_kernelbench_scorer_requires_evaluator_without_env_metadata() -> None:
    sample = Sample(
        prompt="prompt",
        metadata={"ref_code": "class Model: pass"},
    )

    class FakeTrajectory:
        messages = [{"role": "assistant", "content": "```python\nclass ModelNew:\n    pass\n```"}]

    sample.trajectory = cast(Any, FakeTrajectory())
    scorer = KernelBenchSampleScorer()

    with pytest.raises(ValueError, match="requires an evaluator"):
        await scorer.score_samples([sample])


@pytest.mark.trio
async def test_kernelbench_scorer_uses_injected_batch_evaluator() -> None:
    sample = Sample(
        prompt="prompt",
        metadata={"ref_code": "class Model: pass"},
    )
    sample.trajectory = None
    _ = sample.response
    sample_dict_response = "```python\nclass ModelNew:\n    pass\n```"
    sample.metadata["response_override"] = sample_dict_response

    # Provide a minimal trajectory-like object through from_dict-friendly shape.
    class FakeTrajectory:
        messages = [{"role": "assistant", "content": sample_dict_response}]

    sample.trajectory = cast(Any, FakeTrajectory())

    evaluator = FakeBatchEvaluator()
    scorer = KernelBenchSampleScorer(evaluator=evaluator)

    await scorer.score_samples([sample])

    assert evaluator.requests == [
        {
            "kernel_code": "class ModelNew:\n    pass",
            "ref_code": "class Model: pass",
        }
    ]
    assert sample.score is not None
    assert sample.reward == pytest.approx(3.7)
    assert sample.metadata["evaluator_provenance"] == {"backend": "fake"}


@pytest.mark.trio
async def test_kernelbench_scorer_preserves_error_metadata_from_evaluator() -> None:
    sample = Sample(
        prompt="prompt",
        metadata={"ref_code": "class Model: pass"},
    )

    class FakeTrajectory:
        messages = [{"role": "assistant", "content": "```python\nclass ModelNew:\n    pass\n```"}]

    sample.trajectory = cast(Any, FakeTrajectory())
    scorer = KernelBenchSampleScorer(evaluator=ErrorBatchEvaluator())

    await scorer.score_samples([sample])

    assert sample.score is not None
    assert sample.reward == 0.0
    error_metrics = [metric for metric in sample.score.metrics if metric.name == "error"]
    assert len(error_metrics) == 1
    assert error_metrics[0].metadata == {"message": "compile failed"}
    assert sample.metadata["evaluator_provenance"] == {"backend": "fake"}


@pytest.mark.trio
async def test_eval_compute_score_supports_sample_scorer() -> None:
    sample = Sample(id="sample-1")

    score = await _compute_score(None, sample, sample_scorer=FakeSampleScorer())

    assert score.reward == 3.0
    assert sample.reward == 3.0
