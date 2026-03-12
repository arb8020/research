from __future__ import annotations

from typing import Any, cast

import pytest

from examples.rl.kernelbench.scoring import (
    KEVIN_MULTI_TURN_REWARD_WEIGHTS,
    FunctionKernelJudge,
    KernelBenchSampleScorer,
    KernelJudgeDecision,
    KernelJudgeMode,
    KernelJudgePolicy,
)
from rollouts.core import Metric, Score
from rollouts.eval.native import _compute_score
from rollouts.training.types import AttemptRow, ProblemRow, ScoringContext


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
    def __init__(self) -> None:
        self.last_contexts: list[ScoringContext | None] | None = None

    async def score_samples(
        self,
        samples: list[AttemptRow],
        contexts: list[ScoringContext | None] | None = None,
    ) -> list[AttemptRow]:
        self.last_contexts = contexts
        for sample in samples:
            sample.score = Score(metrics=(Metric("reward", 3.0, weight=1.0),))
            sample.reward = 3.0
        return samples


def make_attempt(*, response: str = "", ref_code: str = "class Model: pass") -> AttemptRow:
    sample = AttemptRow(
        attempt_id="attempt-1",
        problem=ProblemRow(problem_id="prompt", payload={"prompt": "prompt"}),
        metadata={"ref_code": ref_code},
    )

    class FakeTrajectory:
        messages = [{"role": "assistant", "content": response}]

    sample.trajectory = cast(Any, FakeTrajectory())
    return sample


@pytest.mark.trio
async def test_kernelbench_scorer_uses_environment_metadata_without_evaluator() -> None:
    sample = AttemptRow(
        attempt_id="attempt-1",
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
    sample = AttemptRow(
        attempt_id="attempt-1",
        problem=ProblemRow(problem_id="prompt", payload={"prompt": "prompt"}),
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
    sample = AttemptRow(
        attempt_id="attempt-1",
        problem=ProblemRow(problem_id="prompt", payload={"prompt": "prompt"}),
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

    assert len(evaluator.requests) == 1
    assert evaluator.requests[0]["kernel_code"] == "class ModelNew:\n    pass"
    assert evaluator.requests[0]["ref_code"] == "class Model: pass"
    assert sample.score is not None
    assert sample.reward == pytest.approx(3.7)
    assert sample.metadata["evaluator_provenance"] == {"backend": "fake"}


@pytest.mark.trio
async def test_kernelbench_scorer_preserves_error_metadata_from_evaluator() -> None:
    sample = AttemptRow(
        attempt_id="attempt-1",
        problem=ProblemRow(problem_id="prompt", payload={"prompt": "prompt"}),
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
    sample = AttemptRow(attempt_id="sample-1")
    scorer = FakeSampleScorer()

    score = await _compute_score(
        None,
        sample,
        sample_scorer=scorer,
        scoring_context=ScoringContext(metadata={"phase": "eval"}),
    )

    assert score.reward == 3.0
    assert sample.reward == 3.0
    assert scorer.last_contexts is not None
    assert scorer.last_contexts[0] is not None
    assert scorer.last_contexts[0].metadata == {"phase": "eval"}


@pytest.mark.trio
async def test_kernelbench_scorer_short_circuits_environment_failures() -> None:
    sample = AttemptRow(
        attempt_id="attempt-1",
        problem=ProblemRow(problem_id="prompt", payload={"prompt": "prompt"}),
        metadata={
            "status": "failed",
            "error": "ProvisionError: no GPU offers found",
            "sandbox_runtime_provenance": {"backend": "modal"},
        },
    )

    class FakeTrajectory:
        messages = [{"role": "assistant", "content": "```python\nclass ModelNew:\n    pass\n```"}]

    sample.trajectory = cast(Any, FakeTrajectory())
    scorer = KernelBenchSampleScorer()

    await scorer.score_samples([sample])

    assert sample.score is not None
    assert sample.reward == 0.0
    error_metrics = [metric for metric in sample.score.metrics if metric.name == "error"]
    assert len(error_metrics) == 1
    assert error_metrics[0].metadata == {"message": "ProvisionError: no GPU offers found"}
    assert sample.metadata["evaluator_provenance"] == {"backend": "modal"}


@pytest.mark.trio
async def test_kernelbench_judge_policy_only_judges_compiled_samples() -> None:
    compiled_sample = make_attempt(response="```python\nclass ModelNew:\n    pass\n```")
    missing_code_sample = make_attempt(response="no kernel here")
    judge_calls: list[list[AttemptRow]] = []

    async def judge_fn(
        samples: list[AttemptRow],
        _execution_results: list[dict[str, object]],
    ) -> list[KernelJudgeDecision | None]:
        judge_calls.append(samples)
        return [KernelJudgeDecision(passed=True, score=0.9) for _ in samples]

    scorer = KernelBenchSampleScorer(
        evaluator=FakeBatchEvaluator(),
        judge=FunctionKernelJudge(judge_fn),
        judge_policy=KernelJudgePolicy(mode=KernelJudgeMode.COMPILED_ONLY),
    )

    await scorer.score_samples([compiled_sample, missing_code_sample])

    assert len(judge_calls) == 1
    assert judge_calls[0] == [compiled_sample]
    assert compiled_sample.metadata["judge"]["passed"] is True
    assert "judge" not in missing_code_sample.metadata


@pytest.mark.trio
async def test_kernelbench_judge_can_gate_reward() -> None:
    sample = make_attempt(response="```python\nclass ModelNew:\n    pass\n```")

    async def judge_fn(
        samples: list[AttemptRow],
        _execution_results: list[dict[str, object]],
    ) -> list[KernelJudgeDecision | None]:
        return [
            KernelJudgeDecision(
                passed=False,
                score=0.1,
                reason="reward hack",
                metadata={"label": "suspicious"},
            )
            for _ in samples
        ]

    scorer = KernelBenchSampleScorer(
        evaluator=FakeBatchEvaluator(),
        judge=FunctionKernelJudge(judge_fn),
        judge_policy=KernelJudgePolicy(mode=KernelJudgeMode.ALL),
        gate_reward_on_judge=True,
    )

    await scorer.score_samples([sample])

    assert sample.score is not None
    assert sample.reward == 0.0
    assert sample.metadata["judge"]["passed"] is False
    assert sample.metadata["judge"]["score"] == 0.1
    assert sample.metadata["judge"]["reason"] == "reward hack"
    assert sample.metadata["judge"]["label"] == "suspicious"
    judge_metrics = {metric.name: metric.value for metric in sample.score.metrics}
    assert judge_metrics["reward"] == 0.0
    assert judge_metrics["judge_passed"] == 0.0
    assert judge_metrics["judge_score"] == 0.1
