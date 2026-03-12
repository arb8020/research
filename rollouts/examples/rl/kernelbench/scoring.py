"""Explicit KernelBench scoring stages shared by RL and evaluation."""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rollouts.core import Metric, Score
from rollouts.environments.resources import BatchKernelEvaluator, KernelEvaluator
from rollouts.training.types import Sample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KernelBenchRewardWeights:
    """Explicit reward weights for KernelBench scoring."""

    compiled: float = 0.2
    correct: float = 1.0
    speedup: float = 1.0


DEFAULT_REWARD_WEIGHTS = KernelBenchRewardWeights()
KEVIN_MULTI_TURN_REWARD_WEIGHTS = KernelBenchRewardWeights(
    compiled=0.0,
    correct=0.3,
    speedup=1.0,
)


@dataclass(frozen=True)
class KernelJudgeDecision:
    """Optional LLM-judge decision layered on top of execution scoring."""

    passed: bool
    score: float | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class KernelJudge(Protocol):
    async def judge_samples(
        self,
        samples: list[Sample],
        execution_results: list[dict[str, Any]],
    ) -> list[KernelJudgeDecision | None]: ...

    def stats(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FunctionKernelJudge:
    """Adapter for simple callable judge implementations."""

    judge_fn: Callable[
        [list[Sample], list[dict[str, Any]]],
        Awaitable[list[KernelJudgeDecision | None]] | list[KernelJudgeDecision | None],
    ]

    async def judge_samples(
        self,
        samples: list[Sample],
        execution_results: list[dict[str, Any]],
    ) -> list[KernelJudgeDecision | None]:
        result = self.judge_fn(samples, execution_results)
        if inspect.isawaitable(result):
            return await result
        return result


def extract_kernel_code(response: str) -> str | None:
    """Extract code from <kernel> tags, fenced python, or raw ModelNew text."""
    match = re.search(r"<kernel>\s*(.*?)\s*</kernel>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "class ModelNew" in code or "ModelNew" in code:
            return code

    if "class ModelNew" in response:
        start = response.find("class ModelNew")
        return response[start:].strip()

    return None


def build_kernelbench_score(
    *,
    compiled: float,
    correct: float,
    speedup: float,
    pass_rate: float,
    has_kernel_code: float,
    reward_weights: KernelBenchRewardWeights,
    error: str | None = None,
    judge_decision: KernelJudgeDecision | None = None,
) -> Score:
    """Build a Score object from explicit KernelBench metrics."""
    reward = (
        reward_weights.compiled * compiled
        + reward_weights.correct * correct
        + reward_weights.speedup * (speedup if correct > 0 else 0.0)
    )

    metrics = [
        Metric("reward", reward, weight=1.0),
        Metric("compiled", compiled, weight=0.0),
        Metric("correct", correct, weight=0.0),
        Metric("speedup", speedup, weight=0.0),
        Metric("pass_rate", pass_rate, weight=0.0),
        Metric("has_kernel_code", has_kernel_code, weight=0.0),
    ]
    if error:
        metrics.append(Metric("error", 0.0, weight=0.0, metadata={"message": error}))
    if judge_decision is not None:
        metrics.append(Metric("judge_passed", 1.0 if judge_decision.passed else 0.0, weight=0.0))
        if judge_decision.score is not None:
            metrics.append(Metric("judge_score", judge_decision.score, weight=0.0))

    return Score(metrics=tuple(metrics))


def _metadata_execution_result(sample: Sample) -> dict[str, Any] | None:
    """Use environment-produced metadata when the rollout already executed scoring."""
    metadata = sample.metadata
    if "best_speedup" not in metadata and "has_correct_kernel" not in metadata:
        return None

    turn_history = metadata.get("turn_history", [])
    compiled_any = any(
        turn.get("compiled", False) for turn in turn_history if isinstance(turn, dict)
    )
    runtime_provenance = metadata.get("evaluator_provenance")
    if runtime_provenance is None:
        for turn in turn_history:
            if isinstance(turn, dict) and isinstance(turn.get("runtime_provenance"), dict):
                runtime_provenance = turn["runtime_provenance"]
                break

    return {
        "compiled": 1.0 if compiled_any else 0.0,
        "correct": 1.0 if metadata.get("has_correct_kernel", False) else 0.0,
        "speedup": float(metadata.get("best_speedup", 0.0)),
        "pass_rate": float(metadata.get("pass_rate", 0.0)),
        "error": metadata.get("error"),
        "runtime_provenance": runtime_provenance,
        "has_kernel_code": 1.0 if extract_kernel_code(sample.response) else 0.0,
        "source": "environment_metadata",
    }


async def _score_one_with_evaluator(
    evaluator: KernelEvaluator,
    sample: Sample,
    timeout: float,
) -> dict[str, Any]:
    response = sample.response
    ref_code = sample.metadata.get("ref_code", "")
    kernel_code = extract_kernel_code(response)

    if not kernel_code or not ref_code:
        return {
            "compiled": 0.0,
            "correct": 0.0,
            "speedup": 0.0,
            "pass_rate": 0.0,
            "error": None,
            "runtime_provenance": None,
            "has_kernel_code": 0.0,
            "source": "missing_kernel_or_ref",
        }

    result = await evaluator.score_one(kernel_code, ref_code, timeout=timeout)
    return {
        "compiled": float(result.get("compiled", 0.0)),
        "correct": float(result.get("correct", 0.0)),
        "speedup": float(result.get("speedup", 0.0)),
        "pass_rate": float(result.get("pass_rate", 0.0)),
        "error": result.get("error"),
        "runtime_provenance": result.get("runtime_provenance"),
        "has_kernel_code": 1.0,
        "source": "evaluator",
    }


@dataclass
class KernelBenchSampleScorer:
    """Explicit scorer shared by KernelBench RL and eval paths.

    Behavior:
    - If the sample already has execution metadata from the environment, use it.
    - Otherwise score generated code via the injected evaluator.
    - Optionally run an LLM judge as a second scoring stage.
    """

    evaluator: KernelEvaluator | None = None
    judge: KernelJudge | None = None
    reward_weights: KernelBenchRewardWeights = DEFAULT_REWARD_WEIGHTS
    timeout: float = 120.0
    gate_reward_on_judge: bool = False

    async def score_samples(self, samples: list[Sample]) -> list[Sample]:
        execution_results = await self._score_execution(samples)
        judge_results = await self._score_judge(samples, execution_results)

        for sample, execution_result, judge_result in zip(
            samples, execution_results, judge_results, strict=False
        ):
            score = build_kernelbench_score(
                compiled=execution_result["compiled"],
                correct=execution_result["correct"],
                speedup=execution_result["speedup"],
                pass_rate=execution_result["pass_rate"],
                has_kernel_code=execution_result["has_kernel_code"],
                reward_weights=self.reward_weights,
                error=execution_result.get("error"),
                judge_decision=judge_result,
            )
            if self.gate_reward_on_judge and judge_result is not None and not judge_result.passed:
                score = Score(
                    metrics=tuple(
                        Metric(m.name, 0.0 if m.name == "reward" else m.value, m.weight, m.metadata)
                        for m in score.metrics
                    )
                )

            sample.score = score
            sample.reward = score.reward
            if execution_result.get("runtime_provenance") is not None:
                sample.metadata["evaluator_provenance"] = execution_result["runtime_provenance"]
            if judge_result is not None:
                sample.metadata["judge"] = {
                    "passed": judge_result.passed,
                    "score": judge_result.score,
                    "reason": judge_result.reason,
                    **judge_result.metadata,
                }

        return samples

    async def score_sample(self, sample: Sample) -> Score:
        await self.score_samples([sample])
        assert sample.score is not None, "score_samples must populate sample.score"
        return sample.score

    def stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        if self.evaluator is not None and hasattr(self.evaluator, "stats"):
            stats["evaluator"] = self.evaluator.stats()
        if self.judge is not None and hasattr(self.judge, "stats"):
            stats["judge"] = self.judge.stats()
        return stats

    async def _score_execution(self, samples: list[Sample]) -> list[dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(samples)
        batchable_indices: list[int] = []
        batch_requests: list[dict[str, Any]] = []

        for idx, sample in enumerate(samples):
            metadata_result = _metadata_execution_result(sample)
            if metadata_result is not None:
                results[idx] = metadata_result
                continue

            if self.evaluator is None:
                raise ValueError(
                    "KernelBenchSampleScorer requires an evaluator for samples without "
                    "KernelBench environment metadata."
                )

            response = sample.response
            ref_code = sample.metadata.get("ref_code", "")
            kernel_code = extract_kernel_code(response)
            if not kernel_code or not ref_code:
                results[idx] = {
                    "compiled": 0.0,
                    "correct": 0.0,
                    "speedup": 0.0,
                    "pass_rate": 0.0,
                    "error": None,
                    "runtime_provenance": None,
                    "has_kernel_code": 0.0,
                    "source": "missing_kernel_or_ref",
                }
                continue

            if isinstance(self.evaluator, BatchKernelEvaluator):
                batchable_indices.append(idx)
                batch_requests.append({"kernel_code": kernel_code, "ref_code": ref_code})
            else:
                results[idx] = await _score_one_with_evaluator(self.evaluator, sample, self.timeout)

        if batchable_indices:
            assert isinstance(self.evaluator, BatchKernelEvaluator)
            batch_results = await self.evaluator.score_batch(batch_requests, timeout=self.timeout)
            for idx, raw_result in zip(batchable_indices, batch_results, strict=False):
                results[idx] = {
                    "compiled": float(raw_result.get("compiled", 0.0)),
                    "correct": float(raw_result.get("correct", 0.0)),
                    "speedup": float(raw_result.get("speedup", 0.0)),
                    "pass_rate": float(raw_result.get("pass_rate", 0.0)),
                    "error": raw_result.get("error"),
                    "runtime_provenance": raw_result.get("runtime_provenance"),
                    "has_kernel_code": 1.0,
                    "source": "batch_evaluator",
                }

        return [result if result is not None else {} for result in results]

    async def _score_judge(
        self,
        samples: list[Sample],
        execution_results: list[dict[str, Any]],
    ) -> list[KernelJudgeDecision | None]:
        if self.judge is None:
            return [None] * len(samples)

        decisions = await self.judge.judge_samples(samples, execution_results)
        assert len(decisions) == len(samples), (
            f"Kernel judge returned {len(decisions)} decisions for {len(samples)} samples"
        )
        return decisions


def make_kernelbench_score_fn(
    scorer: KernelBenchSampleScorer,
) -> Callable[[Sample], Awaitable[Score]]:
    """Compatibility adapter for callers that still need a score_fn."""

    async def score_fn(sample: Sample) -> Score:
        return await scorer.score_sample(sample)

    return score_fn
