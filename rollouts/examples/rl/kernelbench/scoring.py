"""Explicit KernelBench scoring stages shared by RL and evaluation."""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from rollouts.core import Metric, Score
from rollouts.environments.resources import BatchKernelEvaluator, KernelEvaluator
from rollouts.training.types import AttemptRow, ScoringContext

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


class KernelJudgeMode(Enum):
    """Policy for deciding which samples are sent to the judge."""

    ALL = "all"
    COMPILED_ONLY = "compiled_only"
    CORRECT_ONLY = "correct_only"


@dataclass(frozen=True)
class KernelJudgePolicy:
    """Explicit policy for when judge scoring runs."""

    mode: KernelJudgeMode = KernelJudgeMode.COMPILED_ONLY
    max_samples_per_batch: int | None = None


@runtime_checkable
class KernelJudge(Protocol):
    async def judge_samples(
        self,
        samples: list[AttemptRow],
        execution_results: list[dict[str, Any]],
    ) -> list[KernelJudgeDecision | None]: ...

    def stats(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FunctionKernelJudge:
    """Adapter for simple callable judge implementations."""

    judge_fn: Callable[
        [list[AttemptRow], list[dict[str, Any]]],
        Awaitable[list[KernelJudgeDecision | None]] | list[KernelJudgeDecision | None],
    ]

    async def judge_samples(
        self,
        samples: list[AttemptRow],
        execution_results: list[dict[str, Any]],
    ) -> list[KernelJudgeDecision | None]:
        result = self.judge_fn(samples, execution_results)
        if inspect.isawaitable(result):
            return await result
        return result

    def stats(self) -> dict[str, Any]:
        return {}


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


def _metadata_execution_result(sample: AttemptRow) -> dict[str, Any] | None:
    """Use environment-produced metadata when the rollout already executed scoring."""
    metadata = sample.metadata
    status = metadata.get("status")
    error = metadata.get("error")
    if status in {"failed", "provider_error"} and error:
        runtime_provenance = metadata.get("evaluator_provenance") or metadata.get(
            "sandbox_runtime_provenance"
        )
        return {
            "compiled": 0.0,
            "correct": 0.0,
            "speedup": 0.0,
            "pass_rate": 0.0,
            "error": error,
            "runtime_provenance": runtime_provenance,
            "has_kernel_code": 1.0 if extract_kernel_code(sample.response) else 0.0,
            "source": "environment_failure",
        }

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
    sample: AttemptRow,
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
    judge_policy: KernelJudgePolicy = field(default_factory=KernelJudgePolicy)
    reward_weights: KernelBenchRewardWeights = DEFAULT_REWARD_WEIGHTS
    timeout: float = 120.0
    gate_reward_on_judge: bool = False
    _execution_metadata_count: int = field(default=0, init=False, repr=False)
    _execution_evaluator_count: int = field(default=0, init=False, repr=False)
    _execution_missing_code_count: int = field(default=0, init=False, repr=False)
    _judge_requested_count: int = field(default=0, init=False, repr=False)
    _judge_skipped_count: int = field(default=0, init=False, repr=False)
    _judge_gated_count: int = field(default=0, init=False, repr=False)

    async def score_samples(
        self,
        samples: list[AttemptRow],
        contexts: list[ScoringContext | None] | None = None,
    ) -> list[AttemptRow]:
        execution_results = await self._score_execution(samples, contexts=contexts)
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
                self._judge_gated_count += 1
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

    async def score_sample(self, sample: AttemptRow) -> Score:
        await self.score_samples([sample])
        assert sample.score is not None, "score_samples must populate sample.score"
        return sample.score

    def stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "execution": {
                "from_metadata": self._execution_metadata_count,
                "from_evaluator": self._execution_evaluator_count,
                "missing_code_or_ref": self._execution_missing_code_count,
            },
            "judge": {
                "policy": self.judge_policy.mode.value,
                "requested": self._judge_requested_count,
                "skipped": self._judge_skipped_count,
                "gated_rewards": self._judge_gated_count,
            },
        }
        if self.judge_policy.max_samples_per_batch is not None:
            stats["judge"]["max_samples_per_batch"] = self.judge_policy.max_samples_per_batch
        if self.evaluator is not None and hasattr(self.evaluator, "stats"):
            stats["evaluator"] = self.evaluator.stats()
        if self.judge is not None and hasattr(self.judge, "stats"):
            stats["judge_backend"] = self.judge.stats()
        return stats

    async def _score_execution(
        self,
        samples: list[AttemptRow],
        contexts: list[ScoringContext | None] | None = None,
    ) -> list[dict[str, Any]]:
        if contexts is None:
            contexts = [None] * len(samples)
        results: list[dict[str, Any] | None] = [None] * len(samples)
        batchable_indices: list[int] = []
        batch_requests: list[dict[str, Any]] = []

        for idx, (sample, context) in enumerate(zip(samples, contexts, strict=False)):
            metadata_result = _metadata_execution_result(sample)
            if metadata_result is not None:
                self._execution_metadata_count += 1
                results[idx] = metadata_result
                continue

            environment = context.environment if context is not None else None
            if environment is not None:
                runtime_metadata = getattr(environment, "get_runtime_metadata", None)
                if callable(runtime_metadata):
                    try:
                        extra_metadata = runtime_metadata()
                    except Exception:
                        extra_metadata = None
                    if isinstance(extra_metadata, dict):
                        sample.metadata.update(extra_metadata)
                        metadata_result = _metadata_execution_result(sample)
                        if metadata_result is not None:
                            self._execution_metadata_count += 1
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
                self._execution_missing_code_count += 1
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
                self._execution_evaluator_count += 1
                results[idx] = await _score_one_with_evaluator(self.evaluator, sample, self.timeout)

        if batchable_indices:
            assert isinstance(self.evaluator, BatchKernelEvaluator)
            batch_results = await self.evaluator.score_batch(batch_requests, timeout=self.timeout)
            self._execution_evaluator_count += len(batch_results)
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
        samples: list[AttemptRow],
        execution_results: list[dict[str, Any]],
    ) -> list[KernelJudgeDecision | None]:
        if self.judge is None:
            return [None] * len(samples)

        selected_indices = self._select_judge_indices(execution_results)
        self._judge_requested_count += len(selected_indices)
        self._judge_skipped_count += len(samples) - len(selected_indices)
        if not selected_indices:
            return [None] * len(samples)

        judged_samples = [samples[idx] for idx in selected_indices]
        judged_execution_results = [execution_results[idx] for idx in selected_indices]
        decisions = await self.judge.judge_samples(judged_samples, judged_execution_results)
        assert len(decisions) == len(judged_samples), (
            f"Kernel judge returned {len(decisions)} decisions for {len(judged_samples)} samples"
        )

        full_results: list[KernelJudgeDecision | None] = [None] * len(samples)
        for idx, decision in zip(selected_indices, decisions, strict=False):
            full_results[idx] = decision
        return full_results

    def _select_judge_indices(self, execution_results: list[dict[str, Any]]) -> list[int]:
        selected_indices: list[int] = []
        for idx, result in enumerate(execution_results):
            if self.judge_policy.mode == KernelJudgeMode.ALL:
                selected_indices.append(idx)
                continue
            if (
                self.judge_policy.mode == KernelJudgeMode.COMPILED_ONLY
                and result.get("compiled", 0.0) > 0
            ):
                selected_indices.append(idx)
                continue
            if (
                self.judge_policy.mode == KernelJudgeMode.CORRECT_ONLY
                and result.get("correct", 0.0) > 0
            ):
                selected_indices.append(idx)

        max_samples = self.judge_policy.max_samples_per_batch
        if max_samples is not None:
            return selected_indices[:max_samples]
        return selected_indices
