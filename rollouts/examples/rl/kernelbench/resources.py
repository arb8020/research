"""Explicit KernelBench resource ownership for rollout and scoring stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rollouts.environments.kernelbench_multi import (
    KernelBenchMultiTurnEnvironment,
    SandboxPoolKernelEvaluator,
)
from rollouts.gpu_sandbox import SandboxPool

from .scoring import (
    DEFAULT_REWARD_WEIGHTS,
    KernelBenchRewardWeights,
    KernelBenchSampleScorer,
    KernelJudge,
)


async def _maybe_stop_owner(owner: Any) -> None:
    stop = getattr(owner, "stop", None)
    if callable(stop):
        await stop()


@dataclass
class KernelBenchRolloutResources:
    """Own rollout-side environment resources for multi-turn KernelBench."""

    evaluator: SandboxPoolKernelEvaluator
    backend: str = "cuda"
    max_turns: int = 8

    @classmethod
    def from_sandbox_configs(
        cls,
        sandbox_configs: list[Any] | None = None,
        *,
        backend: str = "cuda",
        max_turns: int = 8,
    ) -> KernelBenchRolloutResources:
        pool = SandboxPool(sandbox_configs or [])
        evaluator = SandboxPoolKernelEvaluator(pool)
        return cls(
            evaluator=evaluator,
            backend=backend,
            max_turns=max_turns,
        )

    async def start(self) -> None:
        await self.evaluator.start()

    async def stop(self) -> None:
        await _maybe_stop_owner(self.evaluator.pool)

    def stats(self) -> dict[str, Any]:
        return self.evaluator.stats()

    def __call__(self, sample_data: dict[str, Any]) -> KernelBenchMultiTurnEnvironment:
        return KernelBenchMultiTurnEnvironment(
            ref_code=sample_data.get("ref_code", ""),
            backend=self.backend,
            max_turns=self.max_turns,
            evaluator=self.evaluator,
        )


@dataclass
class KernelBenchScoringResources:
    """Own scorer-side resources for single-turn or judge-based scoring."""

    scorer: KernelBenchSampleScorer
    evaluator_pool: SandboxPool | None = None

    @classmethod
    def from_sandbox_configs(
        cls,
        sandbox_configs: list[Any] | None = None,
        *,
        reward_weights: KernelBenchRewardWeights = DEFAULT_REWARD_WEIGHTS,
        judge: KernelJudge | None = None,
        gate_reward_on_judge: bool = False,
        timeout: float = 120.0,
    ) -> KernelBenchScoringResources:
        pool = SandboxPool(sandbox_configs or [])
        evaluator = SandboxPoolKernelEvaluator(pool)
        scorer = KernelBenchSampleScorer(
            evaluator=evaluator,
            judge=judge,
            reward_weights=reward_weights,
            timeout=timeout,
            gate_reward_on_judge=gate_reward_on_judge,
        )
        return cls(
            scorer=scorer,
            evaluator_pool=pool,
        )

    @classmethod
    def metadata_only(
        cls,
        *,
        reward_weights: KernelBenchRewardWeights = DEFAULT_REWARD_WEIGHTS,
        judge: KernelJudge | None = None,
        gate_reward_on_judge: bool = False,
        timeout: float = 120.0,
    ) -> KernelBenchScoringResources:
        scorer = KernelBenchSampleScorer(
            judge=judge,
            reward_weights=reward_weights,
            timeout=timeout,
            gate_reward_on_judge=gate_reward_on_judge,
        )
        return cls(scorer=scorer)

    async def start(self) -> None:
        evaluator = self.scorer.evaluator
        if evaluator is not None:
            await evaluator.start()

    async def stop(self) -> None:
        if self.evaluator_pool is not None:
            await self.evaluator_pool.stop()

    def stats(self) -> dict[str, Any]:
        return self.scorer.stats()
