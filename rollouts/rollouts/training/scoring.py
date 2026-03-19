from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .types import AttemptResult, AttemptRow, RolloutConfig, RolloutRuntime, Scorer, ScoringContext

if TYPE_CHECKING:
    from ..core import Score


async def score_result(
    scorer: Scorer,
    result: AttemptResult,
    context: ScoringContext | None = None,
) -> Score:
    from ..core import Score

    resolved_context = context if context is not None else ScoringContext()
    score = await scorer.score(result, resolved_context)
    if not isinstance(score, Score):
        raise TypeError(f"scorer.score() must return Score, got {type(score).__name__}")
    return score


def attach_score(result: AttemptResult, score: Score) -> AttemptResult:
    result.score = score
    result.reward = score.reward
    return result


async def score_results(
    scorer: Scorer,
    results: list[AttemptResult],
    contexts: list[ScoringContext | None] | None = None,
) -> list[AttemptResult]:
    if contexts is not None and len(contexts) != len(results):
        raise ValueError("contexts length must match results length")

    scored_results: list[AttemptResult] = []
    for index, result in enumerate(results):
        context = contexts[index] if contexts is not None else None
        scored_results.append(attach_score(result, await score_result(scorer, result, context)))
    return scored_results


async def score_rows(
    scorer: Scorer,
    rows: list[AttemptRow],
    contexts: list[ScoringContext | None] | None = None,
) -> list[AttemptRow]:
    if contexts is not None and len(contexts) != len(rows):
        raise ValueError("contexts length must match rows length")

    results = [row.to_result() for row in rows]
    await score_results(scorer, results, contexts=contexts)

    for index, (row, result) in enumerate(zip(rows, results, strict=True)):
        rows[index] = replace(
            row,
            reward=result.reward,
            score=result.score,
        )
    return rows


def resolve_scorer(
    *,
    config: RolloutConfig | None = None,
    runtime: RolloutRuntime | None = None,
    scorer: Scorer | None = None,
) -> Scorer | None:
    """Resolve the explicit scoring stage with clear precedence."""

    if scorer is not None:
        return scorer
    if runtime is not None and runtime.scorer is not None:
        return runtime.scorer
    if config is not None and config.scorer is not None:
        return config.scorer
    return None
