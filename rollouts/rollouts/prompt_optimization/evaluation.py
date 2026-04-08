"""Template evaluation for GEPA.

Evaluates prompt templates on datasets using the existing rollouts infrastructure.
"""

import logging
from collections.abc import Sequence
from dataclasses import replace
from inspect import isawaitable
from typing import Any

import trio

from ..agents import Actor, RunConfig, rollout
from ..core import Endpoint, Environment, EnvironmentFactory, Trajectory
from ..dtypes import StreamEvent
from ..training.scoring import score_result
from ..training.types import AttemptResult, DatasetRow, Scorer, ScoringContext, Status
from .formatting import format_prompt
from .types import PromptTemplate

logger = logging.getLogger(__name__)


async def _silent_chunk_handler(_: StreamEvent) -> None:
    """Silent handler for streaming events."""
    await trio.lowlevel.checkpoint()


async def evaluate_single_sample(
    template: PromptTemplate,
    sample: dict[str, Any],
    seed: int,
    endpoint: Endpoint,
    scorer: Scorer,
    environment: Environment | None = None,
    run_config: RunConfig | None = None,
) -> float:
    """Evaluate a template on a single sample.

    Args:
        template: PromptTemplate to evaluate
        sample: Problem row payload dict
        seed: Sample index (for logging)
        endpoint: LLM endpoint configuration
        scorer: Explicit scorer over the raw execution result
        environment: Optional environment for tool-using agents
        run_config: Optional run configuration

    Returns:
        Score value (float)
    """
    # Format prompt
    messages = format_prompt(template, sample)

    # Build trajectory with sample metadata
    trajectory = Trajectory(
        messages=messages,
        metadata={"sample_data": sample},
    )

    # Build actor
    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint,
        tools=environment.get_tools() if environment else [],
    )

    # Run single rollout (no agent loop - just one LLM call)
    try:
        result_actor = await rollout(actor, on_chunk=_silent_chunk_handler)
        final_trajectory = result_actor.trajectory
    except Exception as e:
        logger.warning(f"Sample {seed} failed: {e}")
        return 0.0

    # Build raw execution result for scoring.
    ground_truth = sample.get("ground_truth") or sample.get("answer") or sample.get("label")
    result = AttemptResult(
        attempt_id=f"seed_{seed}",
        problem=DatasetRow(
            problem_id=f"seed_{seed}",
            payload=sample,
            ground_truth=ground_truth,
        ),
        trajectory=final_trajectory,
        status=Status.COMPLETED,
    )

    score = await score_result(scorer, result, ScoringContext(environment=environment))
    return score.reward


async def evaluate_template(
    template: PromptTemplate,
    seeds: Sequence[int],
    dataset: Sequence[dict[str, Any]],
    endpoint: Endpoint,
    scorer: Scorer,
    environment_factory: EnvironmentFactory | None = None,
    max_concurrent: int = 10,
) -> PromptTemplate:
    """Evaluate template on multiple samples, return template with score.

    Async pure function: evaluates template on seeds, returns new template with score set.

    Args:
        template: PromptTemplate to evaluate
        seeds: Indices into dataset to evaluate on
        dataset: Full dataset (list of sample dicts)
        endpoint: LLM endpoint configuration
        scorer: Explicit scorer over raw execution results
        environment_factory: Optional factory for per-sample environments
        max_concurrent: Maximum parallel evaluations

    Returns:
        New PromptTemplate with score set to mean reward across seeds

    Example:
        >>> scored_template = await evaluate_template(
        ...     template=my_template,
        ...     seeds=(0, 1, 2, 3, 4),
        ...     dataset=my_dataset,
        ...     endpoint=my_endpoint,
        ...     scorer=my_scorer,
        ... )
        >>> print(f"Score: {scored_template.score}")
    """
    scores: list[float] = []

    async def eval_one(seed: int) -> float:
        sample = dataset[seed]
        if environment_factory:
            env_candidate = environment_factory(sample)
            env = await env_candidate if isawaitable(env_candidate) else env_candidate
        else:
            env = None
        return await evaluate_single_sample(
            template=template,
            sample=sample,
            seed=seed,
            endpoint=endpoint,
            scorer=scorer,
            environment=env,
        )

    # Run evaluations with concurrency limit
    async with trio.open_nursery() as nursery:
        limiter = trio.CapacityLimiter(max_concurrent)

        async def eval_with_limit(seed: int) -> None:
            async with limiter:
                score = await eval_one(seed)
                scores.append(score)

        for seed in seeds:
            nursery.start_soon(eval_with_limit, seed)

    # Compute mean score
    mean_score = sum(scores) / len(scores) if scores else 0.0

    # Return template with score set
    return replace(template, score=mean_score)
