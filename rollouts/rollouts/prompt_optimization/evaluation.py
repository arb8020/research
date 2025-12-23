"""Template evaluation for GEPA.

Evaluates prompt templates on datasets using the existing rollouts infrastructure.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any

import trio

from rollouts.agents import run_agent
from rollouts.dtypes import (
    Actor,
    AgentState,
    Endpoint,
    Environment,
    RunConfig,
    Score,
    StreamEvent,
    Trajectory,
)
from rollouts.training.types import Sample

from .formatting import format_prompt
from .types import PromptTemplate

logger = logging.getLogger(__name__)

# Type alias for score function
ScoreFn = Callable[[Sample], Score] | Callable[[Sample], Awaitable[Score]]

# Type alias for environment factory
EnvironmentFactory = Callable[[dict[str, Any]], Awaitable[Environment]]


async def _silent_chunk_handler(_: StreamEvent) -> None:
    """Silent handler for streaming events."""
    await trio.lowlevel.checkpoint()


async def evaluate_single_sample(
    template: PromptTemplate,
    sample: dict[str, Any],
    seed: int,
    endpoint: Endpoint,
    score_fn: ScoreFn,
    environment: Environment | None = None,
    run_config: RunConfig | None = None,
) -> float:
    """Evaluate a template on a single sample.

    Args:
        template: PromptTemplate to evaluate
        sample: Sample data dict
        seed: Sample index (for logging)
        endpoint: LLM endpoint configuration
        score_fn: Function to compute score from Sample
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

    # Build initial state
    state = AgentState(actor=actor, environment=environment)

    # Use provided run_config or create silent one
    if run_config is None:
        run_config = RunConfig(on_chunk=_silent_chunk_handler)

    # Run agent
    try:
        states = await run_agent(state, run_config)
        final_trajectory = states[-1].actor.trajectory
    except Exception as e:
        logger.warning(f"Sample {seed} failed: {e}")
        return 0.0

    # Build Sample for score function
    eval_sample = Sample(
        id=f"seed_{seed}",
        input=sample,
        ground_truth=sample.get("ground_truth") or sample.get("answer"),
        trajectory=final_trajectory,
    )

    # Compute score (support both sync and async)
    import inspect

    score_result = score_fn(eval_sample)
    if inspect.iscoroutine(score_result):
        score = await score_result
    else:
        score = score_result

    return score.reward


async def evaluate_template(
    template: PromptTemplate,
    seeds: Sequence[int],
    dataset: Sequence[dict[str, Any]],
    endpoint: Endpoint,
    score_fn: ScoreFn,
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
        score_fn: Function to compute score from Sample
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
        ...     score_fn=my_score_fn,
        ... )
        >>> print(f"Score: {scored_template.score}")
    """
    scores: list[float] = []

    async def eval_one(seed: int) -> float:
        sample = dataset[seed]
        env = await environment_factory(sample) if environment_factory else None
        return await evaluate_single_sample(
            template=template,
            sample=sample,
            seed=seed,
            endpoint=endpoint,
            score_fn=score_fn,
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
