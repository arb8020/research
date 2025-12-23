"""GEPA engine - mid and high level orchestration.

Following: functions orchestrate objects, push ifs up.
"""

import logging
from collections.abc import Callable, Sequence

from rollouts.dtypes import Endpoint

from .adapter import GEPAAdapter
from .operations import (
    propose_mutation,
    sample_minibatch,
    select_from_pareto_front,
    update_pareto_front,
)
from .state import GEPAState
from .types import Candidate, GEPAConfig, GEPAResult

logger = logging.getLogger(__name__)


# ─── Mid Level: One Iteration ─────────────────────────────────────────────────


async def gepa_iteration(
    state: GEPAState,
    adapter: GEPAAdapter,
    trainset: Sequence[dict],
    reflection_endpoint: Endpoint,
    config: GEPAConfig,
) -> Candidate | None:
    """Run one GEPA iteration.

    Orchestration function - calls adapter methods and updates state.

    Steps:
    1. Select candidate from Pareto front
    2. Sample minibatch from trainset
    3. Evaluate with capture_traces=True
    4. Skip if all scores are perfect
    5. Select component to update (round-robin)
    6. Build reflective dataset from traces
    7. Propose mutation using reflection LLM
    8. Create new candidate with mutated component
    9. Evaluate new candidate on same minibatch
    10. Accept if improved (sum of scores)

    Args:
        state: Mutable GEPA state
        adapter: GEPAAdapter implementation
        trainset: Training samples
        reflection_endpoint: LLM endpoint for mutations
        config: Optimization config

    Returns:
        New candidate if one was accepted, None otherwise
    """
    # 1. Select candidate from Pareto front
    candidate_idx = select_from_pareto_front(state)
    candidate = state.candidates[candidate_idx]

    # 2. Sample minibatch
    minibatch_ids = sample_minibatch(trainset, state, config.minibatch_size)
    minibatch = [trainset[i] for i in minibatch_ids]

    # 3. Evaluate with traces
    eval_batch = await adapter.evaluate(minibatch, candidate, capture_traces=True)
    state.total_evaluations += len(minibatch)

    # 4. Skip if perfect
    if all(s >= config.perfect_score for s in eval_batch.scores):
        logger.debug("All scores perfect, skipping iteration")
        return None

    # 5. Select component (round-robin)
    component = state.next_component()

    # 6. Build reflective dataset
    reflective_data = adapter.make_reflective_dataset(candidate, eval_batch, [component])

    if component not in reflective_data or not reflective_data[component]:
        logger.warning(f"No reflective data for component {component}")
        return None

    # 7. Propose mutation
    new_text = await propose_mutation(
        candidate, component, reflective_data[component], reflection_endpoint
    )

    # 8. Create new candidate
    new_candidate = {**candidate, component: new_text}

    # 9. Evaluate new candidate on same minibatch
    new_eval = await adapter.evaluate(minibatch, new_candidate, capture_traces=False)
    state.total_evaluations += len(minibatch)

    # 10. Accept if improved
    old_sum = sum(eval_batch.scores)
    new_sum = sum(new_eval.scores)

    if new_sum > old_sum:
        logger.debug(f"Accepted mutation: {old_sum:.3f} -> {new_sum:.3f}")
        return new_candidate

    logger.debug(f"Rejected mutation: {old_sum:.3f} -> {new_sum:.3f}")
    return None


# ─── High Level: Full Optimization Loop ───────────────────────────────────────


async def run_gepa(
    seed_candidate: Candidate,
    dataset: Sequence[dict],
    adapter: GEPAAdapter,
    config: GEPAConfig,
    reflection_endpoint: Endpoint,
    valset: Sequence[dict] | None = None,
    on_iteration: Callable[[int, GEPAState], None] | None = None,
    seed: int | None = None,
) -> GEPAResult:
    """Run GEPA optimization loop.

    Main orchestration function. Runs iterations until evaluation budget exhausted.

    Args:
        seed_candidate: Initial candidate to start from
        dataset: Training samples (used for minibatches)
        adapter: GEPAAdapter implementation
        config: Optimization hyperparameters
        reflection_endpoint: LLM endpoint for proposing mutations
        valset: Validation samples (defaults to dataset)
        on_iteration: Optional callback after each iteration
        seed: Optional RNG seed for reproducibility

    Returns:
        GEPAResult with best candidate and statistics
    """
    valset = valset if valset is not None else dataset
    trainset = dataset

    # Initialize state
    state = GEPAState(seed_candidate)
    if seed is not None:
        state.seed(seed)

    # Initial validation eval
    logger.info("Running initial validation evaluation...")
    initial_eval = await adapter.evaluate(list(valset), seed_candidate, capture_traces=False)
    state.val_scores[0] = {i: s for i, s in enumerate(initial_eval.scores)}
    state.total_evaluations += len(valset)

    initial_score = state.get_best_score()
    logger.info(f"Initial score: {initial_score:.3f}")

    iteration = 0

    # Main loop
    while state.total_evaluations < config.max_evaluations:
        logger.debug(
            f"Iteration {iteration}, evaluations: {state.total_evaluations}/{config.max_evaluations}"
        )

        new_candidate = await gepa_iteration(state, adapter, trainset, reflection_endpoint, config)

        if new_candidate is not None:
            # Full validation eval
            val_eval = await adapter.evaluate(list(valset), new_candidate, capture_traces=False)
            state.total_evaluations += len(valset)

            # Add to population
            new_idx = state.add_candidate(new_candidate, val_eval.scores)

            # Update Pareto front
            update_pareto_front(state, new_idx)

            # Record history
            new_score = state.get_candidate_mean_score(new_idx)
            state.history.append({
                "iteration": iteration,
                "total_evaluations": state.total_evaluations,
                "new_score": new_score,
                "best_score": state.get_best_score(),
                "pareto_front_size": len(state.pareto_front),
                "num_candidates": len(state.candidates),
            })

            logger.info(
                f"Iteration {iteration}: new={new_score:.3f}, "
                f"best={state.get_best_score():.3f}, "
                f"front={len(state.pareto_front)}"
            )

        if on_iteration:
            on_iteration(iteration, state)

        iteration += 1

    # Final results
    best_score = state.get_best_score()
    logger.info(f"Optimization complete: {initial_score:.3f} -> {best_score:.3f}")

    return GEPAResult(
        best_candidate=state.get_best_candidate(),
        best_score=best_score,
        total_evaluations=state.total_evaluations,
        history=tuple(state.history),
    )


# ─── Highest Level: Convenience Function ──────────────────────────────────────


async def optimize_prompt(
    system: str,
    user_template: str,
    dataset: Sequence[dict],
    score_fn: Callable,
    endpoint: Endpoint,
    reflection_endpoint: Endpoint | None = None,
    config: GEPAConfig | None = None,
    environment_factory: Callable | None = None,
    valset: Sequence[dict] | None = None,
    seed: int | None = None,
) -> GEPAResult:
    """Optimize a single system prompt.

    Simplest API - wraps run_gepa with SinglePromptAdapter.

    Args:
        system: Initial system prompt to optimize
        user_template: Template for user messages (with {placeholders})
        dataset: List of sample dicts
        score_fn: Function to compute score from Sample
        endpoint: LLM endpoint for task evaluation
        reflection_endpoint: LLM endpoint for mutations (defaults to endpoint)
        config: Optimization config (defaults to GEPAConfig())
        environment_factory: Optional factory for tool-using agents
        valset: Validation samples (defaults to dataset)
        seed: Optional RNG seed for reproducibility

    Returns:
        GEPAResult with optimized prompt in best_candidate["system"]

    Example:
        >>> result = await optimize_prompt(
        ...     system="Classify the query.",
        ...     user_template="Query: {query}\\nClassify:",
        ...     dataset=my_dataset,
        ...     score_fn=exact_match,
        ...     endpoint=Endpoint(provider="openai", model="gpt-4o-mini"),
        ... )
        >>> print(result.best_candidate["system"])
    """
    from .adapters.single_prompt import SinglePromptAdapter

    adapter = SinglePromptAdapter(
        endpoint=endpoint,
        user_template=user_template,
        score_fn=score_fn,
        environment_factory=environment_factory,
    )

    return await run_gepa(
        seed_candidate={"system": system},
        dataset=dataset,
        adapter=adapter,
        config=config or GEPAConfig(),
        reflection_endpoint=reflection_endpoint or endpoint,
        valset=valset,
        seed=seed,
    )
