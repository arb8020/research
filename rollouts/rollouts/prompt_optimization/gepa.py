"""GEPA (Generative Evolution of Prompt Architectures) optimization.

Evolutionary prompt optimization using LLM-guided mutations and crossover.
"""

import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import trio

from rollouts.dtypes import Endpoint, Message

from .evaluation import EnvironmentFactory, ScoreFn, evaluate_template
from .types import GenerationStats, GEPAConfig, OptimizationResult, PromptTemplate

logger = logging.getLogger(__name__)


# ─── Selection ────────────────────────────────────────────────────────────────


def select_elites(
    population: Sequence[PromptTemplate],
    n: int,
) -> list[PromptTemplate]:
    """Select top-N templates by score.

    Pure function: sorts by score descending, returns top N.

    Args:
        population: List of scored templates
        n: Number of elites to select

    Returns:
        List of top N templates by score

    Example:
        >>> elites = select_elites(population, n=4)
        >>> assert len(elites) == 4
        >>> assert all(e.score is not None for e in elites)
    """
    # Sort by score descending (None scores treated as -inf)
    sorted_pop = sorted(
        population,
        key=lambda t: t.score if t.score is not None else float("-inf"),
        reverse=True,
    )
    return list(sorted_pop[:n])


def tournament_select(
    population: Sequence[PromptTemplate],
    tournament_size: int = 3,
) -> PromptTemplate:
    """Select one template via tournament selection.

    Args:
        population: List of scored templates
        tournament_size: Number of candidates in tournament

    Returns:
        Winner of tournament (highest score)
    """
    candidates = random.sample(list(population), min(tournament_size, len(population)))
    return max(candidates, key=lambda t: t.score if t.score is not None else float("-inf"))


# ─── Mutation ─────────────────────────────────────────────────────────────────

MUTATION_PROMPT = """You are an expert prompt engineer. Your task is to improve a system prompt for an LLM.

Current system prompt:
```
{current_prompt}
```

The current prompt achieves a score of {score:.2%} on the task.

Analyze the prompt and suggest an improved version. Consider:
- Clarity and specificity of instructions
- Task decomposition and step-by-step reasoning
- Output format specifications
- Edge case handling
- Avoiding ambiguity

Respond with ONLY the improved system prompt, nothing else. Do not include markdown code blocks or explanations."""


async def mutate_template(
    template: PromptTemplate,
    endpoint: Endpoint,
    generation: int,
) -> PromptTemplate:
    """Use LLM to propose an improved system prompt.

    Async pure function: queries LLM for mutation, returns new template.

    Args:
        template: Template to mutate
        endpoint: LLM endpoint for mutation proposals
        generation: Current generation number

    Returns:
        New PromptTemplate with mutated system prompt
    """
    from rollouts.agents import rollout
    from rollouts.dtypes import Actor, Trajectory

    # Build mutation prompt
    mutation_request = MUTATION_PROMPT.format(
        current_prompt=template.system,
        score=template.score or 0.0,
    )

    messages = [Message(role="user", content=mutation_request)]
    trajectory = Trajectory(messages=messages)
    actor = Actor(trajectory=trajectory, endpoint=endpoint, tools=[])

    # Silent handler
    async def silent(_: Any) -> None:
        await trio.lowlevel.checkpoint()

    # Get mutation from LLM
    try:
        new_actor = await rollout(actor, on_chunk=silent)
        last_msg = new_actor.trajectory.messages[-1]

        # Extract text content
        if isinstance(last_msg.content, str):
            mutated_system = last_msg.content.strip()
        elif isinstance(last_msg.content, list):
            # Handle content blocks
            mutated_system = ""
            for block in last_msg.content:
                if hasattr(block, "text"):
                    mutated_system += block.text
            mutated_system = mutated_system.strip()
        else:
            mutated_system = template.system  # Fallback

        # Return new template with mutated system prompt
        return PromptTemplate(
            system=mutated_system,
            user_template=template.user_template,
            few_shot_examples=template.few_shot_examples,
            generation=generation,
        )

    except Exception as e:
        logger.warning(f"Mutation failed: {e}, returning original")
        return replace(template, generation=generation)


# ─── Crossover ────────────────────────────────────────────────────────────────


def crossover_templates(
    parent_a: PromptTemplate,
    parent_b: PromptTemplate,
    generation: int,
) -> PromptTemplate:
    """Combine two parent prompts via simple crossover.

    Pure function: creates child by combining parent prompts.

    Strategy: Split each parent's system prompt into sentences,
    interleave sentences from each parent.

    Args:
        parent_a: First parent template
        parent_b: Second parent template
        generation: Current generation number

    Returns:
        New child template with combined system prompt
    """
    import re

    # Split into sentences
    def split_sentences(text: str) -> list[str]:
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    sentences_a = split_sentences(parent_a.system)
    sentences_b = split_sentences(parent_b.system)

    # Interleave sentences
    combined = []
    max_len = max(len(sentences_a), len(sentences_b))
    for i in range(max_len):
        if i < len(sentences_a):
            combined.append(sentences_a[i])
        if i < len(sentences_b) and sentences_b[i] not in combined:
            combined.append(sentences_b[i])

    # Join back
    new_system = " ".join(combined)

    return PromptTemplate(
        system=new_system,
        user_template=parent_a.user_template,  # Keep from parent_a
        few_shot_examples=parent_a.few_shot_examples,
        generation=generation,
    )


# ─── GEPA Orchestration ───────────────────────────────────────────────────────


async def run_gepa(
    initial_template: PromptTemplate,
    config: GEPAConfig,
    dataset: Sequence[dict[str, Any]],
    endpoint: Endpoint,
    mutation_endpoint: Endpoint,
    score_fn: ScoreFn,
    environment_factory: EnvironmentFactory | None = None,
    on_generation: Callable[[int, list[PromptTemplate]], None] | None = None,
) -> OptimizationResult:
    """Run GEPA evolutionary optimization loop.

    Async orchestration: manages population across generations.

    Args:
        initial_template: Starting prompt template
        config: GEPA hyperparameters
        dataset: List of sample dicts
        endpoint: LLM endpoint for task evaluation
        mutation_endpoint: LLM endpoint for proposing mutations (can be cheaper)
        score_fn: Function to compute score from Sample
        environment_factory: Optional factory for per-sample environments
        on_generation: Optional callback after each generation

    Returns:
        OptimizationResult with best template and history

    Example:
        >>> result = await run_gepa(
        ...     initial_template=template,
        ...     config=GEPAConfig(population_size=20, generations=10),
        ...     dataset=my_dataset,
        ...     endpoint=task_endpoint,
        ...     mutation_endpoint=mutation_endpoint,
        ...     score_fn=my_score_fn,
        ... )
        >>> print(f"Best score: {result.best_template.score}")
    """
    # Initialize population with copies of initial template
    population = [replace(initial_template, generation=0) for _ in range(config.population_size)]

    history: list[GenerationStats] = []
    total_evaluations = 0

    logger.info(
        f"Starting GEPA optimization: {config.generations} generations, population={config.population_size}"
    )

    for gen in range(config.generations):
        logger.info(f"Generation {gen + 1}/{config.generations}")

        # ─── Evaluate population on training seeds ────────────────────────
        scored_population: list[PromptTemplate] = []

        async with trio.open_nursery() as nursery:
            limiter = trio.CapacityLimiter(config.max_concurrent)
            results: list[tuple[int, PromptTemplate]] = []

            async def eval_candidate(idx: int, template: PromptTemplate) -> None:
                async with limiter:
                    scored = await evaluate_template(
                        template=template,
                        seeds=config.train_seeds,
                        dataset=dataset,
                        endpoint=endpoint,
                        score_fn=score_fn,
                        environment_factory=environment_factory,
                        max_concurrent=1,  # Already limiting at population level
                    )
                    results.append((idx, scored))

            for i, template in enumerate(population):
                nursery.start_soon(eval_candidate, i, template)

        # Sort by original index to maintain order
        results.sort(key=lambda x: x[0])
        scored_population = [r[1] for r in results]
        total_evaluations += len(scored_population) * len(config.train_seeds)

        # ─── Compute generation statistics ────────────────────────────────
        scores = [t.score for t in scored_population if t.score is not None]
        if scores:
            best_score = max(scores)
            mean_score = sum(scores) / len(scores)
            std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
        else:
            best_score = mean_score = std_score = 0.0

        best_template = select_elites(scored_population, n=1)[0]

        stats = GenerationStats(
            generation=gen,
            best_score=best_score,
            mean_score=mean_score,
            std_score=std_score,
            num_evaluated=len(scored_population),
            best_template_id=best_template.id,
        )
        history.append(stats)

        logger.info(f"  Best: {best_score:.3f}, Mean: {mean_score:.3f} ± {std_score:.3f}")

        # Callback
        if on_generation:
            on_generation(gen, scored_population)

        # ─── Selection & reproduction ─────────────────────────────────────
        if gen < config.generations - 1:  # Skip on last generation
            # Select elites
            elites = select_elites(scored_population, n=config.elite_size)

            # Generate children
            children: list[PromptTemplate] = []
            num_children = config.population_size - config.elite_size

            for _ in range(num_children):
                if random.random() < config.mutation_rate:
                    # Mutation
                    parent = tournament_select(scored_population)
                    child = await mutate_template(parent, mutation_endpoint, gen + 1)
                elif random.random() < config.crossover_rate:
                    # Crossover
                    parent_a = tournament_select(scored_population)
                    parent_b = tournament_select(scored_population)
                    child = crossover_templates(parent_a, parent_b, gen + 1)
                else:
                    # Clone with mutation
                    parent = tournament_select(scored_population)
                    child = await mutate_template(parent, mutation_endpoint, gen + 1)

                children.append(child)

            # New population = elites + children
            population = list(elites) + children

    # ─── Final validation on val_seeds ────────────────────────────────────
    if config.val_seeds:
        logger.info("Validating top candidates on validation set...")
        top_candidates = select_elites(scored_population, n=min(5, len(scored_population)))

        validated: list[PromptTemplate] = []
        for template in top_candidates:
            scored = await evaluate_template(
                template=template,
                seeds=config.val_seeds,
                dataset=dataset,
                endpoint=endpoint,
                score_fn=score_fn,
                environment_factory=environment_factory,
                max_concurrent=config.max_concurrent,
            )
            validated.append(scored)
            total_evaluations += len(config.val_seeds)

        best_template = select_elites(validated, n=1)[0]
        logger.info(f"Best validation score: {best_template.score:.3f}")
    else:
        best_template = select_elites(scored_population, n=1)[0]

    return OptimizationResult(
        best_template=best_template,
        final_population=tuple(scored_population),
        history=tuple(history),
        total_evaluations=total_evaluations,
    )
