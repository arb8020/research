"""GEPA: Prompt optimization for rollouts.

Multi-component prompt optimization using LLM-guided reflective mutations.

Two levels of API:

1. **optimize_prompt()** - Simplest: optimize a single system prompt
2. **run_gepa()** - More control: custom evaluate/reflect functions

Example (Level 1 - simplest):
    >>> from rollouts.prompt_optimization import optimize_prompt, GEPAConfig
    >>> from rollouts.dtypes import Endpoint
    >>>
    >>> result = await optimize_prompt(
    ...     system="Classify the query into a banking intent.",
    ...     user_template="Query: {query}\\nClassify:",
    ...     dataset=my_dataset,
    ...     score_fn=exact_match_score,
    ...     endpoint=Endpoint(provider="openai", model="gpt-4o-mini"),
    ... )
    >>> print(f"Best: {result.best_candidate['system']}")

Example (Level 2 - pure functions):
    >>> from functools import partial
    >>> from rollouts.prompt_optimization import (
    ...     run_gepa, GEPAConfig,
    ...     SystemPromptConfig, evaluate_system_prompt, make_system_prompt_reflective,
    ... )
    >>>
    >>> config = SystemPromptConfig(
    ...     endpoint=endpoint,
    ...     user_template="Query: {query}\\nClassify:",
    ...     score_fn=exact_match_score,
    ... )
    >>>
    >>> result = await run_gepa(
    ...     seed_candidate={"system": "You are a classifier."},
    ...     dataset=my_dataset,
    ...     evaluate_fn=partial(evaluate_system_prompt, config),
    ...     make_reflective_fn=make_system_prompt_reflective,
    ...     config=GEPAConfig(max_evaluations=500),
    ...     reflection_endpoint=reflection_endpoint,
    ... )

Example (Terminal-bench):
    >>> from rollouts.prompt_optimization import (
    ...     TerminalBenchConfig, evaluate_terminal_bench, make_terminal_bench_reflective,
    ... )
    >>>
    >>> config = TerminalBenchConfig(endpoint=endpoint, max_turns=30)
    >>>
    >>> result = await run_gepa(
    ...     seed_candidate={"instruction_prompt": "You are a terminal agent..."},
    ...     dataset=[{"task_id": "fix-permissions"}],
    ...     evaluate_fn=partial(evaluate_terminal_bench, config),
    ...     make_reflective_fn=make_terminal_bench_reflective,
    ...     config=GEPAConfig(max_evaluations=100),
    ...     reflection_endpoint=endpoint,
    ... )
"""

# Type aliases for adapter functions
from .adapter import EvaluateFn, MakeReflectiveFn

# Adapters (configs + pure functions)
from .adapters import (
    # System prompt
    SystemPromptConfig,
    # System + user prompt
    SystemUserPromptConfig,
    # Terminal-bench
    TerminalBenchConfig,
    TerminalBenchTask,
    evaluate_system_prompt,
    evaluate_system_user_prompt,
    evaluate_terminal_bench,
    make_system_prompt_reflective,
    make_system_user_prompt_reflective,
    make_terminal_bench_reflective,
    run_tests_and_score,
)

# Evolutionary GEPA (population-based genetic algorithm)
from .evolutionary import run_evolutionary_gepa

# Low-level operations (for advanced use)
from .operations import (
    dominates,
    propose_mutation,
    sample_minibatch,
    select_from_pareto_front,
    update_pareto_front,
)

# Reflective mutation GEPA (official algorithm)
from .reflective import gepa_iteration, optimize_prompt, run_gepa

# State (for advanced use)
from .state import GEPAState

# Types
from .types import (
    Candidate,
    EvaluationBatch,
    EvolutionaryConfig,
    GenerationStats,
    GEPAConfig,
    GEPAResult,
    OptimizationResult,
    PromptTemplate,
)

__all__ = [
    # Type aliases
    "EvaluateFn",
    "MakeReflectiveFn",
    # Types
    "Candidate",
    "EvaluationBatch",
    "GEPAConfig",
    "GEPAResult",
    "EvolutionaryConfig",
    "GenerationStats",
    "OptimizationResult",
    "PromptTemplate",
    # State
    "GEPAState",
    # Low-level operations
    "propose_mutation",
    "select_from_pareto_front",
    "update_pareto_front",
    "dominates",
    "sample_minibatch",
    # Reflective GEPA
    "gepa_iteration",
    "run_gepa",
    "optimize_prompt",
    # Evolutionary GEPA
    "run_evolutionary_gepa",
    # Adapters - system prompt
    "SystemPromptConfig",
    "evaluate_system_prompt",
    "make_system_prompt_reflective",
    # Adapters - system + user prompt
    "SystemUserPromptConfig",
    "evaluate_system_user_prompt",
    "make_system_user_prompt_reflective",
    # Adapters - terminal-bench
    "TerminalBenchConfig",
    "TerminalBenchTask",
    "evaluate_terminal_bench",
    "make_terminal_bench_reflective",
    "run_tests_and_score",
]
