"""GEPA v2: Prompt optimization for rollouts.

Multi-component prompt optimization using LLM-guided reflective mutations.

Two levels of API (continuous granularity):

1. **optimize_prompt()** - Simplest: optimize a single system prompt
2. **run_gepa()** - More control: custom evaluate/reflect functions, config, validation set

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

Example (Level 2 - more control with pure functions):
    >>> from functools import partial
    >>> from rollouts.prompt_optimization import run_gepa, GEPAConfig
    >>> from rollouts.prompt_optimization.adapters import (
    ...     SystemPromptConfig, evaluate_system_prompt, make_system_prompt_reflective
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

Example (Terminal-bench with pure functions):
    >>> from rollouts.prompt_optimization.adapters import (
    ...     TerminalBenchConfig, evaluate_terminal_bench, make_terminal_bench_reflective
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

# Adapters (configs, pure functions, wrapper classes)
from .adapters import (
    SinglePromptAdapter,
    SinglePromptConfig,
    SystemPromptAdapter,
    # System prompt
    SystemPromptConfig,
    # System + user prompt
    SystemUserPromptAdapter,
    TerminalBenchAdapter,
    # Terminal-bench
    TerminalBenchConfig,
    TerminalBenchTask,
    evaluate_system_prompt,
    evaluate_terminal_bench,
    make_system_prompt_reflective,
    make_terminal_bench_reflective,
    run_tests_and_score,
)

# Reflective mutation (engine.py)
from .engine import gepa_iteration, optimize_prompt, run_gepa

# Evolutionary (gepa.py)
from .gepa import run_evolutionary_gepa

# Low-level operations (for advanced use)
from .operations import (
    dominates,
    propose_mutation,
    sample_minibatch,
    select_from_pareto_front,
    update_pareto_front,
)

# State (for advanced use)
from .state import GEPAState

# Types
from .types import (
    # Reflective mutation types
    Candidate,
    EvaluationBatch,
    # Evolutionary types
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
    # Reflective mutation types
    "Candidate",
    "EvaluationBatch",
    "GEPAConfig",
    "GEPAResult",
    # Evolutionary types
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
    # Reflective mutation
    "gepa_iteration",
    "run_gepa",
    "optimize_prompt",
    # Evolutionary
    "run_evolutionary_gepa",
    # Adapters - system prompt
    "SystemPromptConfig",
    "SinglePromptConfig",
    "evaluate_system_prompt",
    "make_system_prompt_reflective",
    "SystemPromptAdapter",
    "SinglePromptAdapter",
    # Adapters - system + user prompt
    "SystemUserPromptAdapter",
    # Adapters - terminal-bench
    "TerminalBenchConfig",
    "TerminalBenchTask",
    "evaluate_terminal_bench",
    "make_terminal_bench_reflective",
    "run_tests_and_score",
    "TerminalBenchAdapter",
]
