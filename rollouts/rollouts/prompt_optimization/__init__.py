"""GEPA v2: Prompt optimization for rollouts.

Multi-component prompt optimization using LLM-guided reflective mutations.

Three levels of API (continuous granularity):

1. **optimize_prompt()** - Simplest: optimize a single system prompt
2. **run_gepa()** - More control: custom adapter, config, validation set
3. **GEPAAdapter protocol** - Full control: implement for multi-component systems

Example (Level 1 - simplest):
    >>> from ..prompt_optimization import optimize_prompt, GEPAConfig
    >>> from ..dtypes import Endpoint
    >>>
    >>> result = await optimize_prompt(
    ...     system="Classify the query into a banking intent.",
    ...     user_template="Query: {query}\\nClassify:",
    ...     dataset=my_dataset,
    ...     score_fn=exact_match_score,
    ...     endpoint=Endpoint(provider="openai", model="gpt-4o-mini"),
    ... )
    >>> print(f"Best: {result.best_candidate['system']}")

Example (Level 2 - more control):
    >>> from ..prompt_optimization import run_gepa, GEPAConfig, SinglePromptAdapter
    >>>
    >>> adapter = SinglePromptAdapter(
    ...     endpoint=endpoint,
    ...     user_template="Query: {query}\\nClassify:",
    ...     score_fn=exact_match_score,
    ... )
    >>>
    >>> result = await run_gepa(
    ...     seed_candidate={"system": "You are a classifier."},
    ...     dataset=my_dataset,
    ...     adapter=adapter,
    ...     config=GEPAConfig(max_evaluations=500),
    ...     reflection_endpoint=reflection_endpoint,
    ... )

Example (Level 3 - full control):
    >>> class RAGAdapter:
    ...     async def evaluate(self, batch, candidate, capture_traces=False):
    ...         # candidate = {"query_rewriter": "...", "answer_gen": "..."}
    ...         ...
    ...
    ...     def make_reflective_dataset(self, candidate, eval_batch, components):
    ...         ...
    >>>
    >>> result = await run_gepa(
    ...     seed_candidate={"query_rewriter": "...", "answer_gen": "..."},
    ...     dataset=rag_dataset,
    ...     adapter=RAGAdapter(),
    ...     config=GEPAConfig(max_evaluations=1000),
    ...     reflection_endpoint=reflection_endpoint,
    ... )
"""

# Protocol
from .adapter import GEPAAdapter

# Adapters
from .adapters import SinglePromptAdapter

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
    # Protocol
    "GEPAAdapter",
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
    # Adapters
    "SinglePromptAdapter",
]
