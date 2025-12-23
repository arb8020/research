"""GEPA types.

Frozen dataclasses for data that doesn't change.
Following Casey Muratori: transparent, no hidden state.

Two GEPA variants:
1. Reflective mutation (engine.py): Uses Candidate, EvaluationBatch, GEPAConfig, GEPAResult
2. Evolutionary (gepa.py): Uses PromptTemplate, EvolutionaryConfig, GenerationStats, OptimizationResult
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

# A candidate is a dict mapping component names to their text
# For single-prompt optimization: {"system": "You are a classifier..."}
# For RAG: {"query_rewriter": "...", "answer_gen": "...", ...}
Candidate = dict[str, str]


# ─── Evolutionary GEPA Types ───────────────────────────────────────────────────


@dataclass(frozen=True)
class FewShotExample:
    """A single few-shot example with input and output."""

    input: str
    output: str


@dataclass
class PromptTemplate:
    """A prompt template with system prompt, user template, and optional few-shot examples.

    Mutable because score gets set after evaluation.
    """

    system: str
    user_template: str
    few_shot_examples: tuple[FewShotExample, ...] = ()
    generation: int = 0
    score: float | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def format_user(self, sample: dict[str, Any]) -> str:
        """Format user template with sample fields."""
        return self.user_template.format(**sample)


@dataclass(frozen=True)
class EvolutionaryConfig:
    """Configuration for evolutionary GEPA optimization.

    Synth.ai example config:
        population_size=12, generations=20, children_per_gen=6
    """

    # Population
    population_size: int = 12
    generations: int = 20
    elite_size: int = 4  # Top N kept each generation

    # Reproduction rates
    mutation_rate: float = 0.7
    crossover_rate: float = 0.2  # Remaining is clone+mutate

    # Evaluation
    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)  # Indices into dataset for training eval
    val_seeds: tuple[int, ...] | None = None  # Indices for final validation

    # Concurrency
    max_concurrent: int = 10


@dataclass(frozen=True)
class GenerationStats:
    """Statistics for one generation of evolutionary optimization."""

    generation: int
    best_score: float
    mean_score: float
    std_score: float
    num_evaluated: int
    best_template_id: str


@dataclass(frozen=True)
class OptimizationResult:
    """Result of evolutionary GEPA optimization."""

    best_template: PromptTemplate
    final_population: tuple[PromptTemplate, ...]
    history: tuple[GenerationStats, ...]
    total_evaluations: int


@dataclass(frozen=True)
class EvaluationBatch:
    """Result of evaluating a candidate on a batch of samples.

    Frozen dataclass - immutable evaluation result.

    Attributes:
        outputs: Raw outputs per sample (e.g., LLM responses)
        scores: Scores per sample (0.0 to 1.0)
        trajectories: Optional execution traces for reflective mutation
    """

    outputs: tuple[Any, ...]
    scores: tuple[float, ...]
    trajectories: tuple[Any, ...] | None = None

    def __post_init__(self):
        assert len(self.outputs) == len(self.scores)
        if self.trajectories is not None:
            assert len(self.trajectories) == len(self.scores)


@dataclass(frozen=True)
class GEPAConfig:
    """GEPA optimization hyperparameters.

    Frozen dataclass - immutable configuration.

    Attributes:
        max_evaluations: Total evaluation budget (samples evaluated)
        minibatch_size: Samples per iteration for training eval
        perfect_score: Score threshold to skip optimization (already good enough)
    """

    max_evaluations: int = 500
    minibatch_size: int = 4
    perfect_score: float = 1.0

    def __post_init__(self):
        assert self.max_evaluations > 0, "max_evaluations must be positive"
        assert self.minibatch_size > 0, "minibatch_size must be positive"
        assert 0.0 <= self.perfect_score <= 1.0, "perfect_score must be in [0, 1]"


@dataclass(frozen=True)
class GEPAResult:
    """Result of GEPA optimization.

    Frozen dataclass - immutable result.

    Attributes:
        best_candidate: Candidate with highest mean validation score
        best_score: Highest mean validation score achieved
        total_evaluations: Total number of sample evaluations performed
        history: Per-iteration statistics
    """

    best_candidate: Candidate
    best_score: float
    total_evaluations: int
    history: tuple[dict[str, Any], ...]
