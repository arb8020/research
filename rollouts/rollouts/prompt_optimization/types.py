"""GEPA v2 types.

Frozen dataclasses for data that doesn't change.
Following Casey Muratori: transparent, no hidden state.
"""

from dataclasses import dataclass
from typing import Any

# A candidate is a dict mapping component names to their text
# For single-prompt optimization: {"system": "You are a classifier..."}
# For RAG: {"query_rewriter": "...", "answer_gen": "...", ...}
Candidate = dict[str, str]


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
