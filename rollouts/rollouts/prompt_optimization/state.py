"""GEPAState - mutable state for GEPA optimization.

Regular class (not frozen) - legitimate persistent state.
This is the ONLY mutable thing in the GEPA system.

Following: classes only for legitimate state that needs lifecycle management.
"""

import random
from collections.abc import Sequence

from .types import Candidate


class GEPAState:
    """Mutable state for GEPA optimization.

    This class owns:
    - All candidates ever created
    - Pareto front (indices of non-dominated candidates)
    - Per-candidate validation scores
    - Evaluation counter
    - RNG state

    The rest of GEPA is pure functions that operate on this state.
    """

    def __init__(self, seed_candidate: Candidate):
        """Initialize state with seed candidate.

        Args:
            seed_candidate: Initial candidate to start optimization from
        """
        # Candidates
        self.candidates: list[Candidate] = [seed_candidate]

        # Pareto front: indices of non-dominated candidates
        # A candidate is on the front if no other candidate dominates it on ALL examples
        self.pareto_front: set[int] = {0}

        # Validation scores: candidate_idx -> {example_idx -> score}
        self.val_scores: dict[int, dict[int, float]] = {}

        # Evaluation counter
        self.total_evaluations: int = 0

        # History for reporting
        self.history: list[dict] = []

        # RNG for reproducibility
        self.rng: random.Random = random.Random()

        # Component rotation counter (for round-robin component selection)
        self._component_counter: int = 0

    def seed(self, seed: int) -> None:
        """Set RNG seed for reproducibility."""
        self.rng.seed(seed)

    def add_candidate(self, candidate: Candidate, val_scores: Sequence[float]) -> int:
        """Add a new candidate with its validation scores.

        Args:
            candidate: New candidate to add
            val_scores: Scores on validation set (one per example)

        Returns:
            Index of the new candidate
        """
        idx = len(self.candidates)
        self.candidates.append(candidate)
        self.val_scores[idx] = {i: s for i, s in enumerate(val_scores)}
        return idx

    def get_candidate_mean_score(self, idx: int) -> float:
        """Get mean validation score for a candidate."""
        scores = self.val_scores.get(idx, {})
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)

    def get_best_candidate_idx(self) -> int:
        """Get index of candidate with highest mean validation score."""
        if not self.val_scores:
            return 0
        return max(self.val_scores.keys(), key=self.get_candidate_mean_score)

    def get_best_candidate(self) -> Candidate:
        """Get candidate with highest mean validation score."""
        return self.candidates[self.get_best_candidate_idx()]

    def get_best_score(self) -> float:
        """Get highest mean validation score."""
        if not self.val_scores:
            return 0.0
        return self.get_candidate_mean_score(self.get_best_candidate_idx())

    def get_component_names(self) -> list[str]:
        """Get list of component names from seed candidate."""
        return list(self.candidates[0].keys())

    def next_component(self) -> str:
        """Get next component to update (round-robin)."""
        components = self.get_component_names()
        component = components[self._component_counter % len(components)]
        self._component_counter += 1
        return component
