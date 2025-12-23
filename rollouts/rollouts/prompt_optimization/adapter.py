"""GEPAAdapter protocol.

Protocol (structural typing) - no inheritance required.
Following: classes only for legitimate state, protocols for contracts.
"""

from collections.abc import Sequence
from typing import Protocol

from .types import Candidate, EvaluationBatch


class GEPAAdapter(Protocol):
    """Integration point between GEPA engine and task-specific logic.

    Protocol (structural typing) - implement these methods, no inheritance needed.

    For multi-component systems like RAG pipelines, implement this protocol.
    For simple single-prompt optimization, use optimize_prompt() instead.

    Key insight from reference GEPA:
    - evaluate() runs the candidate and returns scores
    - make_reflective_dataset() extracts per-component feedback from traces
    - This separation allows GEPA to optimize each component independently

    Example:
        >>> class MyAdapter:
        ...     async def evaluate(self, batch, candidate, capture_traces=False):
        ...         # Run candidate on batch, return EvaluationBatch
        ...         ...
        ...
        ...     def make_reflective_dataset(self, candidate, eval_batch, components):
        ...         # Extract feedback for each component
        ...         ...
    """

    async def evaluate(
        self,
        batch: Sequence[dict],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate candidate on a batch of samples.

        Args:
            batch: List of sample dicts from dataset
            candidate: Dict mapping component names to their text
            capture_traces: If True, include execution traces in result
                           (needed for reflective mutation)

        Returns:
            EvaluationBatch with:
            - outputs: Raw outputs per sample
            - scores: Scores per sample (0.0 to 1.0)
            - trajectories: Execution traces if capture_traces=True
        """
        ...

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """Extract per-component feedback from execution traces.

        This is the key to reflective mutation. Instead of just saying
        "improve this prompt", we show the LLM:
        - What inputs the component received
        - What outputs it produced
        - What went wrong (feedback)

        Args:
            candidate: Current candidate being optimized
            eval_batch: Evaluation result with trajectories (must have been
                       called with capture_traces=True)
            components_to_update: Which components to extract feedback for

        Returns:
            Dict mapping component name to list of feedback items.
            Each item should have keys like:
            - "Inputs": What the component received
            - "Generated Outputs": What the component produced
            - "Feedback": What went wrong or could be improved

        Example:
            >>> feedback = adapter.make_reflective_dataset(
            ...     candidate, eval_batch, ["system"]
            ... )
            >>> feedback["system"]
            [
                {
                    "Inputs": "Query: How do I reset my PIN?",
                    "Generated Outputs": "card_arrival",
                    "Feedback": "Incorrect. Expected: change_pin",
                },
                ...
            ]
        """
        ...
