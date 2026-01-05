"""Scoring functions for SWE-grep evaluation.

Implements:
1. Retrieval F1: File + line range overlap scoring
2. LLM-as-judge: Answer quality scoring (adapted from BrowseComp)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rollouts.dtypes import Endpoint, Message


@dataclass
class Source:
    """A source citation with file and line range."""

    file: str
    start_line: int
    end_line: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Source:
        return cls(
            file=data["file"],
            start_line=data["start_line"],
            end_line=data["end_line"],
        )

    def line_set(self) -> set[int]:
        """Return set of all line numbers in this source."""
        return set(range(self.start_line, self.end_line + 1))


@dataclass
class RetrievalScore:
    """Retrieval scoring result."""

    precision: float
    recall: float
    f1: float
    iou: float
    matched_sources: int
    total_gt_sources: int
    total_pred_sources: int


def compute_line_iou(pred: Source, gt: Source) -> float:
    """Compute IoU (intersection over union) for line ranges.

    Args:
        pred: Predicted source
        gt: Ground truth source

    Returns:
        IoU score in [0, 1]. Returns 0 if files don't match.
    """
    if pred.file != gt.file:
        return 0.0

    pred_lines = pred.line_set()
    gt_lines = gt.line_set()

    intersection = len(pred_lines & gt_lines)
    union = len(pred_lines | gt_lines)

    if union == 0:
        return 0.0

    return intersection / union


def compute_retrieval_score(
    predicted_sources: list[dict[str, Any]],
    ground_truth_sources: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> RetrievalScore:
    """Compute retrieval F1 score for predicted vs ground truth sources.

    Uses IoU-based matching:
    - Two sources match if they have IoU >= threshold
    - Each predicted source is matched to best ground truth source
    - Computes precision, recall, F1 over matched sources

    Args:
        predicted_sources: Agent's predicted sources
        ground_truth_sources: Ground truth sources from dataset
        iou_threshold: Minimum IoU to consider sources matching (default: 0.5)

    Returns:
        RetrievalScore with precision, recall, F1, and metadata
    """
    if not ground_truth_sources:
        raise ValueError("Ground truth sources cannot be empty")

    pred_sources = [Source.from_dict(s) for s in predicted_sources]
    gt_sources = [Source.from_dict(s) for s in ground_truth_sources]

    # Match each predicted source to best ground truth source
    matched_gt = set()  # Track which GT sources were matched
    total_iou = 0.0
    num_matches = 0

    for pred in pred_sources:
        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(gt_sources):
            iou = compute_line_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx is not None:
            matched_gt.add(best_gt_idx)
            total_iou += best_iou
            num_matches += 1

    # Compute metrics
    precision = num_matches / len(pred_sources) if pred_sources else 0.0
    recall = len(matched_gt) / len(gt_sources)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = total_iou / num_matches if num_matches > 0 else 0.0

    return RetrievalScore(
        precision=precision,
        recall=recall,
        f1=f1,
        iou=avg_iou,
        matched_sources=num_matches,
        total_gt_sources=len(gt_sources),
        total_pred_sources=len(pred_sources),
    )


# ────────────────────── LLM-as-Judge Answer Scoring ──────────────────────────


GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise."""


@dataclass
class AnswerScore:
    """Answer quality scoring result."""

    correct: bool
    grader_reasoning: str
    grader_response: str


async def grade_answer(
    question: str,
    predicted_answer: str,
    correct_answer: str,
    grader_endpoint: Endpoint,
) -> AnswerScore:
    """Grade answer quality using LLM-as-judge.

    Args:
        question: The original query
        predicted_answer: Agent's answer
        correct_answer: Ground truth reference answer
        grader_endpoint: Endpoint for grader model

    Returns:
        AnswerScore with correctness and reasoning
    """
    import re

    from rollouts.agents import run_actor

    grader_prompt = GRADER_TEMPLATE.format(
        question=question,
        response=predicted_answer,
        correct_answer=correct_answer,
    )

    messages = [Message(role="user", content=grader_prompt)]

    # Run grader
    response = await run_actor(
        messages=messages,
        endpoint=grader_endpoint,
        tools=None,
    )

    grader_response = response.content if hasattr(response, "content") else str(response)

    # Extract correctness
    match = re.search(r"correct:\s*(yes|no)", grader_response, re.IGNORECASE)
    is_correct = match.group(1).lower() == "yes" if match else False

    # Extract reasoning
    reasoning_match = re.search(
        r"reasoning:\s*(.+?)(?:\n\ncorrect:|$)", grader_response, re.DOTALL | re.IGNORECASE
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning extracted"

    return AnswerScore(
        correct=is_correct,
        grader_reasoning=reasoning,
        grader_response=grader_response,
    )
