#!/usr/bin/env python3
"""
SWE-grep evaluation: Multi-tool code/document retrieval with structured output.

Tests agent ability to:
1. Use grep/glob/search/read tools to find relevant sources
2. Cite specific file + line ranges
3. Synthesize natural language answers

Scoring:
- Retrieval F1: Precision/recall over (file, line_range) citations with IoU matching
- Answer Quality: LLM-as-judge semantic correctness

Usage:
    python -m examples.eval.swe_grep.eval_swe_grep --corpus /path/to/docs --questions sample_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trio

from rollouts.agents import handle_stop_max_turns, run_agent
from rollouts.dtypes import (
    Actor,
    AgentState,
    Endpoint,
    Message,
    RunConfig,
    Trajectory,
)
from rollouts.environments.swe_grep import SWEGrepEnvironment

from .scoring import compute_retrieval_score, grade_answer

logger = logging.getLogger(__name__)


# ──────────────────────── Configuration ──────────────────────────────────────


@dataclass
class SWEGrepConfig:
    """SWE-grep evaluation configuration."""

    corpus_path: Path
    """Path to document corpus."""

    questions_path: Path
    """Path to questions JSONL file."""

    agent_endpoint: Endpoint
    """Endpoint for agent model."""

    grader_endpoint: Endpoint
    """Endpoint for grader model."""

    max_turns: int = 15
    """Maximum agent turns per query."""

    iou_threshold: float = 0.5
    """Minimum IoU for source matching."""

    max_samples: int | None = None
    """Maximum number of samples to evaluate (None = all)."""

    retrieval_weight: float = 0.5
    """Weight for retrieval score in combined metric."""

    answer_weight: float = 0.5
    """Weight for answer score in combined metric."""


# ──────────────────────── Dataset Loading ────────────────────────────────────


def load_questions(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load questions from JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if max_samples is not None and max_samples < len(questions):
        questions = questions[:max_samples]

    logger.info(f"Loaded {len(questions)} questions from {path}")
    return questions


# ──────────────────────── System Prompt ──────────────────────────────────────


SWE_GREP_SYSTEM_PROMPT = """You are a code/documentation search assistant. Your task is to find relevant sources and answer questions about a codebase or documentation corpus.

## Available Tools

You have access to the following tools:

1. **grep(pattern, path, case_sensitive, literal)**: Search for regex patterns or literal strings in files
   - Returns files with line numbers and excerpts
   - Use for finding specific strings, function names, or patterns

2. **glob(pattern)**: Find files matching glob patterns
   - Examples: `**/*.md`, `src/**/gemm*.cpp`, `**/test_*.py`
   - Use for finding files by name or structure

3. **search(query, top_k)**: Semantic search over corpus
   - Use natural language queries
   - Returns ranked relevant documents

4. **read(path, start_line, end_line)**: Read file content
   - Optionally specify line range
   - Use after finding relevant files to examine content

5. **submit(sources, answer)**: Submit your final answer
   - sources: List of {file, start_line, end_line} citations
   - answer: Natural language answer synthesized from sources
   - This ends the evaluation

## Strategy

1. Start broad: Use glob/search to find candidate files
2. Narrow down: Use grep to find specific matches
3. Verify: Use read to examine relevant sections
4. Cite precisely: Note exact line ranges where information is found
5. Synthesize: Write a clear answer based on the sources
6. Submit: Call submit() with ordered sources and your answer

## Important

- Be precise with line ranges - cite the exact lines that contain the answer
- Order sources by relevance (most relevant first)
- Your answer should directly address the query
- Only submit when you're confident you've found all relevant sources
"""


# ──────────────────────── Evaluation Logic ──────────────────────────────────


async def evaluate_sample(
    sample: dict[str, Any],
    config: SWEGrepConfig,
) -> dict[str, Any]:
    """Evaluate a single SWE-grep sample."""
    query_id = sample["query_id"]
    query = sample["query"]
    ground_truth = sample["ground_truth"]
    gt_sources = ground_truth["sources"]
    gt_answer = ground_truth["answer"]

    logger.info(f"Evaluating {query_id}: {query[:80]}...")

    # Create environment
    environment = SWEGrepEnvironment(
        corpus_path=config.corpus_path,
        use_semantic_search=True,
        max_results=50,
        max_file_lines=2000,
    )

    # Create agent
    trajectory = Trajectory(
        messages=[
            Message(role="system", content=SWE_GREP_SYSTEM_PROMPT),
            Message(role="user", content=query),
        ]
    )

    actor = Actor(
        trajectory=trajectory,
        endpoint=config.agent_endpoint,
        tools=environment.get_tools(),
    )

    state = AgentState(actor=actor, environment=environment)

    # Run agent
    async def silent_handler(_: object) -> None:
        await trio.lowlevel.checkpoint()

    run_config = RunConfig(
        on_chunk=silent_handler,
        handle_stop=handle_stop_max_turns(config.max_turns),
    )

    states = await run_agent(state, run_config)

    # Extract submission
    final_env = states[-1].environment if states else environment
    submission = final_env._submission

    if not submission:
        logger.warning(f"{query_id}: No submission - agent did not call submit()")
        return {
            "query_id": query_id,
            "query": query[:80],
            "error": "no_submission",
            "retrieval_score": 0.0,
            "answer_score": 0.0,
            "combined_score": 0.0,
            "num_turns": len(states),
        }

    pred_sources = submission["sources"]
    pred_answer = submission["answer"]

    # Score retrieval
    retrieval_score = compute_retrieval_score(
        predicted_sources=pred_sources,
        ground_truth_sources=gt_sources,
        iou_threshold=config.iou_threshold,
    )

    logger.info(
        f"  Retrieval: P={retrieval_score.precision:.2f} "
        f"R={retrieval_score.recall:.2f} F1={retrieval_score.f1:.2f}"
    )

    # Score answer
    answer_score = await grade_answer(
        question=query,
        predicted_answer=pred_answer,
        correct_answer=gt_answer,
        grader_endpoint=config.grader_endpoint,
    )

    logger.info(
        f"  Answer: {'✓' if answer_score.correct else '✗'} {answer_score.grader_reasoning[:60]}..."
    )

    # Combined score
    combined_score = config.retrieval_weight * retrieval_score.f1 + config.answer_weight * (
        1.0 if answer_score.correct else 0.0
    )

    return {
        "query_id": query_id,
        "query": query[:80],
        "retrieval": {
            "precision": retrieval_score.precision,
            "recall": retrieval_score.recall,
            "f1": retrieval_score.f1,
            "iou": retrieval_score.iou,
            "matched": retrieval_score.matched_sources,
            "total_pred": retrieval_score.total_pred_sources,
            "total_gt": retrieval_score.total_gt_sources,
        },
        "answer": {
            "correct": answer_score.correct,
            "reasoning": answer_score.grader_reasoning,
        },
        "combined_score": combined_score,
        "num_turns": len(states),
        "predicted_sources": pred_sources,
        "predicted_answer": pred_answer,
    }


async def run_evaluation(config: SWEGrepConfig) -> dict[str, Any]:
    """Run full SWE-grep evaluation."""
    from rollouts._logging import setup_logging

    setup_logging(level="INFO", use_color=True)

    logger.info("=" * 60)
    logger.info("SWE-grep Evaluation")
    logger.info("=" * 60)
    logger.info(f"Corpus: {config.corpus_path}")
    logger.info(f"Questions: {config.questions_path}")
    logger.info(f"Agent: {config.agent_endpoint.provider}/{config.agent_endpoint.model}")
    logger.info(f"Grader: {config.grader_endpoint.provider}/{config.grader_endpoint.model}")

    # Load questions
    questions = load_questions(config.questions_path, config.max_samples)

    if not questions:
        logger.error("No questions loaded!")
        return {"error": "no_questions"}

    # Run evaluation
    results = []
    for question in questions:
        result = await evaluate_sample(question, config)
        results.append(result)
        logger.info("")  # Blank line between samples

    # Compute aggregate metrics
    valid_results = [r for r in results if "error" not in r]
    total = len(results)
    valid = len(valid_results)

    if not valid_results:
        logger.error("No valid results!")
        return {"error": "no_valid_results", "results": results}

    avg_retrieval_f1 = sum(r["retrieval"]["f1"] for r in valid_results) / valid
    avg_answer_correct = sum(1 for r in valid_results if r["answer"]["correct"]) / valid
    avg_combined = sum(r["combined_score"] for r in valid_results) / valid
    avg_turns = sum(r["num_turns"] for r in valid_results) / valid

    logger.info("=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    logger.info(f"Valid samples: {valid}/{total}")
    logger.info(f"Retrieval F1: {avg_retrieval_f1:.2%}")
    logger.info(f"Answer accuracy: {avg_answer_correct:.2%}")
    logger.info(f"Combined score: {avg_combined:.2%}")
    logger.info(f"Avg turns: {avg_turns:.1f}")
    logger.info("=" * 60)

    return {
        "retrieval_f1": avg_retrieval_f1,
        "answer_accuracy": avg_answer_correct,
        "combined_score": avg_combined,
        "avg_turns": avg_turns,
        "valid_samples": valid,
        "total_samples": total,
        "results": results,
    }


# ──────────────────────── CLI ────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-grep Evaluation")

    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to document corpus directory",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to questions JSONL file",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Agent model",
    )
    parser.add_argument(
        "--agent-provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai"],
        help="Agent provider",
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Grader model",
    )
    parser.add_argument(
        "--grader-provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai"],
        help="Grader provider",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Maximum agent turns",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for source matching",
    )

    args = parser.parse_args()

    config = SWEGrepConfig(
        corpus_path=args.corpus,
        questions_path=args.questions,
        agent_endpoint=Endpoint(provider=args.agent_provider, model=args.agent_model),
        grader_endpoint=Endpoint(provider=args.grader_provider, model=args.grader_model),
        max_turns=args.max_turns,
        iou_threshold=args.iou_threshold,
        max_samples=args.max_samples,
    )

    trio.run(run_evaluation, config)


if __name__ == "__main__":
    main()
