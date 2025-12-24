#!/usr/bin/env python3
"""
OOLONG benchmark evaluation for RLM.

OOLONG (Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities)
is a benchmark testing whether models can reason over large quantities of text,
performing classification and counting in-context.

Datasets available on Hugging Face:
- oolongbench/oolong-synth: Synthetic tasks with controlled difficulty
- oolongbench/oolong-real: Real D&D episode transcripts with questions

Task types:
- Counting: "How many entries have label X?"
- Filtering: "Among subset, how many have label Y?"
- Distribution: "What's the most common label?"

From the RLM paper:
"RLM(GPT-5-mini) outperforms GPT-5 by >33% raw score (over double the performance)
on OOLONG's trec_coarse dataset."

Usage:
    # Run with synthetic dataset (easier to start)
    python -m examples.eval.rlm.oolong --dataset synth

    # Run with real D&D dataset
    python -m examples.eval.rlm.oolong --dataset real

    # Quick test with toy version
    python -m examples.eval.rlm.oolong --dataset real --config toy_dnd

References:
- Paper: https://arxiv.org/abs/2511.02817
- Dataset: https://huggingface.co/oolongbench
- RLM Blog: https://alexzhang13.github.io/blog/2025/rlm/
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

import trio
from rollouts.agents import handle_stop_max_turns, run_agent
from rollouts.dtypes import (
    Actor,
    AgentState,
    Message,
    RunConfig,
    Trajectory,
)

from .base_config import (
    RLM_TOOL_SYSTEM_PROMPT,
    DatasetConfig,
    EndpointConfig,
    EvalRunConfig,
    OutputConfig,
    RLMConfig,
    RLMEvalConfig,
    exact_match_score,
    get_endpoint,
    get_sub_endpoint,
    numeric_match_score,
)

logger = logging.getLogger(__name__)


# ──────────────────────── Dataset Config ────────────────────────────────────


@dataclass(frozen=True)
class OolongDatasetConfig(DatasetConfig):
    """OOLONG-specific dataset config."""

    dataset_type: str = "synth"  # "synth" or "real"
    config_name: str | None = None  # e.g., "dnd", "toy_dnd" for real
    split: str = "test"
    seed: int = 42
    max_samples: int | None = 10
    # Filter by context length (for scaling experiments)
    min_context_len: int | None = None
    max_context_len: int | None = None


@dataclass(frozen=True)
class OolongConfig(RLMEvalConfig):
    """OOLONG evaluation config."""

    dataset: OolongDatasetConfig = field(default_factory=OolongDatasetConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(experiment_name="oolong"))


# ──────────────────────── Dataset Loading ────────────────────────────────────


def load_oolong_dataset(config: OolongDatasetConfig) -> list[dict[str, Any]]:
    """Load OOLONG dataset from Hugging Face.

    Returns list of samples with:
        - context: The long context text
        - question: Question to answer
        - answer: Ground truth answer
        - answer_type: LABEL, NUMERIC, etc.
        - task: Specific task type
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    logger.info(f"Loading OOLONG {config.dataset_type} dataset...")

    if config.dataset_type == "synth":
        ds = load_dataset("oolongbench/oolong-synth", split=config.split)
    elif config.dataset_type == "real":
        ds_config = config.config_name or "dnd"
        ds = load_dataset("oolongbench/oolong-real", ds_config, split=config.split)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")

    samples = []
    for row in ds:
        # Handle both synth and real formats
        if config.dataset_type == "synth":
            sample = {
                "id": str(row.get("id", len(samples))),
                "context": row.get("context_window_text", ""),
                "question": row.get("question", ""),
                "answer": str(row.get("answer", "")),
                "answer_type": row.get("answer_type", ""),
                "task": row.get("task", ""),
                "task_group": row.get("task_group", ""),
                "context_len": row.get("context_len", 0),
            }
        else:  # real
            sample = {
                "id": str(row.get("id", len(samples))),
                "context": row.get("context_window_text", ""),
                "question": row.get("question", ""),
                "answer": str(row.get("answer", "")),
                "answer_type": row.get("question_type", ""),
                "task": row.get("question_type", ""),
                "campaign": row.get("campaign", ""),
            }

        sample["context_chars"] = len(sample["context"])

        # Apply context length filters
        if (
            config.min_context_len
            and sample.get("context_len", sample["context_chars"]) < config.min_context_len
        ):
            continue
        if (
            config.max_context_len
            and sample.get("context_len", sample["context_chars"]) > config.max_context_len
        ):
            continue

        samples.append(sample)

    logger.info(f"Loaded {len(samples)} samples")

    # Subsample if requested
    if config.max_samples and config.max_samples < len(samples):
        import random

        rng = random.Random(config.seed)
        samples = rng.sample(samples, config.max_samples)
        logger.info(f"Subsampled to {len(samples)} samples")

    return samples


# ──────────────────────── Evaluation Logic ──────────────────────────────────


OOLONG_SYSTEM_PROMPT = (
    RLM_TOOL_SYSTEM_PROMPT
    + """

## OOLONG Task Guidance

The context contains structured entries that you need to analyze.
Tasks may include:
- Counting entries with specific labels
- Filtering by user/date and counting
- Finding distributions or aggregates

Use the REPL to:
1. Parse the context structure (peek first!)
2. Filter entries based on criteria
3. For classification tasks, categorize each entry
4. Count and aggregate results

Be precise with counting - off-by-one errors will be marked incorrect.
"""
)


async def evaluate_sample(
    sample: dict[str, Any],
    config: OolongConfig,
) -> dict[str, Any]:
    """Evaluate a single OOLONG sample."""
    from rollouts.environments.repl import MessageParsingREPLEnvironment, REPLEnvironment

    endpoint = get_endpoint(config.endpoint)
    sub_endpoint = get_sub_endpoint(config.sub_endpoint)

    context = sample["context"]
    question = sample["question"]
    expected = sample["answer"]
    answer_type = sample.get("answer_type", "")

    # Create environment
    if config.rlm.use_tool_calling:
        environment = REPLEnvironment(
            context=context,
            sub_endpoint=sub_endpoint,
            recursive=config.rlm.recursive,
            max_depth=config.rlm.max_depth,
        )
    else:
        environment = MessageParsingREPLEnvironment(
            context=context,
            sub_endpoint=sub_endpoint,
            recursive=config.rlm.recursive,
            max_depth=config.rlm.max_depth,
        )

    trajectory = Trajectory(
        messages=[
            Message(role="system", content=OOLONG_SYSTEM_PROMPT),
            Message(role="user", content=question),
        ]
    )

    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint,
        tools=environment.get_tools(),
    )

    state = AgentState(actor=actor, environment=environment)

    async def silent_handler(_: object) -> None:
        await trio.lowlevel.checkpoint()

    run_config = RunConfig(
        on_chunk=silent_handler,
        handle_stop=handle_stop_max_turns(config.run.max_turns),
    )

    states = await run_agent(state, run_config)

    # Get result from the final state's environment
    final_answer = None
    if states and states[-1].environment:
        final_answer = states[-1].environment._final_answer

    # Use appropriate scoring based on answer type
    if answer_type == "NUMERIC" or expected.isdigit():
        try:
            expected_num = float(expected)
            score = numeric_match_score(final_answer, expected_num, tolerance=0.01)
        except ValueError:
            score = exact_match_score(final_answer, expected)
    else:
        score = exact_match_score(final_answer, expected)

    return {
        "sample_id": sample["id"],
        "task": sample.get("task", ""),
        "answer_type": answer_type,
        "expected": expected,
        "predicted": final_answer,
        "correct": score.metrics[0].value == 1.0,
        "num_turns": len(states),
        "context_chars": sample["context_chars"],
    }


async def run_evaluation(config: OolongConfig) -> dict[str, Any]:
    """Run full OOLONG evaluation."""
    from rollouts._logging import setup_logging

    setup_logging(level="INFO", use_color=True)

    logger.info("=" * 60)
    logger.info(f"OOLONG Evaluation ({config.dataset.dataset_type})")
    logger.info("=" * 60)
    logger.info(f"Samples: {config.dataset.max_samples}")
    logger.info(f"Model: {config.endpoint.provider}/{config.endpoint.model}")
    logger.info(f"RLM enabled: {config.rlm.enabled}")

    # Load dataset
    samples = load_oolong_dataset(config.dataset)

    if not samples:
        logger.error("No samples loaded!")
        return {"error": "No samples"}

    logger.info(
        f"Context sizes: {min(s['context_chars'] for s in samples):,} - {max(s['context_chars'] for s in samples):,} chars"
    )

    # Run evaluation
    results = []
    for sample in samples:
        result = await evaluate_sample(sample, config)
        results.append(result)
        status = "✓" if result["correct"] else "✗"
        logger.info(
            f"  {status} {result['sample_id']} ({result['task']}): "
            f"{result['predicted']} (expected {result['expected']})"
        )

    # Compute metrics
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    avg_turns = sum(r["num_turns"] for r in results) / total if total > 0 else 0.0

    # Per-task breakdown
    by_task: dict[str, dict] = {}
    for r in results:
        task = r.get("task", "unknown")
        if task not in by_task:
            by_task[task] = {"correct": 0, "total": 0}
        by_task[task]["total"] += 1
        if r["correct"]:
            by_task[task]["correct"] += 1

    logger.info("=" * 60)
    logger.info(f"Overall: {correct}/{total} ({accuracy:.1%})")
    for task, stats in by_task.items():
        task_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        logger.info(f"  {task}: {stats['correct']}/{stats['total']} ({task_acc:.1%})")
    logger.info(f"Average turns: {avg_turns:.1f}")
    logger.info("=" * 60)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_task": by_task,
        "avg_turns": avg_turns,
        "results": results,
    }


# ──────────────────────── CLI ────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="OOLONG RLM Evaluation")

    parser.add_argument(
        "--dataset",
        type=str,
        default="synth",
        choices=["synth", "real"],
        help="Dataset type: synth (synthetic) or real (D&D transcripts)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config name for real dataset: dnd, toy_dnd",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai"],
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Maximum turns (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--min-context",
        type=int,
        default=None,
        help="Minimum context length to filter",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Maximum context length to filter",
    )

    args = parser.parse_args()

    config = OolongConfig(
        endpoint=EndpointConfig(provider=args.provider, model=args.model),
        rlm=RLMConfig(enabled=True),
        dataset=OolongDatasetConfig(
            dataset_type=args.dataset,
            config_name=args.config,
            split=args.split,
            max_samples=args.samples,
            seed=args.seed,
            min_context_len=args.min_context,
            max_context_len=args.max_context,
        ),
        run=EvalRunConfig(max_turns=args.max_turns),
    )

    trio.run(run_evaluation, config)


if __name__ == "__main__":
    main()
