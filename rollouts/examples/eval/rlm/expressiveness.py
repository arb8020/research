#!/usr/bin/env python3
"""
RLM Expressiveness Evaluation - Omar Khattab's Litmus Test.

Tests whether the RLM implementation is truly expressive per Omar Khattab's criteria:
https://x.com/lateinteraction/status/1878896620499075500

Key requirements for "true RLM":
1. O(N) symbolic recursion: Model can't verbalize N tool calls, so recursion must be through CODE
2. Prompt-as-pointer: Large prompts accessible as objects, not just in message history

This eval tests these by:
1. Embedding the TASK INSTRUCTION inside the context (not in the user message)
2. Requiring processing of N items where N is large enough that explicit tool calls fail
3. Requiring semantic judgment (llm_query) for each item - can't be done with pure regex

The litmus test from Omar:
"10M-token prompt starting with 'keep each math question below iff the answer is an even number'"

Pass criteria:
- Model reads instruction from context (not from user message)
- Model iterates programmatically over N items
- Model uses llm_query (or code) to check each answer
- Model returns correct filtered result

Fail criteria:
- Model tries to make N explicit tool calls (not through code)
- Model can't find the instruction (because it's only in context)
- Model gives up / truncates due to volume
- Model hallucinates without actually processing items

Usage:
    # Quick test (100 items, easy)
    python -m examples.eval.rlm.expressiveness --items 100

    # Medium test (1000 items)
    python -m examples.eval.rlm.expressiveness --items 1000

    # Full stress test (10000 items)
    python -m examples.eval.rlm.expressiveness --items 10000 --require-semantic

    # Compare symbolic vs explicit tool calls
    python -m examples.eval.rlm.expressiveness --items 500 --trace-tool-calls
"""

from __future__ import annotations

import argparse
import logging
import random
import re
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
    get_endpoint,
    get_sub_endpoint,
)

logger = logging.getLogger(__name__)


# ──────────────────────── Dataset Config ────────────────────────────────────


@dataclass(frozen=True)
class ExpressivenessDatasetConfig(DatasetConfig):
    """Expressiveness test config."""

    num_items: int = 500  # Number of items to process
    seed: int = 42
    max_samples: int | None = 5
    # Difficulty settings
    require_semantic: bool = False  # If True, answers require LLM judgment (not just regex)
    embed_instruction: bool = True  # If True, instruction is ONLY in context, not user message


@dataclass(frozen=True)
class ExpressivenessConfig(RLMEvalConfig):
    """Expressiveness evaluation config."""

    dataset: ExpressivenessDatasetConfig = field(default_factory=ExpressivenessDatasetConfig)
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(experiment_name="rlm_expressiveness")
    )


# ──────────────────────── Dataset Generation ────────────────────────────────


def generate_math_questions(
    num_items: int,
    seed: int | None = None,
    require_semantic: bool = False,
) -> tuple[str, list[int], list[int]]:
    """Generate math questions where we need to filter by even answers.

    Args:
        num_items: Number of Q&A pairs
        seed: Random seed
        require_semantic: If True, make answers require semantic judgment

    Returns:
        (context_string, list_of_question_ids_with_even_answers, all_question_ids)
    """
    if seed is not None:
        random.seed(seed)

    lines = []

    # CRITICAL: The instruction is embedded in the context, not given separately
    instruction = (
        "=== INSTRUCTION ===\n"
        "Your task: Return the IDs of all questions below where the ANSWER is an EVEN number.\n"
        "Output format: A Python list of integer IDs, e.g., [1, 4, 7, 12]\n"
        "===================\n\n"
    )
    lines.append(instruction)

    even_ids = []
    all_ids = []

    for i in range(1, num_items + 1):
        all_ids.append(i)

        if require_semantic:
            # Harder: answer is written as text, requires understanding
            a = random.randint(1, 50)
            b = random.randint(1, 50)
            answer = a + b
            # Write answer in words or indirect form
            answer_forms = [
                f"The answer is {answer}",
                f"That equals {answer}",
                f"Result: {answer}",
                f"{a} plus {b} gives us {answer}",
                f"Adding these yields {answer}",
            ]
            answer_text = random.choice(answer_forms)
        else:
            # Easier: answer is just a number (can use regex)
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            answer = a + b
            answer_text = str(answer)

        lines.append(f"[Q{i}] What is {a} + {b}?")
        lines.append(f"[A{i}] {answer_text}")
        lines.append("")

        if answer % 2 == 0:
            even_ids.append(i)

    # Add some filler to make context larger
    lines.append("\n" + "=" * 50)
    lines.append("END OF QUESTIONS")
    lines.append("=" * 50)

    context = "\n".join(lines)
    return context, even_ids, all_ids


def generate_samples(config: ExpressivenessDatasetConfig) -> list[dict[str, Any]]:
    """Generate evaluation samples."""
    samples = []
    rng = random.Random(config.seed)

    n_samples = config.max_samples or 5

    for i in range(n_samples):
        sample_seed = config.seed + i

        context, even_ids, all_ids = generate_math_questions(
            num_items=config.num_items,
            seed=sample_seed,
            require_semantic=config.require_semantic,
        )

        samples.append({
            "id": f"expr_{i:04d}",
            "context": context,
            "expected_ids": sorted(even_ids),
            "total_items": len(all_ids),
            "num_even": len(even_ids),
            "context_chars": len(context),
            "require_semantic": config.require_semantic,
            "embed_instruction": config.embed_instruction,
        })

    return samples


# ──────────────────────── Scoring ────────────────────────────────────────────


def score_id_list(predicted: str | None, expected: list[int]) -> dict[str, Any]:
    """Score predicted list of IDs against expected.

    Returns dict with:
        - correct: bool (exact match)
        - precision: float
        - recall: float
        - f1: float
        - predicted_ids: list[int] (parsed)
        - error: str | None
    """
    if predicted is None:
        return {
            "correct": False,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "predicted_ids": [],
            "error": "no answer",
        }

    # Try to parse list from prediction
    # Look for patterns like [1, 2, 3] or 1, 2, 3 or Q1, Q2, Q3
    predicted_ids = []

    # Try direct list parsing
    list_match = re.search(r"\[([^\]]+)\]", predicted)
    if list_match:
        list_content = list_match.group(1)
        # Extract numbers (handles both "3" and "Q3" formats)
        numbers = re.findall(r"\d+", list_content)
        predicted_ids = [int(n) for n in numbers]
    else:
        # Try Q-prefixed IDs like "Q3, Q6, Q7"
        q_matches = re.findall(r"Q(\d+)", predicted, re.IGNORECASE)
        if q_matches:
            predicted_ids = [int(n) for n in q_matches]
        else:
            # Try comma-separated numbers
            numbers = re.findall(r"\b(\d+)\b", predicted)
            if numbers:
                predicted_ids = [int(n) for n in numbers]

    predicted_set = set(predicted_ids)
    expected_set = set(expected)

    if not predicted_set and not expected_set:
        # Both empty is correct
        return {
            "correct": True,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "predicted_ids": [],
            "error": None,
        }

    # Calculate metrics
    true_positives = len(predicted_set & expected_set)
    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Exact match
    correct = predicted_set == expected_set

    return {
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_ids": sorted(predicted_ids),
        "error": None,
    }


# ──────────────────────── Evaluation Logic ──────────────────────────────────


EXPRESSIVENESS_SYSTEM_PROMPT = (
    RLM_TOOL_SYSTEM_PROMPT
    + """

## Important for this task

The task instruction is embedded IN THE CONTEXT ITSELF, not in this message.
You must read the context to find out what you're supposed to do.

Start by peeking at the beginning of the context to find the instruction."""
)


async def evaluate_sample(
    sample: dict[str, Any],
    config: ExpressivenessConfig,
    trace_tool_calls: bool = False,
) -> dict[str, Any]:
    """Evaluate a single expressiveness sample."""
    from rollouts.environments.repl import REPLEnvironment

    endpoint = get_endpoint(config.endpoint)
    sub_endpoint = get_sub_endpoint(config.sub_endpoint)

    context = sample["context"]
    expected_ids = sample["expected_ids"]
    embed_instruction = sample["embed_instruction"]

    # Create environment
    environment = REPLEnvironment(
        context=context,
        sub_endpoint=sub_endpoint,
        max_depth=config.rlm.max_depth,
    )

    # User message depends on whether instruction is embedded
    if embed_instruction:
        user_message = (
            "The context contains both an instruction and data. "
            "Read the context to find out what you need to do, then do it. "
            "Use final_answer to submit your result."
        )
    else:
        user_message = (
            f"The context contains {sample['total_items']} math questions with answers. "
            "Return the IDs of all questions where the answer is an EVEN number. "
            "Output as a Python list of integers."
        )

    trajectory = Trajectory(
        messages=[
            Message(role="system", content=EXPRESSIVENESS_SYSTEM_PROMPT),
            Message(role="user", content=user_message),
        ]
    )

    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint,
        tools=environment.get_tools(),
    )

    state = AgentState(actor=actor, environment=environment)

    # Track tool calls for analysis
    tool_call_counts: dict[str, int] = {"repl": 0, "llm_query": 0, "agent": 0, "final_answer": 0}
    llm_query_from_code = 0  # llm_query called from within repl (symbolic)
    llm_query_direct = 0  # llm_query called as direct tool (not symbolic)

    async def tracking_handler(event: object) -> None:
        nonlocal llm_query_from_code, llm_query_direct
        # Count tool calls by type
        from rollouts.dtypes import ToolCallStart

        if isinstance(event, ToolCallStart):
            name = event.tool_name
            if name in tool_call_counts:
                tool_call_counts[name] += 1

            # Detect if llm_query is direct (bad) vs from code (good)
            if name == "llm_query":
                llm_query_direct += 1

        await trio.lowlevel.checkpoint()

    run_config = RunConfig(
        on_chunk=tracking_handler,
        handle_stop=handle_stop_max_turns(config.run.max_turns),
    )

    states = await run_agent(state, run_config)

    # Get result
    final_answer = None
    if states and states[-1].environment:
        final_answer = states[-1].environment._final_answer

    score = score_id_list(final_answer, expected_ids)

    # Analyze tool usage pattern
    # Good: few direct tool calls, processing happens in repl loops
    # Bad: many direct llm_query tool calls (O(N) explicit calls)
    is_symbolic = tool_call_counts["llm_query"] < sample["total_items"] / 10

    return {
        "sample_id": sample["id"],
        "total_items": sample["total_items"],
        "num_even": sample["num_even"],
        "expected_ids": expected_ids,
        "predicted_ids": score["predicted_ids"],
        "correct": score["correct"],
        "precision": score["precision"],
        "recall": score["recall"],
        "f1": score["f1"],
        "num_turns": len(states),
        "context_chars": sample["context_chars"],
        "tool_calls": tool_call_counts,
        "is_symbolic": is_symbolic,
        "llm_query_direct": llm_query_direct,
        "error": score.get("error"),
    }


async def run_evaluation(
    config: ExpressivenessConfig,
    trace_tool_calls: bool = False,
) -> dict[str, Any]:
    """Run expressiveness evaluation."""
    from rollouts._logging import setup_logging

    setup_logging(level="INFO", use_color=True)

    logger.info("=" * 70)
    logger.info("RLM Expressiveness Evaluation (Omar Khattab Litmus Test)")
    logger.info("=" * 70)
    logger.info(f"Items per sample: {config.dataset.num_items}")
    logger.info(f"Samples: {config.dataset.max_samples}")
    logger.info(f"Require semantic judgment: {config.dataset.require_semantic}")
    logger.info(f"Instruction embedded in context: {config.dataset.embed_instruction}")
    logger.info(f"Model: {config.endpoint.provider}/{config.endpoint.model}")

    # Generate samples
    samples = generate_samples(config.dataset)
    logger.info(f"Generated {len(samples)} samples")
    logger.info(f"Context size: ~{samples[0]['context_chars']:,} chars")
    logger.info(f"Expected even answers per sample: ~{samples[0]['num_even']}")

    # Run evaluation
    results = []
    for sample in samples:
        logger.info(f"\n>>> Evaluating {sample['id']}...")
        result = await evaluate_sample(sample, config, trace_tool_calls)
        results.append(result)

        status = "✓" if result["correct"] else "✗"
        symbolic_status = "symbolic" if result["is_symbolic"] else "EXPLICIT"
        logger.info(
            f"  {status} {result['sample_id']}: "
            f"F1={result['f1']:.2f}, "
            f"turns={result['num_turns']}, "
            f"repl={result['tool_calls']['repl']}, "
            f"llm_query={result['tool_calls']['llm_query']} ({symbolic_status})"
        )

        if result["error"]:
            logger.info(f"     Error: {result['error']}")

        if not result["correct"]:
            # Show what went wrong
            expected = set(result["expected_ids"])
            predicted = set(result["predicted_ids"])
            missing = expected - predicted
            extra = predicted - expected
            if missing:
                logger.info(
                    f"     Missing IDs: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
                )
            if extra:
                logger.info(
                    f"     Extra IDs: {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}"
                )

    # Compute aggregate metrics
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    avg_f1 = sum(r["f1"] for r in results) / total if total > 0 else 0.0
    avg_turns = sum(r["num_turns"] for r in results) / total if total > 0 else 0.0
    symbolic_count = sum(1 for r in results if r["is_symbolic"])

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Exact match accuracy: {correct}/{total} ({accuracy:.1%})")
    logger.info(f"Average F1: {avg_f1:.3f}")
    logger.info(f"Average turns: {avg_turns:.1f}")
    logger.info(f"Symbolic processing: {symbolic_count}/{total} ({symbolic_count / total:.1%})")
    logger.info("=" * 70)

    # Expressiveness verdict
    logger.info("\n>>> EXPRESSIVENESS VERDICT:")
    if accuracy >= 0.8 and symbolic_count >= total * 0.8:
        logger.info("✓ PASS: Model processes items symbolically (through code loops)")
        logger.info("  - High accuracy with embedded instructions")
        logger.info("  - Few direct tool calls (uses code-based iteration)")
    elif accuracy >= 0.8:
        logger.info("⚠ PARTIAL: Model is accurate but uses explicit tool calls")
        logger.info("  - This won't scale to very large N")
        logger.info("  - Need to encourage code-based iteration")
    else:
        logger.info("✗ FAIL: Model struggles with this task")
        if symbolic_count < total * 0.5:
            logger.info("  - Too many explicit tool calls (not symbolic)")
        logger.info("  - May not be reading instruction from context")
        logger.info("  - May be truncating or giving up")

    return {
        "accuracy": accuracy,
        "avg_f1": avg_f1,
        "correct": correct,
        "total": total,
        "symbolic_count": symbolic_count,
        "avg_turns": avg_turns,
        "results": results,
    }


# ──────────────────────── CLI ────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RLM Expressiveness Evaluation (Omar Khattab Litmus Test)"
    )

    parser.add_argument(
        "--items",
        type=int,
        default=500,
        help="Number of items per sample (default: 500)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples (default: 3)",
    )
    parser.add_argument(
        "--require-semantic",
        action="store_true",
        help="Require semantic judgment (answers in natural language)",
    )
    parser.add_argument(
        "--no-embed-instruction",
        action="store_true",
        help="Give instruction in user message (easier, not true RLM test)",
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
        "--trace-tool-calls",
        action="store_true",
        help="Print detailed tool call trace",
    )

    args = parser.parse_args()

    config = ExpressivenessConfig(
        endpoint=EndpointConfig(provider=args.provider, model=args.model),
        rlm=RLMConfig(enabled=True),
        dataset=ExpressivenessDatasetConfig(
            num_items=args.items,
            max_samples=args.samples,
            seed=args.seed,
            require_semantic=args.require_semantic,
            embed_instruction=not args.no_embed_instruction,
        ),
        run=EvalRunConfig(max_turns=args.max_turns),
    )

    trio.run(run_evaluation, config, args.trace_tool_calls)


if __name__ == "__main__":
    main()
