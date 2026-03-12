#!/usr/bin/env python3
"""
Trace Summarization Evaluation for RLM.

Tests RLM's ability to analyze long agent trajectories and extract signal.
This is the core use case for GEPA - analyzing rollouts to identify patterns,
errors, and quality signals.

Tasks tested:
1. Structural queries (regex-able): "How many tool calls failed?"
2. Semantic queries (need llm_query): "What was the root cause of failure?"
3. Aggregation: "Score the agent 1-5 on task completion"
4. Extraction: "List the key decision points"

Usage:
    # Quick test (20 turns, 3 samples)
    python -m examples.eval.rlm.trace_summarization --turns 20 --samples 3

    # Longer traces (100 turns)
    python -m examples.eval.rlm.trace_summarization --turns 100

    # With semantic queries (harder)
    python -m examples.eval.rlm.trace_summarization --turns 50 --include-semantic
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

import trio

from rollouts.agents import Actor, AgentState, RunConfig, handle_stop_max_turns, run_agent
from rollouts.core import Message, Trajectory
from rollouts.dtypes import ToolCallContent

from .base_config import (
    DatasetConfig,
    EndpointConfig,
    EvalRunConfig,
    OutputConfig,
    RLMConfig,
    RLMEvalConfig,
    get_endpoint,
    get_sub_endpoint,
    numeric_match_score,
)

logger = logging.getLogger(__name__)


# ──────────────────────── Synthetic Trace Generation ─────────────────────────


TOOL_NAMES = ["bash", "read_file", "write_file", "search", "edit_file", "list_files"]

ERROR_MESSAGES = [
    "Command failed with exit code 1",
    "File not found: {path}",
    "Permission denied: {path}",
    "Syntax error in edit",
    "Search returned no results",
    "Timeout after 30 seconds",
]

REASONING_SNIPPETS = [
    "I need to first understand the codebase structure.",
    "Let me search for the relevant function.",
    "I'll read the file to understand the current implementation.",
    "Now I'll make the necessary edit.",
    "Let me verify the change worked correctly.",
    "I should run the tests to confirm.",
    "This approach isn't working, let me try a different strategy.",
    "I found the issue - it's in the error handling logic.",
    "The bug is caused by a missing null check.",
    "I'll refactor this to be more robust.",
]

SUCCESS_INDICATORS = [
    "Tests passed successfully",
    "Build completed without errors",
    "Changes applied correctly",
    "Verification successful",
]


@dataclass
class SyntheticTraceConfig:
    """Configuration for generating a synthetic trace."""

    num_turns: int = 50
    error_rate: float = 0.15  # Fraction of tool calls that fail
    seed: int | None = None

    # Planted signals (for ground truth)
    plant_critical_error: bool = True  # Plant a "root cause" error
    critical_error_turn: int | None = None  # Which turn (None = random middle)
    plant_key_insight: bool = True  # Plant an important decision moment
    outcome: str = "success"  # "success", "failure", "partial"


@dataclass
class SyntheticTrace:
    """A generated trace with ground truth labels."""

    messages: list[Message]
    context_str: str  # Serialized for RLM context

    # Ground truth
    total_tool_calls: int
    failed_tool_calls: int
    critical_error_message: str | None
    critical_error_turn: int | None
    key_insight: str | None
    key_insight_turn: int | None
    outcome: str
    outcome_reason: str


def generate_tool_call(
    tool_name: str,
    args: dict[str, Any],
) -> ToolCallContent:
    """Generate a tool call content block."""
    return ToolCallContent(
        type="toolCall",
        id=f"call_{uuid.uuid4().hex[:8]}",
        name=tool_name,
        arguments=args,
    )


def generate_tool_result(
    tool_call_id: str,
    success: bool,
    output: str,
) -> Message:
    """Generate a tool result message."""
    if success:
        content = output
    else:
        content = f"Error: {output}"

    return Message(
        role="tool",
        content=content,
        tool_call_id=tool_call_id,
    )


def generate_synthetic_trace(config: SyntheticTraceConfig) -> SyntheticTrace:
    """Generate a synthetic agent trace with known properties."""
    rng = random.Random(config.seed)

    messages: list[Message] = []
    total_tool_calls = 0
    failed_tool_calls = 0
    critical_error_message = None
    critical_error_turn = None
    key_insight = None
    key_insight_turn = None

    # Decide where to plant critical error
    if config.plant_critical_error:
        critical_error_turn = config.critical_error_turn or rng.randint(
            config.num_turns // 3, 2 * config.num_turns // 3
        )
        critical_error_message = rng.choice([
            "CRITICAL: Database connection failed - connection string malformed",
            "CRITICAL: Memory allocation failed in worker process",
            "CRITICAL: Authentication token expired during operation",
            "CRITICAL: Race condition detected in concurrent write",
        ])

    # Decide where to plant key insight
    if config.plant_key_insight:
        key_insight_turn = rng.randint(config.num_turns // 4, config.num_turns // 2)
        key_insight = rng.choice([
            "KEY INSIGHT: The issue is in the async handler - it's not awaiting the promise",
            "KEY INSIGHT: Found the bug - array index is off by one in the loop",
            "KEY INSIGHT: The config is being loaded before environment variables are set",
            "KEY INSIGHT: Cache invalidation is happening too early",
        ])

    # Initial system/user messages
    task_description = rng.choice([
        "Fix the failing test in tests/test_auth.py",
        "Implement the new caching layer for the API",
        "Debug the memory leak in the worker process",
        "Refactor the database connection handling",
    ])

    messages.append(Message(role="system", content="You are a coding assistant."))
    messages.append(Message(role="user", content=f"Task: {task_description}"))

    # Generate turns
    for turn in range(config.num_turns):
        # Assistant message with reasoning and tool calls
        reasoning = rng.choice(REASONING_SNIPPETS)

        # Check for key insight turn
        if turn == key_insight_turn and key_insight:
            reasoning = key_insight

        # Generate 1-3 tool calls per turn
        num_tools = rng.randint(1, 3)
        tool_calls = []

        for _ in range(num_tools):
            tool_name = rng.choice(TOOL_NAMES)
            total_tool_calls += 1

            # Generate appropriate args
            if tool_name in ["read_file", "write_file", "edit_file"]:
                args = {"path": f"src/{rng.choice(['main', 'utils', 'config', 'api'])}.py"}
            elif tool_name == "bash":
                args = {"command": rng.choice(["pytest", "npm test", "make build", "ls -la"])}
            elif tool_name == "search":
                args = {"query": rng.choice(["TODO", "FIXME", "error", "async"])}
            else:
                args = {}

            tool_calls.append(generate_tool_call(tool_name, args))

        # Create assistant message
        assistant_content: list[Any] = [{"type": "text", "text": reasoning}]
        for tc in tool_calls:
            assistant_content.append({
                "type": "toolCall",
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            })

        messages.append(Message(role="assistant", content=assistant_content))

        # Generate tool results
        for tc in tool_calls:
            # Determine if this call fails
            is_error = rng.random() < config.error_rate

            # Check for critical error turn
            if turn == critical_error_turn and critical_error_message and not is_error:
                is_error = True
                error_msg = critical_error_message
            elif is_error:
                error_msg = rng.choice(ERROR_MESSAGES).format(path="src/main.py")
            else:
                error_msg = ""

            if is_error:
                failed_tool_calls += 1
                output = error_msg
            else:
                output = rng.choice([
                    "File contents:\n```python\ndef main():\n    pass\n```",
                    "Search found 3 matches in src/",
                    "Command completed successfully",
                    "Edit applied to 2 lines",
                    "Listed 5 files",
                ])

            messages.append(generate_tool_result(tc.id, not is_error, output))

    # Final assistant message based on outcome
    if config.outcome == "success":
        final_msg = f"Task completed successfully. {rng.choice(SUCCESS_INDICATORS)}"
        outcome_reason = "All tests pass and changes verified"
    elif config.outcome == "failure":
        final_msg = "I was unable to complete the task due to the errors encountered."
        outcome_reason = critical_error_message or "Multiple tool failures"
    else:  # partial
        final_msg = "I made progress but couldn't fully complete the task."
        outcome_reason = "Some steps succeeded but verification failed"

    messages.append(Message(role="assistant", content=final_msg))

    # Serialize to context string
    context_lines = ["=== AGENT TRACE ===", f"Task: {task_description}", ""]

    for i, msg in enumerate(messages):
        if msg.role == "system":
            continue

        context_lines.append(f"[Turn {i}] [{msg.role.upper()}]")

        if isinstance(msg.content, str):
            context_lines.append(msg.content)
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        context_lines.append(block.get("text", ""))
                    elif block.get("type") == "toolCall":
                        context_lines.append(
                            f"  -> Tool: {block.get('name')}({json.dumps(block.get('arguments', {}))})"
                        )

        if msg.role == "tool":
            if msg.content and isinstance(msg.content, str) and msg.content.startswith("Error:"):
                context_lines.append(f"  [TOOL ERROR] {msg.content}")
            else:
                # Truncate long outputs
                content_str = str(msg.content)[:200]
                context_lines.append(f"  [TOOL OUTPUT] {content_str}")

        context_lines.append("")

    context_lines.append("=== END TRACE ===")
    context_str = "\n".join(context_lines)

    return SyntheticTrace(
        messages=messages,
        context_str=context_str,
        total_tool_calls=total_tool_calls,
        failed_tool_calls=failed_tool_calls,
        critical_error_message=critical_error_message,
        critical_error_turn=critical_error_turn,
        key_insight=key_insight,
        key_insight_turn=key_insight_turn,
        outcome=config.outcome,
        outcome_reason=outcome_reason,
    )


# ──────────────────────── Dataset Config ────────────────────────────────────


@dataclass(frozen=True)
class TraceSummarizationDatasetConfig(DatasetConfig):
    """Trace summarization eval config."""

    num_turns: int = 50  # Turns per trace
    seed: int = 42
    max_samples: int | None = 5
    include_semantic: bool = False  # Include semantic queries (harder)
    error_rate: float = 0.15


@dataclass(frozen=True)
class TraceSummarizationConfig(RLMEvalConfig):
    """Full eval config."""

    dataset: TraceSummarizationDatasetConfig = field(
        default_factory=TraceSummarizationDatasetConfig
    )
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(experiment_name="trace_summarization")
    )


# ──────────────────────── Evaluation Logic ──────────────────────────────────


TRACE_ANALYSIS_SYSTEM_PROMPT = """You are an assistant with access to a REPL environment for analyzing agent traces.

The trace is stored in a Python variable called `context`. It contains a log of an AI agent attempting to complete a coding task, including:
- Assistant reasoning and decisions
- Tool calls (bash, read_file, edit_file, etc.)
- Tool results (including errors)

## Available Tools

### repl
Execute Python code to explore and analyze the trace:
- `context` - the full trace as a string
- `len(context)` - get the size
- `context.split('\\n')` - split into lines
- `re.findall(pattern, context)` - search with regex
- Filter for errors: `[l for l in context.split('\\n') if 'ERROR' in l]`

### llm_query
Query a language model for semantic analysis:
- Use for understanding intent, summarizing sections, identifying root causes
- Example: `llm_query(f"What is the root cause of this error?\\n{error_section}")`

### final_answer
Submit your analysis when done.

## Strategy

1. **Peek first**: Check trace size and structure
2. **Use regex for counts**: Tool calls, errors, specific patterns
3. **Use llm_query for semantics**: Root cause analysis, quality assessment
4. **Answer precisely**: Numbers should be exact, summaries should be concise"""


def generate_samples(config: TraceSummarizationDatasetConfig) -> list[dict[str, Any]]:
    """Generate evaluation samples."""
    samples = []
    rng = random.Random(config.seed)

    n_samples = config.max_samples or 5

    for i in range(n_samples):
        # Vary outcome across samples
        outcome = rng.choice(["success", "failure", "partial"])

        trace_config = SyntheticTraceConfig(
            num_turns=config.num_turns,
            error_rate=config.error_rate,
            seed=config.seed + i,
            outcome=outcome,
        )

        trace = generate_synthetic_trace(trace_config)

        # Build questions and ground truth
        questions = []

        # Q1: Count failed tool calls (structural - regex)
        questions.append({
            "id": "failed_count",
            "question": "In the agent trace stored in `context`, how many tool calls failed (returned errors marked with [TOOL ERROR])? Search the context and count them. Answer with just the number.",
            "answer": trace.failed_tool_calls,
            "type": "numeric",
            "difficulty": "easy",
        })

        # Q2: Count total tool calls (structural)
        questions.append({
            "id": "total_count",
            "question": "In the agent trace stored in `context`, how many total tool calls were made by the agent? Count lines containing '-> Tool:'. Answer with just the number.",
            "answer": trace.total_tool_calls,
            "type": "numeric",
            "difficulty": "easy",
        })

        # Q3: Identify critical error (structural - search for CRITICAL:)
        if config.include_semantic and trace.critical_error_message:
            questions.append({
                "id": "critical_error",
                "question": "In the agent trace in `context`, search for any error message containing 'CRITICAL:'. Extract and return the full CRITICAL error message. Answer with just the error text starting with 'CRITICAL:'.",
                "answer": trace.critical_error_message,
                "type": "text_contains",
                "difficulty": "medium",
            })

        # Q4: Outcome assessment (semantic - look at final message)
        if config.include_semantic:
            questions.append({
                "id": "outcome",
                "question": "Use the repl to read the END of `context` (e.g., `context[-500:]`) and find the FINAL [ASSISTANT] message before '=== END TRACE ==='. Based on what the agent said in that final message, did it succeed, fail, or partially complete? Answer with exactly one word: success, failure, or partial",
                "answer": trace.outcome,
                "type": "exact",
                "difficulty": "medium",
            })

        samples.append({
            "id": f"trace_{i:04d}",
            "context": trace.context_str,
            "questions": questions,
            "trace": trace,
            "context_chars": len(trace.context_str),
        })

    return samples


async def evaluate_question(
    context: str,
    question: dict[str, Any],
    config: TraceSummarizationConfig,
) -> dict[str, Any]:
    """Evaluate a single question about a trace."""
    from rollouts.environments.repl import REPLEnvironment

    endpoint = get_endpoint(config.endpoint)
    sub_endpoint = get_sub_endpoint(config.sub_endpoint)

    environment = REPLEnvironment(
        context=context,
        sub_endpoint=sub_endpoint,
        max_depth=config.rlm.max_depth,
    )

    trajectory = Trajectory(
        messages=[
            Message(role="system", content=TRACE_ANALYSIS_SYSTEM_PROMPT),
            Message(role="user", content=question["question"]),
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

    # Get answer - prefer final_answer tool, fallback to last assistant message
    final_answer = None
    if states and states[-1].environment:
        final_answer = states[-1].environment._final_answer

    # Fallback: if no final_answer tool was used, extract from last assistant message
    if final_answer is None and states:
        for msg in reversed(states[-1].actor.trajectory.messages):
            if msg.role == "assistant" and msg.content:
                if isinstance(msg.content, str):
                    final_answer = msg.content.strip()
                    break
                elif isinstance(msg.content, list):
                    # Extract text blocks (could be dict or TextContent dataclass)
                    for block in msg.content:
                        if hasattr(block, "text") and block.text:
                            final_answer = block.text.strip()
                            break
                        elif isinstance(block, dict) and block.get("type") == "text":
                            final_answer = block.get("text", "").strip()
                            break
                    if final_answer:
                        break

    # Score based on question type
    expected = question["answer"]
    q_type = question["type"]

    if q_type == "numeric":
        score = numeric_match_score(final_answer, expected)
        correct = score.metrics[0].value == 1.0
    elif q_type == "exact":
        # Check for exact match, or if expected word appears in response
        if final_answer is None:
            correct = False
        else:
            answer_lower = str(final_answer).strip().lower()
            expected_lower = str(expected).strip().lower()
            # Exact match or expected word is in the answer
            correct = answer_lower == expected_lower or expected_lower in answer_lower
    elif q_type == "text_contains":
        # Check if the key part of the answer is present
        correct = final_answer is not None and str(expected).lower() in str(final_answer).lower()
    else:
        correct = False

    return {
        "question_id": question["id"],
        "question": question["question"],
        "expected": expected,
        "predicted": final_answer,
        "correct": correct,
        "difficulty": question["difficulty"],
        "num_turns": len(states),
    }


async def evaluate_sample(
    sample: dict[str, Any],
    config: TraceSummarizationConfig,
) -> dict[str, Any]:
    """Evaluate all questions for a single trace."""
    results = []

    for question in sample["questions"]:
        result = await evaluate_question(sample["context"], question, config)
        results.append(result)

    correct = sum(1 for r in results if r["correct"])
    total = len(results)

    return {
        "sample_id": sample["id"],
        "context_chars": sample["context_chars"],
        "questions": results,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
    }


async def run_evaluation(config: TraceSummarizationConfig) -> dict[str, Any]:
    """Run full trace summarization evaluation."""
    from rollouts._logging import setup_logging

    setup_logging(level="INFO", use_color=True)

    logger.info("=" * 70)
    logger.info("Trace Summarization Evaluation")
    logger.info("=" * 70)
    logger.info(f"Turns per trace: {config.dataset.num_turns}")
    logger.info(f"Samples: {config.dataset.max_samples}")
    logger.info(f"Include semantic queries: {config.dataset.include_semantic}")
    logger.info(f"Model: {config.endpoint.provider}/{config.endpoint.model}")

    # Generate samples
    samples = generate_samples(config.dataset)
    logger.info(f"Generated {len(samples)} samples")
    logger.info(f"Context size: ~{samples[0]['context_chars']:,} chars each")

    # Run evaluation
    all_results = []
    all_question_results = []

    for sample in samples:
        logger.info(f"\n>>> Evaluating {sample['id']}...")
        result = await evaluate_sample(sample, config)
        all_results.append(result)

        for qr in result["questions"]:
            all_question_results.append(qr)
            status = "✓" if qr["correct"] else "✗"
            logger.info(
                f"  {status} {qr['question_id']}: {qr['predicted']} (expected {qr['expected']})"
            )

    # Aggregate metrics
    total_correct = sum(r["correct"] for r in all_results)
    total_questions = sum(r["total"] for r in all_results)
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    # By difficulty
    by_difficulty: dict[str, dict] = {}
    for qr in all_question_results:
        diff = qr["difficulty"]
        if diff not in by_difficulty:
            by_difficulty[diff] = {"correct": 0, "total": 0}
        by_difficulty[diff]["total"] += 1
        if qr["correct"]:
            by_difficulty[diff]["correct"] += 1

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Overall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.1%})")

    for diff, stats in sorted(by_difficulty.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        logger.info(f"  {diff}: {stats['correct']}/{stats['total']} ({acc:.1%})")

    logger.info("=" * 70)

    return {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "by_difficulty": by_difficulty,
        "results": all_results,
    }


# ──────────────────────── CLI ────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Summarization RLM Evaluation")

    parser.add_argument(
        "--turns",
        type=int,
        default=50,
        help="Turns per trace (default: 50)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples (default: 3)",
    )
    parser.add_argument(
        "--include-semantic",
        action="store_true",
        help="Include semantic queries (root cause, outcome assessment)",
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
        default=20,
        help="Maximum agent turns (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.15,
        help="Fraction of tool calls that fail (default: 0.15)",
    )

    args = parser.parse_args()

    config = TraceSummarizationConfig(
        endpoint=EndpointConfig(provider=args.provider, model=args.model),
        rlm=RLMConfig(enabled=True),
        dataset=TraceSummarizationDatasetConfig(
            num_turns=args.turns,
            max_samples=args.samples,
            seed=args.seed,
            include_semantic=args.include_semantic,
            error_rate=args.error_rate,
        ),
        run=EvalRunConfig(max_turns=args.max_turns),
    )

    trio.run(run_evaluation, config)


if __name__ == "__main__":
    main()
