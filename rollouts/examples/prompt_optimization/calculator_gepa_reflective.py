"""Reflective GEPA prompt optimization using the Calculator environment.

This example demonstrates:
1. Using reflective GEPA to optimize a system prompt for a tool-using agent
2. The mutation LLM sees the full trajectory (tool calls, results, feedback)
3. Multi-turn agent loops with calculator tools

Reflective GEPA uses Pareto-efficient search with trace-based feedback.
The mutation LLM sees what went wrong (inputs, tool calls, results, feedback)
and proposes targeted fixes.
For comparison, see calculator_gepa_evolutionary.py which uses blind mutations.

Run with:
    python -m examples.prompt_optimization.calculator_gepa_reflective
"""

import logging
import os
import re

import trio

from rollouts.core import Endpoint, Metric, Score
from rollouts.environments.calculator import CalculatorEnvironment
from rollouts.prompt_optimization import GEPAConfig, optimize_prompt
from rollouts.training.scoring import FunctionScorer
from rollouts.training.types import RowAttempt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Dataset ──────────────────────────────────────────────────────────────────

# Simple arithmetic problems for the calculator environment
DATASET = [
    {"question": "What is 15 + 27?", "answer": 42},
    {"question": "Calculate 100 - 37", "answer": 63},
    {"question": "What is 8 * 7?", "answer": 56},
    {"question": "Divide 144 by 12", "answer": 12},
    {"question": "What is 25 + 17 - 10?", "answer": 32},
    {"question": "Calculate (50 + 30) * 2", "answer": 160},
    {"question": "What is 99 / 11?", "answer": 9},
    {"question": "Add 234 and 567", "answer": 801},
    {"question": "Subtract 45 from 100", "answer": 55},
    {"question": "Multiply 13 by 4", "answer": 52},
    {"question": "What is 200 / 8?", "answer": 25},
    {"question": "Calculate 77 + 33", "answer": 110},
    {"question": "What is 1000 - 999?", "answer": 1},
    {"question": "What is 12 * 12?", "answer": 144},
    {"question": "Divide 81 by 9", "answer": 9},
    {"question": "What is 45 + 55?", "answer": 100},
    {"question": "Calculate 7 * 8 + 4", "answer": 60},
    {"question": "What is 90 / 10 - 3?", "answer": 6},
    {"question": "Add 111, 222, and 333", "answer": 666},
    {"question": "What is 5 * 5 * 5?", "answer": 125},
]


# ─── Score Function ───────────────────────────────────────────────────────────


def score_fn(sample: RowAttempt, _context: object) -> Score:
    """Score a calculator sample based on whether the answer is correct.

    The calculator environment uses complete_task tool to submit answers,
    so we check the tool call arguments for the final_result.
    """
    if not sample.trajectory or not sample.trajectory.messages:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    expected = float(sample.ground_truth)

    # Look for complete_task tool call with final_result
    for msg in reversed(sample.trajectory.messages):
        if msg.role == "assistant" and isinstance(msg.content, list):
            for block in msg.content:
                if hasattr(block, "name") and block.name == "complete_task":
                    args = getattr(block, "arguments", {})
                    final_result = args.get("final_result")
                    if final_result is not None:
                        if abs(float(final_result) - expected) < 0.01:
                            return Score(metrics=(Metric("correct", 1.0, weight=1.0),))
                        else:
                            return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    # Fallback: check for number in text response
    for msg in reversed(sample.trajectory.messages):
        if msg.role == "assistant":
            answer_text = ""
            if isinstance(msg.content, str):
                answer_text = msg.content
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text"):
                        answer_text += block.text
            if answer_text:
                numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                for num_str in numbers:
                    try:
                        if abs(float(num_str) - expected) < 0.01:
                            return Score(metrics=(Metric("correct", 1.0, weight=1.0),))
                    except ValueError:
                        continue
                break

    return Score(metrics=(Metric("correct", 0.0, weight=1.0),))


# ─── Environment Factory ──────────────────────────────────────────────────────


async def environment_factory(sample: dict) -> CalculatorEnvironment:
    """Create a fresh calculator environment for each sample."""
    return CalculatorEnvironment()


# ─── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        return

    # ─── Endpoints ────────────────────────────────────────────────────────
    endpoint = Endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_tokens=256,
        temperature=0.0,
    )

    reflection_endpoint = Endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_tokens=1024,
        temperature=0.7,
    )

    # ─── Initial System Prompt ────────────────────────────────────────────
    initial_system = "You are a calculator assistant. Use the available tools to compute answers."

    # ─── Run Reflective GEPA ──────────────────────────────────────────────
    logger.info("Starting reflective GEPA optimization for Calculator...")
    logger.info(f"Initial prompt: {initial_system}")
    logger.info(f"Dataset size: {len(DATASET)} problems")

    result = await optimize_prompt(
        system=initial_system,
        user_template="{question}",
        dataset=DATASET,
        scorer=FunctionScorer(score_fn),
        endpoint=endpoint,
        reflection_endpoint=reflection_endpoint,
        environment_factory=environment_factory,
        max_turns=5,  # Allow up to 5 turns for tool use
        config=GEPAConfig(
            max_evaluations=200,  # Budget for demo
            minibatch_size=4,
        ),
        seed=42,
    )

    # ─── Results ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total evaluations: {result.total_evaluations}")
    logger.info(f"Best score: {result.best_score:.2%}")
    logger.info(f"Iterations with improvements: {len(result.history)}")
    logger.info("")

    if result.history:
        logger.info("Progress:")
        for h in result.history[-5:]:
            logger.info(f"  iter={h['iteration']}, best={h['best_score']:.2%}")

    logger.info("")
    logger.info("Optimized system prompt:")
    logger.info("-" * 40)
    logger.info(result.best_candidate["system"])
    logger.info("-" * 40)


if __name__ == "__main__":
    trio.run(main)
