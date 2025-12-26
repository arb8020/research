"""GEPA prompt optimization example using the Calculator environment.

This example demonstrates:
1. Defining a PromptTemplate for a tool-using agent
2. Configuring GEPA optimization
3. Running optimization to improve the system prompt
4. Using the optimized prompt for evaluation

Run with:
    python -m examples.prompt_optimization.calculator_gepa
"""

import logging
import os

import trio

from rollouts.dtypes import Endpoint, Metric, Score
from rollouts.environments.calculator import CalculatorEnvironment
from rollouts.prompt_optimization import (
    EvolutionaryConfig,
    PromptTemplate,
    run_evolutionary_gepa,
)
from rollouts.training.types import Sample

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


def score_fn(sample: Sample) -> Score:
    """Score a calculator sample based on whether the answer is correct.

    Extracts the final answer from the trajectory and compares to ground truth.
    """
    if not sample.trajectory or not sample.trajectory.messages:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    # Get the last assistant message
    answer_text = ""
    for msg in reversed(sample.trajectory.messages):
        if msg.role == "assistant":
            if isinstance(msg.content, str):
                answer_text = msg.content
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text"):
                        answer_text += block.text
            break

    # Try to extract a number from the response
    import re

    numbers = re.findall(r"-?\d+\.?\d*", answer_text)
    if not numbers:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    # Check if any extracted number matches the answer
    expected = float(sample.ground_truth)
    for num_str in numbers:
        try:
            extracted = float(num_str)
            if abs(extracted - expected) < 0.01:
                return Score(metrics=(Metric("correct", 1.0, weight=1.0),))
        except ValueError:
            continue

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

    # ─── Initial Template ─────────────────────────────────────────────────
    initial_template = PromptTemplate(
        system="You are a calculator assistant. Use the available tools to compute answers.",
        user_template="{question}",
    )

    # ─── GEPA Configuration ───────────────────────────────────────────────
    config = EvolutionaryConfig(
        population_size=8,  # Small for demo
        generations=3,  # Few generations for demo
        mutation_rate=0.4,
        crossover_rate=0.3,
        elite_size=2,
        train_seeds=tuple(range(10)),  # First 10 samples for training
        val_seeds=tuple(range(10, 15)),  # Next 5 for validation
        max_concurrent=4,
    )

    # ─── Endpoints ────────────────────────────────────────────────────────
    task_endpoint = Endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_tokens=256,
        temperature=0.0,
    )

    mutation_endpoint = Endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_tokens=1024,
        temperature=0.7,  # Higher temp for creative mutations
    )

    # ─── Run GEPA ─────────────────────────────────────────────────────────
    logger.info("Starting GEPA optimization...")
    logger.info(f"Initial prompt: {initial_template.system}")

    result = await run_evolutionary_gepa(
        initial_template=initial_template,
        config=config,
        dataset=DATASET,
        endpoint=task_endpoint,
        mutation_endpoint=mutation_endpoint,
        score_fn=score_fn,
        environment_factory=environment_factory,
    )

    # ─── Results ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total evaluations: {result.total_evaluations}")
    logger.info(f"Best validation score: {result.best_template.score:.2%}")
    logger.info("")
    logger.info("Optimized system prompt:")
    logger.info("-" * 40)
    logger.info(result.best_template.system)
    logger.info("-" * 40)
    logger.info("")

    # History
    logger.info("Generation history:")
    for stats in result.history:
        logger.info(
            f"  Gen {stats.generation + 1}: "
            f"best={stats.best_score:.2%}, "
            f"mean={stats.mean_score:.2%} ± {stats.std_score:.2%}"
        )


if __name__ == "__main__":
    trio.run(main)
