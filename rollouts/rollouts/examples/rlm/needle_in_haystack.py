#!/usr/bin/env python3
"""
Needle-in-a-haystack example for RLM (Recursive Language Models).

This example demonstrates RLM's ability to find information in extremely
large contexts (1M+ characters) by:
1. Storing the context as a Python variable (not in messages)
2. Letting the model explore via REPL code execution
3. Using programmatic search before semantic processing

Usage:
    # Tool-based RLM (recommended)
    python -m rollouts.examples.rlm.needle_in_haystack --env repl

    # Block-based RLM (parses ```repl blocks)
    python -m rollouts.examples.rlm.needle_in_haystack --env repl_blocks

Reference: https://github.com/alexzhang13/rlm-minimal
"""

import argparse
import random
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import trio

from rollouts.agents import run_agent
from rollouts.dtypes import Actor, AgentState, Endpoint, Message, RunConfig, Trajectory
from rollouts.environments.repl import MessageParsingREPLEnvironment, REPLEnvironment


def generate_massive_context(num_lines: int = 100_000, seed: int | None = None) -> tuple[str, int]:
    """Generate a massive context with a hidden magic number.

    Args:
        num_lines: Number of lines of random text to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (context_string, magic_number)
    """
    if seed is not None:
        random.seed(seed)

    words = ["blah", "random", "text", "foo", "bar", "hello", "world", "test", "data", "line"]

    # Generate random lines
    lines = []
    for i in range(num_lines):
        num_words = random.randint(3, 10)
        line = " ".join(random.choice(words) for _ in range(num_words))
        lines.append(f"Line {i}: {line}")

    # Insert the magic number somewhere in the middle
    magic_number = random.randint(1000, 9999)
    insert_position = random.randint(num_lines // 3, 2 * num_lines // 3)
    lines[insert_position] = f"Line {insert_position}: The magic number is {magic_number}"

    context = "\n".join(lines)
    print(f"Generated context: {len(context):,} characters, {num_lines:,} lines")
    print(f"Magic number {magic_number} hidden at line {insert_position}")

    return context, magic_number


async def run_rlm_example(
    env_type: str = "repl",
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    num_lines: int = 10_000,
    max_turns: int = 20,
    seed: int | None = 42,
) -> tuple[str | None, int]:
    """Run the needle-in-haystack example with RLM."""

    # Generate context
    context, magic_number = generate_massive_context(num_lines=num_lines, seed=seed)

    # Create endpoint
    import os

    endpoint = Endpoint(
        provider="anthropic",
        model=model.split("/")[-1],
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        max_tokens=4096,
    )

    # Create environment
    if env_type == "repl":
        environment = REPLEnvironment(
            context=context,
            sub_endpoint=endpoint,
            recursive=False,
        )
        system_prompt = """You are an assistant with access to a REPL environment.

The input context is stored in `context`. Use the tools:
- repl: Execute Python code (context variable available)
- llm_query: Query sub-LLM for semantic tasks
- final_answer: Submit your answer

Find the magic number hidden in the context."""
    else:
        environment = MessageParsingREPLEnvironment(
            context=context,
            sub_endpoint=endpoint,
            recursive=False,
        )
        system_prompt = """You are an assistant with access to a REPL environment.

The input context is stored in `context`. Write code in ```repl blocks.
Use FINAL(answer) when done.

Find the magic number hidden in the context."""

    # Create initial state
    trajectory = Trajectory(
        messages=[
            Message(role="system", content=system_prompt),
            Message(
                role="user",
                content="Find the magic number hidden in the context. The context is very large, so use the REPL to search efficiently.",
            ),
        ]
    )

    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint,
        tools=environment.get_tools(),
    )

    state = AgentState(
        actor=actor,
        environment=environment,
    )

    # Run agent
    print(f"\n{'=' * 60}")
    print(f"Running RLM with {env_type} environment")
    print(f"Model: {model}")
    print(f"Context size: {len(context):,} chars")
    print(f"{'=' * 60}\n")

    from rollouts.agents import handle_stop_max_turns, stdout_handler

    run_config = RunConfig(
        on_chunk=stdout_handler,
        handle_stop=handle_stop_max_turns(max_turns),
    )

    states = await run_agent(state, run_config)

    # Check result
    final_answer = environment._final_answer
    print(f"\n{'=' * 60}")
    print(f"Final answer: {final_answer}")
    print(f"Expected: {magic_number}")
    print(f"Correct: {str(magic_number) in str(final_answer) if final_answer else False}")
    print(f"Turns used: {len(states)}")
    print(f"{'=' * 60}")

    return final_answer, magic_number


def main() -> None:
    parser = argparse.ArgumentParser(description="RLM Needle-in-Haystack Example")
    parser.add_argument(
        "--env",
        type=str,
        choices=["repl", "repl_blocks"],
        default="repl",
        help="Environment type: repl (tool-based) or repl_blocks (message parsing)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4-5-20250929",
        help="Model to use",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=10_000,
        help="Number of lines in context (default: 10000)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum turns (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    trio.run(
        run_rlm_example,
        args.env,
        args.model,
        args.lines,
        args.max_turns,
        args.seed,
    )


if __name__ == "__main__":
    main()
