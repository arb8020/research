#!/usr/bin/env python3
"""
CLI entry point for interactive TUI agent.

Usage:
    python -m rollouts.frontends.tui.cli --model gpt-4o-mini
    python -m rollouts.frontends.tui.cli --model claude-sonnet-4-5 --provider anthropic
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import trio

from rollouts.dtypes import Endpoint, Message, Trajectory
from rollouts.environments import CalculatorEnvironment
from rollouts.frontends.tui.interactive_agent import run_interactive_agent


SYSTEM_PROMPTS = {
    "none": "You are a helpful assistant.",
    "calculator": """You are a calculator assistant with access to math tools.

Available tools: add, subtract, multiply, divide, clear, complete_task.
Each tool operates on a running total (starts at 0).

For calculations:
1. Break down the problem into steps
2. Use tools to compute each step
3. Use complete_task when done

Example: For "(5 + 3) * 2", first add(5), then add(3), then multiply(2).""",
}


def create_endpoint(provider: str, model: str, api_base: str | None = None, api_key: str | None = None) -> Endpoint:
    """Create endpoint from CLI arguments."""
    import os

    if api_base is None:
        if provider == "openai":
            api_base = "https://api.openai.com/v1"
        elif provider == "anthropic":
            api_base = "https://api.anthropic.com"
        else:
            api_base = "https://api.openai.com/v1"  # Default

    # Get API key from environment if not provided
    if api_key is None:
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            api_key = ""

    return Endpoint(
        provider=provider,
        model=model,
        api_base=api_base,
        api_key=api_key,
    )


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive TUI agent - chat with an LLM agent in your terminal"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="Provider to use (default: openai)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL (default: provider-specific)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: from environment OPENAI_API_KEY or ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt (default: depends on --env)",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["none", "calculator"],
        default="none",
        help="Environment with tools: none, calculator (default: none)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum number of turns (default: 50)",
    )

    args = parser.parse_args()

    # Create endpoint
    endpoint = create_endpoint(args.provider, args.model, args.api_base, args.api_key)

    # Validate API key is set
    if not endpoint.api_key:
        env_var = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
        print(f"\n❌ Error: No API key found. Please set {env_var} environment variable or use --api-key flag.", file=sys.stderr)
        return 1

    # Create environment
    environment = None
    if args.env == "calculator":
        environment = CalculatorEnvironment()

    # Get system prompt (user-provided or default for env)
    system_prompt = args.system_prompt or SYSTEM_PROMPTS.get(args.env, SYSTEM_PROMPTS["none"])

    # Create initial trajectory
    system_msg = Message(role="system", content=system_prompt)
    trajectory = Trajectory(messages=[system_msg])

    # Run interactive agent
    try:
        states = trio.run(
            run_interactive_agent,
            trajectory,
            endpoint,
            environment,
            args.max_turns,
        )
        return 0
    except KeyboardInterrupt:
        print("\n\n✅ Agent stopped")
        return 0
    except Exception as e:
        print(f"\n\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

