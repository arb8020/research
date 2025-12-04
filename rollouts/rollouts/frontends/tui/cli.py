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
from rollouts.frontends.tui.interactive_agent import run_interactive_agent


def create_endpoint(provider: str, model: str, api_base: str | None = None) -> Endpoint:
    """Create endpoint from CLI arguments."""
    if api_base is None:
        if provider == "openai":
            api_base = "https://api.openai.com/v1"
        elif provider == "anthropic":
            api_base = "https://api.anthropic.com"
        else:
            api_base = "https://api.openai.com/v1"  # Default

    return Endpoint(
        provider=provider,
        model=model,
        api_base=api_base,
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
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt (default: 'You are a helpful assistant.')",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum number of turns (default: 50)",
    )

    args = parser.parse_args()

    # Create endpoint
    endpoint = create_endpoint(args.provider, args.model, args.api_base)

    # Create initial trajectory
    system_msg = Message(role="system", content=args.system_prompt)
    trajectory = Trajectory(messages=[system_msg])

    # Run interactive agent
    try:
        states = trio.run(
            run_interactive_agent,
            initial_trajectory=trajectory,
            endpoint=endpoint,
            environment=None,  # No tools by default
            max_turns=args.max_turns,
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

