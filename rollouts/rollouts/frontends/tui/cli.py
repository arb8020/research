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

from pathlib import Path

from rollouts.agents import AgentState, Actor, run_agent
from rollouts.dtypes import (
    Endpoint,
    Message,
    RunConfig,
    StreamEvent,
    TextDelta,
    Trajectory,
    ToolConfirmResult,
    StopReason,
    ToolCall,
)
from rollouts.environments import CalculatorEnvironment, CodingEnvironment
from rollouts.frontends.tui.interactive_agent import run_interactive_agent
from rollouts.frontends.tui.sessions import (
    Session,
    SessionInfo,
    create_session,
    find_latest_session,
    list_sessions_with_info,
    load_session,
    load_messages,
    load_header,
    append_message,
)


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
    "coding": """You are a coding assistant with access to file and shell tools.

Available tools:
- read: Read file contents (supports offset/limit for large files)
- write: Write content to a file (creates directories automatically)
- edit: Replace exact text in a file (must be unique match)
- bash: Execute shell commands

When working on code:
1. First read relevant files to understand context
2. Make precise edits using the edit tool
3. Use bash to run tests, linting, etc.
4. Prefer small, focused changes over large rewrites""",
}


def format_time_ago(dt: 'datetime') -> str:
    """Format a datetime as relative time (e.g., '2h ago', '3d ago')."""
    from datetime import datetime, timezone
    now = datetime.now()
    diff = now - dt

    seconds = diff.total_seconds()
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        mins = int(seconds / 60)
        return f"{mins}m ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    else:
        days = int(seconds / 86400)
        return f"{days}d ago"


def pick_session(working_dir: Path) -> Session | None:
    """Interactive session picker. Returns None if no sessions or user cancels."""
    sessions = list_sessions_with_info(working_dir)

    if not sessions:
        print("No sessions found for this directory.")
        return None

    print(f"\nSessions for {working_dir}:\n")
    for i, info in enumerate(sessions):
        time_ago = format_time_ago(info.modified)
        preview = info.first_user_message or "(no messages)"
        print(f"  [{i + 1}] {time_ago:>10}  {info.message_count:>3} msgs  {preview}")

    print(f"\n  [0] Cancel")
    print()

    while True:
        try:
            choice = input("Select session: ").strip()
            if not choice:
                continue
            num = int(choice)
            if num == 0:
                return None
            if 1 <= num <= len(sessions):
                return sessions[num - 1].session
            print(f"Please enter 0-{len(sessions)}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            print()
            return None


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
        choices=["none", "calculator", "coding"],
        default="none",
        help="Environment with tools: none, calculator, coding (default: none)",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory for coding environment (default: current directory)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum number of turns (default: 50)",
    )

    # Session management
    parser.add_argument(
        "--continue", "-c",
        dest="continue_session",
        action="store_true",
        help="Continue most recent session",
    )
    parser.add_argument(
        "--session", "-s",
        type=str,
        nargs="?",
        const="",  # Empty string when flag used without value
        default=None,
        help="Resume session: -s to list/pick, -s PATH to resume specific file",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="Don't persist session to disk",
    )

    # Unix utility mode
    parser.add_argument(
        "-p", "--print",
        dest="print_mode",
        type=str,
        default=None,
        metavar="QUERY",
        help="Non-interactive mode: run query and print result",
    )

    args = parser.parse_args()

    # Create endpoint
    endpoint = create_endpoint(args.provider, args.model, args.api_base, args.api_key)

    # Validate API key is set
    if not endpoint.api_key:
        env_var = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
        print(f"\n❌ Error: No API key found. Please set {env_var} environment variable or use --api-key flag.", file=sys.stderr)
        return 1

    # Determine working directory
    working_dir = Path(args.cwd) if args.cwd else Path.cwd()

    # Create environment
    environment = None
    if args.env == "calculator":
        environment = CalculatorEnvironment()
    elif args.env == "coding":
        environment = CodingEnvironment(working_dir=working_dir)

    # Handle session resumption
    session: Session | None = None
    messages: list[Message] = []

    if args.session is not None:
        if args.session == "":
            # --session without path: show picker
            session = pick_session(working_dir)
            if session is None:
                return 0  # User cancelled or no sessions
            messages = load_messages(session)
            header = load_header(session)
            print(f"Resuming session: {session.session_id}")
            print(f"  {len(messages)} messages from {header.timestamp}")
        else:
            # Resume specific session by path
            session = load_session(Path(args.session))
            messages = load_messages(session)
            header = load_header(session)
            print(f"Resuming session: {session.session_id}")
            print(f"  {len(messages)} messages from {header.timestamp}")
    elif args.continue_session:
        # Continue most recent session
        session = find_latest_session(working_dir)
        if session:
            messages = load_messages(session)
            header = load_header(session)
            print(f"Continuing session: {session.session_id}")
            print(f"  {len(messages)} messages from {header.timestamp}")
        else:
            print("No previous session found, starting new session")

    # Create new session if needed (and not --no-session)
    if session is None and not args.no_session:
        session = create_session(working_dir, args.provider, args.model)
        print(f"New session: {session.session_id}")

    # Get system prompt (user-provided or default for env)
    system_prompt = args.system_prompt or SYSTEM_PROMPTS.get(args.env, SYSTEM_PROMPTS["none"])

    # Build trajectory from messages or start fresh
    if messages:
        # Check if first message is system prompt, if not prepend one
        if not messages or messages[0].role != "system":
            messages.insert(0, Message(role="system", content=system_prompt))
        trajectory = Trajectory(messages=messages)
    else:
        system_msg = Message(role="system", content=system_prompt)
        trajectory = Trajectory(messages=[system_msg])

    # Non-interactive print mode
    if args.print_mode:
        return trio.run(run_print_mode, trajectory, endpoint, environment, args.print_mode, session)

    # Run interactive agent
    try:
        states = trio.run(
            run_interactive_agent,
            trajectory,
            endpoint,
            environment,
            args.max_turns,
            session,  # Pass session for persistence
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


async def run_print_mode(
    trajectory: Trajectory,
    endpoint: Endpoint,
    environment,
    query: str,
    session: Session | None,
) -> int:
    """Run in non-interactive print mode - execute query and print result."""

    # Add user query to trajectory
    trajectory = Trajectory(
        messages=trajectory.messages + [Message(role="user", content=query)]
    )

    # Persist user message
    if session:
        append_message(session, Message(role="user", content=query))

    # Create initial state
    initial_state = AgentState(
        actor=Actor(
            trajectory=trajectory,
            endpoint=endpoint,
            tools=environment.get_tools() if environment else [],
        ),
        environment=environment,
    )

    # Simple streaming handler - just print text
    accumulated_text = ""

    async def print_handler(event: StreamEvent) -> None:
        nonlocal accumulated_text
        if isinstance(event, TextDelta):
            print(event.delta, end="", flush=True)
            accumulated_text += event.delta

    # Auto-confirm tools
    async def auto_confirm(tc: ToolCall, state: AgentState, rcfg: RunConfig) -> tuple[AgentState, ToolConfirmResult]:
        return state, ToolConfirmResult(proceed=True)

    # Stop after one turn (no tool handling for now in print mode)
    def handle_stop(state: AgentState) -> AgentState:
        from dataclasses import replace
        # Stop after first turn completes (turn_idx > 0 means we completed a turn)
        if state.turn_idx > 0:
            return replace(state, stop=StopReason.TASK_COMPLETED)
        return state

    run_config = RunConfig(
        on_chunk=print_handler,
        confirm_tool=auto_confirm,
        handle_stop=handle_stop,
    )

    error_occurred = False
    try:
        await run_agent(initial_state, run_config)
    except ValueError as e:
        # Ignore aggregation errors - text was already printed
        if "empty message" not in str(e):
            print(f"\nError: {e}", file=sys.stderr)
            error_occurred = True
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        error_occurred = True

    print()  # Final newline

    # Persist assistant response (even if there was an error, save what we got)
    if session and accumulated_text:
        append_message(session, Message(role="assistant", content=accumulated_text))

    return 1 if error_occurred else 0


if __name__ == "__main__":
    sys.exit(main())

