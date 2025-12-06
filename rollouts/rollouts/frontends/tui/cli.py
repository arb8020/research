#!/usr/bin/env python3
"""
CLI entry point for interactive TUI agent.

Usage:
    python -m rollouts.frontends.tui.cli
    python -m rollouts.frontends.tui.cli --model openai/gpt-4o-mini
    python -m rollouts.frontends.tui.cli --model anthropic/claude-sonnet-4-5 --thinking disabled

Model format is "provider/model" (explicit, no inference).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import trio

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
from rollouts.environments import CalculatorEnvironment, LocalFilesystemEnvironment
from rollouts.frontends.tui.interactive_agent import run_interactive_agent
from rollouts.store import FileSessionStore
from rollouts import AgentSession, EndpointConfig, EnvironmentConfig


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


def format_time_ago(dt_str: str) -> str:
    """Format a datetime string as relative time (e.g., '2h ago', '3d ago')."""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        return "unknown"

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


async def pick_session_async(session_store: FileSessionStore) -> AgentSession | None:
    """Interactive session picker. Returns None if no sessions or user cancels."""
    sessions = await session_store.list(limit=20)

    if not sessions:
        print("No sessions found.")
        return None

    print("\nRecent sessions:\n")
    for i, session in enumerate(sessions):
        time_ago = format_time_ago(session.created_at)
        msg_count = len(session.messages) if session.messages else "?"
        status = session.status.value if session.status else "?"
        print(f"  [{i + 1}] {time_ago:>10}  {msg_count:>3} msgs  [{status}]  {session.session_id}")

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
                # Load full session with messages
                full_session, err = await session_store.get(sessions[num - 1].session_id)
                if err:
                    print(f"Error loading session: {err}")
                    return None
                return full_session
            print(f"Please enter 0-{len(sessions)}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            print()
            return None


def parse_model_string(model_str: str) -> tuple[str, str]:
    """Parse model string into (provider, model_name).

    Requires explicit "provider/model" format (e.g., "anthropic/claude-3-5-haiku-20241022").

    Args:
        model_str: Model string in "provider/model" format

    Returns:
        Tuple of (provider, model_name)

    Raises:
        ValueError: If model_str doesn't contain "/"
    """
    if "/" not in model_str:
        raise ValueError(
            f'Model must be in "provider/model" format (e.g., "anthropic/claude-sonnet-4-5"). '
            f'Got: "{model_str}"'
        )

    provider, model = model_str.split("/", 1)
    return provider, model


def create_endpoint(model_str: str, api_base: str | None = None, api_key: str | None = None, thinking: str = "enabled") -> Endpoint:
    """Create endpoint from CLI arguments.

    Args:
        model_str: Model string in "provider/model" format (e.g., "anthropic/claude-sonnet-4-5")
        api_base: Optional API base URL
        api_key: Optional API key (otherwise from env)
        thinking: Extended thinking setting ("enabled" or "disabled")
    """
    import os
    from rollouts.models import get_model

    # Parse model string
    provider, model = parse_model_string(model_str)

    # Check model capabilities if thinking is enabled
    if thinking == "enabled":
        model_metadata = get_model(provider, model)  # type: ignore
        if model_metadata is not None:
            if not model_metadata.reasoning:
                raise ValueError(
                    f"Model '{model}' does not support extended thinking/reasoning.\n"
                    f"Either:\n"
                    f"  1. Use a model that supports reasoning (e.g., anthropic/claude-3-5-sonnet-20241022)\n"
                    f"  2. Disable thinking with --thinking disabled"
                )
        # If model not in registry, warn but allow (might be custom endpoint)
        elif provider == "anthropic":
            print(f"⚠️  Warning: Model '{model}' not in registry. Cannot verify thinking support.")

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

    # Configure extended thinking for Anthropic
    thinking_config = None
    thinking_budget = 10000
    if provider == "anthropic" and thinking == "enabled":
        thinking_config = {"type": "enabled", "budget_tokens": thinking_budget}

    # max_tokens must be greater than thinking budget
    # Set to a reasonable value that accommodates both thinking and response
    max_tokens = 16384 if thinking_config else 8192

    return Endpoint(
        provider=provider,
        model=model,
        api_base=api_base,
        api_key=api_key,
        thinking=thinking_config,
        max_tokens=max_tokens,
    )


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive TUI agent - chat with an LLM agent in your terminal"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5.1-codex",
        help='Model in "provider/model" format (e.g., "openai/gpt-5.1-codex", "anthropic/claude-sonnet-4-5"). Default: openai/gpt-5.1-codex',
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

    # Theme selection
    parser.add_argument(
        "--theme",
        type=str,
        choices=["dark", "rounded", "minimal"],
        default="minimal",
        help="UI theme (default: minimal)",
    )

    # Extended thinking (Anthropic)
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["enabled", "disabled"],
        default="enabled",
        help="Enable extended thinking for Anthropic models (default: enabled)",
    )

    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and chat state dumps",
    )

    # Visual debug mode
    parser.add_argument(
        "--debug-layout",
        action="store_true",
        help="Show component boundaries and spacing in the UI",
    )

    args = parser.parse_args()

    # Create endpoint
    endpoint = create_endpoint(args.model, args.api_base, args.api_key, args.thinking)

    # Validate API key is set
    if not endpoint.api_key:
        env_var = "OPENAI_API_KEY" if endpoint.provider == "openai" else "ANTHROPIC_API_KEY"
        print(f"\n❌ Error: No API key found. Please set {env_var} environment variable or use --api-key flag.", file=sys.stderr)
        return 1

    # Determine working directory
    working_dir = Path(args.cwd) if args.cwd else Path.cwd()

    # Create environment
    environment = None
    if args.env == "calculator":
        environment = CalculatorEnvironment()
    elif args.env == "coding":
        environment = LocalFilesystemEnvironment(working_dir=working_dir)

    # Session store
    session_store = FileSessionStore() if not args.no_session else None

    # Get system prompt (user-provided or default for env)
    system_prompt = args.system_prompt or SYSTEM_PROMPTS.get(args.env, SYSTEM_PROMPTS["none"])

    # Run async main to handle session operations
    async def async_main() -> int:
        session_id: str | None = None
        messages: list[Message] = []

        if session_store is not None:
            # Handle session resumption
            if args.session is not None:
                if args.session == "":
                    # --session without path: show picker
                    session = await pick_session_async(session_store)
                    if session is None:
                        return 0  # User cancelled or no sessions
                    session_id = session.session_id
                    # Convert SessionMessage to Message
                    messages = [
                        Message(role=m.role, content=m.content, tool_call_id=m.tool_call_id)
                        for m in session.messages
                    ]
                    print(f"Resuming session: {session_id}")
                    print(f"  {len(messages)} messages from {session.created_at}")
                else:
                    # Resume specific session by ID
                    session, err = await session_store.get(args.session)
                    if err or session is None:
                        print(f"Error loading session: {err}", file=sys.stderr)
                        return 1
                    session_id = session.session_id
                    messages = [
                        Message(role=m.role, content=m.content, tool_call_id=m.tool_call_id)
                        for m in session.messages
                    ]
                    print(f"Resuming session: {session_id}")
                    print(f"  {len(messages)} messages from {session.created_at}")
            elif args.continue_session:
                # Continue most recent session
                session, err = await session_store.get_latest()
                if session:
                    session_id = session.session_id
                    messages = [
                        Message(role=m.role, content=m.content, tool_call_id=m.tool_call_id)
                        for m in session.messages
                    ]
                    print(f"Continuing session: {session_id}")
                    print(f"  {len(messages)} messages from {session.created_at}")
                else:
                    print("No previous session found, starting new session")

            # Create new session if needed
            if session_id is None:
                endpoint_config = EndpointConfig(
                    model=endpoint.model,
                    provider=endpoint.provider,
                    temperature=endpoint.temperature,
                )
                env_config = EnvironmentConfig(
                    type=args.env,
                    config={"cwd": str(working_dir)},
                )
                session = await session_store.create(
                    endpoint=endpoint_config,
                    environment=env_config,
                )
                session_id = session.session_id
                print(f"New session: {session_id}")

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
            return await run_print_mode(trajectory, endpoint, environment, args.print_mode, session_store, session_id)

        # Run interactive agent
        try:
            states = await run_interactive_agent(
                trajectory,
                endpoint,
                environment,
                args.max_turns,
                session_store,  # Pass session store for persistence
                session_id,  # Pass session ID
                args.theme,  # Pass theme selection
                args.debug,  # Pass debug flag
                args.debug_layout,  # Pass layout debug flag
            )
            return 0
        except KeyboardInterrupt:
            print("\n\n✅ Agent stopped")
            return 0

    try:
        return trio.run(async_main)
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
    session_store: FileSessionStore | None,
    session_id: str | None,
) -> int:
    """Run in non-interactive print mode - execute query and print result.

    Session persistence is handled by run_agent() via RunConfig.session_store.
    """

    # Add user query to trajectory
    trajectory = Trajectory(
        messages=trajectory.messages + [Message(role="user", content=query)]
    )

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
    async def print_handler(event: StreamEvent) -> None:
        if isinstance(event, TextDelta):
            print(event.delta, end="", flush=True)

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
        session_store=session_store,
        session_id=session_id,
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

    return 1 if error_occurred else 0


if __name__ == "__main__":
    sys.exit(main())

