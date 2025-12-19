#!/usr/bin/env python3
"""
Rollouts CLI - chat with an LLM agent.

Usage:
    python -m rollouts                    # Interactive TUI
    python -m rollouts -p "query"         # Non-interactive, print result
    python -m rollouts --export-md        # Export session to markdown
    python -m rollouts --login-claude     # Login with Claude Pro/Max

Model format is "provider/model" (e.g., "anthropic/claude-opus-4-5-20251101").
For Anthropic: auto-uses OAuth if logged in, otherwise ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import trio

from rollouts import AgentSession
from rollouts.dtypes import Endpoint, Message, Trajectory
from rollouts.environments import (
    CalculatorEnvironment,
    GitWorktreeEnvironment,
    LocalFilesystemEnvironment,
)
from rollouts.store import FileSessionStore

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
    "git": """You are a coding assistant with access to file and shell tools.

All file changes are automatically tracked in an isolated git history.
This gives you full undo capability - every write/edit/bash creates a commit.

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
        msg_count = (
            session.message_count
            if session.message_count is not None
            else len(session.messages)
            if session.messages
            else "?"
        )
        status = session.status.value if session.status else "?"
        print(f"  [{i + 1}] {time_ago:>10}  {msg_count:>3} msgs  [{status}]  {session.session_id}")

    print("\n  [0] Cancel")
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
    """
    if "/" not in model_str:
        raise ValueError(
            f'Model must be in "provider/model" format (e.g., "anthropic/claude-sonnet-4-5"). '
            f'Got: "{model_str}"'
        )

    provider, model = model_str.split("/", 1)
    return provider, model


def get_oauth_client():
    """Get OAuth client for Anthropic. Lazy import to avoid TUI dependencies."""
    from rollouts.frontends.tui.oauth import get_oauth_client as _get_oauth_client

    return _get_oauth_client()


def create_endpoint(
    model_str: str,
    api_base: str | None = None,
    api_key: str | None = None,
    thinking: str = "enabled",
) -> Endpoint:
    """Create endpoint from CLI arguments."""
    import os

    from rollouts.models import get_model

    # Parse model string
    provider, model = parse_model_string(model_str)

    # Check model capabilities if thinking is enabled
    if thinking == "enabled":
        model_metadata = get_model(provider, model)
        if model_metadata is not None:
            if not model_metadata.reasoning:
                raise ValueError(
                    f"Model '{model}' does not support extended thinking/reasoning.\n"
                    f"Either:\n"
                    f"  1. Use a model that supports reasoning (e.g., anthropic/claude-3-5-sonnet-20241022)\n"
                    f"  2. Disable thinking with --thinking disabled"
                )
        else:
            raise ValueError(
                f"Model '{model}' not found in registry.\n"
                f"Use a registered model or add it to rollouts/models.py"
            )

    if api_base is None:
        if provider == "openai":
            api_base = "https://api.openai.com/v1"
        elif provider == "anthropic":
            api_base = "https://api.anthropic.com"
        else:
            api_base = "https://api.openai.com/v1"

    # For Anthropic: auto-detect OAuth if no API key provided
    oauth_token = ""
    if api_key is None and provider == "anthropic":
        client = get_oauth_client()
        tokens = client.tokens
        if tokens:
            if tokens.is_expired():
                try:
                    tokens = trio.run(client.refresh_tokens)
                    oauth_token = tokens.access_token
                    print("🔐 OAuth token refreshed", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️  OAuth token expired and refresh failed: {e}", file=sys.stderr)
                    oauth_token = ""
            else:
                oauth_token = tokens.access_token
            if oauth_token:
                print("🔐 Using OAuth authentication (Claude Pro/Max)", file=sys.stderr)
        if not oauth_token:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if api_key is None:
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
        else:
            api_key = ""

    # Configure extended thinking for Anthropic
    thinking_config = None
    thinking_budget = 10000
    if provider == "anthropic" and thinking == "enabled":
        thinking_config = {"type": "enabled", "budget_tokens": thinking_budget}

    max_tokens = 16384 if thinking_config else 8192

    return Endpoint(
        provider=provider,
        model=model,
        api_base=api_base,
        api_key=api_key,
        oauth_token=oauth_token,
        thinking=thinking_config,
        max_tokens=max_tokens,
    )


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Rollouts - chat with an LLM agent in your terminal"
    )
    # Preset configuration
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Agent preset name (e.g., 'fast_coder', 'careful_coder') or path to preset file",
    )

    # Individual overrides (can override preset values)
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-opus-4-5-20251101",
        help='Model in "provider/model" format. Default: anthropic/claude-opus-4-5-20251101',
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
        help="API key (default: from environment)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt (default: depends on --env or preset)",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["none", "calculator", "coding", "git"],
        default="none",
        help="Environment with tools: none, calculator, coding, git (default: none)",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Tool preset for coding env: full, readonly, no-write (default: full)",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory for coding environment",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum number of turns (default: 50)",
    )
    parser.add_argument(
        "--confirm-tools",
        action="store_true",
        help="Require confirmation before executing tools",
    )

    # Session management
    parser.add_argument(
        "--continue",
        "-c",
        dest="continue_session",
        action="store_true",
        help="Continue most recent session",
    )
    parser.add_argument(
        "--session",
        "-s",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Resume session: -s to list/pick, -s ID to resume specific",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="Don't persist session to disk",
    )

    # Non-interactive mode
    parser.add_argument(
        "-p",
        "--print",
        dest="print_mode",
        type=str,
        nargs="?",
        const="-",
        default=None,
        metavar="QUERY",
        help="Non-interactive mode: run query and print result. Use '-p -' or just '-p' to read from stdin.",
    )
    parser.add_argument(
        "--stream-json",
        action="store_true",
        help="Output NDJSON per turn (for print mode). Each line is a JSON object.",
    )

    # Frontend options
    parser.add_argument(
        "--frontend",
        type=str,
        choices=["tui", "none", "textual"],
        default="tui",
        help="Frontend: tui (default Python TUI), none (stdout), textual (rich TUI)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        choices=["dark", "rounded", "minimal"],
        default="minimal",
        help="TUI theme (default: minimal)",
    )

    # Extended thinking (Anthropic)
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["enabled", "disabled"],
        default="enabled",
        help="Extended thinking for Anthropic models (default: enabled)",
    )

    # Preset listing
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available agent presets and exit",
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--debug-layout",
        action="store_true",
        help="Show TUI component boundaries",
    )

    # OAuth (Claude Pro/Max)
    parser.add_argument(
        "--login-claude",
        action="store_true",
        help="Login with Claude Pro/Max account (OAuth)",
    )
    parser.add_argument(
        "--logout-claude",
        action="store_true",
        help="Logout and revoke Claude OAuth tokens",
    )

    # Export
    parser.add_argument(
        "--export-md",
        type=str,
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="Export session to Markdown (stdout if no FILE)",
    )
    parser.add_argument(
        "--export-html",
        type=str,
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="Export session to HTML (stdout if no FILE)",
    )

    # Session transformations
    parser.add_argument(
        "--handoff",
        type=str,
        metavar="GOAL",
        help="Extract goal-directed context from session to stdout (markdown)",
    )

    # Doctor (session repair)
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Show session diagnostics (use with --session)",
    )
    parser.add_argument(
        "--trim",
        type=int,
        metavar="N",
        help="Remove last N messages from session (creates new fixed session)",
    )

    args = parser.parse_args()

    # --- Non-interactive commands (no endpoint needed) ---

    # List presets
    if args.list_presets:
        from rollouts.agent_presets import list_presets

        presets = list_presets()
        if not presets:
            print("No presets found in rollouts/agent_presets/")
            return 0

        print("Available agent presets:")
        for preset_name in presets:
            print(f"  - {preset_name}")

        print("\nUsage: rollouts --preset <name>")
        print("Example: rollouts --preset sonnet_4")
        return 0

    # OAuth login/logout
    if args.login_claude or args.logout_claude:
        from rollouts.frontends.tui.oauth import OAuthError, login, logout

        if args.logout_claude:
            logout()
            return 0

        async def oauth_action() -> int:
            try:
                await login()
                return 0
            except OAuthError as e:
                print(f"❌ OAuth error: {e}", file=sys.stderr)
                return 1
            except KeyboardInterrupt:
                print("\n⚠️  Login cancelled")
                return 1

        return trio.run(oauth_action)

    # Export
    if args.export_md is not None or args.export_html is not None:
        from rollouts.export import session_to_html, session_to_markdown

        session_store = FileSessionStore()

        async def export_action() -> int:
            if args.session is not None and args.session != "":
                session, err = await session_store.get(args.session)
                if err or session is None:
                    print(f"Error loading session: {err}", file=sys.stderr)
                    return 1
            elif args.session == "" or args.continue_session:
                if args.continue_session:
                    session, err = await session_store.get_latest()
                    if err or session is None:
                        print("No sessions found", file=sys.stderr)
                        return 1
                else:
                    session = await pick_session_async(session_store)
                    if session is None:
                        return 0
            else:
                session, err = await session_store.get_latest()
                if err or session is None:
                    print("No sessions found. Use -s to select a session.", file=sys.stderr)
                    return 1

            if args.export_md is not None:
                output = session_to_markdown(session)
                export_path = args.export_md
            else:
                output = session_to_html(session)
                export_path = args.export_html

            if export_path == "":
                print(output)
            else:
                Path(export_path).write_text(output)
                print(f"Exported to {export_path}")

            return 0

        return trio.run(export_action)

    # Doctor (session repair/diagnostics)
    if args.doctor or args.trim is not None:
        session_store = FileSessionStore()

        async def doctor_action() -> int:
            # Determine which session to doctor
            target_session_id: str | None = None
            if args.session and args.session != "":
                target_session_id = args.session
            elif args.continue_session:
                target_session_id = session_store.get_latest_id_sync()
            else:
                # Default to most recent session
                target_session_id = session_store.get_latest_id_sync()

            if not target_session_id:
                print("No session found. Use --session <id> to specify.", file=sys.stderr)
                return 1

            # Load the session
            session, err = await session_store.get(target_session_id)
            if err or not session:
                print(f"Error loading session: {err}", file=sys.stderr)
                return 1

            # Calculate stats
            total_chars = sum(
                len(str(msg.content)) for msg in session.messages
            )
            estimated_tokens = total_chars // 4  # rough estimate

            print(f"Session: {session.session_id}")
            print(f"  Messages: {len(session.messages)}")
            print(f"  Total chars: {total_chars:,}")
            print(f"  Est. tokens: {estimated_tokens:,}")
            print(f"  Status: {session.status.value}")
            if session.parent_id:
                print(f"  Parent: {session.parent_id}")

            # Show last few messages summary
            if session.messages:
                print("\nLast 5 messages:")
                for msg in session.messages[-5:]:
                    content_preview = str(msg.content)[:80].replace("\n", " ")
                    content_len = len(str(msg.content))
                    print(f"  [{msg.role}] {content_preview}... ({content_len:,} chars)")

            # Trim if requested
            if args.trim is not None:
                if args.trim <= 0:
                    print("\n--trim must be a positive integer", file=sys.stderr)
                    return 1
                if args.trim >= len(session.messages):
                    print(f"\nCannot trim {args.trim} messages from session with {len(session.messages)} messages", file=sys.stderr)
                    return 1

                # Create new session with trimmed messages
                trimmed_messages = session.messages[:-args.trim]
                branch_point = len(trimmed_messages)

                new_session = await session_store.create(
                    endpoint=session.endpoint,
                    environment=session.environment,
                    parent_id=session.session_id,
                    branch_point=branch_point,
                    tags={"doctor": "trimmed", "trimmed_count": str(args.trim)},
                )

                # Copy trimmed messages to new session
                for msg in trimmed_messages:
                    await session_store.append_message(new_session.session_id, msg)

                print(f"\nCreated fixed session: {new_session.session_id}")
                print(f"  Trimmed {args.trim} messages (kept {len(trimmed_messages)})")
                print(f"  Parent: {session.session_id}")
                print(f"\nResume with: rollouts --session {new_session.session_id}")

            return 0

        return trio.run(doctor_action)

    # --- Commands requiring endpoint ---

    # Load preset if specified (must happen before endpoint creation)
    if args.preset:
        from rollouts.agent_presets import load_preset

        try:
            preset = load_preset(args.preset)

            # Apply preset values if individual args not explicitly set
            # Check if arg is still at default value (wasn't overridden by user)
            parser_defaults = {
                "model": "anthropic/claude-opus-4-5-20251101",
                "env": "none",
            }

            # Model: only override if still at default
            if args.model == parser_defaults["model"]:
                args.model = preset.model

            # Env: only override if still at default
            if args.env == parser_defaults["env"]:
                args.env = preset.env

            # System prompt: only override if not explicitly set
            if args.system_prompt is None:
                args.system_prompt = preset.system_prompt

            # Thinking: apply preset value if not explicitly set
            if args.thinking is None:
                args.thinking = preset.thinking

            # Working dir: apply preset if available and not set
            if preset.working_dir and args.cwd is None:
                args.cwd = str(preset.working_dir)

        except Exception as e:
            print(f"Error loading preset '{args.preset}': {e}", file=sys.stderr)
            return 1

    # Load session config if resuming a session (specific ID or --continue)
    # Session config provides defaults; CLI args override
    session_id_for_config: str | None = None
    if args.session and args.session != "":
        session_id_for_config = args.session
    elif args.continue_session:
        session_store_for_config = FileSessionStore()
        session_id_for_config = session_store_for_config.get_latest_id_sync()

    if session_id_for_config:
        session_store_for_config = FileSessionStore()
        session_config, err = session_store_for_config.get_config_sync(session_id_for_config)
        if err:
            print(f"Error loading session config: {err}", file=sys.stderr)
            return 1

        if session_config:
            parser_defaults = {
                "model": "anthropic/claude-opus-4-5-20251101",
                "env": "none",
                "thinking": "enabled",
            }

            # Model: inherit from session if not explicitly set
            if args.model == parser_defaults["model"]:
                endpoint_config = session_config.get("endpoint", {})
                if endpoint_config.get("model"):
                    provider = endpoint_config.get("provider", "anthropic")
                    args.model = f"{provider}/{endpoint_config['model']}"

            # Environment: inherit from session if not explicitly set
            if args.env == parser_defaults["env"]:
                env_config = session_config.get("environment", {})
                env_type = env_config.get("type", "")
                # Map environment class names to CLI env values
                env_map = {
                    "CalculatorEnvironment": "calculator",
                    "LocalFilesystemEnvironment": "coding",
                    "GitWorktreeEnvironment": "git",
                }
                if env_type in env_map:
                    args.env = env_map[env_type]

            # Thinking: inherit from session if not explicitly set
            if args.thinking == parser_defaults["thinking"]:
                endpoint_config = session_config.get("endpoint", {})
                if endpoint_config.get("thinking") is False:
                    args.thinking = "disabled"

            # confirm_tools: inherit from session if not explicitly set
            if not args.confirm_tools:
                env_config = session_config.get("environment", {})
                if env_config.get("config", {}).get("confirm_tools"):
                    args.confirm_tools = True

    working_dir = Path(args.cwd) if args.cwd else Path.cwd()

    # Create endpoint
    try:
        endpoint = create_endpoint(args.model, args.api_base, args.api_key, args.thinking)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    # Validate authentication
    if not endpoint.api_key and not endpoint.oauth_token:
        env_var = "OPENAI_API_KEY" if endpoint.provider == "openai" else "ANTHROPIC_API_KEY"
        print(
            f"❌ No API key found. Set {env_var}, use --api-key, or --login-claude", file=sys.stderr
        )
        return 1

    # Handoff - extract goal-directed context to stdout
    if args.handoff:
        from rollouts.export import run_handoff_command

        session_store = FileSessionStore()

        async def handoff_action() -> int:
            # Require session ID
            if args.session is None:
                print("Error: --handoff requires -s <session_id>", file=sys.stderr)
                return 1

            if args.session == "":
                session = await pick_session_async(session_store)
                if session is None:
                    return 0
            else:
                session, err = await session_store.get(args.session)
                if err or session is None:
                    print(f"Error loading session: {err}", file=sys.stderr)
                    return 1

            handoff_md, err = await run_handoff_command(session, endpoint, args.handoff)
            if err:
                print(f"Error: {err}", file=sys.stderr)
                return 1
            print(handoff_md)
            return 0

        return trio.run(handoff_action)

    # Create environment
    environment = None
    git_env_needs_setup = False
    if args.env == "calculator":
        environment = CalculatorEnvironment()
    elif args.env == "coding":
        from rollouts.environments.coding import TOOL_PRESETS

        tools = args.tools or "full"
        if tools not in TOOL_PRESETS:
            print(
                f"Unknown tool preset: {tools}. Available: {', '.join(TOOL_PRESETS.keys())}",
                file=sys.stderr,
            )
            return 1
        environment = LocalFilesystemEnvironment(working_dir=working_dir, tools=tools)
    elif args.env == "git":
        environment = GitWorktreeEnvironment(working_dir=working_dir)
        git_env_needs_setup = True  # Will call setup() after we have session_id

    session_store = FileSessionStore() if not args.no_session else None
    system_prompt = args.system_prompt or SYSTEM_PROMPTS.get(args.env, SYSTEM_PROMPTS["none"])

    async def async_main() -> int:
        from rollouts.agents import resume_session

        # Resolve session: resume existing or create new (handled by run_agent)
        session_id: str | None = None
        trajectory: Trajectory

        if session_store is not None:
            # Resume specific session
            if args.session is not None:
                if args.session == "":
                    # Interactive picker
                    session = await pick_session_async(session_store)
                    if session is None:
                        return 0
                    session_id = session.session_id
                else:
                    # Direct session ID
                    session_id = args.session

            # Continue latest session
            elif args.continue_session:
                session, err = await session_store.get_latest()
                if session:
                    session_id = session.session_id
                else:
                    print("No previous session found, starting new session")

        # Build trajectory: either from resumed session or fresh
        # Track forking info if config differs from parent
        parent_session_id: str | None = None
        branch_point: int | None = None

        if session_id:
            try:
                state = await resume_session(session_id, session_store, endpoint, environment)
                trajectory = state.actor.trajectory

                # Check if config differs from parent - if so, fork instead of continue
                parent_session, _ = await session_store.get(session_id)
                if parent_session:
                    # Compare endpoint config
                    current_env_type = type(environment).__name__ if environment else "none"
                    parent_env_type = (
                        parent_session.environment.type if parent_session.environment else "none"
                    )
                    parent_confirm_tools = (
                        parent_session.environment.config.get("confirm_tools", False)
                        if parent_session.environment
                        else False
                    )

                    config_differs = (
                        endpoint.model != parent_session.endpoint.model
                        or endpoint.provider != parent_session.endpoint.provider
                        or current_env_type != parent_env_type
                        or args.confirm_tools != parent_confirm_tools
                    )

                    if config_differs:
                        # Fork: create child session instead of continuing parent
                        parent_session_id = session_id
                        branch_point = len(trajectory.messages)
                        session_id = None  # Will create new session in run_agent
                        print(f"Forking from session: {parent_session_id}")
                        print(f"  Config changed: model={endpoint.model}, env={current_env_type}")
                        print(f"  Branch point: {branch_point} messages")
                    else:
                        print(f"Resuming session: {parent_session.session_id}")
                        print(f"  {len(trajectory.messages)} messages")
                else:
                    print(f"Resuming session: {session_id}")
                    print(f"  {len(trajectory.messages)} messages")
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

            # Ensure system prompt is present
            if not trajectory.messages or trajectory.messages[0].role != "system":
                trajectory = Trajectory(
                    messages=[Message(role="system", content=system_prompt)]
                    + list(trajectory.messages)
                )
        else:
            # Fresh session - just system prompt, run_agent will create session
            trajectory = Trajectory(messages=[Message(role="system", content=system_prompt)])

        # Check for stdin input (piped or redirected)
        initial_prompt: str | None = None
        if not sys.stdin.isatty():
            initial_prompt = sys.stdin.read().strip() or None

        # Non-interactive print mode - use frontends
        if args.print_mode is not None:
            query = args.print_mode
            if query == "-":
                # Read from stdin (already read above if piped)
                query = initial_prompt or ""
                if not query:
                    print("Error: no input from stdin", file=sys.stderr)
                    return 1

            from rollouts.frontends import JsonFrontend, NoneFrontend, run_interactive

            if args.stream_json:
                frontend = JsonFrontend(include_thinking=True)
                # Set tools for init event
                if environment:
                    frontend.set_tools([t.function.name for t in environment.get_tools()])
            else:
                frontend = NoneFrontend(show_tool_calls=True, show_thinking=False)

            try:
                await run_interactive(
                    trajectory,
                    endpoint,
                    frontend=frontend,
                    environment=environment,
                    max_turns=args.max_turns,
                    session_store=session_store,
                    session_id=session_id,
                    initial_prompt=query,
                    single_turn=True,
                )
                return 0
            except KeyboardInterrupt:
                return 0
            except Exception as e:
                if args.stream_json:
                    import json
                    print(json.dumps({"type": "error", "error": str(e)}), flush=True)
                else:
                    print(f"\nError: {e}", file=sys.stderr)
                return 1

        # Interactive mode - select frontend
        if args.frontend == "none":
            # Simple stdout frontend
            from rollouts.frontends import NoneFrontend, run_interactive

            frontend = NoneFrontend(show_tool_calls=True, show_thinking=True)
            try:
                await run_interactive(
                    trajectory,
                    endpoint,
                    frontend=frontend,
                    environment=environment,
                    max_turns=args.max_turns,
                    session_store=session_store,
                    session_id=session_id,
                    parent_session_id=parent_session_id,
                    branch_point=branch_point,
                    confirm_tools=args.confirm_tools,
                    initial_prompt=initial_prompt,
                )
                return 0
            except KeyboardInterrupt:
                print("\n\n✅ Agent stopped")
                return 0

        elif args.frontend == "textual":
            # Textual-based rich TUI (coming soon)
            print(
                "Textual frontend not yet implemented. Use --frontend=tui for now.", file=sys.stderr
            )
            return 1

        else:
            # Default: Python TUI (existing implementation)
            from rollouts.frontends.tui.interactive_agent import run_interactive_agent

            try:
                await run_interactive_agent(
                    trajectory,
                    endpoint,
                    environment,
                    args.max_turns,
                    session_store,
                    session_id,
                    args.theme,
                    args.debug,
                    args.debug_layout,
                    parent_session_id,
                    branch_point,
                    args.confirm_tools,
                    initial_prompt,
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


if __name__ == "__main__":
    sys.exit(main())
