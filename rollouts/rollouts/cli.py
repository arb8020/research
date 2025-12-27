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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import trio

from rollouts import AgentSession
from rollouts.dtypes import Endpoint, Message, Trajectory
from rollouts.environments import (
    CalculatorEnvironment,
    GitWorktreeEnvironment,
    LocalFilesystemEnvironment,
)
from rollouts.store import FileSessionStore

if TYPE_CHECKING:
    from rollouts.environments.base import Environment
    from rollouts.import_cc import ClaudeCodeSession

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
    "repl": """You are an assistant with access to a REPL environment for processing large contexts.

The input context is stored in a Python variable called `context`. You explore it programmatically.

Available tools:
- repl: Execute Python code (context variable available, plus re module)
- llm_query: Query a sub-LLM for semantic tasks on text chunks
- final_answer: Submit your final answer

Strategy:
1. Peek first: context[:1000], len(context)
2. Search: re.findall(pattern, context), list comprehensions
3. Chunk for semantics: llm_query("Classify: " + chunk)
4. Answer: final_answer(your_result)""",
    "repl_blocks": """You are an assistant with access to a REPL environment for processing large contexts.

The input context is stored in a Python variable called `context`. You explore it programmatically.

Write code in ```repl or ```python blocks to execute. Use FINAL(answer) when done.

Example:
```repl
print(len(context))
matches = [l for l in context.split('\\n') if 'keyword' in l]
print(matches[:5])
```

When you have the answer: FINAL(42)""",
}

# Default values for detecting if user overrode args
PARSER_DEFAULTS = {
    "model": "anthropic/claude-opus-4-5-20251101",
    "env": "none",
    "thinking": "enabled",
}


@dataclass
class CLIConfig:
    """Parsed CLI configuration."""

    # Model/endpoint
    model: str = PARSER_DEFAULTS["model"]
    api_base: str | None = None
    api_key: str | None = None
    thinking: str = PARSER_DEFAULTS["thinking"]

    # Environment
    env: str = PARSER_DEFAULTS["env"]
    tools: str | None = None
    cwd: str | None = None
    confirm_tools: bool = False
    context: str | None = None  # For REPL environments

    # Session
    continue_session: bool = False
    session: str | None = None
    no_session: bool = False

    # Interaction
    print_mode: str | None = None
    stream_json: bool = False
    quiet: bool = False
    initial_prompt: str | None = None

    # Frontend
    frontend: str = "tui"
    theme: str = "minimal"
    debug: bool = False
    debug_layout: bool = False

    # Preset
    preset: str | None = None
    system_prompt: str | None = None

    # Commands (mutually exclusive actions)
    list_presets: bool = False
    login_claude: bool = False
    logout_claude: bool = False
    export_md: str | None = None
    export_html: str | None = None
    handoff: str | None = None
    slice: str | None = None
    slice_goal: str | None = None
    import_cc: str | None = None
    doctor: bool = False
    trim: int | None = None
    fix: bool = False

    # Derived (populated after arg processing)
    working_dir: Path = field(default_factory=Path.cwd)
    endpoint: Endpoint | None = None
    environment: Environment | None = None
    session_store: FileSessionStore | None = None
    trajectory: Trajectory | None = None


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
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
        default=PARSER_DEFAULTS["model"],
        help=f'Model in "provider/model" format. Default: {PARSER_DEFAULTS["model"]}',
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
        default=PARSER_DEFAULTS["env"],
        help=(
            "Environment with tools. Options: none, calculator, coding, git, repl, repl_blocks. "
            "Compose with '+': coding+repl, git+repl (default: none)"
        ),
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
        "--confirm-tools",
        action="store_true",
        help="Require confirmation before executing tools",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Context string for REPL environments (alternative to --context-file)",
    )
    parser.add_argument(
        "--context-file",
        type=str,
        default=None,
        help="Path to file containing context for REPL environments",
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
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print final assistant response (for print mode). Hides tool calls and intermediate output.",
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
        default=PARSER_DEFAULTS["thinking"],
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
    parser.add_argument(
        "--slice",
        type=str,
        metavar="SPEC",
        help=(
            "Slice session messages. SPEC format: '0:4, summarize:5:15, 16:, inject:\"msg\"'. "
            "Creates new child session. Outputs session ID."
        ),
    )
    parser.add_argument(
        "--slice-goal",
        type=str,
        metavar="GOAL",
        help="Focus summaries in --slice on this goal (optional)",
    )

    # Import from Claude Code
    parser.add_argument(
        "--import-cc",
        type=str,
        nargs="?",
        const="",
        default=None,
        metavar="SESSION_ID",
        help="Import session from Claude Code. Use --import-cc to list/pick, --import-cc <id> for specific session",
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
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix detected issues (duplicate tool results, etc.)",
    )

    return parser


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

    Note: Returns str instead of Provider literal since user input is dynamic.
    Callers should validate the provider against known providers if needed.
    """
    if "/" not in model_str:
        raise ValueError(
            f'Model must be in "provider/model" format (e.g., "anthropic/claude-sonnet-4-5"). '
            f'Got: "{model_str}"'
        )

    provider, model = model_str.split("/", 1)
    return provider, model


def get_oauth_client() -> object:
    """Get OAuth client for Anthropic. Lazy import to avoid TUI dependencies."""
    from rollouts.frontends.tui.oauth import get_oauth_client as _get_oauth_client

    return _get_oauth_client()


def create_endpoint(
    model_str: str,
    api_base: str | None = None,
    api_key: str | None = None,
    thinking: str = "enabled",
    quiet: bool = False,
) -> Endpoint:
    """Create endpoint from CLI arguments."""
    import os

    from rollouts.models import get_model

    # Parse model string
    provider, model = parse_model_string(model_str)

    # Check model capabilities if thinking is enabled
    if thinking == "enabled":
        from typing import cast

        from rollouts.models import Provider

        model_metadata = get_model(cast(Provider, provider), model)
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

    # Explicit auth flow - no silent fallbacks to avoid surprise billing
    # 1. --api-key flag → use that (user's explicit choice)
    # 2. No flag + OAuth exists → use OAuth
    # 3. No flag + no OAuth → ERROR (prompt to login)
    oauth_token = ""
    if provider == "anthropic":
        if api_key is not None:
            # User explicitly passed --api-key, use it
            print("🔑 Using API key (explicit)", file=sys.stderr)
        else:
            # Try OAuth - never silently fall back to ANTHROPIC_API_KEY
            client = get_oauth_client()
            tokens = client.tokens
            if tokens:
                if tokens.is_expired():
                    try:
                        tokens = trio.run(client.refresh_tokens)
                        oauth_token = tokens.access_token
                        print("🔐 OAuth token refreshed", file=sys.stderr)
                    except Exception as e:
                        print(f"❌ OAuth token expired and refresh failed: {e}", file=sys.stderr)
                        print(
                            "   Run `rollouts login` to re-authenticate, or use --api-key",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                else:
                    oauth_token = tokens.access_token
                if not quiet:
                    print("🔐 Using OAuth authentication (Claude Pro/Max)", file=sys.stderr)
            else:
                # No OAuth tokens - require explicit action
                print("❌ No authentication configured for Anthropic", file=sys.stderr)
                print(
                    "   Run `rollouts login` to authenticate with Claude Pro/Max", file=sys.stderr
                )
                print("   Or use --api-key to use API billing", file=sys.stderr)
                sys.exit(1)
            api_key = ""  # Ensure no API key when using OAuth

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


# =============================================================================
# Command handlers - each handles a specific CLI subcommand
# =============================================================================


def cmd_list_presets() -> int:
    """Handle --list-presets command."""
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


def cmd_oauth(login: bool) -> int:
    """Handle --login-claude and --logout-claude commands."""
    from rollouts.frontends.tui.oauth import OAuthError, logout
    from rollouts.frontends.tui.oauth import login as do_login

    if not login:
        logout()
        return 0

    async def oauth_action() -> int:
        try:
            await do_login()
        except OAuthError as e:
            print(f"❌ OAuth error: {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\n⚠️  Login cancelled")
            return 1
        else:
            return 0

    return trio.run(oauth_action)


def cmd_export(
    config: CLIConfig,
    session_store: FileSessionStore,
) -> int:
    """Handle --export-md and --export-html commands."""
    from rollouts.export import session_to_html, session_to_markdown

    async def export_action() -> int:
        if config.session is not None and config.session != "":
            session, err = await session_store.get(config.session)
            if err or session is None:
                print(f"Error loading session: {err}", file=sys.stderr)
                return 1
        elif config.session == "" or config.continue_session:
            if config.continue_session:
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

        if config.export_md is not None:
            output = session_to_markdown(session)
            export_path = config.export_md
        else:
            output = session_to_html(session)
            export_path = config.export_html

        # export_path should never be None here since we check export_md/export_html before calling
        assert export_path is not None, "export_path should be set by export_md or export_html"

        if export_path == "":
            print(output)
        else:
            Path(export_path).write_text(output)
            print(f"Exported to {export_path}")

        return 0

    return trio.run(export_action)


def cmd_doctor(config: CLIConfig, session_store: FileSessionStore) -> int:
    """Handle --doctor, --trim, and --fix commands."""

    async def doctor_action() -> int:
        # Determine which session to doctor
        target_session_id: str | None = None
        if config.session and config.session != "":
            target_session_id = config.session
        elif config.continue_session:
            target_session_id = session_store.get_latest_id_sync()
        else:
            target_session_id = session_store.get_latest_id_sync()

        if not target_session_id:
            print("No session found. Use --session <id> to specify.", file=sys.stderr)
            return 1

        session, err = await session_store.get(target_session_id)
        if err or not session:
            print(f"Error loading session: {err}", file=sys.stderr)
            return 1

        # Calculate stats
        total_chars = sum(len(str(msg.content)) for msg in session.messages)
        estimated_tokens = total_chars // 4

        print(f"Session: {session.session_id}")
        print(f"  Messages: {len(session.messages)}")
        print(f"  Total chars: {total_chars:,}")
        print(f"  Est. tokens: {estimated_tokens:,}")
        print(f"  Status: {session.status.value}")
        if session.parent_id:
            print(f"  Parent: {session.parent_id}")

        # Diagnose issues
        issues = _diagnose_session_issues(session)

        if issues:
            print(f"\n⚠️  Found {len(issues)} issue(s):")
            for issue_type, desc, _ in issues:
                print(f"  [{issue_type}] {desc}")

        # Auto-fix if requested
        if config.fix and issues:
            return await _fix_session_issues(session, issues, session_store)

        # Show last few messages summary
        if session.messages:
            print("\nLast 5 messages:")
            for msg in session.messages[-5:]:
                content_preview = str(msg.content)[:80].replace("\n", " ")
                content_len = len(str(msg.content))
                print(f"  [{msg.role}] {content_preview}... ({content_len:,} chars)")

        # Trim if requested
        if config.trim is not None:
            return await _trim_session(session, config.trim, session_store)

        return 0

    return trio.run(doctor_action)


def _diagnose_session_issues(
    session: AgentSession,
) -> list[tuple[str, str, list[int]]]:
    """Diagnose issues in a session. Returns list of (issue_type, description, affected_indices)."""
    issues: list[tuple[str, str, list[int]]] = []

    # Check for duplicate tool results
    tool_result_ids: dict[str, list[int]] = {}
    for i, msg in enumerate(session.messages):
        if msg.role == "tool" and msg.tool_call_id:
            if msg.tool_call_id not in tool_result_ids:
                tool_result_ids[msg.tool_call_id] = []
            tool_result_ids[msg.tool_call_id].append(i)

    duplicate_results = {k: v for k, v in tool_result_ids.items() if len(v) > 1}
    if duplicate_results:
        for tool_id, indices in duplicate_results.items():
            issues.append((
                "duplicate_tool_result",
                f"Tool result '{tool_id[:20]}...' appears {len(indices)} times at messages {indices}",
                indices[1:],
            ))

    # Check for orphaned tool results
    tool_call_ids: set[str] = set()
    for msg in session.messages:
        if msg.role == "assistant" and isinstance(msg.content, list):
            for block in msg.content:
                # Handle both dict (raw JSON) and ToolCallContent (loaded via dacite)
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    tool_call_ids.add(block.get("id", ""))
                elif hasattr(block, "type") and block.type == "toolCall":
                    tool_call_ids.add(getattr(block, "id", ""))

    for i, msg in enumerate(session.messages):
        if msg.role == "tool" and msg.tool_call_id:
            if msg.tool_call_id not in tool_call_ids:
                issues.append((
                    "orphaned_tool_result",
                    f"Tool result '{msg.tool_call_id[:20]}...' at message {i} has no matching tool_use",
                    [i],
                ))

    # Check for oversized messages
    for i, msg in enumerate(session.messages):
        content_len = len(str(msg.content))
        if content_len > 100_000:
            issues.append((
                "oversized_message",
                f"Message {i} ({msg.role}) is {content_len:,} chars ({content_len // 4:,} est. tokens)",
                [i],
            ))

    return issues


async def _fix_session_issues(
    session: AgentSession,
    issues: list[tuple[str, str, list[int]]],
    session_store: FileSessionStore,
) -> int:
    """Fix auto-fixable issues by creating a new session."""
    indices_to_remove: set[int] = set()
    for issue_type, _, affected in issues:
        if issue_type in ("duplicate_tool_result", "orphaned_tool_result"):
            indices_to_remove.update(affected)

    if not indices_to_remove:
        print("\nNo auto-fixable issues found. Use --trim N for oversized messages.")
        return 0

    fixed_messages = [msg for i, msg in enumerate(session.messages) if i not in indices_to_remove]

    new_session = await session_store.create(
        endpoint=session.endpoint,
        environment=session.environment,
        parent_id=session.session_id,
        branch_point=len(fixed_messages),
        tags={"doctor": "fixed", "removed_indices": str(sorted(indices_to_remove))},
    )

    for msg in fixed_messages:
        await session_store.append_message(new_session.session_id, msg)

    print(f"\nCreated fixed session: {new_session.session_id}")
    print(f"  Removed {len(indices_to_remove)} message(s) at indices: {sorted(indices_to_remove)}")
    print(f"  Parent: {session.session_id}")
    print(f"\nResume with: rollouts --session {new_session.session_id}")
    return 0


async def _trim_session(
    session: AgentSession,
    trim_count: int,
    session_store: FileSessionStore,
) -> int:
    """Trim messages from session by creating a new session."""
    if trim_count <= 0:
        print("\n--trim must be a positive integer", file=sys.stderr)
        return 1
    if trim_count >= len(session.messages):
        print(
            f"\nCannot trim {trim_count} messages from session with {len(session.messages)} messages",
            file=sys.stderr,
        )
        return 1

    trimmed_messages = session.messages[:-trim_count]
    branch_point = len(trimmed_messages)

    new_session = await session_store.create(
        endpoint=session.endpoint,
        environment=session.environment,
        parent_id=session.session_id,
        branch_point=branch_point,
        tags={"doctor": "trimmed", "trimmed_count": str(trim_count)},
    )

    for msg in trimmed_messages:
        await session_store.append_message(new_session.session_id, msg)

    print(f"\nCreated fixed session: {new_session.session_id}")
    print(f"  Trimmed {trim_count} messages (kept {len(trimmed_messages)})")
    print(f"  Parent: {session.session_id}")
    print(f"\nResume with: rollouts --session {new_session.session_id}")
    return 0


def cmd_handoff(config: CLIConfig, session_store: FileSessionStore) -> int:
    """Handle --handoff command."""
    from rollouts.export import run_handoff_command

    async def handoff_action() -> int:
        if config.session is None:
            print("Error: --handoff requires -s <session_id>", file=sys.stderr)
            return 1

        if config.session == "":
            session = await pick_session_async(session_store)
            if session is None:
                return 0
        else:
            session, err = await session_store.get(config.session)
            if err or session is None:
                print(f"Error loading session: {err}", file=sys.stderr)
                return 1

        assert config.endpoint is not None
        assert config.handoff is not None
        handoff_md, err = await run_handoff_command(session, config.endpoint, config.handoff)
        if err:
            print(f"Error: {err}", file=sys.stderr)
            return 1
        print(handoff_md)
        return 0

    return trio.run(handoff_action)


def cmd_slice(config: CLIConfig, session_store: FileSessionStore) -> int:
    """Handle --slice command.

    Output (stderr):
        Slicing: <source_id> (N messages, ~M tokens)
        Spec: <slice_spec>
        Created: <child_id> (N messages, ~M tokens)
        Reduction: X% fewer tokens

    Output (stdout):
        <child_session_id>
    """
    from rollouts.slice import run_slice_command

    def estimate_tokens(messages: list) -> int:
        """Rough token estimate: chars / 4."""
        total_chars = sum(
            len(m.content) if isinstance(m.content, str) else len(str(m.content)) for m in messages
        )
        return total_chars // 4

    async def slice_action() -> int:
        if config.session is None:
            print("Error: --slice requires -s <session_id>", file=sys.stderr)
            return 1

        if config.session == "":
            session = await pick_session_async(session_store)
            if session is None:
                return 0
        else:
            session, err = await session_store.get(config.session)
            if err or session is None:
                print(f"Error loading session: {err}", file=sys.stderr)
                return 1

        assert config.endpoint is not None
        assert config.slice is not None

        # Handle "count" command specially - just show message count
        if config.slice.strip().lower() == "count":
            print(len(session.messages))
            return 0

        source_tokens = estimate_tokens(session.messages)
        print(
            f"Slicing: {session.session_id} ({len(session.messages)} messages, ~{source_tokens:,} tokens)",
            file=sys.stderr,
        )
        print(f"Spec: {config.slice}", file=sys.stderr)

        child, err = await run_slice_command(
            session=session,
            spec=config.slice,
            endpoint=config.endpoint,
            session_store=session_store,
            summarize_goal=config.slice_goal,
        )

        if err:
            print(f"Error: {err}", file=sys.stderr)
            return 1

        assert child is not None

        # Load child messages to get accurate count
        child_full, _ = await session_store.get(child.session_id)
        if child_full:
            child_tokens = estimate_tokens(child_full.messages)
            reduction = (1 - child_tokens / source_tokens) * 100 if source_tokens > 0 else 0
            print(
                f"Created: {child.session_id} ({len(child_full.messages)} messages, ~{child_tokens:,} tokens)",
                file=sys.stderr,
            )
            if reduction > 0:
                print(f"Reduction: {reduction:.0f}% fewer tokens", file=sys.stderr)
        else:
            print(f"Created: {child.session_id}", file=sys.stderr)

        # Print session ID to stdout for piping
        print(child.session_id)
        return 0

    return trio.run(slice_action)


def cmd_import_cc(config: CLIConfig, session_store: FileSessionStore) -> int:
    """Handle --import-cc command."""
    from rollouts.import_cc import (
        import_claude_code_session,
        list_claude_code_sessions,
    )

    async def import_action() -> int:
        # If session ID provided, find and import it directly
        if config.import_cc and config.import_cc != "":
            # Find session by ID (could be partial match)
            sessions = list_claude_code_sessions(limit=100)
            matches = [s for s in sessions if config.import_cc in s.session_id]

            if not matches:
                print(f"No Claude Code session found matching: {config.import_cc}", file=sys.stderr)
                return 1

            if len(matches) > 1:
                print(f"Multiple sessions match '{config.import_cc}':", file=sys.stderr)
                for s in matches[:5]:
                    print(f"  {s.session_id} ({s.project_path.name})", file=sys.stderr)
                print("Please provide a more specific ID", file=sys.stderr)
                return 1

            session = matches[0]
        else:
            # Interactive picker
            session = await pick_cc_session_async()
            if session is None:
                return 0

        # Import the session
        print(f"Importing: {session.session_id}", file=sys.stderr)
        print(f"  Project: {session.project_path}", file=sys.stderr)
        print(f"  Messages: {session.message_count}", file=sys.stderr)
        if session.model:
            print(f"  Model: {session.model}", file=sys.stderr)

        new_id, err = await import_claude_code_session(session, session_store)
        if err:
            print(f"Error: {err}", file=sys.stderr)
            return 1

        print(f"\nImported as: {new_id}", file=sys.stderr)
        print(f"Resume with: rollouts -s {new_id}")
        return 0

    return trio.run(import_action)


async def pick_cc_session_async() -> ClaudeCodeSession | None:
    """Interactive Claude Code session picker."""
    from rollouts.import_cc import list_claude_code_sessions

    sessions = list_claude_code_sessions(limit=20)

    if not sessions:
        print("No Claude Code sessions found in ~/.claude/projects/")
        return None

    print("\nClaude Code sessions:\n")
    for i, session in enumerate(sessions):
        time_ago = format_time_ago(session.last_modified.isoformat())
        model_short = session.model.split("-")[0] if session.model else "?"
        project_name = session.project_path.name
        print(
            f"  [{i + 1:2}] {time_ago:>10}  {session.message_count:>4} msgs  "
            f"{model_short:<10}  {project_name}/{session.session_id[:8]}"
        )

    print("\n  [0] Cancel")
    print()

    while True:
        try:
            choice = input("Select session to import: ").strip()
            if not choice:
                continue
            num = int(choice)
            if num == 0:
                return None
            if 1 <= num <= len(sessions):
                return sessions[num - 1]
            print(f"Please enter 0-{len(sessions)}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            print()
            return None


# =============================================================================
# Config loading - preset and session config merging
# =============================================================================


def apply_preset(config: CLIConfig) -> bool:
    """Apply preset configuration if specified. Returns False on error."""
    if not config.preset:
        return True

    from rollouts.agent_presets import load_preset

    try:
        preset = load_preset(config.preset)
    except Exception as e:
        print(f"Error loading preset '{config.preset}': {e}", file=sys.stderr)
        return False

    # Model: only override if still at default
    if config.model == PARSER_DEFAULTS["model"]:
        config.model = preset.model

    # Env: only override if still at default
    if config.env == PARSER_DEFAULTS["env"]:
        config.env = preset.env

    # System prompt: only override if not explicitly set
    if config.system_prompt is None:
        config.system_prompt = preset.system_prompt

    # Thinking: apply preset value
    if preset.thinking:
        config.thinking = preset.thinking

    # Working dir: apply preset if available and not set
    if preset.working_dir and config.cwd is None:
        config.cwd = str(preset.working_dir)

    return True


def apply_session_config(config: CLIConfig) -> bool:
    """Apply session configuration if resuming. Returns False on error."""
    session_id_for_config: str | None = None
    if config.session and config.session != "":
        session_id_for_config = config.session
    elif config.continue_session:
        session_store = FileSessionStore()
        session_id_for_config = session_store.get_latest_id_sync()

    if not session_id_for_config:
        return True

    session_store = FileSessionStore()
    session_config, err = session_store.get_config_sync(session_id_for_config)
    if err:
        print(f"Error loading session config: {err}", file=sys.stderr)
        return False

    if not session_config:
        return True

    # Model: inherit from session if not explicitly set
    if config.model == PARSER_DEFAULTS["model"]:
        endpoint_config = session_config.get("endpoint", {})
        if endpoint_config.get("model"):
            provider = endpoint_config.get("provider", "anthropic")
            config.model = f"{provider}/{endpoint_config['model']}"

    # Environment: inherit from session if not explicitly set
    if config.env == PARSER_DEFAULTS["env"]:
        env_config = session_config.get("environment", {})
        env_type = env_config.get("type", "")
        env_map = {
            "CalculatorEnvironment": "calculator",
            "LocalFilesystemEnvironment": "coding",
            "GitWorktreeEnvironment": "git",
        }
        if env_type in env_map:
            config.env = env_map[env_type]

    # Thinking: inherit from session if not explicitly set
    if config.thinking == PARSER_DEFAULTS["thinking"]:
        endpoint_config = session_config.get("endpoint", {})
        if endpoint_config.get("thinking") is False:
            config.thinking = "disabled"

    # confirm_tools: inherit from session if not explicitly set
    if not config.confirm_tools:
        env_config = session_config.get("environment", {})
        if env_config.get("config", {}).get("confirm_tools"):
            config.confirm_tools = True

    return True


def create_environment(config: CLIConfig) -> tuple[Environment | None, bool]:
    """Create environment from config. Returns (environment, success)."""
    if config.env == "calculator":
        return CalculatorEnvironment(), True

    if config.env == "coding":
        from rollouts.environments.coding import TOOL_PRESETS

        tools = config.tools or "full"
        if tools not in TOOL_PRESETS:
            print(
                f"Unknown tool preset: {tools}. Available: {', '.join(TOOL_PRESETS.keys())}",
                file=sys.stderr,
            )
            return None, False
        return LocalFilesystemEnvironment(working_dir=config.working_dir, tools=tools), True

    if config.env == "git":
        return GitWorktreeEnvironment(working_dir=config.working_dir), True

    if config.env == "repl":
        from rollouts.environments.repl import REPLEnvironment

        # Context must be provided via --context or --context-file
        context = config.context or ""
        if not context:
            print(
                "Warning: No context provided for REPL environment. "
                "Use --context or --context-file to provide input.",
                file=sys.stderr,
            )
        return REPLEnvironment(context=context, sub_endpoint=config.endpoint), True

    if config.env == "repl_blocks":
        from rollouts.environments.repl import MessageParsingREPLEnvironment

        context = config.context or ""
        if not context:
            print(
                "Warning: No context provided for REPL environment. "
                "Use --context or --context-file to provide input.",
                file=sys.stderr,
            )
        return MessageParsingREPLEnvironment(context=context, sub_endpoint=config.endpoint), True

    # Composed environments: coding+repl, git+repl, etc.
    # TODO: Consider auto-composition when --context is provided with coding/git envs
    # For now, explicit composition via comma-separated env names
    if "+" in config.env:
        from rollouts.environments.compose import compose
        from rollouts.environments.repl import REPLEnvironment

        env_names = config.env.split("+")
        environments = []

        for env_name in env_names:
            if env_name == "coding":
                environments.append(
                    LocalFilesystemEnvironment(
                        working_dir=config.working_dir, tools=config.tools or "full"
                    )
                )
            elif env_name == "git":
                environments.append(GitWorktreeEnvironment(working_dir=config.working_dir))
            elif env_name == "repl":
                context = config.context or ""
                if not context:
                    print(
                        "Warning: No context provided for REPL environment. "
                        "Use --context or --context-file to provide input.",
                        file=sys.stderr,
                    )
                environments.append(REPLEnvironment(context=context, sub_endpoint=config.endpoint))
            elif env_name == "calculator":
                environments.append(CalculatorEnvironment())
            else:
                print(f"Unknown environment in composition: {env_name}", file=sys.stderr)
                return None, False

        return compose(*environments), True

    return None, True


# =============================================================================
# Main agent runner
# =============================================================================


async def run_agent(config: CLIConfig) -> int:
    """Run the interactive agent."""
    from rollouts.agents import resume_session

    session_store = config.session_store
    session_id: str | None = None
    trajectory: Trajectory

    # Resolve session
    if session_store is not None:
        if config.session is not None:
            if config.session == "":
                session = await pick_session_async(session_store)
                if session is None:
                    return 0
                session_id = session.session_id
            else:
                session_id = config.session
        elif config.continue_session:
            session, _err = await session_store.get_latest()
            if session:
                session_id = session.session_id
            else:
                print("No previous session found, starting new session")

    # Build trajectory
    parent_session_id: str | None = None
    branch_point: int | None = None

    # Build system prompt - use dynamic builder if we have an environment with tools
    if config.system_prompt:
        # User provided explicit prompt - use as-is
        system_prompt = config.system_prompt
    elif config.environment:
        # Build dynamic prompt with actual tools
        from rollouts.prompt import build_system_prompt

        # Get environment-provided system prompt if available
        env_system_prompt = None
        if hasattr(config.environment, "get_system_prompt"):
            env_system_prompt = config.environment.get_system_prompt()

        system_prompt = build_system_prompt(
            env_name=config.env,
            tools=config.environment.get_tools(),
            cwd=config.working_dir,
            env_system_prompt=env_system_prompt,
        )
    else:
        # Fallback to static prompts
        system_prompt = SYSTEM_PROMPTS.get(config.env, SYSTEM_PROMPTS["none"])

    if session_id and session_store:
        try:
            assert config.endpoint is not None
            state = await resume_session(
                session_id, session_store, config.endpoint, config.environment
            )
            trajectory = state.actor.trajectory

            parent_session, _ = await session_store.get(session_id)
            if parent_session:
                current_env_type = (
                    type(config.environment).__name__ if config.environment else "none"
                )
                parent_env_type = (
                    parent_session.environment.type if parent_session.environment else "none"
                )
                parent_confirm_tools = (
                    parent_session.environment.config.get("confirm_tools", False)
                    if parent_session.environment
                    else False
                )

                config_differs = (
                    config.endpoint.model != parent_session.endpoint.model
                    or config.endpoint.provider != parent_session.endpoint.provider
                    or current_env_type != parent_env_type
                    or config.confirm_tools != parent_confirm_tools
                )

                if config_differs:
                    parent_session_id = session_id
                    branch_point = len(trajectory.messages)
                    session_id = None
                    print(f"Forking from session: {parent_session_id}")
                    print(
                        f"  Config changed: model={config.endpoint.model}, env={current_env_type}"
                    )
                    print(f"  Branch point: {branch_point} messages")
                else:
                    print(f"Resuming session: {parent_session.session_id}")
                    # TODO: Token counting is not accurate when resuming - should count tokens
                    # from resumed messages, not just new messages in this session
                    print(f"  {len(trajectory.messages)} messages")
            else:
                print(f"Resuming session: {session_id}")
                print(f"  {len(trajectory.messages)} messages")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        if not trajectory.messages or trajectory.messages[0].role != "system":
            trajectory = Trajectory(
                messages=[Message(role="system", content=system_prompt)] + list(trajectory.messages)
            )
    else:
        trajectory = Trajectory(messages=[Message(role="system", content=system_prompt)])

    # Check for stdin input
    initial_prompt = config.initial_prompt
    if initial_prompt is None and not sys.stdin.isatty():
        initial_prompt = sys.stdin.read().strip() or None

    # Non-interactive print mode
    if config.print_mode is not None:
        return await _run_print_mode(config, trajectory, session_id, initial_prompt)

    # Interactive mode
    return await _run_interactive_mode(
        config, trajectory, session_id, parent_session_id, branch_point, initial_prompt
    )


async def _run_print_mode(
    config: CLIConfig,
    trajectory: Trajectory,
    session_id: str | None,
    initial_prompt: str | None,
) -> int:
    """Run in non-interactive print mode."""
    assert config.endpoint is not None, "endpoint must be set for print mode"

    from rollouts.frontends import JsonFrontend, NoneFrontend, run_interactive

    query = config.print_mode
    if query == "-":
        query = initial_prompt or ""
        if not query:
            print("Error: no input from stdin", file=sys.stderr)
            return 1

    if config.stream_json:
        frontend = JsonFrontend(include_thinking=True)
        if config.environment:
            frontend.set_tools([t.function.name for t in config.environment.get_tools()])
    elif config.quiet:
        frontend = NoneFrontend(show_tool_calls=False, show_thinking=False)
    else:
        frontend = NoneFrontend(show_tool_calls=True, show_thinking=False)

    try:
        await run_interactive(
            trajectory,
            config.endpoint,
            frontend=frontend,
            environment=config.environment,
            session_store=config.session_store,
            session_id=session_id,
            initial_prompt=query,
            single_turn=True,
        )
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        if config.stream_json:
            import json

            print(json.dumps({"type": "error", "error": str(e)}), flush=True)
        else:
            print(f"\nError: {e}", file=sys.stderr)
        return 1
    else:
        return 0


async def _run_interactive_mode(
    config: CLIConfig,
    trajectory: Trajectory,
    session_id: str | None,
    parent_session_id: str | None,
    branch_point: int | None,
    initial_prompt: str | None,
) -> int:
    """Run in interactive mode with selected frontend."""
    assert config.endpoint is not None, "endpoint must be set for interactive mode"

    if config.frontend == "none":
        from rollouts.frontends import NoneFrontend, run_interactive

        frontend = NoneFrontend(show_tool_calls=True, show_thinking=True)
        try:
            await run_interactive(
                trajectory,
                config.endpoint,
                frontend=frontend,
                environment=config.environment,
                session_store=config.session_store,
                session_id=session_id,
                parent_session_id=parent_session_id,
                branch_point=branch_point,
                confirm_tools=config.confirm_tools,
                initial_prompt=initial_prompt,
            )
        except KeyboardInterrupt:
            print("\n\n✅ Agent stopped")
        return 0

    if config.frontend == "textual":
        print(
            "Textual frontend not yet implemented. Use --frontend=tui for now.",
            file=sys.stderr,
        )
        return 1

    # Default: Python TUI
    from rollouts.frontends.tui.interactive_agent import run_interactive_agent

    try:
        await run_interactive_agent(
            trajectory,
            config.endpoint,
            config.environment,
            config.session_store,
            session_id,
            config.theme,
            config.debug,
            config.debug_layout,
            parent_session_id,
            branch_point,
            config.confirm_tools,
            initial_prompt,
        )
    except KeyboardInterrupt:
        print("\n\n✅ Agent stopped")
    return 0


# =============================================================================
# Main entry point
# =============================================================================


def main() -> int:
    """Main CLI entry point - dispatcher for all CLI commands."""
    parser = create_parser()
    args = parser.parse_args()

    # Build config from parsed args
    # Handle context from --context or --context-file
    context = args.context
    if args.context_file:
        try:
            context = Path(args.context_file).read_text()
        except Exception as e:
            print(f"Error reading context file: {e}", file=sys.stderr)
            return 1

    config = CLIConfig(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        thinking=args.thinking,
        env=args.env,
        tools=args.tools,
        cwd=args.cwd,
        confirm_tools=args.confirm_tools,
        context=context,
        continue_session=args.continue_session,
        session=args.session,
        no_session=args.no_session,
        print_mode=args.print_mode,
        stream_json=args.stream_json,
        quiet=args.quiet,
        frontend=args.frontend,
        theme=args.theme,
        debug=args.debug,
        debug_layout=args.debug_layout,
        preset=args.preset,
        system_prompt=args.system_prompt,
        list_presets=args.list_presets,
        login_claude=args.login_claude,
        logout_claude=args.logout_claude,
        export_md=args.export_md,
        export_html=args.export_html,
        handoff=args.handoff,
        slice=args.slice,
        slice_goal=args.slice_goal,
        import_cc=args.import_cc,
        doctor=args.doctor,
        trim=args.trim,
        fix=args.fix,
    )

    # === Commands that don't need endpoint ===

    if config.list_presets:
        return cmd_list_presets()

    if config.login_claude or config.logout_claude:
        return cmd_oauth(login=config.login_claude)

    if config.export_md is not None or config.export_html is not None:
        return cmd_export(config, FileSessionStore())

    if config.doctor or config.trim is not None or config.fix:
        return cmd_doctor(config, FileSessionStore())

    if config.import_cc is not None:
        return cmd_import_cc(config, FileSessionStore())

    # === Commands requiring endpoint ===

    # Apply preset and session config
    if not apply_preset(config):
        return 1
    if not apply_session_config(config):
        return 1

    # Set working directory
    config.working_dir = Path(config.cwd) if config.cwd else Path.cwd()

    # Create endpoint
    try:
        config.endpoint = create_endpoint(
            config.model, config.api_base, config.api_key, config.thinking, config.quiet
        )
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    # Validate authentication
    if not config.endpoint.api_key and not config.endpoint.oauth_token:
        env_var = "OPENAI_API_KEY" if config.endpoint.provider == "openai" else "ANTHROPIC_API_KEY"
        print(
            f"❌ No API key found. Set {env_var}, use --api-key, or --login-claude",
            file=sys.stderr,
        )
        return 1

    # Handoff command (needs endpoint)
    if config.handoff:
        return cmd_handoff(config, FileSessionStore())

    # Slice command (needs endpoint for summarize)
    if config.slice:
        return cmd_slice(config, FileSessionStore())

    # Create environment
    environment, ok = create_environment(config)
    if not ok:
        return 1
    config.environment = environment

    # Set up session store
    config.session_store = FileSessionStore() if not config.no_session else None

    # Run the agent
    try:
        return trio.run(run_agent, config)
    except Exception as e:
        print(f"\n\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
