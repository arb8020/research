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

from .dtypes import AgentSession, Endpoint, Message, Trajectory
from .environments import (
    CalculatorEnvironment,
    GitWorktreeEnvironment,
    LocalFilesystemEnvironment,
)
from .store import FileSessionStore

if TYPE_CHECKING:
    from .environments.base import Environment

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

# Token and thinking budget defaults
DEFAULT_THINKING_BUDGET = 10000
MAX_TOKENS_WITH_THINKING = 16384
MAX_TOKENS_DEFAULT = 8192

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
    log_file: str | None = None

    # Driver/backend
    driver: str = "sdk"  # sdk, claude, codex, cursor
    cursor_api_key: str | None = None  # For cursor driver

    # Preset
    preset: str | None = None
    system_prompt: str | None = None

    # Template mode (-t)
    template: str | None = None
    template_args: dict[str, str] | None = None
    interactive: bool = False  # -i to attach TUI with template
    list_templates: bool = False
    _template_config: object | None = None  # Loaded TemplateConfig (internal)
    _bash_allowlist: list[str] | None = None  # From template (internal)

    # Model/driver picker
    pick: bool = False  # Interactive model/driver picker

    # Commands (mutually exclusive actions)
    list_models: bool = False
    sync_models: bool = False
    write_models: bool = False  # --write flag for --sync-models
    list_presets: bool = False
    login_claude: bool = False
    logout_claude: bool = False
    list_claude_profiles: bool = False
    set_default_profile: str | None = None
    profile: str | None = None
    export_md: str | None = None
    export_html: str | None = None
    handoff: str | None = None
    fast_handoff: bool = False  # Use fast single-call mode for handoff
    slice: str | None = None
    slice_goal: str | None = None
    doctor: bool = False
    trim: int | None = None
    fix: bool = False

    # Tmux-style session management
    send: tuple[str, str] | None = None  # (session_id, message)
    send_file: tuple[str, str] | None = None  # (session_id, file_path)
    attach: str | None = None  # session_id to attach
    status: str | None = None  # session_id or "" for list
    ls: bool = False
    ls_all: bool = False
    detached: bool = False

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
        "--cursor-api-key",
        type=str,
        default=None,
        help="Cursor API key for --driver cursor (can also use CURSOR_API_KEY env var)",
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
        choices=["tui", "none", "minimal", "textual"],
        default="tui",
        help="Frontend: tui (default Python TUI), none (stdout), minimal (OpenCode-style icons), textual (rich TUI)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        choices=["dark", "rounded", "minimal"],
        default="minimal",
        help="TUI theme (default: minimal)",
    )

    # Driver/backend options
    parser.add_argument(
        "--driver",
        type=str,
        choices=["sdk", "claude", "codex", "cursor"],
        default="sdk",
        help="Backend driver: sdk (default, direct API), claude (Claude Code CLI), codex (Codex CLI), cursor (Cursor Agent CLI)",
    )

    # Extended thinking (Anthropic)
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["enabled", "disabled"],
        default=PARSER_DEFAULTS["thinking"],
        help="Extended thinking for Anthropic models (default: enabled)",
    )

    # Model/driver picker
    parser.add_argument(
        "--pick",
        action="store_true",
        help="Interactive model/driver picker (choose between rollouts, claude-code, codex)",
    )

    # Model management
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models in registry, highlight missing from provider APIs",
    )
    parser.add_argument(
        "--sync-models",
        action="store_true",
        help="Fetch latest models from provider APIs/docs and update registry",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="With --sync-models: write changes to models.py on disk",
    )

    # Preset listing
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available agent presets and exit",
    )

    # Template mode
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        default=None,
        help="Run with a template (constrained agent). Defaults to detached mode. Use -i to attach TUI.",
    )
    parser.add_argument(
        "--args",
        type=str,
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Template variable (can be repeated). Example: --args corpus=./docs/ --args format=json",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Attach TUI when using -t template (default is detached/headless)",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available templates and exit",
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (sets LOG_LEVEL=DEBUG)",
    )
    parser.add_argument(
        "--debug-layout",
        action="store_true",
        help="Show TUI component boundaries",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Write logs to file (JSONL format, includes API requests at DEBUG level)",
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
    parser.add_argument(
        "--list-claude-profiles",
        action="store_true",
        help="List available Claude OAuth profiles",
    )
    parser.add_argument(
        "--set-default-profile",
        type=str,
        metavar="PROFILE",
        help="Set a profile as the default (copies to default.json)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Claude OAuth profile to use (default: 'default' or ROLLOUTS_PROFILE env var)",
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
        "--fast",
        action="store_true",
        help="Use fast single-call handoff (default uses agent mode for better context)",
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

    # Tmux-style session management
    parser.add_argument(
        "--send",
        type=str,
        nargs=2,
        metavar=("SESSION_ID", "MESSAGE"),
        help="Send message to waiting session and resume",
    )
    parser.add_argument(
        "--send-file",
        type=str,
        nargs=2,
        metavar=("SESSION_ID", "FILE"),
        help="Send file contents to waiting session and resume",
    )
    parser.add_argument(
        "--attach",
        type=str,
        metavar="SESSION_ID",
        help="Attach TUI to existing session",
    )
    parser.add_argument(
        "--status",
        type=str,
        nargs="?",
        const="",
        metavar="SESSION_ID",
        help="Show session status (list active if no ID)",
    )
    parser.add_argument(
        "--ls",
        action="store_true",
        help="List active sessions (running/waiting)",
    )
    parser.add_argument(
        "--ls-all",
        action="store_true",
        help="List all sessions including completed/failed",
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Run detached: exit when agent needs input instead of blocking",
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


# TODO: Remove OAuth code. OAuth was used to let Pro/Max users avoid API billing,
# but it added confusing auth precedence (OAuth > env var). Now SDK driver just uses
# API keys. OAuth code remains in frontends/tui/oauth.py and providers/anthropic.py
# (_get_fresh_oauth_token). Also remove --login-claude/--logout-claude flags and
# cmd_oauth function. Delete ~/.rollouts/oauth/ handling.
def get_oauth_client(profile: str = "default") -> object:
    """Get OAuth client for Anthropic. Lazy import to avoid TUI dependencies."""
    from .frontends.tui.oauth import get_oauth_client as _get_oauth_client

    return _get_oauth_client(profile)


def create_endpoint(
    model_str: str,
    api_base: str | None = None,
    api_key: str | None = None,
    thinking: str = "enabled",
    quiet: bool = False,
    profile: str = "default",
    driver: str = "sdk",
) -> Endpoint:
    """Create endpoint from CLI arguments."""

    from .models import get_model

    # Parse model string
    provider, model = parse_model_string(model_str)

    # Check model capabilities if thinking is enabled
    if thinking == "enabled":
        from typing import cast

        from .models import Provider

        model_metadata = get_model(cast(Provider, provider), model)
        if model_metadata is not None:
            if not model_metadata.reasoning:
                # Auto-disable thinking for models that don't support it
                print(
                    f"⚠️  Model '{model}' doesn't support extended thinking, disabling.",
                    file=sys.stderr,
                )
                thinking = "disabled"
        else:
            # Unknown model - warn but continue (might work)
            print(
                f"⚠️  Model '{model}' not in registry, thinking support unknown.",
                file=sys.stderr,
            )

    if api_base is None:
        # Try to get base_url from model metadata first
        from typing import cast

        from .models import Provider, get_model

        model_metadata = get_model(cast(Provider, provider), model)
        if model_metadata and model_metadata.base_url:
            api_base = model_metadata.base_url
        elif provider == "openai":
            api_base = "https://api.openai.com/v1"
        elif provider == "anthropic":
            api_base = "https://api.anthropic.com"
        else:
            api_base = "https://api.openai.com/v1"

    # Auth flow for SDK driver: simple API key lookup
    # 1. --api-key flag → use that (user's explicit choice)
    # 2. $ANTHROPIC_API_KEY env var → use that
    # 3. credentials.toml active profile → use that
    # OAuth is only for external drivers (claude, codex, cursor) which handle their own auth
    oauth_token = ""
    is_claude_code_api_key = False
    if api_key is not None and not quiet:
        print("🔑 Using API key (--api-key flag)", file=sys.stderr)

    if api_key is None:
        # Try credential store first, then env vars
        from .credentials import get_api_key

        api_key = get_api_key(provider) or ""

    # Configure extended thinking for Anthropic
    thinking_config = None
    if provider == "anthropic" and thinking == "enabled":
        thinking_config = {"type": "enabled", "budget_tokens": DEFAULT_THINKING_BUDGET}

    max_tokens = MAX_TOKENS_WITH_THINKING if thinking_config else MAX_TOKENS_DEFAULT

    return Endpoint(
        provider=provider,
        model=model,
        api_base=api_base,
        api_key=api_key,
        oauth_token=oauth_token,
        is_claude_code_api_key=is_claude_code_api_key,
        thinking=thinking_config,
        max_tokens=max_tokens,
    )


# =============================================================================
# Command handlers - each handles a specific CLI subcommand
# =============================================================================


def cmd_list_models() -> int:
    """Handle --list-models command."""
    import os

    from .models import MODELS, fetch_anthropic_models

    print("Models in registry:\n")

    for provider, models in MODELS.items():
        if not models:
            continue
        print(f"{provider}:")
        for model_id, meta in models.items():
            cost_str = f"${meta.cost.input:.2f}/${meta.cost.output:.2f}"
            print(
                f"  {model_id:<40} {cost_str:<12} {meta.context_window // 1000}K ctx, {meta.max_tokens // 1000}K out"
            )
        print()

    # Check for missing models from Anthropic API
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        print("Checking Anthropic API for new models...")
        try:
            import trio

            api_models = trio.run(fetch_anthropic_models, api_key)
            api_ids = {m["id"] for m in api_models}
            registry_ids = set(MODELS.get("anthropic", {}).keys())
            missing = api_ids - registry_ids

            if missing:
                print(f"\nMissing from registry ({len(missing)}):")
                for model_id in sorted(missing):
                    print(f"  + {model_id}")
                print("\nRun --sync-models to add them.")
            else:
                print("Registry is up to date with Anthropic API.")
        except Exception as e:
            print(f"Could not fetch from API: {e}")
    else:
        print("Set ANTHROPIC_API_KEY to check for new models from API.")

    return 0


def cmd_sync_models(write: bool = False) -> int:
    """Handle --sync-models command."""
    import os

    from .models import (
        ModelDiff,
        fetch_anthropic_docs,
        sync_anthropic_models,
        update_models_file,
        write_models_to_disk,
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY required for --sync-models")
        return 1

    print("Fetching models from Anthropic API and docs...")

    import trio

    async def do_sync() -> tuple[ModelDiff, dict[str, dict]]:
        diff = await sync_anthropic_models(api_key)
        docs = await fetch_anthropic_docs()
        return diff, docs

    diff, docs = trio.run(do_sync)

    if diff.missing:
        print(f"\nNew models ({len(diff.missing)}):")
        for model_id in diff.missing:
            if model_id in docs:
                d = docs[model_id]
                print(
                    f"  + {model_id}: ${d['input_cost']:.2f}/${d['output_cost']:.2f}, {d['context_window'] // 1000}K ctx"
                )
            else:
                print(f"  + {model_id}: (no metadata available)")

    if diff.updated:
        print(f"\nUpdated models ({len(diff.updated)}):")
        for model_id, changes in diff.updated.items():
            print(f"  ~ {model_id}:")
            for field, (old, new) in changes.items():
                print(f"      {field}: {old} -> {new}")

    if diff.extra:
        print(f"\nDeprecated/unlisted ({len(diff.extra)}):")
        for model_id in diff.extra:
            print(f"  - {model_id}")

    if not diff.missing and not diff.updated:
        print("\nRegistry is already up to date.")
        return 0

    # Apply updates in memory
    result = update_models_file(diff, docs)
    print(f"\n{result}")

    if write:
        # Write to disk
        models_path = write_models_to_disk()
        print(f"\nWrote updated models to {models_path}")
    else:
        print("\nNote: Changes applied to runtime registry only.")
        print("Use --sync-models --write to persist to models.py")

    return 0


def cmd_list_presets() -> int:
    """Handle --list-presets command."""
    from .agent_presets import list_presets

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


def cmd_list_templates() -> int:
    """Handle --list-templates command."""
    from .templates import list_templates

    templates = list_templates()
    if not templates:
        print("No templates found.")
        print("Templates are searched in:")
        print("  - ./rollouts/templates/ (project)")
        print("  - ~/.rollouts/templates/ (user)")
        print("  - rollouts/templates/ (built-in)")
        return 0

    print("Available templates:")
    for template_name in templates:
        print(f"  - {template_name}")

    print('\nUsage: rollouts -t <template> [--args key=value] "prompt"')
    print('Example: rollouts -t ask-docs "How do bank conflicts occur?"')
    return 0


def cmd_oauth(login: bool, profile: str) -> int:
    """Handle --login-claude and --logout-claude commands."""
    from .frontends.tui.oauth import OAuthError, logout
    from .frontends.tui.oauth import login as do_login

    if not login:
        logout(profile)
        return 0

    async def oauth_action() -> int:
        try:
            await do_login(profile)
        except OAuthError as e:
            print(f"❌ OAuth error: {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\n⚠️  Login cancelled")
            return 1
        else:
            return 0

    return trio.run(oauth_action)


def cmd_list_profiles() -> int:
    """Handle --list-claude-profiles command."""
    from .frontends.tui.oauth import list_profiles

    profiles = list_profiles()

    if not profiles:
        print("No Claude OAuth profiles found.")
        print("Run: rollouts --login-claude")
        return 0

    for profile in profiles:
        print(profile)

    return 0


def cmd_set_default_profile(profile: str) -> int:
    """Handle --set-default-profile command."""
    from .frontends.tui.oauth import set_default_profile

    _, err = set_default_profile(profile)
    if err:
        print(f"❌ {err}", file=sys.stderr)
        return 1

    print(f"✅ Set '{profile}' as default profile")
    return 0


def cmd_export(
    config: CLIConfig,
    session_store: FileSessionStore,
) -> int:
    """Handle --export-md and --export-html commands."""
    from .export import session_to_html, session_to_markdown

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
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    tool_call_ids.add(block.get("id", ""))

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
    from .export import run_handoff_command

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

        if config.fast_handoff:
            # Fast mode: single LLM call summarization
            print(f"Extracting context (fast) for: {config.handoff}", file=sys.stderr)
            print(
                f"From session: {session.session_id} ({len(session.messages)} messages)",
                file=sys.stderr,
            )
            print(file=sys.stderr)

            handoff_md, err = await run_handoff_command(session, config.endpoint, config.handoff)
        else:
            # Default: agent mode - use sub-agent with tools for better context
            from .environments.handoff import generate_handoff_context_agent

            print(f"Extracting context for: {config.handoff}", file=sys.stderr)
            print(
                f"From session: {session.session_id} ({len(session.messages)} messages)",
                file=sys.stderr,
            )
            print(file=sys.stderr)

            handoff_md, err = await generate_handoff_context_agent(
                session_id=session.session_id,
                goal=config.handoff,
                endpoint=config.endpoint,
                sessions_dir=session_store.base_dir,
                working_dir=Path.cwd(),
            )

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
    from .slice import run_slice_command

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


# =============================================================================
# Tmux-style session management commands
# =============================================================================


def cmd_ls(session_store: FileSessionStore, include_all: bool = False) -> int:
    """Handle --ls and --ls-all commands."""
    from .dtypes import SessionStatus

    async def ls_action() -> int:
        # Get all sessions
        sessions = await session_store.list(limit=100)

        if not include_all:
            # Filter to active only (pending, waiting)
            sessions = [
                s for s in sessions if s.status in (SessionStatus.PENDING, SessionStatus.WAITING)
            ]

        if not sessions:
            if include_all:
                print("No sessions found.")
            else:
                print("No active sessions. Use --ls-all to see all sessions.")
            return 0

        # Print header
        print(f"{'SESSION ID':<28} {'STATUS':<12} {'MODEL':<25} {'UPDATED':<12}")
        print("-" * 77)

        for session in sessions:
            status_str = session.status.value
            if session.status == SessionStatus.WAITING:
                # Check if there's a pending input
                pending = await session_store.read_pending_input(session.session_id)
                if pending:
                    q_type = pending.get("type", "")
                    if q_type == "ask_user":
                        questions = pending.get("questions", [])
                        if questions:
                            status_str = f"waiting: {questions[0].get('question', '')[:20]}..."
                    else:
                        status_str = "waiting: input needed"

            model = f"{session.endpoint.provider}/{session.endpoint.model}"
            if len(model) > 25:
                model = model[:22] + "..."

            updated = format_time_ago(session.updated_at) if session.updated_at else "?"

            print(f"{session.session_id:<28} {status_str:<12} {model:<25} {updated:<12}")

        return 0

    return trio.run(ls_action)


def cmd_status(session_store: FileSessionStore, session_id: str) -> int:
    """Handle --status command."""
    from .dtypes import SessionStatus

    async def status_action() -> int:
        session, err = await session_store.get(session_id)
        if err or not session:
            print(f"Session not found: {session_id}", file=sys.stderr)
            return 1

        print(f"Session: {session.session_id}")
        print(f"Status:  {session.status.value}")
        print(f"Model:   {session.endpoint.provider}/{session.endpoint.model}")
        print(f"Messages: {len(session.messages)}")
        if session.updated_at:
            print(f"Updated: {session.updated_at}")

        # Show pending input if waiting
        if session.status == SessionStatus.WAITING:
            pending = await session_store.read_pending_input(session_id)
            if pending:
                print("\n--- Pending Input ---")
                p_type = pending.get("type", "unknown")
                if p_type == "ask_user":
                    for q in pending.get("questions", []):
                        print(f"Q: {q.get('question', '')}")
                        options = q.get("options", [])
                        if options:
                            print(f"   Options: {', '.join(options)}")
                elif p_type == "no_tools":
                    last_msg = pending.get("last_message", "")
                    print(f"Agent stopped. Last message:\n{last_msg[:200]}...")

        return 0

    return trio.run(status_action)


def cmd_send(
    config: CLIConfig,
    session_store: FileSessionStore,
    session_id: str,
    message: str,
) -> int:
    """Handle --send command: send message to waiting session and resume."""
    from .dtypes import SessionStatus

    async def send_action() -> int | None:
        # Check session exists and is waiting
        session, err = await session_store.get(session_id)
        if err or not session:
            print(f"Session not found: {session_id}", file=sys.stderr)
            return 1

        if session.status != SessionStatus.WAITING:
            print(
                f"Session is not waiting for input (status: {session.status.value})",
                file=sys.stderr,
            )
            return 1

        # Clear pending input
        await session_store.clear_pending_input(session_id)

        print(f"Resuming {session_id}...", file=sys.stderr)

        # Set up config to resume with the message as initial prompt.
        # TODO(cleanup): We use initial_prompt instead of appending to messages.jsonl
        # because the runner waits for input before running the agent. When resuming,
        # it sees the existing trajectory and asks for new input, ignoring any message
        # we append. Using initial_prompt bypasses this. A cleaner fix would be for the
        # runner to detect "trajectory has unprocessed user message" and skip waiting.
        config.session = session_id
        config.initial_prompt = message
        # Keep detached mode for send (no TUI)
        config.detached = True

        return None  # Signal to continue to main agent flow

    result = trio.run(send_action)
    if result is None:
        # Continue to main agent flow
        return -1  # Special return code to continue
    return result


def cmd_attach(config: CLIConfig, session_id: str) -> int:
    """Handle --attach command: attach TUI to existing session."""
    # Just set up config to resume the session with TUI
    config.session = session_id
    config.frontend = "tui"
    config.detached = False  # Attached mode
    return -1  # Signal to continue to main agent flow


# =============================================================================
# Config loading - preset and session config merging
# =============================================================================


def apply_preset(config: CLIConfig) -> bool:
    """Apply preset configuration if specified. Returns False on error."""
    if not config.preset:
        return True

    from .agent_presets import load_preset

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


def apply_template(config: CLIConfig) -> bool:
    """Apply template configuration if specified. Returns False on error."""
    if not config.template:
        return True

    from .templates import TemplateConfig, load_template

    try:
        template: TemplateConfig = load_template(config.template)
    except Exception as e:
        print(f"Error loading template '{config.template}': {e}", file=sys.stderr)
        return False

    # Store loaded template for later use
    config._template_config = template

    # Model: only override if template specifies one and CLI didn't override
    if template.model is not None and config.model == PARSER_DEFAULTS["model"]:
        config.model = template.model

    # Thinking: apply template value if specified
    if template.thinking is not None and config.thinking == PARSER_DEFAULTS["thinking"]:
        config.thinking = "enabled" if template.thinking else "disabled"

    # Force coding environment for templates (they use file tools)
    config.env = "coding"

    # Store tools filter and bash allowlist
    config.tools = ",".join(template.tools)  # Will be parsed by create_environment
    config._bash_allowlist = template.bash_allowlist

    # Interpolate system prompt with template args
    try:
        config.system_prompt = template.interpolate_prompt(config.template_args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

    # Template mode: detached by default unless -i specified
    if not config.interactive:
        config.detached = True

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
        from .environments.coding import TOOL_PRESETS

        tools = config.tools or "full"

        # Handle template tools (comma-separated list like "read,grep,bash")
        if "," in tools:
            tools_list = [t.strip() for t in tools.split(",")]
            return LocalFilesystemEnvironment(
                working_dir=config.working_dir,
                tools=tools_list,
                bash_allowlist=config._bash_allowlist,
            ), True

        # Handle preset names
        if tools not in TOOL_PRESETS:
            print(
                f"Unknown tool preset: {tools}. Available: {', '.join(TOOL_PRESETS.keys())}",
                file=sys.stderr,
            )
            return None, False
        return LocalFilesystemEnvironment(
            working_dir=config.working_dir,
            tools=tools,
            bash_allowlist=config._bash_allowlist,
        ), True

    if config.env == "git":
        return GitWorktreeEnvironment(working_dir=config.working_dir), True

    if config.env == "repl":
        from .environments.repl import REPLEnvironment

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
        from .environments.repl import MessageParsingREPLEnvironment

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
        from .environments.compose import compose
        from .environments.repl import REPLEnvironment

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
            elif env_name == "ask_user":
                from .environments.ask_user import AskUserQuestionEnvironment

                environments.append(AskUserQuestionEnvironment())
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
    from .agents import resume_session

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
        from .prompt import build_system_prompt

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

    from .frontends import JsonFrontend, NoneFrontend, run_interactive

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
    elif config.frontend == "minimal":
        from .frontends import MinimalFrontend

        env_name = config.environment.__class__.__name__ if config.environment else None
        frontend = MinimalFrontend(
            show_tool_calls=True,
            show_thinking=False,
            agent=env_name,
            model=config.endpoint.model,
        )
    else:
        frontend = NoneFrontend(show_tool_calls=True, show_thinking=False)

    from .frontends.runner import RunnerConfig

    try:
        await run_interactive(
            trajectory,
            config.endpoint,
            frontend=frontend,
            environment=config.environment,
            config=RunnerConfig(
                session_store=config.session_store,
                session_id=session_id,
                initial_prompt=query,
                single_turn=True,
            ),
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
    """Run in interactive mode with selected frontend and driver."""
    assert config.endpoint is not None, "endpoint must be set for interactive mode"

    from functools import partial

    from .frontends import MinimalFrontend, NoneFrontend, TUIFrontend, run_interactive
    from .frontends.runner import RunFn, RunnerConfig

    # Select run_fn based on driver
    run_fn: RunFn | None = None
    if config.driver == "claude":
        from .drivers.run_claude import run_claude

        run_fn = partial(run_claude, model=config.endpoint.model or "sonnet", cwd=config.cwd)
    elif config.driver == "codex":
        from .drivers.run_codex import run_codex

        # Don't pass model - codex uses its own model config
        # The endpoint.model is for Anthropic/SDK, not codex
        run_fn = partial(run_codex, model=None, cwd=config.cwd)
    elif config.driver == "cursor":
        from .drivers.run_cursor import run_cursor

        # Cursor uses its own API key and model config
        run_fn = partial(run_cursor, model=None, cwd=config.cwd, api_key=config.cursor_api_key)
    # else: sdk - use default run_agent (run_fn=None)

    # Select frontend
    env_name = config.environment.__class__.__name__ if config.environment else None
    if config.frontend == "none":
        frontend = NoneFrontend(show_tool_calls=True, show_thinking=True)
    elif config.frontend == "minimal":
        frontend = MinimalFrontend(
            show_tool_calls=True,
            show_thinking=True,
            agent=env_name,
            model=config.endpoint.model,
        )
    elif config.frontend == "textual":
        print("Textual frontend not yet implemented. Use --frontend=tui for now.", file=sys.stderr)
        return 1
    else:
        # Default: TUI (only for SDK driver)
        frontend = TUIFrontend(
            theme=config.theme,
            environment=config.environment,
            debug=config.debug,
            debug_layout=config.debug_layout,
            driver=config.driver,
        )

    # Detached mode uses simple frontend
    if config.detached:
        frontend = NoneFrontend(show_tool_calls=True, show_thinking=False)

    try:
        await run_interactive(
            trajectory,
            config.endpoint,
            frontend=frontend,
            environment=config.environment,
            config=RunnerConfig(
                session_store=config.session_store,
                session_id=session_id,
                parent_session_id=parent_session_id,
                branch_point=branch_point,
                confirm_tools=config.confirm_tools,
                initial_prompt=initial_prompt,
                detached=config.detached,
                cwd=config.cwd,
                run_fn=run_fn,
            ),
        )
    except KeyboardInterrupt:
        print("\n\n✅ Agent stopped")
    return 0


# =============================================================================
# Main entry point
# =============================================================================


def auth_main(args: list[str]) -> int:
    """Handle auth subcommand: rollouts auth <login|status|switch>."""
    from .credentials import (
        CREDENTIALS_FILE,
        KNOWN_PROVIDERS,
        PROVIDER_ENV_MAP,
        get_active_profile,
        key_preview,
        load_profiles,
        set_active_profile,
        set_profile_key,
    )

    if not args or args[0] in ("-h", "--help"):
        print("Usage: rollouts auth <command>")
        print()
        print("Commands:")
        print("  login <provider>   Save API key for a provider")
        print("  status             Show configured credentials")
        print("  switch <profile>   Switch active profile")
        print()
        print(f"Providers: {', '.join(sorted(KNOWN_PROVIDERS))}")
        print(f"Config: {CREDENTIALS_FILE}")
        return 0

    cmd = args[0]

    if cmd == "login":
        if len(args) < 2:
            print("Usage: rollouts auth login <provider> [--profile NAME]", file=sys.stderr)
            print(f"Providers: {', '.join(sorted(KNOWN_PROVIDERS))}", file=sys.stderr)
            return 1

        provider = args[1]
        profile = "default"

        # Parse --profile
        if "--profile" in args:
            idx = args.index("--profile")
            if idx + 1 < len(args):
                profile = args[idx + 1]

        if provider not in KNOWN_PROVIDERS:
            print(f"Unknown provider: {provider}", file=sys.stderr)
            print(f"Known providers: {', '.join(sorted(KNOWN_PROVIDERS))}", file=sys.stderr)
            return 1

        # Prompt for API key
        import getpass

        api_key = getpass.getpass(f"Enter {provider} API key: ")
        if not api_key.strip():
            print("No API key provided", file=sys.stderr)
            return 1

        set_profile_key(profile, provider, api_key.strip())
        print(f"✓ Saved {provider} API key to profile '{profile}'")
        return 0

    elif cmd == "status":
        import os

        profiles = load_profiles()
        active_name, profile_creds = get_active_profile()

        # All providers use same simple precedence: env var > credentials.toml
        # (OAuth is only for external drivers like claude/codex, not shown here)
        for provider in sorted(KNOWN_PROVIDERS):
            env_var_name = PROVIDER_ENV_MAP.get(provider)
            env_val = os.environ.get(env_var_name) if env_var_name else None
            toml_val = profile_creds.get(provider)

            if not env_val and not toml_val:
                continue

            print(f"{provider.capitalize()} (in precedence order):")
            active_found = False

            # 1. Env var
            if env_val:
                active_found = True
                print(f"  1. ${env_var_name}: {key_preview(env_val)} <- active")
            else:
                print(f"  1. ${env_var_name}: (not set)")

            # 2. credentials.toml
            if toml_val:
                marker = " <- active" if not active_found else ""
                print(
                    f"  2. credentials.toml: {key_preview(toml_val)} (profile:{active_name}){marker}"
                )
            else:
                print(f"  2. credentials.toml: (not set in profile:{active_name})")

            print()

        # Show providers with no config
        unconfigured = []
        for provider in sorted(KNOWN_PROVIDERS):
            env_var_name = PROVIDER_ENV_MAP.get(provider)
            env_val = os.environ.get(env_var_name) if env_var_name else None
            toml_val = profile_creds.get(provider)
            if not env_val and not toml_val:
                unconfigured.append(provider)

        if unconfigured:
            print(f"Not configured: {', '.join(unconfigured)}")
            print()

        print(f"Config: {CREDENTIALS_FILE}")

        return 0

    elif cmd == "switch":
        if len(args) < 2:
            print("Usage: rollouts auth switch <profile>", file=sys.stderr)
            profiles = load_profiles()
            if profiles:
                print(f"Available: {', '.join(profiles.keys())}", file=sys.stderr)
            return 1

        profile = args[1]
        try:
            set_active_profile(profile)
            print(f"✓ Switched to profile '{profile}'")
            return 0
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1

    else:
        print(f"Unknown auth command: {cmd}", file=sys.stderr)
        print("Use: rollouts auth --help", file=sys.stderr)
        return 1


def main() -> int:
    """Main CLI entry point - dispatcher for all CLI commands."""
    # Intercept subcommands before argparse (flat parser doesn't support subparsers)
    if len(sys.argv) > 1 and sys.argv[1] == "auth":
        return auth_main(sys.argv[2:])

    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        from .tui.monitor_cli import monitor_main

        return monitor_main(sys.argv[2:])

    # "rollouts agent" subcommand replaced with --driver flag
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        print("The 'agent' subcommand has been replaced with --driver:", file=sys.stderr)
        print("  rollouts --driver claude    # Start with Claude Code", file=sys.stderr)
        print("  rollouts --driver codex     # Start with Codex", file=sys.stderr)
        print("  rollouts                    # Start with SDK (default)", file=sys.stderr)
        print()
        print("You can also swap mid-session with /swap claude or /swap rollouts", file=sys.stderr)
        return 1

    # Load .env file for API keys (if present)
    from dotenv import load_dotenv

    load_dotenv()

    parser = create_parser()
    args = parser.parse_args()

    # Setup logging early (before any other imports that might log)
    # --debug sets DEBUG level, --log-file writes JSONL to file
    if args.debug or args.log_file:
        from ._logging import setup_logging

        setup_logging(
            level="DEBUG" if args.debug else "INFO",
            log_file=args.log_file,
            use_color=True,  # Colorized console output
            logger_levels={
                # Suppress noisy third-party loggers unless we're debugging
                "httpx": "WARNING",
                "httpcore": "WARNING",
                "anthropic": "DEBUG" if args.debug else "WARNING",
            },
        )

    # Build config from parsed args
    # Handle context from --context or --context-file
    context = args.context
    if args.context_file:
        try:
            context = Path(args.context_file).read_text()
        except Exception as e:
            print(f"Error reading context file: {e}", file=sys.stderr)
            return 1

    # Parse --args into dict
    template_args: dict[str, str] | None = None
    if args.args:
        template_args = {}
        for arg in args.args:
            if "=" not in arg:
                print(f"Invalid --args format: {arg!r}. Use KEY=VALUE", file=sys.stderr)
                return 1
            key, value = arg.split("=", 1)
            template_args[key] = value

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
        log_file=args.log_file,
        driver=args.driver,
        cursor_api_key=getattr(args, "cursor_api_key", None),
        preset=args.preset,
        system_prompt=args.system_prompt,
        pick=args.pick,
        list_models=args.list_models,
        sync_models=args.sync_models,
        write_models=args.write,
        list_presets=args.list_presets,
        login_claude=args.login_claude,
        logout_claude=args.logout_claude,
        list_claude_profiles=args.list_claude_profiles,
        set_default_profile=args.set_default_profile,
        profile=args.profile,
        export_md=args.export_md,
        export_html=args.export_html,
        handoff=args.handoff,
        fast_handoff=args.fast,
        slice=args.slice,
        slice_goal=args.slice_goal,
        doctor=args.doctor,
        trim=args.trim,
        fix=args.fix,
        send=tuple(args.send) if args.send else None,
        send_file=tuple(args.send_file) if args.send_file else None,
        attach=args.attach,
        status=args.status,
        ls=args.ls,
        ls_all=args.ls_all,
        detached=args.detached,
        template=args.template,
        template_args=template_args,
        interactive=args.interactive,
        list_templates=args.list_templates,
    )

    # === Commands that don't need endpoint ===

    if config.list_models:
        return cmd_list_models()

    if config.sync_models:
        return cmd_sync_models(write=config.write_models)

    if config.list_presets:
        return cmd_list_presets()

    if config.list_templates:
        return cmd_list_templates()

    # Determine profile from CLI arg or env var
    import os

    profile = config.profile or os.environ.get("ROLLOUTS_PROFILE", "default")

    if config.list_claude_profiles:
        return cmd_list_profiles()

    if config.set_default_profile is not None:
        return cmd_set_default_profile(config.set_default_profile)

    if config.login_claude or config.logout_claude:
        return cmd_oauth(login=config.login_claude, profile=profile)

    if config.export_md is not None or config.export_html is not None:
        return cmd_export(config, FileSessionStore())

    if config.doctor or config.trim is not None or config.fix:
        return cmd_doctor(config, FileSessionStore())

    # === Tmux-style session commands (don't need endpoint) ===

    if config.ls or config.ls_all:
        return cmd_ls(FileSessionStore(), include_all=config.ls_all)

    if config.status is not None:
        if config.status == "":
            # No session ID, show list instead
            return cmd_ls(FileSessionStore(), include_all=False)
        return cmd_status(FileSessionStore(), config.status)

    if config.send:
        session_id, message = config.send
        result = cmd_send(config, FileSessionStore(), session_id, message)
        if result != -1:
            return result
        # result == -1 means continue to main agent flow

    if config.send_file:
        session_id, file_path = config.send_file
        try:
            message = Path(file_path).read_text()
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            return 1
        result = cmd_send(config, FileSessionStore(), session_id, message)
        if result != -1:
            return result
        # result == -1 means continue to main agent flow

    if config.attach:
        result = cmd_attach(config, config.attach)
        if result != -1:
            return result
        # result == -1 means continue to main agent flow

    # === Commands requiring endpoint ===

    # Apply preset, template, and session config
    if not apply_preset(config):
        return 1
    if not apply_template(config):
        return 1
    if not apply_session_config(config):
        return 1

    # Set working directory
    config.working_dir = Path(config.cwd) if config.cwd else Path.cwd()

    # Create endpoint
    try:
        config.endpoint = create_endpoint(
            config.model,
            config.api_base,
            config.api_key,
            config.thinking,
            config.quiet,
            profile,
            driver=config.driver,
        )
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    # Validate authentication (skip for external drivers - they handle their own auth)
    if config.driver == "sdk" and not config.endpoint.api_key and not config.endpoint.oauth_token:
        provider = config.endpoint.provider
        print(
            f"❌ No API key found for {provider}.",
            file=sys.stderr,
        )
        print(
            f"   Run: rollouts auth login {provider}",
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
    except BaseException as e:
        from .providers.base import AuthenticationError

        # Check for auth errors (may be wrapped in ExceptionGroup by Trio)
        if isinstance(e, AuthenticationError):
            print(f"\n❌ {e}", file=sys.stderr)
            return 1
        if hasattr(e, "exceptions"):  # ExceptionGroup (Python 3.11+)
            auth_errors = [exc for exc in e.exceptions if isinstance(exc, AuthenticationError)]
            if auth_errors:
                print(f"\n❌ {auth_errors[0]}", file=sys.stderr)
                return 1
        # Other errors: print full traceback
        print(f"\n\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
