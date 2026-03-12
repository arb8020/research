from __future__ import annotations

import argparse

from ..environments.factory import get_environment_names
from .config import PARSER_DEFAULTS

ENVIRONMENT_HELP = ", ".join(["none", *get_environment_names()])


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Rollouts - chat with an LLM agent in your terminal"
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Agent preset name (e.g., 'fast_coder', 'careful_coder') or path to preset file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=PARSER_DEFAULTS["model"],
        help=f'Model in "provider/model" format. Default: {PARSER_DEFAULTS["model"]}',
    )
    parser.add_argument(
        "--api-base", type=str, default=None, help="API base URL (default: provider-specific)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="API key (default: from environment)"
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
            f"Environment with tools. Options: {ENVIRONMENT_HELP}. "
            "Compose with '+': coding+repl, git+repl, orchestrate+coding (default: none)"
        ),
    )
    parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Tool preset for coding env: full, readonly, no-write (default: full)",
    )
    parser.add_argument(
        "--cwd", type=str, default=None, help="Working directory for coding environment"
    )
    parser.add_argument(
        "--confirm-tools", action="store_true", help="Require confirmation before executing tools"
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
    parser.add_argument(
        "--tbench-task-id", type=str, default=None, help="Terminal-Bench task ID for --env tbench"
    )
    parser.add_argument(
        "--tbench-dataset-name",
        type=str,
        default="terminal-bench-core",
        help="Terminal-Bench dataset name (default: terminal-bench-core)",
    )
    parser.add_argument(
        "--tbench-dataset-version",
        type=str,
        default="head",
        help="Terminal-Bench dataset version (default: head)",
    )
    parser.add_argument(
        "--tbench-surface",
        type=str,
        default="terminal",
        help="Terminal-Bench tool surface (default: terminal)",
    )
    parser.add_argument(
        "--tbench-logging-dir",
        type=str,
        default=None,
        help="Terminal-Bench run directory (default: auto-generated runs/tb_<task>_<timestamp>)",
    )
    parser.add_argument(
        "--tbench-rebuild",
        action="store_true",
        help="Rebuild the Terminal-Bench task image before starting",
    )
    parser.add_argument(
        "--tbench-no-cleanup",
        action="store_true",
        help="Keep Terminal-Bench containers/artifacts alive after the run",
    )
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
    parser.add_argument("--no-session", action="store_true", help="Don't persist session to disk")
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
    parser.add_argument(
        "--driver",
        type=str,
        choices=["sdk", "claude", "codex", "cursor"],
        default="sdk",
        help="Backend driver: sdk (default, direct API), claude (Claude Code CLI), codex (Codex CLI), cursor (Cursor Agent CLI)",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["enabled", "disabled"],
        default=PARSER_DEFAULTS["thinking"],
        help="Extended thinking for Anthropic models (default: enabled)",
    )
    parser.add_argument(
        "--pick",
        action="store_true",
        help="Interactive model/driver picker (choose between rollouts, claude-code, codex)",
    )
    parser.add_argument(
        "--list-models",
        type=str,
        nargs="?",
        const="",
        default=None,
        metavar="SEARCH",
        help="List available models (with optional fuzzy search pattern)",
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
    parser.add_argument(
        "--list-presets", action="store_true", help="List available agent presets and exit"
    )
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
        "--list-templates", action="store_true", help="List available templates and exit"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging (sets LOG_LEVEL=DEBUG)"
    )
    parser.add_argument("--debug-layout", action="store_true", help="Show TUI component boundaries")
    parser.add_argument(
        "--log-file",
        type=str,
        help="Write logs to file (JSONL format, includes API requests at DEBUG level)",
    )
    parser.add_argument(
        "--login-claude", action="store_true", help="Login with Claude Pro/Max account (OAuth)"
    )
    parser.add_argument(
        "--logout-claude", action="store_true", help="Logout and revoke Claude OAuth tokens"
    )
    parser.add_argument(
        "--list-claude-profiles", action="store_true", help="List available Claude OAuth profiles"
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
        help="Slice session messages. SPEC format: '0:4, summarize:5:15, 16:, inject:\"msg\"'. Creates new child session. Outputs session ID.",
    )
    parser.add_argument(
        "--slice-goal",
        type=str,
        metavar="GOAL",
        help="Focus summaries in --slice on this goal (optional)",
    )
    parser.add_argument(
        "--doctor", action="store_true", help="Show session diagnostics (use with --session)"
    )
    parser.add_argument(
        "--trim",
        type=int,
        metavar="N",
        help="Remove last N messages from session (creates new fixed session)",
    )
    parser.add_argument(
        "--fix", action="store_true", help="Auto-fix detected issues (duplicate tool results, etc.)"
    )
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
        "--attach", type=str, metavar="SESSION_ID", help="Attach TUI to existing session"
    )
    parser.add_argument(
        "--status",
        type=str,
        nargs="?",
        const="",
        metavar="SESSION_ID",
        help="Show session status (list active if no ID)",
    )
    parser.add_argument("--ls", action="store_true", help="List active sessions (running/waiting)")
    parser.add_argument(
        "--ls-all", action="store_true", help="List all sessions including completed/failed"
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Run detached: exit when agent needs input instead of blocking",
    )
    return parser
