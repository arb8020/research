"""
Slash commands for the TUI.

Supports:
1. Built-in commands (/model, /thinking, /slice)
2. File-based commands from ~/.rollouts/commands/ and .rollouts/commands/

Reference: /tmp/pi-mono/packages/coding-agent/src/core/slash-commands.ts
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .interactive_agent import InteractiveAgentRunner


# =============================================================================
# Types
# =============================================================================


@dataclass
class SlashCommand:
    """A slash command definition."""

    name: str
    description: str
    arg_hint: str | None = None  # Ghost text hint for arguments, e.g., "[spec]"
    content: str | None = None  # For file-based commands
    source: str = "(builtin)"  # "(builtin)", "(user)", "(project)"


@dataclass
class SlashCommandResult:
    """Result of executing a slash command."""

    handled: bool = True  # If False, pass the message to the LLM
    message: str | None = None  # Display to user (ghost message)
    expanded_text: str | None = None  # For file commands, send this to LLM instead
    persist_changes: dict[str, Any] | None = None  # Session metadata to persist


# =============================================================================
# Built-in Commands
# =============================================================================


BUILTIN_COMMANDS: list[SlashCommand] = [
    SlashCommand("model", "Switch model", arg_hint="[provider/model]"),
    SlashCommand("thinking", "Toggle extended thinking", arg_hint="[on|off|budget]"),
    SlashCommand(
        "slice",
        "Slice session context",
        arg_hint="[0:5, summarize:5:20, 20:]",
    ),
]


async def handle_slash_command(
    runner: InteractiveAgentRunner,
    text: str,
) -> SlashCommandResult:
    """Handle a slash command.

    Returns a SlashCommandResult indicating how to proceed.
    """
    if not text.startswith("/"):
        return SlashCommandResult(handled=False)

    # Parse command name and args
    space_index = text.find(" ")
    if space_index == -1:
        command = text[1:]
        args = ""
    else:
        command = text[1:space_index]
        args = text[space_index + 1 :].strip()

    # Try built-in commands first
    if command == "model":
        return await _handle_model(runner, args)
    elif command == "thinking":
        return await _handle_thinking(runner, args)
    elif command == "slice":
        return await _handle_slice(runner, args)

    # Try file-based commands
    file_commands = load_file_commands()
    expanded = expand_file_command(text, file_commands)
    if expanded is not None:
        # File command found - return expanded text to send to LLM
        return SlashCommandResult(handled=False, expanded_text=expanded)

    # Unknown command - suggest correction
    suggestion = _find_similar_command(command)
    if suggestion:
        return SlashCommandResult(
            handled=True,
            message=f"Unknown command: /{command}\nDid you mean /{suggestion}?",
        )

    # No suggestion - just show error
    return SlashCommandResult(
        handled=True,
        message=f"Unknown command: /{command}\nType /model, /thinking, or /slice",
    )


def _find_similar_command(command: str) -> str | None:
    """Find a similar command name for typo suggestions."""
    all_commands = [c.name for c in BUILTIN_COMMANDS] + [c.name for c in load_file_commands()]

    # Simple prefix match
    for name in all_commands:
        if name.startswith(command[:2]) or command.startswith(name[:2]):
            return name

    return None


# =============================================================================
# /model Command
# =============================================================================


async def _handle_model(runner: InteractiveAgentRunner, args: str) -> SlashCommandResult:
    """Handle /model command."""
    from dataclasses import replace as dc_replace

    from rollouts.models import get_model, get_models, get_providers

    if not args:
        # Show current model
        return SlashCommandResult(
            message=f"Current model: {runner.endpoint.provider}/{runner.endpoint.model}"
        )

    # Parse provider/model
    if "/" not in args:
        # List available models for partial match
        available = []
        for provider in get_providers():
            for model in get_models(provider):
                if args.lower() in model.id.lower() or args.lower() in provider.lower():
                    available.append(f"  {provider}/{model.id}")

        if available:
            msg = "Invalid format. Use: /model provider/model-id\n\nMatching models:\n"
            msg += "\n".join(available[:10])
            if len(available) > 10:
                msg += f"\n  ... and {len(available) - 10} more"
        else:
            msg = "Invalid format. Use: /model provider/model-id"

        return SlashCommandResult(message=msg)

    provider, model_id = args.split("/", 1)

    # Validate model exists
    model_meta = get_model(provider, model_id)  # type: ignore[arg-type]
    if not model_meta:
        # Show similar models
        available = []
        for m in get_models(provider):  # type: ignore[arg-type]
            if model_id.lower() in m.id.lower():
                available.append(f"  {provider}/{m.id}")

        if available:
            msg = f"Unknown model: {args}\n\nSimilar models:\n" + "\n".join(available[:5])
        else:
            msg = f"Unknown model: {args}\nUse Tab to autocomplete available models."

        return SlashCommandResult(message=msg)

    # Update endpoint
    runner.endpoint = dc_replace(runner.endpoint, provider=provider, model=model_id)

    # Persist to session
    if runner.session_store and runner.session_id:
        await runner.session_store.update(
            runner.session_id,
            endpoint=runner.endpoint,
        )

    return SlashCommandResult(
        message=f"Switched to: {provider}/{model_id}",
    )


# =============================================================================
# /thinking Command
# =============================================================================


def _get_thinking_budget(endpoint: Any) -> int | None:
    """Extract thinking budget from endpoint.thinking dict."""
    thinking = endpoint.thinking
    if thinking is None:
        return None
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        return thinking.get("budget_tokens")
    return None


def _make_thinking_config(budget: int | None) -> dict[str, Any] | None:
    """Create thinking config dict for Anthropic."""
    if budget is None:
        return None
    return {"type": "enabled", "budget_tokens": budget}


async def _handle_thinking(runner: InteractiveAgentRunner, args: str) -> SlashCommandResult:
    """Handle /thinking command."""
    from dataclasses import replace as dc_replace

    from rollouts.models import get_model

    model_meta = get_model(runner.endpoint.provider, runner.endpoint.model)  # type: ignore[arg-type]

    if not args:
        # Show current status
        budget = _get_thinking_budget(runner.endpoint)
        if budget and budget > 0:
            status = f"on (budget: {budget} tokens)"
        else:
            status = "off"
        return SlashCommandResult(message=f"Thinking: {status}")

    # Parse argument
    args_lower = args.lower()
    if args_lower == "off":
        new_budget = None
    elif args_lower == "on":
        new_budget = 10000  # Default budget
    else:
        try:
            new_budget = int(args)
            if new_budget <= 0:
                new_budget = None
            elif new_budget < 1024:
                return SlashCommandResult(
                    message=f"Thinking budget must be >= 1024 tokens (Anthropic requirement)\nGot: {new_budget}"
                )
        except ValueError:
            return SlashCommandResult(
                message=f"Invalid argument: {args}\nUse: /thinking [on|off|<budget>]"
            )

    # Check model supports thinking
    if new_budget and model_meta and not model_meta.reasoning:
        return SlashCommandResult(
            message=f"Cannot enable thinking for {runner.endpoint.model}\nModel does not support extended thinking."
        )

    # Build new thinking config
    new_thinking = _make_thinking_config(new_budget)

    # Update endpoint (also ensure max_tokens > budget and temperature=1.0 for Anthropic)
    new_max_tokens = runner.endpoint.max_tokens
    new_temperature = runner.endpoint.temperature
    if new_budget and runner.endpoint.provider == "anthropic":
        if new_max_tokens <= new_budget:
            new_max_tokens = new_budget + 4096  # Give room for response
        new_temperature = 1.0  # Anthropic requires temp=1.0 with thinking

    runner.endpoint = dc_replace(
        runner.endpoint,
        thinking=new_thinking,
        max_tokens=new_max_tokens,
        temperature=new_temperature,
    )

    # Persist to session
    if runner.session_store and runner.session_id:
        await runner.session_store.update(
            runner.session_id,
            endpoint=runner.endpoint,
        )

    if new_budget:
        status = f"on (budget: {new_budget} tokens)"
    else:
        status = "off"

    return SlashCommandResult(message=f"Thinking: {status}")


# =============================================================================
# /slice Command
# =============================================================================


async def _handle_slice(runner: InteractiveAgentRunner, args: str) -> SlashCommandResult:
    """Handle /slice command."""
    from rollouts.slice import parse_slice_spec, run_slice_command

    # Get current message count
    messages = runner.trajectory.messages if runner.trajectory else []

    if not args or args.lower() == "count":
        return SlashCommandResult(message=f"Session has {len(messages)} messages")

    # Validate we have session store
    if not runner.session_store or not runner.session_id:
        return SlashCommandResult(message="Cannot slice: no session configured")

    # Parse spec to validate
    try:
        segments = parse_slice_spec(args)
    except ValueError as e:
        return SlashCommandResult(message=f"Invalid slice spec: {e}")

    # Load full session
    session, err = await runner.session_store.get(runner.session_id)
    if err or not session:
        return SlashCommandResult(message=f"Cannot load session: {err}")

    # Execute slice
    child, err = await run_slice_command(
        session=session,
        spec=args,
        endpoint=runner.endpoint,
        session_store=runner.session_store,
    )

    if err:
        return SlashCommandResult(message=f"Slice failed: {err}")

    if not child:
        return SlashCommandResult(message="Slice produced no result")

    # Switch to the child session
    switched = await runner.switch_session(child.session_id)
    if switched:
        return SlashCommandResult(message=f"Switched to child session: {child.session_id}")
    else:
        # Fallback if switch failed
        return SlashCommandResult(
            message=f"Created child session: {child.session_id}\n\nSwitch failed. Run:\n  rollouts -s {child.session_id}"
        )


# =============================================================================
# File-Based Commands
# =============================================================================


def get_user_commands_dir() -> Path:
    """Get user-level commands directory (~/.rollouts/commands/)."""
    return Path.home() / ".rollouts" / "commands"


def get_project_commands_dir(cwd: Path | None = None) -> Path:
    """Get project-level commands directory (.rollouts/commands/)."""
    if cwd is None:
        cwd = Path.cwd()
    return cwd / ".rollouts" / "commands"


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from markdown content.

    Reference: pi-mono slash-commands.ts L19-43

    Returns (frontmatter_dict, remaining_content).
    """
    frontmatter: dict[str, str] = {}

    if not content.startswith("---"):
        return frontmatter, content

    # Find end of frontmatter
    end_index = content.find("\n---", 3)
    if end_index == -1:
        return frontmatter, content

    frontmatter_block = content[4:end_index]
    remaining_content = content[end_index + 4 :].strip()

    # Simple YAML parsing - just key: value pairs
    for line in frontmatter_block.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key:
                frontmatter[key] = value

    return frontmatter, remaining_content


def parse_command_args(args_string: str) -> list[str]:
    """Parse command arguments respecting quoted strings.

    Reference: pi-mono slash-commands.ts L46-71
    """
    args: list[str] = []
    current = ""
    in_quote: str | None = None

    for char in args_string:
        if in_quote:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in "\"'":
            in_quote = char
        elif char in " \t":
            if current:
                args.append(current)
                current = ""
        else:
            current += char

    if current:
        args.append(current)

    return args


def substitute_args(content: str, args: list[str]) -> str:
    """Substitute argument placeholders in command content.

    Reference: pi-mono slash-commands.ts L76-90

    Supports $1, $2, ... for positional args and $@ for all args.
    """
    result = content

    # Replace $@ with all args joined
    result = result.replace("$@", " ".join(args))

    # Replace $1, $2, etc. with positional args
    def replace_positional(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        return args[index] if index < len(args) else ""

    result = re.sub(r"\$(\d+)", replace_positional, result)

    return result


def load_commands_from_dir(
    directory: Path,
    source: str,
    subdir: str = "",
) -> list[SlashCommand]:
    """Recursively load .md files as slash commands.

    Reference: pi-mono slash-commands.ts L102-175
    """
    commands: list[SlashCommand] = []

    if not directory.exists():
        return commands

    try:
        for entry in directory.iterdir():
            if entry.is_dir():
                # Recurse into subdirectory
                new_subdir = f"{subdir}:{entry.name}" if subdir else entry.name
                commands.extend(load_commands_from_dir(entry, source, new_subdir))
            elif entry.suffix == ".md":
                try:
                    raw_content = entry.read_text()
                    frontmatter, content = parse_frontmatter(raw_content)

                    name = entry.stem  # Filename without .md

                    # Build source string
                    source_str = f"({source}:{subdir})" if subdir else f"({source})"

                    # Get description from frontmatter or first line
                    description = frontmatter.get("description", "")
                    if not description:
                        first_line = next(
                            (line.strip() for line in content.split("\n") if line.strip()),
                            "",
                        )
                        description = first_line[:60]
                        if len(first_line) > 60:
                            description += "..."

                    # Append source to description
                    if description:
                        description = f"{description} {source_str}"
                    else:
                        description = source_str

                    commands.append(
                        SlashCommand(
                            name=name,
                            description=description,
                            content=content,
                            source=source_str,
                        )
                    )
                except Exception:
                    # Skip files that can't be read
                    pass
    except Exception:
        # Skip directories that can't be read
        pass

    return commands


# Cache for file commands (cleared on each call for now)
_file_commands_cache: list[SlashCommand] | None = None


def load_file_commands(cwd: Path | None = None) -> list[SlashCommand]:
    """Load all file-based slash commands.

    Loads from:
    1. ~/.rollouts/commands/ (user-level)
    2. .rollouts/commands/ (project-level)
    """
    # TODO: Add caching with file watcher
    commands: list[SlashCommand] = []

    # User-level commands
    user_dir = get_user_commands_dir()
    commands.extend(load_commands_from_dir(user_dir, "user"))

    # Project-level commands
    project_dir = get_project_commands_dir(cwd)
    commands.extend(load_commands_from_dir(project_dir, "project"))

    return commands


def expand_file_command(text: str, file_commands: list[SlashCommand]) -> str | None:
    """Expand a file-based slash command.

    Reference: pi-mono slash-commands.ts L202-218

    Returns the expanded content, or None if not a file command.
    """
    if not text.startswith("/"):
        return None

    # Parse command name and args
    space_index = text.find(" ")
    if space_index == -1:
        command_name = text[1:]
        args_string = ""
    else:
        command_name = text[1:space_index]
        args_string = text[space_index + 1 :]

    # Find matching file command
    for cmd in file_commands:
        if cmd.name == command_name and cmd.content is not None:
            args = parse_command_args(args_string)
            return substitute_args(cmd.content, args)

    return None


# =============================================================================
# Utilities
# =============================================================================


def get_all_commands() -> list[SlashCommand]:
    """Get all available slash commands (builtin + file-based)."""
    commands = list(BUILTIN_COMMANDS)
    commands.extend(load_file_commands())
    return commands


def get_command_arg_hint(command_name: str) -> str | None:
    """Get the argument hint for a command (e.g., '[spec]' for /slice).

    Args:
        command_name: The command name without the leading /

    Returns:
        The arg_hint string, or None if no hint available
    """
    for cmd in BUILTIN_COMMANDS:
        if cmd.name == command_name:
            return cmd.arg_hint
    return None
