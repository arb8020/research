"""
Coding environment with minimal toolset for code editing tasks.

Tools: read, write, edit, bash
Inspired by pi-mono's minimalist approach.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
import markdownify
import trio

if TYPE_CHECKING:
    from ..dtypes import DetailLevel
    from ..frontends.tui.theme import Theme

from ..agents import AgentState, RunConfig
from ..core import (
    Message,
    Tool,
    ToolCall,
    ToolFunction,
    ToolFunctionParameter,
    ToolRenderConfig,
    ToolResult,
)
from ._formatting import (
    format_tool,
    get_text_output,
    replace_tabs,
    shorten_path,
)
from .resources import CodingWorkspaceResource, CommandExecutionResult, CommandRunner

MAX_LINES = 2000
MAX_LINE_LENGTH = 2000
MAX_OUTPUT_SIZE = 30_000  # 30KB (matches Claude Code's default)

# Directory for storing large tool outputs
DEFAULT_TOOL_OUTPUT_DIR = Path.home() / ".rollouts" / "tool_outputs"

# Web fetch constants
WEB_FETCH_MAX_SIZE = 10 * 1024 * 1024  # 10MB max download
WEB_FETCH_MAX_CONTENT = 100_000  # 100KB max content after conversion
WEB_FETCH_CACHE_TTL = 900  # 15 minutes
WEB_FETCH_TIMEOUT = 30  # seconds


def expand_path(file_path: str) -> Path:
    """Expand ~ to home directory and resolve path."""
    if file_path == "~":
        return Path.home()
    if file_path.startswith("~/"):
        return Path.home() / file_path[2:]
    return Path(file_path).resolve()


# ── Tool Formatters ───────────────────────────────────────────────────────────
# These format tool calls for display in the TUI.
# Signature: (tool_name, args, result, expanded, theme) -> str


# ── Tool Render Configs ────────────────────────────────────────────────────────
# Simple tools just define a config. Complex tools (edit, write) use custom_formatter.

BASH_RENDER_CONFIG = ToolRenderConfig(
    header_fn=lambda name, args: f"bash(command={repr(args.get('command', '...'))})",
    max_lines=5,
    success_summary="Command completed",
    error_summary="Command failed",
)


def format_read(
    tool_name: str,
    args: dict,
    result: dict | None,
    detail_level: DetailLevel | bool,
    theme: Theme | None = None,
) -> str:
    """Format read tool execution."""
    from ..dtypes import DetailLevel

    # Handle backward compat: bool -> DetailLevel
    if isinstance(detail_level, bool):
        detail_level = DetailLevel.EXPANDED if detail_level else DetailLevel.STANDARD

    path = shorten_path(args.get("file_path") or args.get("path") or "")
    offset = args.get("offset")
    limit = args.get("limit")

    params = f"file_path={repr(path if path else '...')}"
    if offset is not None:
        params += f", offset={offset}"
    if limit is not None:
        params += f", limit={limit}"

    header = f"read({params})"

    # Compact mode: one-liner with success/fail indicator
    if detail_level == DetailLevel.COMPACT:
        if result is None:
            return header
        is_error = result.get("isError", False)
        indicator = "✗" if is_error else "✓"
        return f"{header} {indicator}"

    text = header

    # Just show line count summary (not full content)
    if result:
        output = get_text_output(result)
        total_lines = len(output.split("\n"))
        text += f"\n⎿ Read {total_lines} line{'s' if total_lines != 1 else ''}"

    return text


def format_write(
    tool_name: str,
    args: dict,
    result: dict | None,
    detail_level: DetailLevel | bool,
    theme: Theme | None = None,
) -> str:
    """Format write tool execution with line numbers and gray styling.

    Args:
        detail_level: DetailLevel enum or bool for backward compat.
            - COMPACT: one-liner with ✓/✗
            - STANDARD: 10 lines
            - EXPANDED: all lines
    """
    from ..dtypes import DetailLevel

    # Handle backward compat: bool -> DetailLevel
    if isinstance(detail_level, bool):
        detail_level = DetailLevel.EXPANDED if detail_level else DetailLevel.STANDARD

    path = shorten_path(args.get("file_path") or args.get("path") or "")
    file_content = args.get("content", "")
    lines = file_content.split("\n") if file_content else []
    total_lines = len(lines)

    header = f"write(file_path={repr(path if path else '...')})"

    # Compact mode: one-liner with success/fail indicator
    if detail_level == DetailLevel.COMPACT:
        if result is None:
            return header
        is_error = result.get("isError", False)
        indicator = "✗" if is_error else "✓"
        return f"{header} {indicator}"

    if not file_content:
        return header

    # Standard vs Expanded line counts
    if detail_level == DetailLevel.EXPANDED:
        max_lines = len(lines)
    else:
        max_lines = 10

    display_lines = lines[:max_lines]
    remaining = len(lines) - max_lines

    text = header
    text += f"\n⎿ Wrote {total_lines} line{'s' if total_lines != 1 else ''} to {path or '...'}"

    # Format with line numbers
    line_num_width = len(str(total_lines))
    for i, line in enumerate(display_lines, start=1):
        line_num = str(i).rjust(line_num_width)
        formatted = f"{line_num}   {replace_tabs(line)}"
        text += "\n  " + (theme.diff_context_fg(formatted) if theme else formatted)

    if remaining > 0:
        text += f"\n  ... ({remaining} more lines)"

    return text


def format_edit(
    tool_name: str,
    args: dict,
    result: dict | None,
    detail_level: DetailLevel | bool,
    theme: Theme | None = None,
) -> str:
    """Format edit tool execution with colored diff."""
    from ..dtypes import DetailLevel

    # Handle backward compat: bool -> DetailLevel
    if isinstance(detail_level, bool):
        detail_level = DetailLevel.EXPANDED if detail_level else DetailLevel.STANDARD

    path = shorten_path(args.get("file_path") or args.get("path") or "")

    header = f"edit(file_path={repr(path if path else '...')}, ...)"

    # Compact mode: one-liner with success/fail indicator
    if detail_level == DetailLevel.COMPACT:
        if result is None:
            return header
        is_error = result.get("isError", False)
        indicator = "✗" if is_error else "✓"
        return f"{header} {indicator}"

    text = header

    if result:
        # Check for diff in details
        # New structure: {"content": [...], "details": {...}, "isError": bool}
        # Legacy structure: {"content": {"content": [...], "details": {...}}, "isError": bool}
        details = result.get("details", {})
        if not details:
            # Try legacy structure
            content = result.get("content", {})
            details = content.get("details", {}) if isinstance(content, dict) else {}
        diff_str = details.get("diff") if details else None

        is_error = result.get("isError", False)

        if diff_str and theme:
            # Count additions and removals using regex to avoid false matches
            import re

            diff_lines = diff_str.split("\n")
            additions = sum(1 for line in diff_lines if re.match(r"^\s*\d+\s+\+\s", line))
            removals = sum(1 for line in diff_lines if re.match(r"^\s*\d+\s+-\s", line))

            # Build summary like "Updated file.py with 2 additions and 1 removal"
            if is_error:
                summary = "Edit failed"
            else:
                parts = []
                if additions:
                    parts.append(f"{additions} addition{'s' if additions != 1 else ''}")
                if removals:
                    parts.append(f"{removals} removal{'s' if removals != 1 else ''}")
                if parts:
                    summary = f"Updated {path or '...'} with {' and '.join(parts)}"
                else:
                    summary = f"Updated {path or '...'}"

            # Render colored diff
            text += f"\n⎿ {summary}"

            for line in diff_lines:
                # Format: "  607 - content" or "  607 + content" or "  607   content"
                # The marker (-, +, or space) is right after the line number
                # We need to check character positions, not just substring match
                import re

                # Match line number followed by marker (-, +, or spaces)
                match = re.match(r"^(\s*\d+)\s+([-+])\s", line)
                if match and match.group(2) == "-":
                    text += "\n  " + theme.diff_removed_fg(line)
                elif match and match.group(2) == "+":
                    text += "\n  " + theme.diff_added_fg(line)
                else:
                    text += "\n  " + theme.diff_context_fg(line)
        elif diff_str:
            # No theme - plain diff
            summary = "Edit failed" if is_error else f"Updated {path or '...'}"
            text += f"\n⎿ {summary}"
            for line in diff_str.split("\n"):
                text += "\n  " + line
        else:
            # Fallback to plain output
            summary = "Edit failed" if is_error else f"Updated {path or '...'}"
            output = get_text_output(result)
            if output:
                text += f"\n⎿ {summary}"
                for line in output.split("\n"):
                    text += "\n  " + line

    return text


def format_web_fetch(
    tool_name: str,
    args: dict,
    result: dict | None,
    detail_level: DetailLevel | bool,
    theme: Theme | None = None,
) -> str:
    """Format web_fetch tool execution."""
    from ..dtypes import DetailLevel

    # Handle backward compat: bool -> DetailLevel
    if isinstance(detail_level, bool):
        detail_level = DetailLevel.EXPANDED if detail_level else DetailLevel.STANDARD

    url = args.get("url", "")
    _prompt = args.get("prompt", "")  # Available for future display use

    # Shorten URL for display
    try:
        parsed = urlparse(url)
        display_url = (
            f"{parsed.netloc}{parsed.path[:30]}..."
            if len(parsed.path) > 30
            else f"{parsed.netloc}{parsed.path}"
        )
    except Exception:
        display_url = url[:50] + "..." if len(url) > 50 else url

    header = f"web_fetch(url={repr(display_url)})"

    # Compact mode: one-liner with success/fail indicator
    if detail_level == DetailLevel.COMPACT:
        if result is None:
            return header
        is_error = result.get("isError", False)
        indicator = "✗" if is_error else "✓"
        return f"{header} {indicator}"

    text = header

    if result:
        output = get_text_output(result).strip()
        is_error = result.get("isError", False)

        if is_error:
            text += "\n⎿ Fetch failed"
            if output:
                text += f": {output[:100]}"
        else:
            # Show truncated response
            lines = output.split("\n") if output else []
            expanded = detail_level == DetailLevel.EXPANDED
            max_lines = len(lines) if expanded else 8
            display_lines = lines[:max_lines]
            remaining = len(lines) - max_lines

            text += "\n⎿ Fetched and processed"
            for line in display_lines:
                text += "\n  " + line[:120]
            if remaining > 0:
                text += f"\n  ... ({remaining} more lines)"

    return text


def compute_edit_line_range(
    old_content: str, new_content: str, old_text: str, new_text: str
) -> tuple[int, int]:
    """Compute the line range affected by a text replacement.

    Returns (start_line, end_line) where lines are 1-indexed and inclusive.
    The range covers the lines that were modified in the new content.
    """
    # Find where the replacement occurred
    replace_start = old_content.find(old_text)
    assert replace_start >= 0, "old_text must exist in old_content"

    # Count lines before the replacement to get start line
    lines_before = old_content[:replace_start].count("\n")
    start_line = lines_before + 1  # 1-indexed

    # Count lines in the new text to get end line
    new_text_lines = new_text.count("\n") + 1
    end_line = start_line + new_text_lines - 1

    return start_line, end_line


def generate_diff(old_content: str, new_content: str, context_lines: int = 3) -> str:
    """Generate unified diff string with line numbers in gutter.

    Args:
        old_content: Original file content
        new_content: New file content
        context_lines: Number of context lines to show around changes

    Returns:
        Diff string formatted as (line number in gutter, marker after):
             605                    tool_call,
             606                    current_state,
             607 -                  None,
             607                    cancel_scope=rcfg.cancel_scope,
             608                )
    """
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")

    # Simple line-by-line diff
    output = []
    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))

    # For simplicity, use a basic diff algorithm
    # Find common prefix and suffix
    i = 0
    while i < len(old_lines) and i < len(new_lines) and old_lines[i] == new_lines[i]:
        i += 1

    j_old = len(old_lines) - 1
    j_new = len(new_lines) - 1
    while j_old >= i and j_new >= i and old_lines[j_old] == new_lines[j_new]:
        j_old -= 1
        j_new -= 1

    # Show context before changes
    context_start = max(0, i - context_lines)
    if context_start > 0:
        output.append("     ...")

    for line_idx in range(context_start, i):
        line_num = str(line_idx + 1).rjust(line_num_width)
        output.append(f"{line_num}   {old_lines[line_idx]}")

    # Show removed lines (use old line numbers)
    for line_idx in range(i, j_old + 1):
        line_num = str(line_idx + 1).rjust(line_num_width)
        output.append(f"{line_num} - {old_lines[line_idx]}")

    # Show added lines (use new line numbers, continuing from where removed ended)
    new_line_start = i
    for idx, line_idx in enumerate(range(i, j_new + 1)):
        line_num = str(new_line_start + idx + 1).rjust(line_num_width)
        output.append(f"{line_num} + {new_lines[line_idx]}")

    # Show context after changes (use new line numbers)
    context_end = min(len(new_lines), j_new + 2 + context_lines)
    for line_idx in range(j_new + 1, context_end):
        line_num = str(line_idx + 1).rjust(line_num_width)
        output.append(f"{line_num}   {new_lines[line_idx]}")

    if context_end < len(new_lines):
        output.append("     ...")

    return "\n".join(output)


# Tool preset configurations
TOOL_PRESETS = {
    "full": ["read", "write", "edit", "bash", "web_fetch"],
    "readonly": ["read"],
    "no-write": ["read", "edit", "bash", "web_fetch"],
}


# Threshold for AI summarization (characters) - below this, raw content is fine
WEB_FETCH_SUMMARIZE_THRESHOLD = 5000


async def _summarize_content(
    content: str,
    prompt: str,
    provider: str,
    model: str,
) -> tuple[str | None, str | None]:
    """Summarize web content using a small model. Pure function.

    Returns (summary, None) on success, (None, error) on failure.
    """
    try:
        if provider == "anthropic":
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": f"Extract the relevant information from this web page based on the following prompt. Be concise but complete.\n\nPrompt: {prompt}\n\n---\n\nWeb content:\n{content}",
                    }
                ],
            )
            await client.close()
            text_block = response.content[0]
            assert text_block.type == "text"
            return text_block.text, None

        elif provider == "openai":
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            response = await client.chat.completions.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": f"Extract the relevant information from this web page based on the following prompt. Be concise but complete.\n\nPrompt: {prompt}\n\n---\n\nWeb content:\n{content}",
                    }
                ],
            )
            await client.aclose()
            return response.choices[0].message.content, None

        elif provider == "google":
            import google.generativeai as genai

            model_client = genai.GenerativeModel(model)
            response = await model_client.generate_content_async(
                f"Extract the relevant information from this web page based on the following prompt. Be concise but complete.\n\nPrompt: {prompt}\n\n---\n\nWeb content:\n{content}"
            )
            return response.text, None

        else:
            return None, f"Unknown summarizer provider: {provider}"

    except Exception as e:
        return None, f"Summarization failed: {e}"


@dataclass
class LocalWorkspaceResource:
    working_dir: str
    path_resolver: Callable[[str], Path] = field(default=expand_path, repr=False)

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        candidate = path
        if not (path == "~" or path.startswith("~/") or os.path.isabs(path)):
            candidate = str(Path(current_working_dir) / path)
        return str(self.path_resolver(candidate))

    async def read_file(self, path: str) -> bytes:
        return Path(path).read_bytes()

    async def write_file(self, path: str, content: bytes) -> None:
        abs_path = Path(path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(content)


class LocalCommandRunner:
    async def run(
        self,
        command: str,
        *,
        cwd: str,
        timeout: float,
        session_id: str | None = None,
        cancel_scope: Any | None = None,
    ) -> CommandExecutionResult:
        from ._subprocess import run_command

        returncode, stdout, stderr = await run_command(command, cwd=cwd, timeout=timeout)
        return CommandExecutionResult(returncode=returncode, stdout=stdout, stderr=stderr, cwd=cwd)


class CodingEnvironment:
    """Generic coding environment over injected workspace and command backends."""

    def __init__(
        self,
        workspace: CodingWorkspaceResource,
        command_runner: CommandRunner,
        *,
        tools: str | list[str] = "full",
        bash_allowlist: list[str] | None = None,
        summarize_web_fetch: bool = True,
        summarizer_provider: str = "anthropic",
        summarizer_model: str = "claude-3-5-haiku-latest",
        output_dir: Path = DEFAULT_TOOL_OUTPUT_DIR,
        current_working_dir: str | None = None,
        web_fetch_cache: dict[str, tuple[float, dict[str, Any]]] | None = None,
        content_summarizer: Callable[
            [str, str, str, str],
            Awaitable[tuple[str | None, str | None]],
        ] = _summarize_content,
    ) -> None:
        self.workspace = workspace
        self.command_runner = command_runner
        self.tools = tools
        self.bash_allowlist = bash_allowlist
        self.summarize_web_fetch = summarize_web_fetch
        self.summarizer_provider = summarizer_provider
        self.summarizer_model = summarizer_model
        self.output_dir = output_dir
        self.current_working_dir = current_working_dir or workspace.working_dir
        self.web_fetch_cache = web_fetch_cache or {}
        self.content_summarizer = content_summarizer

        if isinstance(self.tools, str):
            if self.tools not in TOOL_PRESETS:
                raise ValueError(
                    f"Unknown tool preset: {self.tools}. Available: {list(TOOL_PRESETS.keys())}"
                )
            self._tool_filter = TOOL_PRESETS[self.tools]
        else:
            self._tool_filter = self.tools

    def get_name(self) -> str:
        return "coding"

    def get_status_info(self) -> dict[str, str] | None:
        cwd = self.current_working_dir
        home = os.path.expanduser("~")
        if cwd.startswith(home):
            cwd = "~" + cwd[len(home) :]
        return {"cwd": cwd}

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        return tool_call.name == "bash"

    def get_tool_render_config(self, tool_name: str) -> ToolRenderConfig | None:
        configs: dict[str, ToolRenderConfig] = {
            "bash": BASH_RENDER_CONFIG,
            "read": ToolRenderConfig(custom_formatter=format_read),
            "write": ToolRenderConfig(custom_formatter=format_write),
            "edit": ToolRenderConfig(custom_formatter=format_edit),
            "web_fetch": ToolRenderConfig(custom_formatter=format_web_fetch),
        }
        return configs.get(tool_name)

    def get_tool_formatter(
        self, tool_name: str
    ) -> Callable[[str, dict, dict | None, bool, Theme | None], str] | None:
        config = self.get_tool_render_config(tool_name)
        if config and config.custom_formatter:
            return config.custom_formatter
        if config:
            return lambda name, args, result, expanded, theme: format_tool(
                name, args, result, expanded, theme, config
            )
        return None

    def get_tools(self) -> list[Tool]:
        all_tools = self._get_all_tools() + self._get_extra_tools()
        return [t for t in all_tools if t.function.name in self._tool_filter]

    def _get_bash_description(self) -> str:
        base = "Execute a bash command in the current working directory. Returns stdout and stderr."
        if self.bash_allowlist:
            allowed = ", ".join(f"'{p}'" for p in self.bash_allowlist)
            return f"{base} RESTRICTION: Only commands starting with: {allowed}"
        return base

    def _get_all_tools(self) -> list[Tool]:
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="read",
                    description="Read the contents of a file. Defaults to first 2000 lines. Use offset/limit for large files.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read (relative or absolute)",
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Line number to start reading from (1-indexed)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of lines to read",
                            },
                        },
                    ),
                    required=["path"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="write",
                    description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "path": {
                                "type": "string",
                                "description": "Path to the file to write (relative or absolute)",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file",
                            },
                        },
                    ),
                    required=["path", "content"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="edit",
                    description="Edit a file by replacing exact text. The old_text must match exactly (including whitespace). Use this for precise, surgical edits.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "path": {
                                "type": "string",
                                "description": "Path to the file to edit (relative or absolute)",
                            },
                            "old_text": {
                                "type": "string",
                                "description": "Exact text to find and replace (must match exactly)",
                            },
                            "new_text": {
                                "type": "string",
                                "description": "New text to replace the old text with",
                            },
                        },
                    ),
                    required=["path", "old_text", "new_text"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="bash",
                    description=self._get_bash_description(),
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "command": {"type": "string", "description": "Bash command to execute"},
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (default: 120)",
                            },
                        },
                    ),
                    required=["command"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="web_fetch",
                    description="Fetch content from a URL and extract information. Converts HTML to markdown. Use this to read documentation, articles, or any web content.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "url": {
                                "type": "string",
                                "description": "The URL to fetch (must be valid http/https URL)",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "What information to extract from the page (e.g., 'summarize this article', 'find the API endpoints')",
                            },
                        },
                    ),
                    required=["url", "prompt"],
                ),
            ),
        ]

    def _get_extra_tools(self) -> list[Tool]:
        return []

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        try:
            if tool_call.name == "read":
                return await self._exec_read(tool_call)
            if tool_call.name == "write":
                return await self._exec_write(tool_call)
            if tool_call.name == "edit":
                return await self._exec_edit(tool_call)
            if tool_call.name == "bash":
                session_id = current_state.session_id if current_state is not None else None
                return await self._exec_bash(tool_call, session_id, cancel_scope)
            if tool_call.name == "web_fetch":
                session_id = current_state.session_id if current_state is not None else None
                return await self._exec_web_fetch(tool_call, session_id)

            extra = await self._exec_extra_tool(tool_call)
            if extra is not None:
                return extra
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Unknown tool: {tool_call.name}",
            )
        except trio.Cancelled:
            raise
        except Exception as e:
            return ToolResult(tool_call_id=tool_call.id, is_error=True, content="", error=str(e))

    async def _exec_extra_tool(self, tool_call: ToolCall) -> ToolResult | None:
        return None

    async def _exec_read(self, tool_call: ToolCall) -> ToolResult:
        path_str = tool_call.args["path"]
        offset = tool_call.args.get("offset")
        limit = tool_call.args.get("limit")
        abs_path = self.workspace.resolve_path(self.current_working_dir, path_str)

        try:
            content = (await self.workspace.read_file(abs_path)).decode("utf-8")
        except FileNotFoundError:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"File not found: {path_str}",
            )
        except IsADirectoryError:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Not a file: {path_str}",
            )
        except UnicodeDecodeError:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Cannot read binary file: {path_str}",
            )

        lines = content.split("\n")
        start_line = (offset - 1) if offset else 0
        max_lines = limit or MAX_LINES
        end_line = min(start_line + max_lines, len(lines))
        if start_line >= len(lines):
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Offset {offset} is beyond end of file ({len(lines)} lines total)",
            )

        selected_lines = lines[start_line:end_line]
        had_truncated = False
        formatted_lines = []
        for line in selected_lines:
            if len(line) > MAX_LINE_LENGTH:
                had_truncated = True
                formatted_lines.append(line[:MAX_LINE_LENGTH])
            else:
                formatted_lines.append(line)

        output_text = "\n".join(formatted_lines)
        notices = []
        if had_truncated:
            notices.append(f"Some lines were truncated to {MAX_LINE_LENGTH} characters")
        if end_line < len(lines):
            remaining = len(lines) - end_line
            notices.append(
                f"{remaining} more lines not shown. Use offset={end_line + 1} to continue"
            )
        if notices:
            output_text += f"\n\n... ({'. '.join(notices)})"

        return ToolResult(tool_call_id=tool_call.id, is_error=False, content=output_text)

    async def _exec_write(self, tool_call: ToolCall) -> ToolResult:
        path_str = tool_call.args["path"]
        content = tool_call.args["content"]
        abs_path = self.workspace.resolve_path(self.current_working_dir, path_str)
        is_create = True
        try:
            await self.workspace.read_file(abs_path)
            is_create = False
        except (FileNotFoundError, IsADirectoryError):
            is_create = True

        await self.workspace.write_file(abs_path, content.encode("utf-8"))
        line_count = content.count("\n") + 1
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Successfully wrote {len(content)} bytes to {path_str}",
            details={
                "file_path": abs_path,
                "start_line": 1,
                "end_line": line_count,
                "operation": "create" if is_create else "edit",
            },
        )

    async def _exec_edit(self, tool_call: ToolCall) -> ToolResult:
        path_str = tool_call.args["path"]
        old_text = tool_call.args["old_text"]
        new_text = tool_call.args["new_text"]
        abs_path = self.workspace.resolve_path(self.current_working_dir, path_str)

        try:
            content = (await self.workspace.read_file(abs_path)).decode("utf-8")
        except FileNotFoundError:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"File not found: {path_str}",
            )
        except IsADirectoryError:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Not a file: {path_str}",
            )
        except UnicodeDecodeError:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Cannot read binary file: {path_str}",
            )

        if old_text not in content:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Could not find the exact text in {path_str}. The old text must match exactly including all whitespace and newlines.",
            )

        occurrences = content.count(old_text)
        if occurrences > 1:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Found {occurrences} occurrences of the text in {path_str}. The text must be unique. Please provide more context to make it unique.",
            )

        index = content.find(old_text)
        new_content = content[:index] + new_text + content[index + len(old_text) :]
        if content == new_content:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"No changes made to {path_str}. The replacement produced identical content.",
            )

        await self.workspace.write_file(abs_path, new_content.encode("utf-8"))
        diff_str = generate_diff(content, new_content)
        start_line, end_line = compute_edit_line_range(content, new_content, old_text, new_text)
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Successfully replaced text in {path_str}. Changed {len(old_text)} characters to {len(new_text)} characters.",
            details={
                "diff": diff_str,
                "file_path": abs_path,
                "start_line": start_line,
                "end_line": end_line,
                "operation": "edit",
            },
        )

    def _check_bash_allowlist(self, command: str) -> str | None:
        if self.bash_allowlist is None:
            return None
        cmd = command.lstrip()
        for prefix in self.bash_allowlist:
            if cmd.startswith(prefix):
                return None
        allowed_str = ", ".join(f"'{p}'" for p in self.bash_allowlist)
        return (
            f"Command not allowed. This environment only permits commands starting with: {allowed_str}\n"
            f"Attempted command: {command[:100]}{'...' if len(command) > 100 else ''}"
        )

    async def _exec_bash(
        self,
        tool_call: ToolCall,
        session_id: str | None = None,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        command = tool_call.args["command"]
        timeout = tool_call.args.get("timeout", 120)
        if error := self._check_bash_allowlist(command):
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=error,
            )

        try:
            result = await self.command_runner.run(
                command,
                cwd=self.current_working_dir,
                timeout=timeout,
                session_id=session_id,
                cancel_scope=cancel_scope,
            )
            if result.cwd:
                self.current_working_dir = result.cwd

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += result.stderr

            output_file_path: str | None = None
            if len(output) > MAX_OUTPUT_SIZE:
                output_file_path = self._write_large_output(output, tool_call.id, session_id)
                total_lines = output.count("\n") + 1
                total_kb = len(output) // 1024
                preview_size = MAX_OUTPUT_SIZE // 2
                head = output[:preview_size]
                tail = output[-preview_size:]
                output = (
                    f"{head}\n\n"
                    f"... [{total_kb}KB total, {total_lines} lines - full output saved to file]\n\n"
                    f"... (last {preview_size // 1024}KB of output):\n\n"
                    f"{tail}\n\n"
                    f"Full output: {output_file_path}\n"
                    f"Use `read path={output_file_path}` to see specific sections, "
                    f"or `bash command='tail -100 {output_file_path}'` to see the end."
                )

            if result.returncode != 0:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content=output or "(no output)",
                    error=f"Command exited with code {result.returncode}",
                    details={"output_file": output_file_path} if output_file_path else None,
                )

            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=False,
                content=output or "(no output)",
                details={"output_file": output_file_path} if output_file_path else None,
            )
        except TimeoutError:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Command timed out after {timeout} seconds",
            )
        except trio.Cancelled:
            raise

    def _write_large_output(self, output: str, tool_call_id: str, session_id: str | None) -> str:
        session_dir = session_id or "anonymous"
        output_dir = self.output_dir / session_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(c for c in tool_call_id if c.isalnum() or c in "-_")[:64]
        if not safe_id:
            import uuid

            safe_id = str(uuid.uuid4())
        output_file = output_dir / f"{safe_id}.txt"
        resolved = output_file.resolve()
        assert resolved.is_relative_to(output_dir.resolve()), "Path traversal detected"
        output_file.write_text(output, encoding="utf-8")
        return str(output_file)

    async def _exec_web_fetch(
        self, tool_call: ToolCall, session_id: str | None = None
    ) -> ToolResult:
        import time

        url = tool_call.args["url"]
        prompt = tool_call.args["prompt"]

        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Invalid URL scheme: {parsed.scheme}. Must be http or https.",
                )
            if not parsed.netloc:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error="Invalid URL: missing hostname",
                )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id, is_error=True, content="", error=f"Invalid URL: {e}"
            )

        original_host = parsed.netloc
        if parsed.scheme == "http":
            url = url.replace("http://", "https://", 1)

        now = time.time()
        if url in self.web_fetch_cache:
            cached_time, cached_result = self.web_fetch_cache[url]
            if now - cached_time < WEB_FETCH_CACHE_TTL:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=False,
                    content=f"[Cached] URL: {url}\nPrompt: {prompt}\n\n---\n\n{cached_result['content']}",
                )

        try:
            async with httpx.AsyncClient(
                timeout=WEB_FETCH_TIMEOUT,
                follow_redirects=False,
            ) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; rollouts/1.0)",
                        "Accept": "text/html, text/markdown, */*",
                    },
                )
                redirect_count = 0
                while response.is_redirect and redirect_count < 5:
                    redirect_url = response.headers.get("location", "")
                    if not redirect_url:
                        break
                    if redirect_url.startswith("/"):
                        redirect_url = f"https://{urlparse(str(response.url)).netloc}{redirect_url}"
                    redirect_parsed = urlparse(redirect_url)
                    redirect_host = redirect_parsed.netloc
                    if redirect_host and redirect_host != original_host:
                        return ToolResult(
                            tool_call_id=tool_call.id,
                            is_error=False,
                            content=f"Redirect detected: {url} redirects to a different host.\n\nRedirect URL: {redirect_url}\n\nPlease make a new web_fetch request with this URL if you want to follow the redirect.",
                        )
                    response = await client.get(
                        redirect_url,
                        headers={
                            "User-Agent": "Mozilla/5.0 (compatible; rollouts/1.0)",
                            "Accept": "text/html, text/markdown, */*",
                        },
                    )
                    redirect_count += 1
                response.raise_for_status()
        except httpx.TimeoutException:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Request timed out after {WEB_FETCH_TIMEOUT} seconds",
            )
        except httpx.HTTPStatusError as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            )
        except httpx.RequestError as e:
            return ToolResult(
                tool_call_id=tool_call.id, is_error=True, content="", error=f"Request failed: {e}"
            )

        content_length = len(response.content)
        if content_length > WEB_FETCH_MAX_SIZE:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Content too large: {content_length} bytes (max {WEB_FETCH_MAX_SIZE})",
            )

        try:
            text = response.text
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Failed to decode response: {e}",
            )

        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            try:
                text = markdownify.markdownify(text, heading_style="ATX", strip=["script", "style"])
            except Exception:
                pass

        self.web_fetch_cache[url] = (now, {"content": text, "status": response.status_code})
        for cached_url in list(self.web_fetch_cache.keys()):
            cached_time, _ = self.web_fetch_cache[cached_url]
            if now - cached_time > WEB_FETCH_CACHE_TTL:
                del self.web_fetch_cache[cached_url]

        final_content = text
        summarized = False
        if self.summarize_web_fetch and len(text) > WEB_FETCH_SUMMARIZE_THRESHOLD and prompt:
            summary, _error = await self.content_summarizer(
                text, prompt, self.summarizer_provider, self.summarizer_model
            )
            if summary:
                final_content = summary
                summarized = True

        header = f"URL: {url}"
        if summarized:
            header += f"\n[Summarized by {self.summarizer_model}]"
        header += f"\nPrompt: {prompt}\n\n---\n\n"

        output_file_path: str | None = None
        if len(final_content) > WEB_FETCH_MAX_CONTENT:
            output_file_path = self._write_large_output(
                header + final_content, tool_call.id, session_id
            )
            total_kb = len(final_content) // 1024
            preview_size = WEB_FETCH_MAX_CONTENT // 2
            head = final_content[:preview_size]
            tail = final_content[-preview_size:]
            final_content = (
                f"{head}\n\n"
                f"... [{total_kb}KB total - full content saved to file]\n\n"
                f"... (last {preview_size // 1024}KB):\n\n"
                f"{tail}\n\n"
                f"Full content: {output_file_path}\n"
                f"Use `read path={output_file_path}` to see specific sections."
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=header + final_content,
            details={"output_file": output_file_path} if output_file_path else None,
        )


@dataclass
class LocalFilesystemEnvironment(CodingEnvironment):
    """Local filesystem assembly for the generic coding environment."""

    working_dir: Path = field(default_factory=Path.cwd)
    tools: str | list[str] = "full"
    bash_allowlist: list[str] | None = None
    summarize_web_fetch: bool = True
    summarizer_provider: str = "anthropic"
    summarizer_model: str = "claude-3-5-haiku-latest"
    output_dir: Path = field(default_factory=lambda: DEFAULT_TOOL_OUTPUT_DIR)
    path_resolver: Callable[[str], Path] = field(default=expand_path, repr=False)
    web_fetch_cache: dict[str, tuple[float, dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )
    content_summarizer: Callable[
        [str, str, str, str],
        Awaitable[tuple[str | None, str | None]],
    ] = field(default=_summarize_content, repr=False)

    def __post_init__(self) -> None:
        workspace = LocalWorkspaceResource(
            working_dir=str(self.working_dir),
            path_resolver=self.path_resolver,
        )
        CodingEnvironment.__init__(
            self,
            workspace,
            LocalCommandRunner(),
            tools=self.tools,
            bash_allowlist=self.bash_allowlist,
            summarize_web_fetch=self.summarize_web_fetch,
            summarizer_provider=self.summarizer_provider,
            summarizer_model=self.summarizer_model,
            output_dir=self.output_dir,
            current_working_dir=str(self.working_dir),
            web_fetch_cache=self.web_fetch_cache,
            content_summarizer=self.content_summarizer,
        )

    async def serialize(self) -> dict:
        return {
            "env_kind": "coding",
            "working_dir": self.current_working_dir,
            "tools": self.tools,
            "bash_allowlist": self.bash_allowlist,
            "summarize_web_fetch": self.summarize_web_fetch,
            "summarizer_provider": self.summarizer_provider,
            "summarizer_model": self.summarizer_model,
            "output_dir": str(self.output_dir),
        }

    @staticmethod
    async def deserialize(data: dict) -> LocalFilesystemEnvironment:
        return LocalFilesystemEnvironment(
            working_dir=Path(data["working_dir"]),
            tools=data.get("tools", "full"),
            bash_allowlist=data.get("bash_allowlist"),
            summarize_web_fetch=data.get("summarize_web_fetch", True),
            summarizer_provider=data.get("summarizer_provider", "anthropic"),
            summarizer_model=data.get("summarizer_model", "claude-3-5-haiku-latest"),
            output_dir=Path(data.get("output_dir", DEFAULT_TOOL_OUTPUT_DIR)),
        )
