"""
Coding environment with minimal toolset for code editing tasks.

Tools: read, write, edit, bash
Inspired by pi-mono's minimalist approach.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
import markdownify
import trio

from ..dtypes import (
    AgentState,
    Message,
    RunConfig,
    Tool,
    ToolCall,
    ToolFunction,
    ToolFunctionParameter,
    ToolResult,
)

MAX_LINES = 2000
MAX_LINE_LENGTH = 2000
MAX_OUTPUT_SIZE = 30_000  # 30KB (matches Claude Code's default)

# Web fetch constants
WEB_FETCH_MAX_SIZE = 10 * 1024 * 1024  # 10MB max download
WEB_FETCH_MAX_CONTENT = 100_000  # 100KB max content after conversion
WEB_FETCH_CACHE_TTL = 900  # 15 minutes
WEB_FETCH_TIMEOUT = 30  # seconds

# Simple in-memory cache for web fetches
_web_fetch_cache: dict[str, tuple[float, dict]] = {}


def expand_path(file_path: str) -> Path:
    """Expand ~ to home directory and resolve path."""
    if file_path == "~":
        return Path.home()
    if file_path.startswith("~/"):
        return Path.home() / file_path[2:]
    return Path(file_path).resolve()


# ── Tool Formatting Utilities ─────────────────────────────────────────────────


def _shorten_path(path: str) -> str:
    """Convert absolute path to tilde notation if in home directory."""
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def _replace_tabs(text: str) -> str:
    """Replace tabs with spaces for consistent rendering."""
    return text.replace("\t", "   ")


def _get_text_output(result: dict | None) -> str:
    """Extract text output from tool result.

    Result structure (after fix): {"content": [{"type": "text", "text": "..."}], "details": {...}, "isError": bool}
    Legacy structure: {"content": {"content": [...]}, "isError": bool}
    """
    if not result:
        return ""

    content = result.get("content", {})

    # If content is a string, return it directly
    if isinstance(content, str):
        return content

    # If content is a list, extract text blocks directly
    if isinstance(content, list):
        text_blocks = [c for c in content if isinstance(c, dict) and c.get("type") == "text"]
        text_output = "\n".join(c.get("text", "") for c in text_blocks if c.get("text"))

        # Strip ANSI codes and carriage returns
        import re

        text_output = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text_output)
        text_output = text_output.replace("\r", "")
        return text_output

    # Legacy: If content is a dict with a "content" key, extract from that
    if isinstance(content, dict):
        content_list = content.get("content", [])
        if isinstance(content_list, list):
            text_blocks = [
                c for c in content_list if isinstance(c, dict) and c.get("type") == "text"
            ]
            text_output = "\n".join(c.get("text", "") for c in text_blocks if c.get("text"))

            # Strip ANSI codes and carriage returns
            import re

            text_output = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text_output)
            text_output = text_output.replace("\r", "")
            return text_output

    return ""


# ── Tool Formatters ───────────────────────────────────────────────────────────
# These format tool calls for display in the TUI.
# Signature: (tool_name, args, result, expanded, theme) -> str


def format_bash(tool_name: str, args: dict, result: dict | None, expanded: bool, theme=None) -> str:
    """Format bash tool execution."""
    command = args.get("command", "")
    text = f"bash(command={repr(command or '...')})"

    if result:
        output = _get_text_output(result).strip()
        if output:
            lines = output.split("\n")
            max_lines = len(lines) if expanded else 5
            display_lines = lines[:max_lines]
            remaining = len(lines) - max_lines

            is_error = result.get("isError", False)
            summary = "Command failed" if is_error else "Command completed"
            text += f"\n⎿ {summary}"
            for line in display_lines:
                # Use base gray color (like diff context) for bash output
                if theme:
                    text += "\n  " + theme.diff_context_fg(line)
                else:
                    text += "\n  " + line
            if remaining > 0:
                text += f"\n  ... ({remaining} more lines)"

    return text


def format_read(tool_name: str, args: dict, result: dict | None, expanded: bool, theme=None) -> str:
    """Format read tool execution."""
    path = _shorten_path(args.get("file_path") or args.get("path") or "")
    offset = args.get("offset")
    limit = args.get("limit")

    params = f"file_path={repr(path if path else '...')}"
    if offset is not None:
        params += f", offset={offset}"
    if limit is not None:
        params += f", limit={limit}"

    text = f"read({params})"

    if result:
        output = _get_text_output(result)
        lines = output.split("\n")
        total_lines = len(lines)
        summary = f"Read {total_lines} line{'s' if total_lines != 1 else ''}"
        text += f"\n⎿ {summary}"
        # Lines commented out - just show summary for now
        # max_lines = len(lines) if expanded else 10
        # display_lines = lines[:max_lines]
        # remaining = len(lines) - max_lines
        # for line in display_lines:
        #     text += "\n  " + _replace_tabs(line)
        # if remaining > 0:
        #     text += f"\n  ... ({remaining} more lines)"

    return text


def format_write(
    tool_name: str, args: dict, result: dict | None, expanded: bool, theme=None
) -> str:
    """Format write tool execution with line numbers and gray styling (like edit)."""
    path = _shorten_path(args.get("file_path") or args.get("path") or "")
    file_content = args.get("content", "")
    lines = file_content.split("\n") if file_content else []
    total_lines = len(lines)

    text = f"write(file_path={repr(path if path else '...')})"

    if file_content:
        max_lines = len(lines) if expanded else 10
        display_lines = lines[:max_lines]
        remaining = len(lines) - max_lines

        summary = f"Wrote {total_lines} line{'s' if total_lines != 1 else ''} to {path or '...'}"
        text += f"\n⎿ {summary}"

        # Calculate line number width for alignment
        line_num_width = len(str(total_lines))

        for i, line in enumerate(display_lines, start=1):
            line_num = str(i).rjust(line_num_width)
            formatted_line = f"{line_num}   {_replace_tabs(line)}"
            # Use gray color (like diff context) for consistent styling
            if theme:
                text += "\n  " + theme.diff_context_fg(formatted_line)
            else:
                text += "\n  " + formatted_line

        if remaining > 0:
            text += f"\n  ... ({remaining} more lines)"

    return text


def format_edit(tool_name: str, args: dict, result: dict | None, expanded: bool, theme=None) -> str:
    """Format edit tool execution with colored diff."""
    path = _shorten_path(args.get("file_path") or args.get("path") or "")

    text = f"edit(file_path={repr(path if path else '...')}, old_string=..., new_string=...)"

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
            output = _get_text_output(result)
            if output:
                text += f"\n⎿ {summary}"
                for line in output.split("\n"):
                    text += "\n  " + line

    return text


def format_web_fetch(
    tool_name: str, args: dict, result: dict | None, expanded: bool, theme=None
) -> str:
    """Format web_fetch tool execution."""
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

    text = f"web_fetch(url={repr(display_url)})"

    if result:
        output = _get_text_output(result).strip()
        is_error = result.get("isError", False)

        if is_error:
            text += "\n⎿ Fetch failed"
            if output:
                text += f": {output[:100]}"
        else:
            # Show truncated response
            lines = output.split("\n") if output else []
            max_lines = len(lines) if expanded else 8
            display_lines = lines[:max_lines]
            remaining = len(lines) - max_lines

            text += "\n⎿ Fetched and processed"
            for line in display_lines:
                text += "\n  " + line[:120]
            if remaining > 0:
                text += f"\n  ... ({remaining} more lines)"

    return text


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


@dataclass
class LocalFilesystemEnvironment:
    """Local filesystem environment with read, write, edit, bash tools.

    Args:
        working_dir: Working directory for file operations and bash commands
        tools: Tool filter - either a preset name ("full", "readonly", "no-write")
               or a list of tool names (e.g., ["read", "edit"]). Defaults to "full".
    """

    working_dir: Path = field(default_factory=Path.cwd)
    tools: str | list[str] = "full"

    def __post_init__(self):
        # Resolve preset name to tool list
        if isinstance(self.tools, str):
            if self.tools in TOOL_PRESETS:
                self._tool_filter = TOOL_PRESETS[self.tools]
            else:
                raise ValueError(
                    f"Unknown tool preset: {self.tools}. Available: {list(TOOL_PRESETS.keys())}"
                )
        else:
            self._tool_filter = self.tools

    def get_name(self) -> str:
        """Return environment name identifier."""
        return "coding"

    def get_status_info(self) -> dict[str, str] | None:
        """Return cwd for status line display."""
        cwd = str(self.working_dir)
        # Shorten home directory to ~
        home = os.path.expanduser("~")
        if cwd.startswith(home):
            cwd = "~" + cwd[len(home) :]
        return {"cwd": cwd}

    async def serialize(self) -> dict:
        return {"working_dir": str(self.working_dir), "tools": self.tools}

    @staticmethod
    async def deserialize(data: dict) -> "LocalFilesystemEnvironment":
        return LocalFilesystemEnvironment(
            working_dir=Path(data["working_dir"]),
            tools=data.get("tools", "full"),
        )

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """Only bash commands require confirmation by default."""
        return tool_call.name == "bash"

    def get_tool_formatter(self, tool_name: str):
        """Return formatter function for the given tool.

        Returns a function with signature:
            (tool_name, args, result, expanded, theme) -> str

        Returns None for unknown tools (uses generic fallback).
        """
        formatters = {
            "bash": format_bash,
            "read": format_read,
            "write": format_write,
            "edit": format_edit,
            "web_fetch": format_web_fetch,
        }
        return formatters.get(tool_name)

    def get_tools(self) -> list[Tool]:
        all_tools = self._get_all_tools()
        return [t for t in all_tools if t.function.name in self._tool_filter]

    def _get_all_tools(self) -> list[Tool]:
        """Return all available tools (before filtering)."""
        return [
            # read tool
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
            # write tool
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
            # edit tool
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
            # bash tool
            Tool(
                type="function",
                function=ToolFunction(
                    name="bash",
                    description="Execute a bash command in the current working directory. Returns stdout and stderr.",
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
            # web_fetch tool
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

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """No feedback needed for coding environment."""
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: "AgentState",
        run_config: "RunConfig",
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Execute tool call."""
        try:
            if tool_call.name == "read":
                return await self._exec_read(tool_call)
            elif tool_call.name == "write":
                return await self._exec_write(tool_call)
            elif tool_call.name == "edit":
                return await self._exec_edit(tool_call)
            elif tool_call.name == "bash":
                return await self._exec_bash(tool_call, cancel_scope)
            elif tool_call.name == "web_fetch":
                return await self._exec_web_fetch(tool_call)
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Unknown tool: {tool_call.name}",
                )
        except trio.Cancelled:
            # Re-raise cancellation so agent loop can handle it
            raise
        except Exception as e:
            return ToolResult(tool_call_id=tool_call.id, is_error=True, content="", error=str(e))

    async def _exec_read(self, tool_call: ToolCall) -> ToolResult:
        """Read file contents."""
        path_str = tool_call.args["path"]
        offset = tool_call.args.get("offset")
        limit = tool_call.args.get("limit")

        abs_path = expand_path(path_str)

        if not abs_path.exists():
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"File not found: {path_str}",
            )

        if not abs_path.is_file():
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Not a file: {path_str}",
            )

        try:
            content = abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Cannot read binary file: {path_str}",
            )

        lines = content.split("\n")

        # Apply offset and limit
        start_line = (offset - 1) if offset else 0  # 1-indexed to 0-indexed
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

        # Truncate long lines
        had_truncated = False
        formatted_lines = []
        for line in selected_lines:
            if len(line) > MAX_LINE_LENGTH:
                had_truncated = True
                formatted_lines.append(line[:MAX_LINE_LENGTH])
            else:
                formatted_lines.append(line)

        output_text = "\n".join(formatted_lines)

        # Add notices
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
        """Write content to file."""
        path_str = tool_call.args["path"]
        content = tool_call.args["content"]

        abs_path = expand_path(path_str)

        # Create parent directories
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        abs_path.write_text(content, encoding="utf-8")

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Successfully wrote {len(content)} bytes to {path_str}",
        )

    async def _exec_edit(self, tool_call: ToolCall) -> ToolResult:
        """Edit file by replacing exact text."""
        path_str = tool_call.args["path"]
        old_text = tool_call.args["old_text"]
        new_text = tool_call.args["new_text"]

        abs_path = expand_path(path_str)

        if not abs_path.exists():
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"File not found: {path_str}",
            )

        try:
            content = abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Cannot read binary file: {path_str}",
            )

        # Check if old text exists
        if old_text not in content:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Could not find the exact text in {path_str}. The old text must match exactly including all whitespace and newlines.",
            )

        # Count occurrences
        occurrences = content.count(old_text)
        if occurrences > 1:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Found {occurrences} occurrences of the text in {path_str}. The text must be unique. Please provide more context to make it unique.",
            )

        # Perform replacement (manual to avoid $ interpretation)
        index = content.find(old_text)
        new_content = content[:index] + new_text + content[index + len(old_text) :]

        if content == new_content:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"No changes made to {path_str}. The replacement produced identical content.",
            )

        abs_path.write_text(new_content, encoding="utf-8")

        # Generate diff for UI display
        diff_str = generate_diff(content, new_content)

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Successfully replaced text in {path_str}. Changed {len(old_text)} characters to {len(new_text)} characters.",
            details={"diff": diff_str},
        )

    async def _exec_bash(
        self, tool_call: ToolCall, cancel_scope: trio.CancelScope | None = None
    ) -> ToolResult:
        """Execute bash command with proper cancellation support."""
        from ._subprocess import run_command

        command = tool_call.args["command"]
        timeout = tool_call.args.get("timeout", 120)

        try:
            returncode, stdout, stderr = await run_command(
                command, cwd=str(self.working_dir), timeout=timeout
            )

            output = ""
            if stdout:
                output += stdout
            if stderr:
                if output:
                    output += "\n"
                output += stderr

            # Truncate if too large
            if len(output) > MAX_OUTPUT_SIZE:
                removed_kb = (len(output) - MAX_OUTPUT_SIZE) // 1024
                output = (
                    output[:MAX_OUTPUT_SIZE]
                    + f"\n\n... [output truncated - {removed_kb}KB removed]"
                )

            if returncode != 0:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content=output or "(no output)",
                    error=f"Command exited with code {returncode}",
                )

            return ToolResult(
                tool_call_id=tool_call.id, is_error=False, content=output or "(no output)"
            )

        except TimeoutError:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Command timed out after {timeout} seconds",
            )
        except trio.Cancelled:
            raise  # Re-raise so the agent loop handles it

    async def _exec_web_fetch(self, tool_call: ToolCall) -> ToolResult:
        """Fetch content from URL, convert to markdown, return with context."""
        import time

        url = tool_call.args["url"]
        prompt = tool_call.args["prompt"]

        # Validate URL
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

        # Upgrade http to https
        if parsed.scheme == "http":
            url = url.replace("http://", "https://", 1)

        # Check cache
        now = time.time()
        if url in _web_fetch_cache:
            cached_time, cached_result = _web_fetch_cache[url]
            if now - cached_time < WEB_FETCH_CACHE_TTL:
                # Return cached content with the new prompt context
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=False,
                    content=f"[Cached] URL: {url}\nPrompt: {prompt}\n\n---\n\n{cached_result['content']}",
                )

        # Fetch the URL
        try:
            async with httpx.AsyncClient(
                timeout=WEB_FETCH_TIMEOUT,
                follow_redirects=True,
                max_redirects=5,
            ) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; rollouts/1.0)",
                        "Accept": "text/html, text/markdown, */*",
                    },
                )
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

        # Check content size
        content_length = len(response.content)
        if content_length > WEB_FETCH_MAX_SIZE:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Content too large: {content_length} bytes (max {WEB_FETCH_MAX_SIZE})",
            )

        # Decode content
        try:
            text = response.text
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Failed to decode response: {e}",
            )

        # Convert HTML to markdown if needed
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            try:
                text = markdownify.markdownify(text, heading_style="ATX", strip=["script", "style"])
            except Exception:
                # Fall back to raw text if conversion fails
                pass

        # Truncate if too large
        if len(text) > WEB_FETCH_MAX_CONTENT:
            text = text[:WEB_FETCH_MAX_CONTENT] + "\n\n...[content truncated]"

        # Cache the result
        _web_fetch_cache[url] = (now, {"content": text, "status": response.status_code})

        # Clean old cache entries
        for cached_url in list(_web_fetch_cache.keys()):
            cached_time, _ = _web_fetch_cache[cached_url]
            if now - cached_time > WEB_FETCH_CACHE_TTL:
                del _web_fetch_cache[cached_url]

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"URL: {url}\nPrompt: {prompt}\n\n---\n\n{text}",
        )
