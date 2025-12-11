"""
Coding environment with minimal toolset for code editing tasks.

Tools: read, write, edit, bash
Inspired by pi-mono's minimalist approach.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import subprocess
import shlex

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
MAX_OUTPUT_SIZE = 10 * 1024 * 1024  # 10MB


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
        return "~" + path[len(home):]
    return path


def _replace_tabs(text: str) -> str:
    """Replace tabs with spaces for consistent rendering."""
    return text.replace("\t", "   ")


def _get_text_output(result: dict | None) -> str:
    """Extract text output from tool result.

    Result structure: {"content": {"content": [{"type": "text", "text": "..."}], ...}, "isError": bool}
    """
    if not result:
        return ""

    content = result.get("content", {})

    # If content is a string, return it directly
    if isinstance(content, str):
        return content

    # If content is a dict with a "content" key, extract from that
    if isinstance(content, dict):
        content_list = content.get("content", [])
        if isinstance(content_list, list):
            text_blocks = [c for c in content_list if isinstance(c, dict) and c.get("type") == "text"]
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
        max_lines = len(lines) if expanded else 10
        display_lines = lines[:max_lines]
        remaining = len(lines) - max_lines

        total_lines = len(lines)
        summary = f"Read {total_lines} line{'s' if total_lines != 1 else ''}"
        text += f"\n⎿ {summary}"
        for line in display_lines:
            text += "\n  " + _replace_tabs(line)
        if remaining > 0:
            text += f"\n  ... ({remaining} more lines)"

    return text


def format_write(tool_name: str, args: dict, result: dict | None, expanded: bool, theme=None) -> str:
    """Format write tool execution."""
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
        for line in display_lines:
            text += "\n  " + _replace_tabs(line)
        if remaining > 0:
            text += f"\n  ... ({remaining} more lines)"

    return text


def format_edit(tool_name: str, args: dict, result: dict | None, expanded: bool, theme=None) -> str:
    """Format edit tool execution with colored diff."""
    path = _shorten_path(args.get("file_path") or args.get("path") or "")

    text = f"edit(file_path={repr(path if path else '...')}, old_string=..., new_string=...)"

    if result:
        # Check for diff in details (result is wrapped in {"content": ..., "isError": ...})
        content = result.get("content", {})
        details = content.get("details", {}) if isinstance(content, dict) else {}
        diff_str = details.get("diff") if details else None

        is_error = result.get("isError", False)

        if diff_str and theme:
            # Count additions and removals
            diff_lines = diff_str.split("\n")
            additions = sum(1 for line in diff_lines if " + " in line)
            removals = sum(1 for line in diff_lines if " - " in line)

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
                # New format: "  607 - content" or "  607 + content" or "  607   content"
                # Find the marker position (after line number, before content)
                if " - " in line:
                    text += "\n  " + theme.diff_removed_fg(line)
                elif " + " in line:
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


@dataclass
class LocalFilesystemEnvironment:
    """Local filesystem environment with read, write, edit, bash tools."""

    working_dir: Path = field(default_factory=Path.cwd)

    def get_name(self) -> str:
        """Return environment name identifier."""
        return "coding"

    def get_status_info(self) -> dict[str, str] | None:
        """Return cwd for status line display."""
        cwd = str(self.working_dir)
        # Shorten home directory to ~
        home = os.path.expanduser("~")
        if cwd.startswith(home):
            cwd = "~" + cwd[len(home):]
        return {"cwd": cwd}

    async def serialize(self) -> dict:
        return {"working_dir": str(self.working_dir)}

    @staticmethod
    async def deserialize(data: dict) -> 'LocalFilesystemEnvironment':
        return LocalFilesystemEnvironment(working_dir=Path(data["working_dir"]))

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
        }
        return formatters.get(tool_name)

    def get_tools(self) -> list[Tool]:
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
                            "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
                            "limit": {"type": "integer", "description": "Maximum number of lines to read"},
                        }
                    ),
                    required=["path"]
                )
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
                            "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                            "content": {"type": "string", "description": "Content to write to the file"},
                        }
                    ),
                    required=["path", "content"]
                )
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
                            "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                            "old_text": {"type": "string", "description": "Exact text to find and replace (must match exactly)"},
                            "new_text": {"type": "string", "description": "New text to replace the old text with"},
                        }
                    ),
                    required=["path", "old_text", "new_text"]
                )
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
                            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120)"},
                        }
                    ),
                    required=["command"]
                )
            ),
        ]

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """No feedback needed for coding environment."""
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: 'AgentState',
        run_config: 'RunConfig',
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
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Unknown tool: {tool_call.name}"
                )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=str(e)
            )

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
                error=f"File not found: {path_str}"
            )

        if not abs_path.is_file():
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Not a file: {path_str}"
            )

        try:
            content = abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Cannot read binary file: {path_str}"
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
                error=f"Offset {offset} is beyond end of file ({len(lines)} lines total)"
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
            notices.append(f"{remaining} more lines not shown. Use offset={end_line + 1} to continue")

        if notices:
            output_text += f"\n\n... ({'. '.join(notices)})"

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=output_text
        )

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
            content=f"Successfully wrote {len(content)} bytes to {path_str}"
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
                error=f"File not found: {path_str}"
            )

        try:
            content = abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Cannot read binary file: {path_str}"
            )

        # Check if old text exists
        if old_text not in content:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Could not find the exact text in {path_str}. The old text must match exactly including all whitespace and newlines."
            )

        # Count occurrences
        occurrences = content.count(old_text)
        if occurrences > 1:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Found {occurrences} occurrences of the text in {path_str}. The text must be unique. Please provide more context to make it unique."
            )

        # Perform replacement (manual to avoid $ interpretation)
        index = content.find(old_text)
        new_content = content[:index] + new_text + content[index + len(old_text):]

        if content == new_content:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"No changes made to {path_str}. The replacement produced identical content."
            )

        abs_path.write_text(new_content, encoding="utf-8")

        # Generate diff for UI display
        diff_str = generate_diff(content, new_content)

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Successfully replaced text in {path_str}. Changed {len(old_text)} characters to {len(new_text)} characters.",
            details={"diff": diff_str}
        )

    async def _exec_bash(self, tool_call: ToolCall, cancel_scope: trio.CancelScope | None = None) -> ToolResult:
        """Execute bash command."""
        command = tool_call.args["command"]
        timeout = tool_call.args.get("timeout", 120)

        try:
            # Run command with trio
            result = await trio.to_thread.run_sync(
                lambda: subprocess.run(
                    ["sh", "-c", command],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(self.working_dir),
                ),
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += result.stderr

            # Truncate if too large
            if len(output) > MAX_OUTPUT_SIZE:
                output = output[:MAX_OUTPUT_SIZE] + "\n\n... (output truncated)"

            if result.returncode != 0:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content=output or "(no output)",
                    error=f"Command exited with code {result.returncode}"
                )

            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=False,
                content=output or "(no output)"
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Command timed out after {timeout} seconds"
            )
        except trio.Cancelled:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="Command aborted"
            )
