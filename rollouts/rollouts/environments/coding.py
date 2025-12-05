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


@dataclass
class CodingEnvironment:
    """Minimal coding environment with read, write, edit, bash tools."""

    working_dir: Path = field(default_factory=Path.cwd)

    def get_name(self) -> str:
        """Return environment name identifier."""
        return "coding"

    async def serialize(self) -> dict:
        return {"working_dir": str(self.working_dir)}

    @staticmethod
    async def deserialize(data: dict) -> 'CodingEnvironment':
        return CodingEnvironment(working_dir=Path(data["working_dir"]))

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """Only bash commands require confirmation by default."""
        return tool_call.name == "bash"

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
        checkpoint_store=None,
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

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Successfully replaced text in {path_str}. Changed {len(old_text)} characters to {len(new_text)} characters."
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
