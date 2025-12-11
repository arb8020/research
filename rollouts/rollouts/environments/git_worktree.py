"""
Git Worktree Environment - isolated git-based coding environment.

Key design:
- Creates a separate .rollouts/ directory, NOT touching user's .git
- Each session gets its own worktree for isolation
- Auto-commits on every write/edit/bash for full history
- Restore = git checkout (fast, deterministic)

Layout:
    <working_dir>/
        .rollouts/
            repo.git/              # bare repo (our isolated git)
            worktrees/
                <session_id>/      # worktree for each session
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import subprocess
import shutil

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


def _shorten_path(path: str, working_dir: Path) -> str:
    """Convert path to relative if under working_dir, or tilde notation."""
    try:
        rel = Path(path).relative_to(working_dir)
        return str(rel)
    except ValueError:
        pass
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home):]
    return path


@dataclass
class GitWorktreeEnvironment:
    """Git-based coding environment with worktree isolation and auto-commit.

    Each session works in an isolated worktree. Every file operation is
    automatically committed, creating a full history that can be restored.

    The user's existing .git is completely ignored - we maintain our own
    bare repo in .rollouts/repo.git.
    """

    working_dir: Path = field(default_factory=Path.cwd)
    session_id: str | None = None

    # Internal state
    _rollouts_dir: Path | None = field(default=None, repr=False)
    _bare_repo: Path | None = field(default=None, repr=False)
    _worktree_path: Path | None = field(default=None, repr=False)
    _current_branch: str | None = field(default=None, repr=False)
    _commit_count: int = field(default=0, repr=False)

    def get_name(self) -> str:
        return "git-worktree"

    def get_status_info(self) -> dict[str, str] | None:
        """Return git info for status line."""
        info = {}
        # Shorten working dir
        cwd = str(self.working_dir)
        home = os.path.expanduser("~")
        if cwd.startswith(home):
            cwd = "~" + cwd[len(home):]
        info["cwd"] = cwd

        if self._current_branch:
            info["branch"] = self._current_branch
        if self._commit_count > 0:
            info["commits"] = str(self._commit_count)
        return info

    async def on_session_start(self, session_id: str) -> None:
        """Initialize git repo and create worktree for this session.

        Called automatically by run_agent before any tools execute.
        """
        self.session_id = session_id
        self._rollouts_dir = self.working_dir / ".rollouts"
        self._bare_repo = self._rollouts_dir / "repo.git"
        self._worktree_path = self._rollouts_dir / "worktrees" / session_id
        self._current_branch = f"session-{session_id}"

        # Create .rollouts directory
        self._rollouts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize bare repo if needed
        if not self._bare_repo.exists():
            await self._run_git(["init", "--bare", str(self._bare_repo)], cwd=self._rollouts_dir)

            # Create initial commit with current directory contents
            await self._create_initial_commit()

        # Create worktree for this session
        if not self._worktree_path.exists():
            # Create branch and worktree
            # First, get the main branch name
            main_branch = await self._get_main_branch()

            await self._run_git(
                ["worktree", "add", "-b", self._current_branch, str(self._worktree_path), main_branch],
                cwd=self._bare_repo
            )

    async def _create_initial_commit(self) -> None:
        """Create initial commit with current working directory contents."""
        # Create a temporary worktree to make the initial commit
        temp_worktree = self._rollouts_dir / "worktrees" / "_init"
        temp_worktree.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize as a regular repo first
            await self._run_git(["init"], cwd=temp_worktree)
            await self._run_git(["config", "user.email", "rollouts@local"], cwd=temp_worktree)
            await self._run_git(["config", "user.name", "Rollouts"], cwd=temp_worktree)

            # Copy current directory contents (excluding .rollouts and .git)
            await self._copy_working_dir_to(temp_worktree)

            # Add and commit
            await self._run_git(["add", "-A"], cwd=temp_worktree)
            await self._run_git(
                ["commit", "-m", "Initial snapshot", "--allow-empty"],
                cwd=temp_worktree
            )

            # Push to bare repo
            await self._run_git(
                ["remote", "add", "origin", str(self._bare_repo)],
                cwd=temp_worktree
            )
            await self._run_git(
                ["push", "-u", "origin", "main"],
                cwd=temp_worktree
            )
        finally:
            # Clean up temp worktree
            shutil.rmtree(temp_worktree, ignore_errors=True)

    async def _copy_working_dir_to(self, dest: Path) -> None:
        """Copy working directory contents to dest, excluding .rollouts and .git."""
        def should_copy(src: Path) -> bool:
            name = src.name
            # Skip our own directory and user's git
            if name in (".rollouts", ".git"):
                return False
            # Skip common large/generated directories
            if name in ("node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"):
                return False
            return True

        def copy_tree(src: Path, dst: Path) -> None:
            for item in src.iterdir():
                if not should_copy(item):
                    continue
                dst_item = dst / item.name
                if item.is_dir():
                    dst_item.mkdir(exist_ok=True)
                    copy_tree(item, dst_item)
                else:
                    shutil.copy2(item, dst_item)

        await trio.to_thread.run_sync(lambda: copy_tree(self.working_dir, dest))

    async def _get_main_branch(self) -> str:
        """Get the main branch name from bare repo."""
        try:
            result = await self._run_git(
                ["symbolic-ref", "--short", "HEAD"],
                cwd=self._bare_repo
            )
            return result.strip() or "main"
        except Exception:
            return "main"

    async def _run_git(self, args: list[str], cwd: Path | None = None) -> str:
        """Run a git command and return stdout."""
        if cwd is None:
            cwd = self._worktree_path or self.working_dir

        result = await trio.to_thread.run_sync(
            lambda: subprocess.run(
                ["git"] + args,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=30,
            )
        )

        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")

        return result.stdout

    async def _auto_commit(self, message: str) -> None:
        """Auto-commit all changes with the given message."""
        if not self._worktree_path:
            return

        try:
            # Stage all changes
            await self._run_git(["add", "-A"])

            # Commit (allow empty to track bash commands that don't change files)
            await self._run_git(["commit", "-m", message, "--allow-empty"])

            self._commit_count += 1
        except RuntimeError:
            # Ignore commit failures (e.g., nothing to commit)
            pass

    async def serialize(self) -> dict:
        """Capture current state for session persistence."""
        head_commit = None
        if self._worktree_path and self._worktree_path.exists():
            try:
                head_commit = (await self._run_git(["rev-parse", "HEAD"])).strip()
            except Exception:
                pass

        return {
            "working_dir": str(self.working_dir),
            "session_id": self.session_id,
            "branch": self._current_branch,
            "head_commit": head_commit,
            "commit_count": self._commit_count,
        }

    @staticmethod
    async def deserialize(data: dict) -> 'GitWorktreeEnvironment':
        """Restore environment from serialized state."""
        env = GitWorktreeEnvironment(working_dir=Path(data["working_dir"]))

        session_id = data.get("session_id")
        if session_id:
            await env.on_session_start(session_id)

            # Checkout specific commit if provided
            head_commit = data.get("head_commit")
            if head_commit and env._worktree_path:
                try:
                    await env._run_git(["checkout", head_commit])
                except Exception:
                    pass

            env._commit_count = data.get("commit_count", 0)

        return env

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """Only bash commands require confirmation."""
        return tool_call.name == "bash"

    def get_tool_formatter(self, tool_name: str):
        """Return formatter for the given tool."""
        # Reuse formatters from coding.py
        from .coding import format_bash, format_read, format_write, format_edit
        formatters = {
            "bash": format_bash,
            "read": format_read,
            "write": format_write,
            "edit": format_edit,
        }
        return formatters.get(tool_name)

    def get_tools(self) -> list[Tool]:
        """Return available tools - same as LocalFilesystemEnvironment."""
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="read",
                    description="Read the contents of a file. Defaults to first 2000 lines.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "path": {"type": "string", "description": "Path to the file to read"},
                            "offset": {"type": "integer", "description": "Line number to start from (1-indexed)"},
                            "limit": {"type": "integer", "description": "Max lines to read"},
                        }
                    ),
                    required=["path"]
                )
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="write",
                    description="Write content to a file. Creates parent directories automatically.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "path": {"type": "string", "description": "Path to the file"},
                            "content": {"type": "string", "description": "Content to write"},
                        }
                    ),
                    required=["path", "content"]
                )
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="edit",
                    description="Edit a file by replacing exact text. old_text must match exactly.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "path": {"type": "string", "description": "Path to the file"},
                            "old_text": {"type": "string", "description": "Exact text to find"},
                            "new_text": {"type": "string", "description": "Replacement text"},
                        }
                    ),
                    required=["path", "old_text", "new_text"]
                )
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="bash",
                    description="Execute a bash command. Returns stdout and stderr.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "command": {"type": "string", "description": "Command to execute"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120)"},
                        }
                    ),
                    required=["command"]
                )
            ),
        ]

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """No feedback needed."""
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Execute tool call with auto-commit."""
        # Determine working directory - use worktree if set up
        work_dir = self._worktree_path or self.working_dir

        try:
            if tool_call.name == "read":
                return await self._exec_read(tool_call, work_dir)
            elif tool_call.name == "write":
                result = await self._exec_write(tool_call, work_dir)
                # Auto-commit after write
                path = tool_call.args.get("path", "unknown")
                await self._auto_commit(f"write: {_shorten_path(path, work_dir)}")
                return result
            elif tool_call.name == "edit":
                result = await self._exec_edit(tool_call, work_dir)
                # Auto-commit after edit
                path = tool_call.args.get("path", "unknown")
                await self._auto_commit(f"edit: {_shorten_path(path, work_dir)}")
                return result
            elif tool_call.name == "bash":
                result = await self._exec_bash(tool_call, work_dir, cancel_scope)
                # Auto-commit after bash (may have changed files)
                cmd = tool_call.args.get("command", "")[:50]
                await self._auto_commit(f"bash: {cmd}")
                return result
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

    def _resolve_path(self, path_str: str, work_dir: Path) -> Path:
        """Resolve path relative to work_dir or as absolute."""
        if path_str.startswith("~"):
            if path_str == "~":
                return Path.home()
            return Path.home() / path_str[2:]

        path = Path(path_str)
        if path.is_absolute():
            return path
        return work_dir / path

    async def _exec_read(self, tool_call: ToolCall, work_dir: Path) -> ToolResult:
        """Read file contents."""
        path_str = tool_call.args["path"]
        offset = tool_call.args.get("offset")
        limit = tool_call.args.get("limit")

        abs_path = self._resolve_path(path_str, work_dir)

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
        start_line = (offset - 1) if offset else 0
        max_lines = limit or MAX_LINES
        end_line = min(start_line + max_lines, len(lines))

        if start_line >= len(lines):
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Offset {offset} beyond end of file ({len(lines)} lines)"
            )

        selected_lines = lines[start_line:end_line]

        # Truncate long lines
        formatted_lines = []
        for line in selected_lines:
            if len(line) > MAX_LINE_LENGTH:
                formatted_lines.append(line[:MAX_LINE_LENGTH])
            else:
                formatted_lines.append(line)

        output_text = "\n".join(formatted_lines)

        if end_line < len(lines):
            remaining = len(lines) - end_line
            output_text += f"\n\n... ({remaining} more lines)"

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=output_text
        )

    async def _exec_write(self, tool_call: ToolCall, work_dir: Path) -> ToolResult:
        """Write content to file."""
        path_str = tool_call.args["path"]
        content = tool_call.args["content"]

        abs_path = self._resolve_path(path_str, work_dir)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Wrote {len(content)} bytes to {path_str}"
        )

    async def _exec_edit(self, tool_call: ToolCall, work_dir: Path) -> ToolResult:
        """Edit file by replacing exact text."""
        from .coding import generate_diff

        path_str = tool_call.args["path"]
        old_text = tool_call.args["old_text"]
        new_text = tool_call.args["new_text"]

        abs_path = self._resolve_path(path_str, work_dir)

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

        if old_text not in content:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Text not found in {path_str}"
            )

        occurrences = content.count(old_text)
        if occurrences > 1:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Found {occurrences} occurrences - text must be unique"
            )

        # Replace
        index = content.find(old_text)
        new_content = content[:index] + new_text + content[index + len(old_text):]

        if content == new_content:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="No changes made"
            )

        abs_path.write_text(new_content, encoding="utf-8")

        diff_str = generate_diff(content, new_content)

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Edited {path_str}",
            details={"diff": diff_str}
        )

    async def _exec_bash(
        self,
        tool_call: ToolCall,
        work_dir: Path,
        cancel_scope: trio.CancelScope | None = None
    ) -> ToolResult:
        """Execute bash command."""
        command = tool_call.args["command"]
        timeout = tool_call.args.get("timeout", 120)

        try:
            result = await trio.to_thread.run_sync(
                lambda: subprocess.run(
                    ["sh", "-c", command],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(work_dir),
                )
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += result.stderr

            if len(output) > MAX_OUTPUT_SIZE:
                output = output[:MAX_OUTPUT_SIZE] + "\n\n... (truncated)"

            if result.returncode != 0:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content=output or "(no output)",
                    error=f"Exit code {result.returncode}"
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
                error=f"Timeout after {timeout}s"
            )
        except trio.Cancelled:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="Aborted"
            )
