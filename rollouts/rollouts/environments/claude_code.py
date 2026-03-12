"""ClaudeCodeEnvironment - Drive Claude Code as a subprocess.

This environment wraps Claude Code CLI, allowing rollouts to:
1. Spawn Claude Code with a system prompt and working directory
2. Send it a task and observe the full trajectory
3. Collect tool calls, results, and final output

Usage:
    env = ClaudeCodeEnvironment(
        working_dir="/path/to/repo",
        model="claude-sonnet-4-20250514",
    )

    # The environment exposes a single tool: run_agent
    # When called, it spawns Claude Code and runs until completion
    result = await env.exec_tool(
        ToolCall(name="run_agent", args={"task": "Fix the bug in auth.py"}),
        state,
        run_config,
    )

    # Result contains the full trajectory as JSON
    result.content  # JSON with messages, tool calls, outcome

For evals:
    # Run Claude Code on a task and get structured output
    env = ClaudeCodeEnvironment(working_dir=repo_path, max_turns=20)
    trajectory = await env.run_task("Implement feature X")

    # Inspect what happened
    trajectory.messages      # All messages exchanged
    trajectory.tool_calls    # All tools Claude Code used
    trajectory.stop_reason   # Why it stopped
    trajectory.tokens_used   # Token counts

Design notes:
- Uses --print --verbose --output-format stream-json for NDJSON output
- Subprocess lifecycle managed per-task (spawn, run, collect, terminate)
- Tool confirmation auto-approved (eval mode)
- Can optionally stream events to a callback for live monitoring

Open questions / TODOs:
- [ ] max_turns: Claude Code CLI doesn't have --max-turns, need to implement via
      system prompt instruction or by terminating after N turns
- [ ] Bidirectional mode: Current implementation uses query mode (one-shot).
      For multi-turn or interactive use, need --input-format stream-json
- [ ] Permission handling: Current implementation auto-approves tools. For eval,
      may want --dangerously-skip-permissions or custom permission mode
- [ ] Session reuse: Could use --resume to continue sessions, but current design
      is one-shot per task
- [ ] API key inheritance: Subprocess inherits env vars, so ANTHROPIC_API_KEY
      must be set in the environment
- [ ] Stderr handling: Currently ignored, may want to capture for debugging
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import trio

    from ..agents import AgentState, RunConfig
    from ..core import Tool, ToolCall, ToolResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaudeCodeTrajectory:
    """Result from running Claude Code on a task.

    Immutable record of what happened during the run.
    """

    messages: tuple[dict[str, Any], ...]  # Raw NDJSON messages
    stop_reason: str  # "success", "max_turns", "error", "aborted"
    num_turns: int
    duration_ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None

    def get_assistant_messages(self) -> list[dict[str, Any]]:
        """Get all assistant messages from the trajectory."""
        return [m for m in self.messages if m.get("type") == "assistant"]

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Get all tool calls made during the run."""
        calls = []
        for msg in self.messages:
            if msg.get("type") == "assistant":
                content = msg.get("message", {}).get("content", [])
                for block in content:
                    if block.get("type") == "tool_use":
                        calls.append(block)
        return calls

    def get_final_text(self) -> str:
        """Get the final text output from Claude Code."""
        assistant_msgs = self.get_assistant_messages()
        if not assistant_msgs:
            return ""

        last_msg = assistant_msgs[-1]
        content = last_msg.get("message", {}).get("content", [])
        text_parts = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts)


@dataclass
class ClaudeCodeEnvironment:
    """Environment that drives Claude Code as a subprocess.

    This is NOT a normal tool-providing environment. Instead, it wraps
    Claude Code and exposes a single "run_agent" tool that:
    1. Spawns Claude Code with the given task
    2. Streams events for monitoring
    3. Returns the full trajectory when done

    Attributes:
        working_dir: Directory where Claude Code runs (cwd for the subprocess)
        model: Model to use (e.g., "claude-sonnet-4-20250514")
        max_turns: Maximum turns before stopping (passed to Claude Code)
        system_prompt: Optional system prompt override
        allowed_tools: Optional list of allowed tools (None = all)
        timeout_seconds: Timeout for the entire run
    """

    working_dir: Path
    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 50
    system_prompt: str | None = None
    allowed_tools: list[str] | None = None
    timeout_seconds: float = 600.0  # 10 minutes default

    # Runtime state (not serialized)
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _collected_messages: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.working_dir = Path(self.working_dir)
        self._collected_messages = []

    # ── Environment Protocol ──────────────────────────────────────────────────

    def get_tools(self) -> list[Tool]:
        """Return the single 'run_agent' tool."""
        from ..dtypes import Tool, ToolFunction, ToolFunctionParameter

        return [
            Tool(
                function=ToolFunction(
                    name="run_agent",
                    description="Run Claude Code on a task and return the trajectory",
                    parameters=ToolFunctionParameter(
                        properties={
                            "task": {
                                "type": "string",
                                "description": "The task/prompt to send to Claude Code",
                            },
                        },
                    ),
                    required=["task"],
                )
            )
        ]

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Execute a tool call - runs Claude Code on the task."""
        from ..dtypes import ToolResult

        if tool_call.name != "run_agent":
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                error=f"Unknown tool: {tool_call.name}",
                content="",
            )

        task = tool_call.args.get("task", "")
        if not task:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                error="Missing 'task' argument",
                content="",
            )

        try:
            trajectory = await self.run_task(task)

            # Format trajectory as JSON for the calling agent
            result_data = {
                "stop_reason": trajectory.stop_reason,
                "num_turns": trajectory.num_turns,
                "duration_ms": trajectory.duration_ms,
                "final_text": trajectory.get_final_text(),
                "tool_calls_count": len(trajectory.get_tool_calls()),
                "tokens_in": trajectory.tokens_in,
                "tokens_out": trajectory.tokens_out,
            }

            if trajectory.error:
                result_data["error"] = trajectory.error

            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=trajectory.stop_reason == "error",
                error=trajectory.error,
                content=json.dumps(result_data, indent=2),
            )

        except Exception as e:
            logger.exception(f"Claude Code run failed: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                error=str(e),
                content="",
            )

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """No confirmation needed - this is for evals."""
        return False

    def get_tool_formatter(self, tool_name: str) -> Any:
        """No custom formatter."""
        return None

    def get_status_info(self) -> dict[str, str] | None:
        """Return status info for TUI."""
        return {
            "env": "claude-code",
            "model": self.model,
            "cwd": str(self.working_dir),
        }

    def get_system_prompt(self) -> str | None:
        """No additional system prompt."""
        return None

    async def serialize(self) -> dict[str, Any]:
        """Serialize environment state."""
        return {
            "working_dir": str(self.working_dir),
            "model": self.model,
            "max_turns": self.max_turns,
            "system_prompt": self.system_prompt,
            "allowed_tools": self.allowed_tools,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    async def deserialize(cls, data: dict[str, Any]) -> ClaudeCodeEnvironment:
        """Deserialize environment state."""
        return cls(
            working_dir=Path(data["working_dir"]),
            model=data.get("model", "claude-sonnet-4-20250514"),
            max_turns=data.get("max_turns", 50),
            system_prompt=data.get("system_prompt"),
            allowed_tools=data.get("allowed_tools"),
            timeout_seconds=data.get("timeout_seconds", 600.0),
        )

    # ── Core Implementation ───────────────────────────────────────────────────

    async def run_task(
        self,
        task: str,
        on_message: Any | None = None,
    ) -> ClaudeCodeTrajectory:
        """Run Claude Code on a task and return the trajectory.

        This is the main entry point for running Claude Code programmatically.

        Args:
            task: The task/prompt to send to Claude Code
            on_message: Optional callback for each NDJSON message (for streaming)

        Returns:
            ClaudeCodeTrajectory with the full run record
        """
        import time

        # Find claude binary
        claude_bin = shutil.which("claude")
        if claude_bin is None:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )

        # Build command
        # Note: Claude Code CLI uses --print for non-interactive mode
        # --verbose is required for stream-json output
        cmd = [
            claude_bin,
            "--print",  # Non-interactive mode
            "--verbose",  # Required for stream-json
            "--output-format",
            "stream-json",
            "--model",
            self.model,
        ]

        if self.system_prompt:
            cmd.extend(["--system-prompt", self.system_prompt])

        if self.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])

        # Add the task as the final argument (positional prompt)
        cmd.append(task)

        logger.info(f"Starting Claude Code: {' '.join(cmd[:5])}...")

        start_time = time.perf_counter()
        self._collected_messages = []

        try:
            # Spawn subprocess
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.working_dir),
                env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "rollouts-eval"},
            )

            # Read stdout line by line (NDJSON)
            stop_reason = "success"
            num_turns = 0
            tokens_in = None
            tokens_out = None
            error_msg = None

            assert self._process.stdout is not None

            while True:
                try:
                    line = await asyncio.wait_for(
                        self._process.stdout.readline(),
                        timeout=self.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    stop_reason = "timeout"
                    error_msg = f"Timeout after {self.timeout_seconds}s"
                    break

                if not line:
                    break

                # Parse NDJSON line
                try:
                    msg = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    continue

                self._collected_messages.append(msg)

                if on_message:
                    await on_message(msg)

                # Extract metadata from result message
                if msg.get("type") == "result":
                    num_turns = msg.get("num_turns", 0)

                    # Check is_error flag first, then subtype
                    if msg.get("is_error"):
                        stop_reason = "error"
                        error_msg = msg.get("result", "Unknown error")
                    elif msg.get("subtype") == "success":
                        stop_reason = "success"
                    elif msg.get("subtype") == "error":
                        stop_reason = "error"
                        error_msg = msg.get("error")

                    # Token counts from usage object
                    usage = msg.get("usage", {})
                    if usage:
                        tokens_in = usage.get("input_tokens")
                        tokens_out = usage.get("output_tokens")

            # Wait for process to finish
            await self._process.wait()

            duration_ms = int((time.perf_counter() - start_time) * 1000)

            return ClaudeCodeTrajectory(
                messages=tuple(self._collected_messages),
                stop_reason=stop_reason,
                num_turns=num_turns,
                duration_ms=duration_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                error=error_msg,
            )

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return ClaudeCodeTrajectory(
                messages=tuple(self._collected_messages),
                stop_reason="error",
                num_turns=0,
                duration_ms=duration_ms,
                error=str(e),
            )
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up subprocess."""
        if self._process is not None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
            self._process = None


# ── Convenience functions for evals ───────────────────────────────────────────


async def run_claude_code(
    task: str,
    working_dir: Path | str,
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = 50,
    timeout_seconds: float = 600.0,
) -> ClaudeCodeTrajectory:
    """Convenience function to run Claude Code on a task.

    Example:
        trajectory = await run_claude_code(
            task="Fix the bug in auth.py",
            working_dir="/path/to/repo",
        )
        print(f"Completed in {trajectory.num_turns} turns")
        print(trajectory.get_final_text())
    """
    env = ClaudeCodeEnvironment(
        working_dir=Path(working_dir),
        model=model,
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
    )
    return await env.run_task(task)
