"""CodexEnvironment - Drive OpenAI Codex CLI as a subprocess.

This environment wraps the Codex CLI (codex exec), allowing rollouts to:
1. Spawn Codex with a prompt and working directory
2. Send it a task and observe the full trajectory
3. Collect tool calls, results, and final output

Usage:
    env = CodexEnvironment(
        working_dir="/path/to/repo",
        model="o3",
    )

    # The environment exposes a single tool: run_agent
    # When called, it spawns Codex and runs until completion
    result = await env.exec_tool(
        ToolCall(name="run_agent", args={"task": "Fix the bug in auth.py"}),
        state,
        run_config,
    )

    # Result contains the full trajectory as JSON
    result.content  # JSON with messages, tool calls, outcome

For evals:
    # Run Codex on a task and get structured output
    env = CodexEnvironment(working_dir=repo_path)
    trajectory = await env.run_task("Implement feature X")

    # Inspect what happened
    trajectory.messages      # All JSONL events
    trajectory.items         # Completed items (agent messages, tool calls)
    trajectory.stop_reason   # Why it stopped
    trajectory.tokens_used   # Token counts

Design notes:
- Uses `codex exec --json` for JSONL output
- Subprocess lifecycle managed per-task (spawn, run, collect, terminate)
- Uses --dangerously-bypass-approvals-and-sandbox for eval mode
- Can optionally stream events to a callback for live monitoring

Codex JSONL event types:
- thread.started: {thread_id}
- turn.started: {}
- item.started: {item: {id, type, ...}}
- item.completed: {item: {id, type, text?, tool_call?, ...}}
- turn.completed: {usage: {input_tokens, output_tokens, ...}}
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
class CodexTrajectory:
    """Result from running Codex on a task.

    Immutable record of what happened during the run.
    """

    events: tuple[dict[str, Any], ...]  # Raw JSONL events
    stop_reason: str  # "success", "error", "timeout"
    num_turns: int
    duration_ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None
    thread_id: str | None = None

    def get_completed_items(self) -> list[dict[str, Any]]:
        """Get all completed items from the trajectory."""
        items = []
        for event in self.events:
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                items.append(item)
        return items

    def get_agent_messages(self) -> list[str]:
        """Get all agent message texts from completed items."""
        messages = []
        for item in self.get_completed_items():
            if item.get("type") == "agent_message":
                text = item.get("text", "")
                if text:
                    messages.append(text)
        return messages

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Get all tool calls made during the run."""
        calls = []
        for item in self.get_completed_items():
            if item.get("type") == "tool_call":
                calls.append(item)
        return calls

    def get_final_text(self) -> str:
        """Get the final text output from Codex."""
        messages = self.get_agent_messages()
        return messages[-1] if messages else ""


@dataclass
class CodexEnvironment:
    """Environment that drives Codex CLI as a subprocess.

    This is NOT a normal tool-providing environment. Instead, it wraps
    Codex and exposes a single "run_agent" tool that:
    1. Spawns Codex with the given task
    2. Streams events for monitoring
    3. Returns the full trajectory when done

    Attributes:
        working_dir: Directory where Codex runs (cwd for the subprocess)
        model: Model to use (e.g., "o3", "o4-mini")
        sandbox_mode: Sandbox policy ("read-only", "workspace-write", "danger-full-access")
        timeout_seconds: Timeout for the entire run
        bypass_approvals: If True, use --dangerously-bypass-approvals-and-sandbox
    """

    working_dir: Path
    model: str = "o3"
    sandbox_mode: str = "workspace-write"
    timeout_seconds: float = 600.0  # 10 minutes default
    bypass_approvals: bool = True  # For eval mode

    # Runtime state (not serialized)
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _collected_events: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.working_dir = Path(self.working_dir)
        self._collected_events = []

    # ── Environment Protocol ──────────────────────────────────────────────────

    def get_tools(self) -> list[Tool]:
        """Return the single 'run_agent' tool."""
        from ..dtypes import Tool, ToolFunction, ToolFunctionParameter

        return [
            Tool(
                function=ToolFunction(
                    name="run_agent",
                    description="Run Codex on a task and return the trajectory",
                    parameters=ToolFunctionParameter(
                        properties={
                            "task": {
                                "type": "string",
                                "description": "The task/prompt to send to Codex",
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
        """Execute a tool call - runs Codex on the task."""
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
                "thread_id": trajectory.thread_id,
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
            logger.exception(f"Codex run failed: {e}")
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
            "env": "codex",
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
            "sandbox_mode": self.sandbox_mode,
            "timeout_seconds": self.timeout_seconds,
            "bypass_approvals": self.bypass_approvals,
        }

    @classmethod
    async def deserialize(cls, data: dict[str, Any]) -> CodexEnvironment:
        """Deserialize environment state."""
        return cls(
            working_dir=Path(data["working_dir"]),
            model=data.get("model", "o3"),
            sandbox_mode=data.get("sandbox_mode", "workspace-write"),
            timeout_seconds=data.get("timeout_seconds", 600.0),
            bypass_approvals=data.get("bypass_approvals", True),
        )

    # ── Core Implementation ───────────────────────────────────────────────────

    async def run_task(
        self,
        task: str,
        on_event: Any | None = None,
    ) -> CodexTrajectory:
        """Run Codex on a task and return the trajectory.

        This is the main entry point for running Codex programmatically.

        Args:
            task: The task/prompt to send to Codex
            on_event: Optional callback for each JSONL event (for streaming)

        Returns:
            CodexTrajectory with the full run record
        """
        import time

        # Find codex binary
        codex_bin = shutil.which("codex")
        if codex_bin is None:
            raise RuntimeError("Codex CLI not found. Install from: https://github.com/openai/codex")

        # Build command
        cmd = [
            codex_bin,
            "exec",
            "--json",  # JSONL output
            "--model",
            self.model,
            "--cd",
            str(self.working_dir),
        ]

        if self.bypass_approvals:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            cmd.extend(["--sandbox", self.sandbox_mode])

        # Add the task as the final argument
        cmd.append(task)

        logger.info(f"Starting Codex: {' '.join(cmd[:5])}...")

        start_time = time.perf_counter()
        self._collected_events = []

        try:
            # Spawn subprocess
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ},
            )

            # Read stdout line by line (JSONL)
            stop_reason = "success"
            num_turns = 0
            tokens_in = None
            tokens_out = None
            error_msg = None
            thread_id = None

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

                # Parse JSONL line
                try:
                    event = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    continue

                self._collected_events.append(event)

                if on_event:
                    await on_event(event)

                # Extract metadata
                event_type = event.get("type", "")

                if event_type == "thread.started":
                    thread_id = event.get("thread_id")

                elif event_type == "turn.completed":
                    num_turns += 1
                    usage = event.get("usage", {})
                    if usage:
                        # Accumulate tokens across turns
                        if tokens_in is None:
                            tokens_in = 0
                        if tokens_out is None:
                            tokens_out = 0
                        tokens_in += usage.get("input_tokens", 0)
                        tokens_out += usage.get("output_tokens", 0)

                elif event_type == "error":
                    stop_reason = "error"
                    error_msg = event.get("message", "Unknown error")

            # Wait for process to finish
            await self._process.wait()

            # Check exit code
            if self._process.returncode != 0 and stop_reason == "success":
                stop_reason = "error"
                # Try to read stderr for error message
                if self._process.stderr:
                    stderr_data = await self._process.stderr.read()
                    if stderr_data:
                        error_msg = stderr_data.decode().strip()[:500]

            duration_ms = int((time.perf_counter() - start_time) * 1000)

            return CodexTrajectory(
                events=tuple(self._collected_events),
                stop_reason=stop_reason,
                num_turns=num_turns,
                duration_ms=duration_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                error=error_msg,
                thread_id=thread_id,
            )

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return CodexTrajectory(
                events=tuple(self._collected_events),
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


async def run_codex(
    task: str,
    working_dir: Path | str,
    model: str = "o3",
    timeout_seconds: float = 600.0,
) -> CodexTrajectory:
    """Convenience function to run Codex on a task.

    Example:
        trajectory = await run_codex(
            task="Fix the bug in auth.py",
            working_dir="/path/to/repo",
        )
        print(f"Completed in {trajectory.num_turns} turns")
        print(trajectory.get_final_text())
    """
    env = CodexEnvironment(
        working_dir=Path(working_dir),
        model=model,
        timeout_seconds=timeout_seconds,
    )
    return await env.run_task(task)
