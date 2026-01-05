"""Terminal-Bench environment for rollouts.

Provides tools to interact with terminal-bench tasks running in Docker containers
with tmux sessions. This closely mirrors terminal-bench's Terminus agent patterns.

Tools:
- send_keys: Send keystrokes to the tmux session (matches Terminus Command model)
- capture_terminal: Capture current terminal state
- task_complete: Signal that the task is complete

Reference: terminal_bench/agents/terminus_1.py

Usage:
    from rollouts.environments.terminal_bench import TerminalBenchEnvironment

    env = await TerminalBenchEnvironment.create(task_id="hello-world")
    # Use with run_agent...
    await env.cleanup()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

from ..dtypes import (
    AgentState,
    Message,
    RunConfig,
    StopReason,
    Tool,
    ToolCall,
    ToolFunction,
    ToolFunctionParameter,
    ToolResult,
)

if TYPE_CHECKING:
    from terminal_bench.terminal.terminal import Terminal
    from terminal_bench.terminal.tmux_session import TmuxSession

logger = logging.getLogger(__name__)


@dataclass
class TerminalBenchEnvironment:
    """Environment for terminal-bench tasks.

    Wraps a Docker container with a tmux session, providing tools for
    sending keystrokes and capturing terminal output. Closely mirrors
    the Terminus agent's interaction pattern.
    """

    task_id: str
    instruction: str
    terminal: Terminal
    session: TmuxSession
    logging_dir: Path | None = None

    # Track interaction history for trajectories
    interactions: list[dict[str, Any]] = field(default_factory=list)
    _task_completed: bool = False
    _terminal_cm: Any = None  # Context manager for cleanup

    async def serialize(self) -> dict:
        """Serialize environment state (for checkpointing).

        Note: We store a reference to self because Docker state can't be truly serialized.
        This works for within-session use but won't persist across restarts.
        """
        return {
            "env_kind": "terminal_bench",
            "task_id": self.task_id,
            "instruction": self.instruction,
            "interactions": self.interactions,
            "task_completed": self._task_completed,
            "_env_ref": self,  # Keep reference to the live environment
        }

    @staticmethod
    async def deserialize(data: dict) -> TerminalBenchEnvironment:
        """Deserialize environment state.

        For terminal-bench, we return the cached live environment reference
        since Docker state cannot be truly serialized.
        """
        if "_env_ref" in data:
            env = data["_env_ref"]
            env.interactions = data.get("interactions", env.interactions)
            env._task_completed = data.get("task_completed", env._task_completed)
            return env

        raise NotImplementedError(
            "TerminalBenchEnvironment cannot be deserialized from cold storage - "
            "Docker container state is not persistable."
        )

    @classmethod
    async def create(
        cls,
        task_id: str,
        dataset_name: str = "terminal-bench-core",
        dataset_version: str = "head",
        logging_dir: Path | str | None = None,
        no_rebuild: bool = True,
        cleanup: bool = True,
    ) -> TerminalBenchEnvironment:
        """Create a new terminal-bench environment for a task.

        Args:
            task_id: The terminal-bench task ID
            dataset_name: Dataset name (default: terminal-bench-core)
            dataset_version: Dataset version (default: head)
            logging_dir: Directory for logs/recordings
            no_rebuild: Skip Docker image rebuild if exists
            cleanup: Remove Docker container after use

        Returns:
            Initialized TerminalBenchEnvironment
        """
        from terminal_bench.dataset.dataset import Dataset
        from terminal_bench.handlers.trial_handler import Task, TaskPaths
        from terminal_bench.terminal.terminal import spin_up_terminal

        # Load dataset and find task
        dataset = Dataset(name=dataset_name, version=dataset_version, task_ids=[task_id])
        if not dataset._tasks:
            raise ValueError(f"Task {task_id} not found in {dataset_name}/{dataset_version}")

        task_path = dataset._tasks[0]
        task = Task.from_yaml(task_path / "task.yaml")
        task_paths = TaskPaths(task_path)

        # Setup logging directory
        if logging_dir is None:
            logging_dir = Path(f"runs/tb_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        logging_dir = Path(logging_dir)
        logging_dir.mkdir(parents=True, exist_ok=True)

        # Spin up the terminal (Docker container + tmux)
        terminal_cm = spin_up_terminal(
            client_container_name=f"tb_{task_id}",
            client_image_name=f"tb_{task_id}_image",
            docker_image_name_prefix="terminal-bench",
            docker_compose_path=task_paths.docker_compose_path,
            sessions_logs_path=logging_dir / "sessions",
            agent_logs_path=logging_dir / "agent",
            commands_path=logging_dir / "commands.log",
            no_rebuild=no_rebuild,
            cleanup=cleanup,
            livestream=False,
            disable_recording=task.disable_asciinema,
        )

        # Enter the context manager
        terminal = await trio.to_thread.run_sync(terminal_cm.__enter__)

        # Create tmux session
        session = await trio.to_thread.run_sync(
            lambda: terminal.create_session(
                "agent", is_active_stream=False, as_configured_user=True
            )
        )

        env = cls(
            task_id=task_id,
            instruction=task.instruction,
            terminal=terminal,
            session=session,
            logging_dir=logging_dir,
        )
        env._terminal_cm = terminal_cm

        return env

    async def cleanup(self) -> None:
        """Clean up Docker container and resources."""
        if self._terminal_cm is not None:
            await trio.to_thread.run_sync(lambda: self._terminal_cm.__exit__(None, None, None))

    def get_tools(self) -> list[Tool]:
        """Return tools for interacting with the terminal.

        Tool descriptions match terminal-bench's Terminus Command model
        for consistency with their agent patterns.
        """
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="send_keys",
                    # Description mirrors terminal-bench's Command.keystrokes + is_blocking
                    description="""Send keystrokes to the terminal.

Keystrokes to execute in the terminal. Use tmux-style escape sequences for modifier keys 
(e.g. C-c for ctrl-c). Modifier keys must be sent as their own commands otherwise the 
characters will be interpreted literally.

Set is_blocking=true to wait for and return the terminal output after executing these 
keystrokes. This will append '; tmux wait -S done' to your command.

DO NOT block on modifier keys or inside interactive programs (e.g. vim or less). Only 
block when the command is executed in the command line, is not interactive, and you 
expect the output to be returned with no intervention. When in doubt, don't block.

End commands with a newline character to execute them (e.g., "ls -la\\n").""",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "keystrokes": {
                                "type": "string",
                                "description": "Keystrokes to send. End with newline to execute a command.",
                            },
                            "is_blocking": {
                                "type": "boolean",
                                "description": "Whether to wait for command to complete. Default: true",
                            },
                            "timeout_sec": {
                                "type": "number",
                                "description": "The number of expected seconds to wait for the command to complete. Default: 30",
                            },
                        },
                    ),
                    required=["keystrokes"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="capture_terminal",
                    description="Capture the current terminal screen content. Use this to see command output or check the current state.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "full_history": {
                                "type": "boolean",
                                "description": "Capture full scrollback history, not just visible screen. Default: false",
                            },
                        },
                    ),
                    required=[],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="task_complete",
                    # Matches terminal-bench's CommandBatchResponse.is_task_complete
                    description="""Signal that the task is complete.

Call this when the task is complete. Make sure to check that the command you last 
executed worked before saying you're done.""",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "summary": {
                                "type": "string",
                                "description": "Brief summary of what you did to complete the task.",
                            },
                        },
                    ),
                    required=["summary"],
                ),
            ),
        ]

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """No confirmation needed for terminal operations."""
        return False

    def get_tool_formatter(self, tool_name: str) -> None:
        """Return formatter for tool output display."""
        return None

    def get_status_info(self) -> dict[str, str] | None:
        """Return status info for TUI."""
        return {
            "task": self.task_id,
            "completed": str(self._task_completed),
        }

    def get_system_prompt(self) -> str | None:
        """Return terminal-bench specific system prompt.

        This mirrors the structure of Terminus's prompt template.
        """
        return f"""You are solving a terminal-bench task in a Linux Docker container.

## Task
{self.instruction}

## Available Tools
- send_keys: Send keystrokes to the terminal (end commands with \\n to execute)
- capture_terminal: See the current terminal output
- task_complete: Signal when the task is done

## Strategy
1. First capture_terminal to see the initial state
2. Explore the environment (ls, pwd, cat files)
3. Execute commands to solve the task
4. Verify your solution worked
5. Call task_complete when done

## Important
- Always end shell commands with \\n to execute them
- Use is_blocking=true for commands where you need to see output
- Use is_blocking=false for interactive programs (vim, less, etc.)
- For interactive programs, send individual keystrokes (e.g., C-c for Ctrl+C)
- Make sure to verify your solution before calling task_complete"""

    async def on_session_start(self, session_id: str) -> None:
        """Called when agent session starts."""
        logger.info(f"Terminal-bench session started: {session_id} for task {self.task_id}")

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """No feedback injection needed."""
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Execute a tool call."""
        try:
            if tool_call.name == "send_keys":
                return await self._exec_send_keys(tool_call)
            elif tool_call.name == "capture_terminal":
                return await self._exec_capture_terminal(tool_call)
            elif tool_call.name == "task_complete":
                return await self._exec_task_complete(tool_call)
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Unknown tool: {tool_call.name}",
                )
        except trio.Cancelled:
            raise
        except Exception as e:
            logger.exception(f"Error executing {tool_call.name}")
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=str(e),
            )

    async def _exec_send_keys(self, tool_call: ToolCall) -> ToolResult:
        """Send keystrokes to terminal.

        Mirrors terminal-bench's Terminus._execute_commands logic.
        """
        keystrokes = tool_call.args.get("keystrokes", tool_call.args.get("keys", ""))
        is_blocking = tool_call.args.get("is_blocking", True)
        timeout_sec = tool_call.args.get("timeout_sec", 30.0)

        # Convert \n to actual newline for tmux (agent sends \\n in JSON)
        if isinstance(keystrokes, str):
            keystrokes = keystrokes.replace("\\n", "\n")

        # Match Terminus's blocking logic: don't block on EOF or background commands
        # See: terminal_bench/agents/terminus_1.py _execute_commands
        effective_blocking = is_blocking
        if is_blocking:
            stripped = keystrokes.strip()
            if stripped.endswith("EOF") or stripped.endswith("&"):
                effective_blocking = False

        # Record interaction
        self.interactions.append({
            "type": "send_keys",
            "keystrokes": keystrokes,
            "is_blocking": effective_blocking,
            "timeout_sec": timeout_sec,
            "timestamp": datetime.now().isoformat(),
        })

        try:

            def send() -> str:
                self.session.send_keys(
                    keys=keystrokes,
                    block=effective_blocking,
                    max_timeout_sec=timeout_sec,
                )
                return self.session.capture_pane()

            terminal_output = await trio.to_thread.run_sync(send)

            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=False,
                content=terminal_output,
            )

        except TimeoutError:
            # Match Terminus's timeout handling
            terminal_output = await trio.to_thread.run_sync(lambda: self.session.capture_pane())
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content=terminal_output,
                error=f"Command timed out after {timeout_sec} seconds. Terminal state shown above.",
            )

    async def _exec_capture_terminal(self, tool_call: ToolCall) -> ToolResult:
        """Capture terminal screen."""
        full_history = tool_call.args.get("full_history", False)

        def capture() -> str:
            return self.session.capture_pane(capture_entire=full_history)

        output = await trio.to_thread.run_sync(capture)

        self.interactions.append({
            "type": "capture_terminal",
            "full_history": full_history,
            "timestamp": datetime.now().isoformat(),
        })

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=output,
        )

    async def _exec_task_complete(self, tool_call: ToolCall) -> ToolResult:
        """Mark task as complete.

        Mirrors Terminus's is_task_complete signal.
        """
        summary = tool_call.args.get("summary", "Task completed")

        self._task_completed = True
        self.interactions.append({
            "type": "task_complete",
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        })

        # Save interactions log
        if self.logging_dir:
            log_file = self.logging_dir / "interactions.json"
            log_file.write_text(json.dumps(self.interactions, indent=2))

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Task marked as complete. Summary: {summary}",
            stop_reason=StopReason.TASK_COMPLETED,
        )

    def get_trajectory(self) -> dict[str, Any]:
        """Get the full trajectory for GEPA reflective dataset.

        Returns data in format compatible with GEPA's TerminusAdapter.
        """
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "interactions": self.interactions,
            "success": self._task_completed,
            "failed_reason": "" if self._task_completed else "Task not completed",
        }


# ─── Test Scoring ─────────────────────────────────────────────────────────────


async def run_tests_and_score(
    env: TerminalBenchEnvironment,
) -> tuple[float, bool, str]:
    """Run terminal-bench tests and compute score.

    Mirrors terminal-bench's test harness logic.

    Returns:
        Tuple of (score, success, failure_reason)
    """
    from terminal_bench.dataset.dataset import Dataset
    from terminal_bench.handlers.trial_handler import Task, TaskPaths
    from terminal_bench.parsers.parser_factory import ParserFactory
    from terminal_bench.terminal.docker_compose_manager import DockerComposeManager

    # Get task paths
    dataset = Dataset(name="terminal-bench-core", version="head", task_ids=[env.task_id])
    task_path = dataset._tasks[0]
    task = Task.from_yaml(task_path / "task.yaml")
    task_paths = TaskPaths(task_path)

    def setup_and_run_tests() -> tuple[str | None, str | None]:
        # Copy test script
        env.terminal.copy_to_container(
            paths=[task_paths.run_tests_path],
            container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
        )

        # Copy test directory if it exists
        if task_paths.test_dir.exists():
            env.terminal.copy_to_container(
                paths=[task_paths.test_dir],
                container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
            )

        # Create a new session for tests (not the agent session)
        test_session = env.terminal.create_session(
            "tests", is_active_stream=False, as_configured_user=False
        )

        # Run tests
        test_script = DockerComposeManager.CONTAINER_TEST_DIR / task_paths.run_tests_path.name
        try:
            test_session.send_keys(
                [f"bash {test_script}", "Enter"],
                block=True,
                max_timeout_sec=task.max_test_timeout_sec,
            )
        except TimeoutError:
            return None, "TEST_TIMEOUT"

        return test_session.capture_pane(capture_entire=True), None

    test_output, timeout_error = await trio.to_thread.run_sync(setup_and_run_tests)

    if timeout_error:
        return 0.0, False, timeout_error

    # Parse results using terminal-bench's parser
    try:
        parser = ParserFactory.get_parser(task.parser_name)
        results = parser.parse(test_output)

        if results is None:
            return 0.0, False, "PARSE_ERROR"

        # Compute score as fraction of passed tests
        passed = sum(1 for r in results.values() if str(r) == "UnitTestStatus.PASSED")
        total = len(results)
        score = passed / total if total > 0 else 0.0

        success = all(str(r) == "UnitTestStatus.PASSED" for r in results.values())
        failure_reason = "" if success else f"Failed {total - passed}/{total} tests"

        return score, success, failure_reason

    except Exception as e:
        logger.exception(f"Error parsing test results: {e}")
        return 0.0, False, f"PARSE_ERROR: {e}"
