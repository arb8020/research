"""Terminal-Bench environment built around an injected task resource.

The important boundary is:
- the task resource owns container/session/test execution
- the environment exposes the tool contract used by rollouts agents

That matches Harbor and Terminal-Bench's installed-agent model more closely than
having the environment directly own every terminal-bench primitive.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import posixpath
import re
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

from ..agents import AgentState, RunConfig
from ..core import (
    Message,
    StopReason,
    Tool,
    ToolCall,
    ToolFunction,
    ToolFunctionParameter,
    ToolResult,
)
from .resources import TerminalTaskResource, TerminalTaskResult

if TYPE_CHECKING:
    from terminal_bench.handlers.trial_handler import Task, TaskPaths
    from terminal_bench.terminal.terminal import Terminal
    from terminal_bench.terminal.tmux_session import TmuxSession

logger = logging.getLogger(__name__)
TBENCH_SURFACES = ("terminal", "coding")


def _docker_label(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "run"


def _docker_resource_names(task_id: str, logging_path: Path) -> tuple[str, str]:
    assert task_id.strip(), "task_id cannot be empty"
    digest = hashlib.sha1(str(logging_path.resolve()).encode("utf-8")).hexdigest()[:10]
    assert digest, "logging_path digest cannot be empty"
    prefix = _docker_label(task_id)[:32]
    base = f"tb-{prefix}-{digest}"
    assert base.startswith("tb-"), "docker resource base name must start with tb-"
    return base, f"{base}-image"


@dataclass
class DefaultTerminalBenchTaskResource:
    task_id: str
    instruction: str
    task: Task
    task_paths: TaskPaths
    terminal: Terminal
    session: TmuxSession
    working_dir: str
    logging_dir: Path | None = None
    _terminal_cm: Any = None
    _test_session_count: int = 0

    @classmethod
    async def create(
        cls,
        task_id: str,
        dataset_name: str = "terminal-bench-core",
        dataset_version: str = "head",
        logging_dir: Path | str | None = None,
        no_rebuild: bool = True,
        cleanup: bool = True,
    ) -> DefaultTerminalBenchTaskResource:
        from terminal_bench.dataset.dataset import Dataset
        from terminal_bench.handlers.trial_handler import Task, TaskPaths
        from terminal_bench.terminal.terminal import spin_up_terminal

        dataset = Dataset(name=dataset_name, version=dataset_version, task_ids=[task_id])
        if not dataset._tasks:
            raise ValueError(f"Task {task_id} not found in {dataset_name}/{dataset_version}")

        task_path = dataset._tasks[0]
        task = Task.from_yaml(task_path / "task.yaml")
        task_paths = TaskPaths(task_path)

        if logging_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logging_dir = Path(f"runs/tb_{task_id}_{timestamp}")
        logging_path = Path(logging_dir)
        logging_path.mkdir(parents=True, exist_ok=True)
        container_name, image_name = _docker_resource_names(task_id, logging_path)

        terminal_cm = spin_up_terminal(
            client_container_name=container_name,
            client_image_name=image_name,
            docker_image_name_prefix="terminal-bench",
            docker_compose_path=task_paths.docker_compose_path,
            sessions_logs_path=logging_path / "sessions",
            agent_logs_path=logging_path / "agent",
            commands_path=logging_path / "commands.log",
            no_rebuild=no_rebuild,
            cleanup=cleanup,
            livestream=False,
            disable_recording=task.disable_asciinema,
        )
        terminal = await trio.to_thread.run_sync(terminal_cm.__enter__)
        session = await trio.to_thread.run_sync(
            lambda: terminal.create_session(
                "agent",
                is_active_stream=False,
                as_configured_user=True,
            )
        )
        working_dir = terminal.container.attrs.get("Config", {}).get("WorkingDir") or "/"

        return cls(
            task_id=task_id,
            instruction=task.instruction,
            task=task,
            task_paths=task_paths,
            terminal=terminal,
            session=session,
            working_dir=working_dir,
            logging_dir=logging_path,
            _terminal_cm=terminal_cm,
        )

    async def send_keys(
        self,
        keystrokes: str,
        *,
        is_blocking: bool = True,
        timeout_sec: float = 30.0,
    ) -> str:
        effective_blocking = is_blocking
        stripped = keystrokes.strip()
        if is_blocking and (stripped.endswith("EOF") or stripped.endswith("&")):
            effective_blocking = False

        if isinstance(keystrokes, str):
            keystrokes = keystrokes.replace("\\n", "\n")

        def send() -> str:
            self.session.send_keys(
                keys=keystrokes,
                block=effective_blocking,
                max_timeout_sec=timeout_sec,
            )
            return self.session.capture_pane()

        return await trio.to_thread.run_sync(send)

    async def capture_terminal(self, *, full_history: bool = False) -> str:
        return await trio.to_thread.run_sync(
            lambda: self.session.capture_pane(capture_entire=full_history)
        )

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        if posixpath.isabs(path):
            return posixpath.normpath(path)
        return posixpath.normpath(posixpath.join(current_working_dir, path))

    async def read_file(self, path: str) -> bytes:
        def _read() -> bytes:
            assert self.terminal.container is not None
            try:
                stream, _stat = self.terminal.container.get_archive(path)
            except Exception as exc:
                raise FileNotFoundError(path) from exc

            archive = io.BytesIO()
            for chunk in stream:
                archive.write(chunk)
            archive.seek(0)

            with tarfile.open(fileobj=archive, mode="r:*") as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                if not members:
                    raise IsADirectoryError(path)
                extracted = tar.extractfile(members[0])
                if extracted is None:
                    raise FileNotFoundError(path)
                return extracted.read()

        return await trio.to_thread.run_sync(_read)

    async def write_file(self, path: str, content: bytes) -> None:
        def _write() -> None:
            assert self.terminal.container is not None
            parent_dir = posixpath.dirname(path) or "/"
            file_name = posixpath.basename(path)
            mkdir_result = self.terminal.container.exec_run(["mkdir", "-p", parent_dir])
            if mkdir_result.exit_code != 0:
                raise RuntimeError(f"Failed to create directory: {parent_dir}")

            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                info = tarfile.TarInfo(name=file_name)
                info.size = len(content)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(content))
            tar_stream.seek(0)

            ok = self.terminal.container.put_archive(parent_dir, tar_stream.read())
            if not ok:
                raise RuntimeError(f"Failed to write file: {path}")

        await trio.to_thread.run_sync(_write)

    async def run_tests(self) -> TerminalTaskResult:
        from terminal_bench.parsers.parser_factory import ParserFactory
        from terminal_bench.terminal.docker_compose_manager import DockerComposeManager

        def setup_and_run_tests() -> tuple[str | None, str | None]:
            self.terminal.copy_to_container(
                paths=[self.task_paths.run_tests_path],
                container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
            )
            if self.task_paths.test_dir.exists():
                self.terminal.copy_to_container(
                    paths=[self.task_paths.test_dir],
                    container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
                )

            if self.task.run_tests_in_same_shell:
                test_session = self.session
            else:
                session_name = f"tests-{self._test_session_count}"
                self._test_session_count += 1
                test_session = self.terminal.create_session(
                    session_name,
                    is_active_stream=False,
                    as_configured_user=False,
                )

            test_script = (
                DockerComposeManager.CONTAINER_TEST_DIR / self.task_paths.run_tests_path.name
            )
            try:
                test_session.send_keys(
                    [f"bash {test_script}", "Enter"],
                    block=True,
                    max_timeout_sec=self.task.max_test_timeout_sec,
                )
            except TimeoutError:
                return None, "TEST_TIMEOUT"

            return test_session.capture_pane(capture_entire=True), None

        test_output, timeout_error = await trio.to_thread.run_sync(setup_and_run_tests)
        if timeout_error:
            return TerminalTaskResult(
                score=0.0,
                success=False,
                failure_reason=timeout_error,
            )

        if test_output is None:
            return TerminalTaskResult(
                score=0.0,
                success=False,
                failure_reason="TEST_OUTPUT_MISSING",
            )

        try:
            parser = ParserFactory.get_parser(self.task.parser_name)
            results = parser.parse(test_output)
            if results is None:
                return TerminalTaskResult(
                    score=0.0,
                    success=False,
                    failure_reason="PARSE_ERROR",
                    output=test_output,
                )

            passed = sum(1 for result in results.values() if str(result) == "UnitTestStatus.PASSED")
            total = len(results)
            score = passed / total if total else 0.0
            success = all(str(result) == "UnitTestStatus.PASSED" for result in results.values())
            failure_reason = "" if success else f"Failed {total - passed}/{total} tests"
            return TerminalTaskResult(
                score=score,
                success=success,
                failure_reason=failure_reason,
                output=test_output,
            )
        except Exception as exc:
            logger.exception("Error parsing terminal-bench test output")
            return TerminalTaskResult(
                score=0.0,
                success=False,
                failure_reason=f"PARSE_ERROR: {exc}",
                output=test_output,
            )

    async def close(self) -> None:
        if self._terminal_cm is not None:
            await trio.to_thread.run_sync(lambda: self._terminal_cm.__exit__(None, None, None))


@dataclass
class TerminalBenchEnvironment:
    resource: TerminalTaskResource
    interactions: list[dict[str, Any]] = field(default_factory=list)
    _task_completed: bool = False

    @property
    def task_id(self) -> str:
        return self.resource.task_id

    @property
    def instruction(self) -> str:
        return self.resource.instruction

    @property
    def logging_dir(self) -> Path | None:
        return getattr(self.resource, "logging_dir", None)

    @property
    def max_agent_timeout_sec(self) -> float | None:
        task = getattr(self.resource, "task", None)
        return getattr(task, "max_agent_timeout_sec", None)

    @property
    def max_test_timeout_sec(self) -> float | None:
        task = getattr(self.resource, "task", None)
        return getattr(task, "max_test_timeout_sec", None)

    @property
    def terminal(self) -> Any | None:
        return getattr(self.resource, "terminal", None)

    @property
    def session(self) -> Any | None:
        return getattr(self.resource, "session", None)

    async def serialize(self) -> dict[str, Any]:
        return {
            "env_kind": "terminal_bench",
            "task_id": self.task_id,
            "instruction": self.instruction,
            "interactions": self.interactions,
            "task_completed": self._task_completed,
            "_resource_ref": self.resource,
        }

    @staticmethod
    async def deserialize(data: dict[str, Any]) -> TerminalBenchEnvironment:
        resource = data.get("_resource_ref")
        if resource is None:
            raise NotImplementedError(
                "TerminalBenchEnvironment cannot be deserialized from cold storage "
                "because the live terminal resource is not persistable."
            )
        return TerminalBenchEnvironment(
            resource=resource,
            interactions=data.get("interactions", []),
            _task_completed=data.get("task_completed", False),
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
        resource: TerminalTaskResource | None = None,
    ) -> TerminalBenchEnvironment:
        if resource is None:
            resource = await DefaultTerminalBenchTaskResource.create(
                task_id=task_id,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                logging_dir=logging_dir,
                no_rebuild=no_rebuild,
                cleanup=cleanup,
            )
        return cls(resource=resource)

    async def close(self) -> None:
        await self.resource.close()

    def get_tools(self) -> list[Tool]:
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="send_keys",
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
        return False

    def get_tool_formatter(self, tool_name: str) -> None:
        return None

    def get_status_info(self) -> dict[str, str] | None:
        return {
            "task": self.task_id,
            "completed": str(self._task_completed),
        }

    def get_system_prompt(self) -> str | None:
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
        logger.info(
            "Terminal-bench session started: %s for task %s",
            session_id,
            self.task_id,
        )

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
            if tool_call.name == "send_keys":
                return await self._exec_send_keys(tool_call)
            if tool_call.name == "capture_terminal":
                return await self._exec_capture_terminal(tool_call)
            if tool_call.name == "task_complete":
                return await self._exec_task_complete(tool_call)
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Unknown tool: {tool_call.name}",
            )
        except trio.Cancelled:
            raise
        except Exception as exc:
            logger.exception("Error executing terminal-bench tool %s", tool_call.name)
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=str(exc),
            )

    async def _exec_send_keys(self, tool_call: ToolCall) -> ToolResult:
        keystrokes = tool_call.args.get("keystrokes", tool_call.args.get("keys", ""))
        is_blocking = tool_call.args.get("is_blocking", True)
        timeout_sec = tool_call.args.get("timeout_sec", 30.0)

        self.interactions.append({
            "type": "send_keys",
            "keystrokes": keystrokes,
            "is_blocking": is_blocking,
            "timeout_sec": timeout_sec,
            "timestamp": datetime.now().isoformat(),
        })

        try:
            terminal_output = await self.resource.send_keys(
                keystrokes,
                is_blocking=is_blocking,
                timeout_sec=timeout_sec,
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=False,
                content=terminal_output,
            )
        except TimeoutError:
            terminal_output = await self.resource.capture_terminal(full_history=False)
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content=terminal_output,
                error=f"Command timed out after {timeout_sec} seconds. Terminal state shown above.",
            )

    async def _exec_capture_terminal(self, tool_call: ToolCall) -> ToolResult:
        full_history = tool_call.args.get("full_history", False)
        output = await self.resource.capture_terminal(full_history=full_history)
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
        summary = tool_call.args.get("summary", "Task completed")
        self._task_completed = True
        self.interactions.append({
            "type": "task_complete",
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        })

        if self.logging_dir is not None:
            log_file = self.logging_dir / "interactions.json"
            log_file.write_text(json.dumps(self.interactions, indent=2))

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Task marked as complete. Summary: {summary}",
            stop_reason=StopReason.TASK_COMPLETED,
        )

    async def run_tests(self) -> TerminalTaskResult:
        return await self.resource.run_tests()

    def get_trajectory(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "interactions": self.interactions,
            "success": self._task_completed,
            "failed_reason": "" if self._task_completed else "Task not completed",
        }


async def run_tests_and_score(
    env: TerminalBenchEnvironment,
) -> tuple[float, bool, str]:
    result = await env.run_tests()
    return result.score, result.success, result.failure_reason


async def create_tbench_resource(
    *,
    task_id: str,
    dataset_name: str = "terminal-bench-core",
    dataset_version: str = "head",
    logging_dir: Path | str | None = None,
    no_rebuild: bool = True,
    cleanup: bool = True,
) -> TerminalTaskResource:
    return await DefaultTerminalBenchTaskResource.create(
        task_id=task_id,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        logging_dir=logging_dir,
        no_rebuild=no_rebuild,
        cleanup=cleanup,
    )


async def create_tbench_environment(
    *,
    surface: str = "terminal",
    resource: TerminalTaskResource,
) -> Any:
    if surface == "terminal":
        return TerminalBenchEnvironment(resource=resource)
    if surface == "coding":
        from .terminal_bench_coding import (
            TerminalBenchCodingEnvironment,
            supports_terminal_workspace,
        )

        if not supports_terminal_workspace(resource):
            raise ValueError("Terminal-Bench coding surface requires workspace file access")
        return TerminalBenchCodingEnvironment(resource=resource)

    raise ValueError(
        f"Unknown Terminal-Bench surface: {surface}. "
        f"Available surfaces: {', '.join(TBENCH_SURFACES)}"
    )
