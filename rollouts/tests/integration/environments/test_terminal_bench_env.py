from __future__ import annotations

from pathlib import Path

import pytest

from rollouts.core import ToolCall
from rollouts.environments.resources import TerminalTaskResult
from rollouts.environments.terminal_bench import (
    TerminalBenchEnvironment,
    create_tbench_environment,
    run_tests_and_score,
)
from rollouts.environments.terminal_bench_coding import TerminalBenchCodingEnvironment


class FakeTerminalTaskResource:
    def __init__(self, logging_dir: Path) -> None:
        self.task_id = "hello-world"
        self.instruction = "print hello"
        self.logging_dir = logging_dir
        self.working_dir = "/workspace"
        self.closed = False
        self.sent: list[tuple[str, bool, float]] = []
        self.capture_calls: list[bool] = []
        self.files: dict[str, bytes] = {
            "/workspace/app.py": b"print('hello')\n",
        }

    async def send_keys(
        self,
        keystrokes: str,
        *,
        is_blocking: bool = True,
        timeout_sec: float = 30.0,
    ) -> str:
        self.sent.append((keystrokes, is_blocking, timeout_sec))
        if "cd src" in keystrokes:
            self.working_dir = "/workspace/src"
        if "__ROLLOUTS_CWD__=" in keystrokes:
            return f"ran: {keystrokes}\n__ROLLOUTS_CWD__={self.working_dir}\n"
        return f"ran: {keystrokes}"

    async def capture_terminal(self, *, full_history: bool = False) -> str:
        self.capture_calls.append(full_history)
        return "terminal output"

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        if path.startswith("/"):
            return path
        return f"{current_working_dir}/{path}"

    async def read_file(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_file(self, path: str, content: bytes) -> None:
        self.files[path] = content

    async def run_tests(self) -> TerminalTaskResult:
        return TerminalTaskResult(score=1.0, success=True, failure_reason="")

    async def close(self) -> None:
        self.closed = True


@pytest.mark.trio
async def test_terminal_bench_environment_uses_injected_resource(tmp_path: Path) -> None:
    resource = FakeTerminalTaskResource(tmp_path)
    env = TerminalBenchEnvironment(resource=resource)

    send_result = await env.exec_tool(
        ToolCall(
            id="tc-1",
            name="send_keys",
            args={"keystrokes": "ls\\n", "is_blocking": True, "timeout_sec": 5.0},
        ),
        current_state=None,
        run_config=None,
    )
    capture_result = await env.exec_tool(
        ToolCall(id="tc-2", name="capture_terminal", args={"full_history": True}),
        current_state=None,
        run_config=None,
    )
    complete_result = await env.exec_tool(
        ToolCall(id="tc-3", name="task_complete", args={"summary": "done"}),
        current_state=None,
        run_config=None,
    )
    score, success, failure_reason = await run_tests_and_score(env)
    await env.cleanup()

    assert send_result.content == "ran: ls\\n"
    assert capture_result.content == "terminal output"
    assert complete_result.stop_reason is not None
    assert resource.sent == [("ls\\n", True, 5.0)]
    assert resource.capture_calls == [True]
    assert score == 1.0
    assert success is True
    assert failure_reason == ""
    assert resource.closed is True
    assert (tmp_path / "interactions.json").exists()


@pytest.mark.trio
async def test_terminal_bench_serialize_round_trip_keeps_live_resource(
    tmp_path: Path,
) -> None:
    resource = FakeTerminalTaskResource(tmp_path)
    env = TerminalBenchEnvironment(resource=resource)

    payload = await env.serialize()
    restored = await TerminalBenchEnvironment.deserialize(payload)

    assert restored.resource is resource
    assert restored.task_id == "hello-world"


@pytest.mark.trio
async def test_create_tbench_environment_uses_selected_surface(tmp_path: Path) -> None:
    resource = FakeTerminalTaskResource(tmp_path)

    env = await create_tbench_environment(surface="terminal", resource=resource)

    assert isinstance(env, TerminalBenchEnvironment)
    assert env.resource is resource


@pytest.mark.trio
async def test_create_tbench_environment_supports_coding_surface(tmp_path: Path) -> None:
    resource = FakeTerminalTaskResource(tmp_path)

    env = await create_tbench_environment(surface="coding", resource=resource)

    assert isinstance(env, TerminalBenchCodingEnvironment)
    assert env.resource is resource


@pytest.mark.trio
async def test_create_tbench_environment_rejects_unknown_surface(tmp_path: Path) -> None:
    resource = FakeTerminalTaskResource(tmp_path)

    with pytest.raises(ValueError, match="Unknown Terminal-Bench surface"):
        await create_tbench_environment(surface="invalid", resource=resource)


@pytest.mark.trio
async def test_terminal_bench_coding_environment_uses_workspace_resource(
    tmp_path: Path,
) -> None:
    resource = FakeTerminalTaskResource(tmp_path)
    env = TerminalBenchCodingEnvironment(resource=resource)

    read_result = await env.exec_tool(
        ToolCall(id="tc-read", name="read", args={"path": "app.py"}),
        current_state=None,
        run_config=None,
    )
    edit_result = await env.exec_tool(
        ToolCall(
            id="tc-edit",
            name="edit",
            args={"path": "app.py", "old_text": "hello", "new_text": "world"},
        ),
        current_state=None,
        run_config=None,
    )
    bash_result = await env.exec_tool(
        ToolCall(id="tc-bash", name="bash", args={"command": "cd src && pwd", "timeout": 5}),
        current_state=None,
        run_config=None,
    )
    write_result = await env.exec_tool(
        ToolCall(id="tc-write", name="write", args={"path": "notes.txt", "content": "done\n"}),
        current_state=None,
        run_config=None,
    )

    assert read_result.content == "print('hello')\n"
    assert edit_result.is_error is False
    assert resource.files["/workspace/app.py"] == b"print('world')\n"
    assert bash_result.content.startswith("ran:")
    assert env.current_working_dir == "/workspace/src"
    assert write_result.is_error is False
    assert resource.files["/workspace/src/notes.txt"] == b"done\n"
