from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class KernelEvaluator(Protocol):
    async def start(self) -> None: ...

    async def score_one(
        self,
        kernel_code: str,
        ref_code: str,
        timeout: float,
    ) -> dict[str, Any]: ...


@runtime_checkable
class BatchKernelEvaluator(KernelEvaluator, Protocol):
    async def score_batch(
        self,
        requests: list[dict[str, Any]],
        timeout: float,
    ) -> list[dict[str, Any]]: ...

    def stats(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TerminalTaskResult:
    score: float
    success: bool
    failure_reason: str
    output: str | None = None


@dataclass(frozen=True)
class CommandExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    cwd: str | None = None


@runtime_checkable
class CodingWorkspaceResource(Protocol):
    working_dir: str

    def resolve_path(self, current_working_dir: str, path: str) -> str: ...

    async def read_file(self, path: str) -> bytes: ...

    async def write_file(self, path: str, content: bytes) -> None: ...


@runtime_checkable
class CommandRunner(Protocol):
    async def run(
        self,
        command: str,
        *,
        cwd: str,
        timeout: float,
        session_id: str | None = None,
        cancel_scope: Any | None = None,
    ) -> CommandExecutionResult: ...


@runtime_checkable
class SandboxWorkspaceResource(CodingWorkspaceResource, CommandRunner, Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def describe_runtime(self) -> dict[str, Any]: ...

    def stats(self) -> dict[str, Any]: ...


@runtime_checkable
class TerminalTaskResource(Protocol):
    task_id: str
    instruction: str

    async def send_keys(
        self,
        keystrokes: str,
        *,
        is_blocking: bool = True,
        timeout_sec: float = 30.0,
    ) -> str: ...

    async def capture_terminal(self, *, full_history: bool = False) -> str: ...

    async def run_tests(self) -> TerminalTaskResult: ...

    async def close(self) -> None: ...


@runtime_checkable
class TerminalWorkspaceResource(TerminalTaskResource, CodingWorkspaceResource, Protocol):
    pass
