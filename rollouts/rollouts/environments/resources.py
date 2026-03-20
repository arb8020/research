from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

# TODO(resource-boundary): resources currently blur three distinct things:
# - serializable state needed to rehydrate a capability later
# - live capability handles like workspaces/sessions/evaluators
# - environment-facing tool surfaces built from those capabilities
#
# The intended split is:
# 1. ResourceState: serializable handle/reference, no tools
# 2. ResourceHandle: live capability object, no task semantics
# 3. Environment: owns tool exposure plus how resource state participates in
#    environment ser/deser
#
# KernelBench is the forcing example:
# - workspace/evaluator should be injected as capabilities
# - workspace-style SDK-agent KernelBench should expose coding/kernel tools at
#   the environment layer
# - external-agent KernelBench can still use attempt_executor, but should
#   consume the same underlying resource/state boundary


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


@dataclass(frozen=True)
class SessionExecSpec:
    """Process spec at the execution-substrate boundary.

    This is intentionally smaller than the current workspace-facing protocol.
    A session knows how to execute a command and move bytes to/from remote
    paths; workspace semantics are layered on top.
    """

    command: str
    cwd: str
    timeout: float
    session_id: str | None = None
    cancel_scope: Any | None = None


@runtime_checkable
class RemoteSession(Protocol):
    """Live execution/session substrate.

    Concrete implementations may lower this into Modal sandbox calls, SSH/Bifrost
    exec + SFTP, or `docker exec` + `docker cp`.
    """

    async def exec(self, spec: SessionExecSpec) -> CommandExecutionResult: ...

    async def upload_bytes(self, remote_path: str, content: bytes) -> None: ...

    async def download_bytes(self, remote_path: str) -> bytes: ...

    async def close(self) -> None: ...


@runtime_checkable
class InspectableRemoteSession(RemoteSession, Protocol):
    async def describe_runtime(self) -> dict[str, Any]: ...

    def stats(self) -> dict[str, Any]: ...


@runtime_checkable
class CodingWorkspaceResource(Protocol):
    # TODO(session-first): this is a convenience facade for environment/tool
    # code, not the honest execution substrate. New backends should prefer
    # implementing `InspectableRemoteSession` and only expose this shape via
    # `SessionBackedWorkspaceHandle`.
    working_dir: str

    def resolve_path(self, current_working_dir: str, path: str) -> str: ...

    async def read_file(self, path: str) -> bytes: ...

    async def write_file(self, path: str, content: bytes) -> None: ...


@runtime_checkable
class CommandRunner(Protocol):
    # TODO(session-first): fold this into the lower-level session boundary over
    # time. Keeping it separate preserves compatibility with existing coding
    # environments while we migrate concrete backends.
    async def run(
        self,
        command: str,
        *,
        cwd: str,
        timeout: float,
        session_id: str | None = None,
        cancel_scope: Any | None = None,
    ) -> CommandExecutionResult: ...


@dataclass
class SessionBackedWorkspaceHandle:
    """Task-facing workspace façade layered over a lower-level remote session.

    The environment layer should usually depend on a workspace handle. Modal,
    RunPod/SSH, or local Docker should implement the lower-level
    `InspectableRemoteSession`, then opt into the existing workspace semantics
    by wrapping it here.

    TODO(session-first): once the concrete backends migrate, make this the
    primary way workspace semantics are constructed instead of having backends
    implement the flattened workspace protocol directly.
    """

    session: InspectableRemoteSession
    working_dir: str

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        if not path:
            return current_working_dir
        pure = PurePosixPath(path)
        if pure.is_absolute():
            return str(pure)
        return str(PurePosixPath(current_working_dir) / pure)

    async def read_file(self, path: str) -> bytes:
        resolved = self.resolve_path(self.working_dir, path)
        return await self.session.download_bytes(resolved)

    async def write_file(self, path: str, content: bytes) -> None:
        resolved = self.resolve_path(self.working_dir, path)
        await self.session.upload_bytes(resolved, content)

    async def run(
        self,
        command: str,
        *,
        cwd: str,
        timeout: float,
        session_id: str | None = None,
        cancel_scope: Any | None = None,
    ) -> CommandExecutionResult:
        return await self.session.exec(
            SessionExecSpec(
                command=command,
                cwd=self.resolve_path(self.working_dir, cwd),
                timeout=timeout,
                session_id=session_id,
                cancel_scope=cancel_scope,
            )
        )

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        await self.session.close()

    async def describe_runtime(self) -> dict[str, Any]:
        return await self.session.describe_runtime()

    def stats(self) -> dict[str, Any]:
        return self.session.stats()


@runtime_checkable
class SandboxWorkspaceResource(CodingWorkspaceResource, CommandRunner, Protocol):
    # TODO(session-first): this currently reads like the core sandbox protocol,
    # but it is really an adapter shape. After the backend migration, narrow
    # environment-facing code to `SessionBackedWorkspaceHandle` or another
    # explicit workspace facade built on `InspectableRemoteSession`.
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
