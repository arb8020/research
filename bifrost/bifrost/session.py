"""Public async execution-session surface for bifrost."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Protocol, runtime_checkable

from .async_client import AsyncBifrostClient
from .types import (
    EnvironmentVariables,
    ExecResult,
    ObservedProcessHandle,
    ProcessOutputLine,
    ProcessSpec,
    ServiceHandle,
    ServiceSpec,
    WorkspaceHandle,
    WorkspaceMaterializationSpec,
)


@runtime_checkable
class ExecutionSession(Protocol):
    """Provider-agnostic live async execution session."""

    backend: str

    def current_workspace(self) -> WorkspaceHandle | None:
        """Return the last materialized workspace, if one exists."""

    async def materialize(self, spec: WorkspaceMaterializationSpec) -> WorkspaceHandle:
        """Materialize a project snapshot according to a backend-agnostic plan."""

    async def materialize_workspace(
        self,
        workspace_path: str,
        bootstrap_cmd: str | list[str] | None = None,
        on_bootstrap_step: Callable[[str, int, int], None] | None = None,
        allow_dirty: bool = False,
    ) -> WorkspaceHandle:
        """Materialize a project snapshot into a remote workspace."""

    async def exec(
        self,
        command: str,
        env: EnvironmentVariables | dict[str, str] | None = None,
        working_dir: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute a raw command synchronously."""

    async def run(self, spec: ProcessSpec, timeout: float | None = None) -> ExecResult:
        """Execute a structured one-shot process synchronously."""

    async def start_process(
        self,
        spec: ProcessSpec,
        *,
        name: str | None = None,
        workspace: WorkspaceHandle | None = None,
        timeout: float | None = None,
        log_file: str | None = None,
    ) -> ObservedProcessHandle:
        """Launch and observe a live attached process."""

    async def serve_service(
        self,
        service: ServiceSpec,
        *,
        name: str,
        workspace: WorkspaceHandle | None = None,
        log_file: str | None = None,
    ) -> ServiceHandle:
        """Launch a long-lived service with explicit readiness semantics."""

    async def stream_exec(
        self,
        command: str,
        env: EnvironmentVariables | dict[str, str] | None = None,
        working_dir: str | None = None,
    ) -> AsyncIterator[ProcessOutputLine]:
        """Execute a raw command and stream typed stdout/stderr output."""


async def connect(ssh_connection: str, ssh_key_path: str, timeout: int = 30) -> ExecutionSession:
    """Connect to an execution resource via the current async SSH backend."""

    client = AsyncBifrostClient(
        ssh_connection=ssh_connection,
        ssh_key_path=ssh_key_path,
        timeout=timeout,
    )
    await client._get_connection()
    return client
