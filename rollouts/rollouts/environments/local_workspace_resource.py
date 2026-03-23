"""Local tempdir-backed workspace resource.

Satisfies both CodingWorkspaceResource and CommandRunner, so CodingEnvironment
can take it directly as workspace= and command_runner=.

Usage pattern (following the Environment.deserialize convention):

    def row_to_state(row: dict) -> dict:
        return {
            "source_dir": row["challenge_root"],
            "working_dir": "/workspace",  # relative path inside copy
        }

    class MyEnvironment:
        @staticmethod
        async def deserialize(state: dict) -> MyEnvironment:
            resource = await LocalWorkspaceResource.create(
                source_dir=Path(state["source_dir"]),
            )
            return MyEnvironment(resource=resource)

        async def close(self) -> None:
            await self.resource.close()  # removes tempdir
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trio

from .resources import CommandExecutionResult
from .runtime_probe import build_gpu_runtime_probe_script


@dataclass
class LocalWorkspaceResource:
    """Workspace resource backed by a local tempdir copy of a source directory.

    The source directory is copied into a fresh tempdir on start(). The agent
    operates against the copy — the original source is never modified.

    Satisfies CodingWorkspaceResource and CommandRunner, so it can be passed
    directly to CodingEnvironment as both workspace= and command_runner=.

    TODO(session-first): model this as a local `InspectableRemoteSession`
    analogue plus `SessionBackedWorkspaceHandle` so local, Docker, Modal, and
    SSH-backed sandboxes all share the same substrate boundary.
    """

    source_dir: Path
    logical_working_dir: str | None = None
    _working_dir: str = field(default="", repr=False)
    _tempdir: str | None = field(default=None, repr=False)
    _runtime_description: dict[str, Any] | None = field(default=None, repr=False)

    @classmethod
    async def create(cls, source_dir: Path) -> LocalWorkspaceResource:
        """Create and start a workspace from source_dir."""
        resource = cls(source_dir=source_dir)
        await resource.start()
        return resource

    async def start(self) -> None:
        if self._tempdir is not None:
            return
        assert self.source_dir.is_dir(), f"source_dir does not exist: {self.source_dir}"

        def _copy() -> str:
            tmp = tempfile.mkdtemp(prefix="rollouts-workspace-")
            shutil.copytree(
                self.source_dir,
                tmp,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
            )
            return tmp

        self._tempdir = await trio.to_thread.run_sync(_copy)
        self._working_dir = self.logical_working_dir or self._tempdir

    async def close(self) -> None:
        if self._tempdir is None:
            return

        def _rm() -> None:
            shutil.rmtree(self._tempdir, ignore_errors=True)

        await trio.to_thread.run_sync(_rm)
        self._tempdir = None
        self._working_dir = ""

    # ── CodingWorkspaceResource ───────────────────────────────────────────────

    @property
    def working_dir(self) -> str:
        return self._working_dir or self.logical_working_dir or str(self.source_dir)

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        if not path:
            return current_working_dir
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(Path(current_working_dir) / p)

    def _host_root(self) -> Path:
        return Path(self._tempdir) if self._tempdir is not None else self.source_dir

    def _to_host_path(self, path: str) -> str:
        if self.logical_working_dir is None:
            return path
        logical_root = Path(self.logical_working_dir)
        pure = Path(path)
        if pure.is_absolute():
            try:
                relative = pure.relative_to(logical_root)
            except ValueError:
                return str(pure)
            return str(self._host_root() / relative)
        return str(self._host_root() / pure)

    async def read_file(self, path: str) -> bytes:
        resolved = self.resolve_path(self.working_dir, path)
        host_path = self._to_host_path(resolved)
        return await trio.to_thread.run_sync(lambda: Path(host_path).read_bytes())

    async def write_file(self, path: str, content: bytes) -> None:
        resolved = self.resolve_path(self.working_dir, path)
        host_path = self._to_host_path(resolved)

        def _write() -> None:
            p = Path(host_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)

        await trio.to_thread.run_sync(_write)

    # ── CommandRunner ─────────────────────────────────────────────────────────

    async def run(
        self,
        command: str,
        *,
        cwd: str,
        timeout: float,
        session_id: str | None = None,
        cancel_scope: Any | None = None,
    ) -> CommandExecutionResult:
        del session_id, cancel_scope

        def _run() -> tuple[str, str, int]:
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=self._to_host_path(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout, result.stderr, result.returncode

        try:
            stdout, stderr, returncode = await trio.to_thread.run_sync(_run)
        except subprocess.TimeoutExpired:
            return CommandExecutionResult(
                returncode=124,
                stdout="",
                stderr=f"command timed out after {timeout}s",
                cwd=cwd,
            )

        return CommandExecutionResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            cwd=cwd,
        )

    # ── Optional introspection ────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "kind": "local_workspace_resource",
            "source_dir": str(self.source_dir),
            "logical_working_dir": self.logical_working_dir,
            "working_dir": self._working_dir,
            "started": self._tempdir is not None,
            "runtime": self._runtime_description,
        }

    async def describe_runtime(self) -> dict[str, Any]:
        script = build_gpu_runtime_probe_script()
        command = f"""
python3 << 'RUNTIME_PROBE_EOF'
{script}
RUNTIME_PROBE_EOF
"""
        result = await self.run(
            command,
            cwd=self.working_dir,
            timeout=60.0,
        )
        if result.returncode != 0:
            self._runtime_description = {
                "runtime_ok": False,
                "error": result.stderr or result.stdout or "runtime probe failed",
                "errors": [result.stderr or result.stdout or "runtime probe failed"],
                "kind": "local_workspace_resource",
                "source_dir": str(self.source_dir),
                "working_dir": self._working_dir,
            }
            return self._runtime_description
        try:
            runtime = json.loads(result.stdout)
        except json.JSONDecodeError:
            runtime = {
                "runtime_ok": False,
                "error": f"invalid runtime probe output: {result.stdout}",
                "errors": [result.stderr] if result.stderr else [],
            }
        runtime.setdefault("kind", "local_workspace_resource")
        runtime.setdefault("source_dir", str(self.source_dir))
        runtime.setdefault("working_dir", self._working_dir)
        self._runtime_description = runtime
        return runtime

    def serialize_state(self) -> dict[str, Any]:
        return {
            "kind": "local_workspace_resource",
            "source_dir": str(self.source_dir),
            "logical_working_dir": self.logical_working_dir,
            "working_dir": self._working_dir,
            "tempdir": self._tempdir,
            "runtime": self._runtime_description,
        }

    @classmethod
    def deserialize_state(cls, data: dict[str, Any]) -> LocalWorkspaceResource:
        resource = cls(
            source_dir=Path(data["source_dir"]),
            logical_working_dir=data.get("logical_working_dir"),
        )
        resource._working_dir = str(data.get("working_dir", ""))
        resource._tempdir = data.get("tempdir")
        resource._runtime_description = data.get("runtime")
        return resource
