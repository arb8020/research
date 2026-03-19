"""Docker-backed workspace resource.

Satisfies both CodingWorkspaceResource and CommandRunner, so CodingEnvironment
can take it directly as workspace= and command_runner=.

Each instance owns one Docker container. start() runs the container, close()
removes it. The source directory (if provided) is copied into the container's
working_dir on start.

Requires: Docker daemon running, `docker` CLI on PATH.

Usage pattern (following the Environment.deserialize convention):

    def row_to_state(row: dict) -> dict:
        return {
            "image": "python:3.12-slim",
            "source_dir": row["challenge_root"],
            "working_dir": "/workspace",
        }

    class MyEnvironment:
        @staticmethod
        async def deserialize(state: dict) -> MyEnvironment:
            resource = await DockerWorkspaceResource.create(
                image=state["image"],
                source_dir=Path(state["source_dir"]),
                working_dir=state["working_dir"],
            )
            return MyEnvironment(resource=resource)

        async def close(self) -> None:
            await self.resource.close()
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trio

from .resources import CommandExecutionResult


@dataclass
class DockerWorkspaceResourceConfig:
    image: str
    working_dir: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)
    # Shell commands run inside the container after it starts and source is copied in.
    # Use for dependency installation, tool setup, etc.
    setup_commands: tuple[str, ...] = ()


@dataclass
class DockerWorkspaceResource:
    """Workspace resource backed by a Docker container.

    The container runs a long-lived `sleep infinity` process. Commands execute
    via `docker exec`. The source directory is copied into the container's
    working_dir with `docker cp` on start.

    Satisfies CodingWorkspaceResource and CommandRunner, so it can be passed
    directly to CodingEnvironment as both workspace= and command_runner=.
    """

    config: DockerWorkspaceResourceConfig
    source_dir: Path | None = None
    _container_id: str | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)

    @classmethod
    async def create(
        cls,
        image: str,
        source_dir: Path | None = None,
        working_dir: str = "/workspace",
        env: dict[str, str] | None = None,
        setup_commands: tuple[str, ...] = (),
    ) -> DockerWorkspaceResource:
        """Create and start a workspace."""
        config = DockerWorkspaceResourceConfig(
            image=image,
            working_dir=working_dir,
            env=env or {},
            setup_commands=setup_commands,
        )
        resource = cls(config=config, source_dir=source_dir)
        await resource.start()
        return resource

    async def start(self) -> None:
        assert not self._started, "DockerWorkspaceResource already started"

        env_flags: list[str] = []
        for k, v in self.config.env.items():
            env_flags += ["-e", f"{k}={v}"]

        def _run_container() -> str:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--rm",  # auto-remove when stopped
                    *env_flags,
                    self.config.image,
                    "sleep",
                    "infinity",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()

        self._container_id = await trio.to_thread.run_sync(_run_container)
        self._started = True

        # Ensure working_dir exists inside the container
        await self.run(
            f"mkdir -p {self.config.working_dir}",
            cwd=self.config.working_dir,
            timeout=10.0,
        )

        # Copy source directory in if provided
        if self.source_dir is not None:
            assert self.source_dir.is_dir(), f"source_dir does not exist: {self.source_dir}"
            await self._copy_source_in(self.source_dir)

        # Run setup commands
        for cmd in self.config.setup_commands:
            result = await self.run(cmd, cwd=self.config.working_dir, timeout=300.0)
            if result.returncode != 0:
                raise RuntimeError(
                    f"setup command failed (rc={result.returncode}): {cmd}\nstderr: {result.stderr}"
                )

    async def close(self) -> None:
        if self._container_id is None:
            return

        def _stop() -> None:
            subprocess.run(
                ["docker", "stop", "-t", "5", self._container_id],
                capture_output=True,
                check=False,
            )

        await trio.to_thread.run_sync(_stop)
        self._container_id = None
        self._started = False

    async def reset(self) -> None:
        """Stop and restart the container with a fresh copy of the source."""
        await self.close()
        self._started = False
        await self.start()

    # ── CodingWorkspaceResource ───────────────────────────────────────────────

    @property
    def working_dir(self) -> str:
        return self.config.working_dir

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        if not path:
            return current_working_dir
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(Path(current_working_dir) / p)

    async def read_file(self, path: str) -> bytes:
        assert self._container_id is not None, "DockerWorkspaceResource not started"
        resolved = self.resolve_path(self.working_dir, path)

        def _read() -> bytes:
            result = subprocess.run(
                ["docker", "exec", self._container_id, "cat", resolved],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to read {resolved}: {result.stderr.decode()}")
            return result.stdout

        return await trio.to_thread.run_sync(_read)

    async def write_file(self, path: str, content: bytes) -> None:
        assert self._container_id is not None, "DockerWorkspaceResource not started"
        resolved = self.resolve_path(self.working_dir, path)

        # Write to local temp file, docker cp it in
        def _write() -> None:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(content)
                tmp_path = f.name
            try:
                # Ensure parent dir exists in container
                parent = str(Path(resolved).parent)
                subprocess.run(
                    ["docker", "exec", self._container_id, "mkdir", "-p", parent],
                    capture_output=True,
                    check=True,
                )
                subprocess.run(
                    ["docker", "cp", tmp_path, f"{self._container_id}:{resolved}"],
                    capture_output=True,
                    check=True,
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

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
        assert self._container_id is not None, "DockerWorkspaceResource not started"

        def _exec() -> tuple[str, str, int]:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "--workdir",
                    cwd,
                    self._container_id,
                    "bash",
                    "-lc",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout, result.stderr, result.returncode

        try:
            stdout, stderr, returncode = await trio.to_thread.run_sync(_exec)
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _copy_source_in(self, source_dir: Path) -> None:
        """Copy source_dir contents into working_dir inside the container."""
        assert self._container_id is not None

        # docker cp doesn't support ignore patterns, so copy to a local tempdir first
        def _cp() -> None:
            with tempfile.TemporaryDirectory(prefix="rollouts-docker-cp-") as tmp:
                staging = Path(tmp) / "workspace"
                shutil.copytree(
                    source_dir,
                    staging,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
                )
                subprocess.run(
                    [
                        "docker",
                        "cp",
                        f"{staging}/.",
                        f"{self._container_id}:{self.config.working_dir}",
                    ],
                    capture_output=True,
                    check=True,
                )

        await trio.to_thread.run_sync(_cp)

    # ── Optional introspection ────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "kind": "docker_workspace",
            "image": self.config.image,
            "container_id": self._container_id,
            "working_dir": self.config.working_dir,
            "started": self._started,
        }

    async def describe_runtime(self) -> dict[str, Any]:
        if not self._started:
            return {"kind": "docker", "started": False}
        result = await self.run(
            "python3 --version 2>&1; uname -r",
            cwd=self.config.working_dir,
            timeout=10.0,
        )
        return {
            "kind": "docker",
            "image": self.config.image,
            "container_id": self._container_id,
            "runtime_info": result.stdout.strip(),
        }
