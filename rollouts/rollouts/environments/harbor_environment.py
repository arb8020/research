"""Harbor-backed environment for rollouts agents.

Runs Terminal-Bench 2.0 and other Harbor-flavored benchmarks by wrapping a
Harbor container environment (DockerEnvironment, ModalEnvironment, etc.)
behind our CodingEnvironment abstraction. Our rollouts runtime drives the
tool surface; Harbor owns container provisioning and, if invoked, the
verifier.

Architectural choice (native path): this is the "forward tool calls INTO
the sandbox" shape. The LLM (brain) sits in our process; its tool calls
route through HarborCommandRunner → harbor_env.exec(command) → the
container executes. The agent is not inside the sandbox — the container
is a pure effect interpreter.

The complementary shape, "agent runs INSIDE the sandbox and we read out
its session log," is how external runtimes (claude-code CLI, codex CLI,
ACP) work against non-Harbor workspaces today. See the bottom-of-file
TODO and external_agent_environments.py for the Harbor-workspace-as-
SandboxWorkspaceResource adapter that would let external agents run
inside Harbor containers. Not implemented yet.

We consume Harbor as a library (no CLI shelling). Harbor's `BaseEnvironment`
gives us `start`, `exec`, `upload_file`/`download_file`, `stop`. We adapt
those into our `CodingWorkspaceResource` and `CommandRunner` protocols so
`CodingEnvironment` just works over a Harbor container without knowing it's
Harbor-backed.

Install (out-of-band, see rollouts/pyproject.toml's note on why this is
not a workspace extra):

    uv pip install 'harbor @ git+https://github.com/laude-institute/harbor.git@e0fcdc2'

See also:

- /docs/design/session_ownership.md — the ownership model we're extending.
- rollouts/rollouts/agents/runtime_refactor.md — the session refactor this
  sits on top of.
- rollouts/rollouts/environments/external_agent_environments.py — sibling
  stubs for running external agents (claude-code, codex) against a Harbor
  workspace. Not the same thing as HarborEnvironment: that one is "our
  runtime drives Harbor"; those are "someone else's runtime drives Harbor
  and we observe."
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, TypeVar

import trio

from .coding import CodingEnvironment
from .resources import CommandExecutionResult

_T = TypeVar("_T")


async def _aio_in_thread(coro_factory: Callable[[], Awaitable[_T]]) -> _T:
    """Run a Harbor asyncio coroutine from inside trio.

    Harbor uses asyncio internally (e.g. `asyncio.create_subprocess_exec` in
    DockerEnvironment). Our runtime uses trio. These don't share event
    loops. To bridge, we spawn a thread via `trio.to_thread.run_sync`, open
    a fresh asyncio loop inside it via `asyncio.run`, and drive the Harbor
    coroutine to completion there. Trio cancellation still applies at the
    thread boundary.

    Each call is its own short-lived asyncio loop. Sufficient for our
    Harbor usage (tens of calls per sample). If this becomes a hotpath or
    we need shared asyncio state across calls, switch to trio_asyncio.
    """

    def _run() -> _T:
        return asyncio.run(coro_factory())

    return await trio.to_thread.run_sync(_run, abandon_on_cancel=True)


if TYPE_CHECKING:
    # Harbor is an optional out-of-band install; don't import at module load.
    pass

logger = logging.getLogger(__name__)


# ── Harbor availability ─────────────────────────────────────────────────────

_HARBOR_IMPORT_ERROR: str | None = None
try:
    from harbor.environments.base import BaseEnvironment as _HarborBaseEnvironment  # noqa: F401
    from harbor.environments.docker.docker import DockerEnvironment as _HarborDockerEnvironment
    from harbor.models.task.config import EnvironmentConfig as _HarborEnvironmentConfig
    from harbor.models.trial.paths import TrialPaths as _HarborTrialPaths

    HARBOR_AVAILABLE = True
except ImportError as exc:
    HARBOR_AVAILABLE = False
    _HARBOR_IMPORT_ERROR = str(exc)
    _HarborDockerEnvironment = None  # type: ignore[assignment]
    _HarborEnvironmentConfig = None  # type: ignore[assignment]
    _HarborTrialPaths = None  # type: ignore[assignment]


def _require_harbor() -> None:
    """Raise a helpful ImportError if harbor isn't installed."""
    if HARBOR_AVAILABLE:
        return
    raise ImportError(
        "Harbor is not installed. Install with:\n"
        "    uv pip install 'harbor @ git+https://github.com/laude-institute/harbor.git@e0fcdc2'\n"
        "\n"
        "Harbor is NOT a rollouts workspace dep (it pulls supabase which collides "
        "with other workspace members). Install it out-of-band into your venv "
        "before using HarborEnvironment.\n"
        "\n"
        f"Underlying import error: {_HARBOR_IMPORT_ERROR}"
    )


# ── Workspace resource adapter ──────────────────────────────────────────────


@dataclass
class HarborWorkspaceResource:
    """Adapts a Harbor `BaseEnvironment` to our `CodingWorkspaceResource` shape.

    Harbor exposes `upload_file(local, remote)` / `download_file(remote, local)`
    at the bytes-level. Our workspace protocol wants `read_file(path) -> bytes`
    and `write_file(path, bytes)`. Bridge via a temp file.
    """

    harbor_env: Any  # BaseEnvironment; Any-typed to keep harbor optional
    working_dir: str = "/app"

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        """Posix-style path resolution inside a Linux container."""
        p = PurePosixPath(path)
        if p.is_absolute():
            return str(p)
        return str(PurePosixPath(current_working_dir) / p)

    async def read_file(self, path: str) -> bytes:
        # Harbor download_file takes (remote_source, local_target).
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            local = Path(tmp.name)
        try:
            try:
                await _aio_in_thread(lambda: self.harbor_env.download_file(path, local))
            except RuntimeError as exc:
                # Harbor raises a generic RuntimeError when `docker compose cp`
                # fails, including for legitimate "file does not exist" cases.
                # Translate to FileNotFoundError so the coding environment's
                # existence-check pattern (try read_file; except
                # FileNotFoundError: treat as create) works as intended.
                msg = str(exc).lower()
                if "could not find the file" in msg or "no such file" in msg:
                    raise FileNotFoundError(path) from exc
                raise
            return local.read_bytes()
        finally:
            local.unlink(missing_ok=True)

    async def write_file(self, path: str, content: bytes) -> None:
        # Harbor upload_file takes (local_source, remote_target).
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            local = Path(tmp.name)
            local.write_bytes(content)
        try:
            await _aio_in_thread(lambda: self.harbor_env.upload_file(local, path))
        finally:
            local.unlink(missing_ok=True)


# ── Command runner adapter ──────────────────────────────────────────────────


@dataclass
class HarborCommandRunner:
    """Adapts a Harbor `BaseEnvironment.exec` to our `CommandRunner` protocol."""

    harbor_env: Any  # BaseEnvironment

    async def run(
        self,
        command: str,
        *,
        cwd: str,
        timeout: float,
        session_id: str | None = None,
        cancel_scope: Any | None = None,
    ) -> CommandExecutionResult:
        # `session_id` and `cancel_scope` are part of our protocol but Harbor's
        # exec doesn't thread them; we accept and ignore. If cancellation matters
        # later, trio's cancel scope around the await should still work.
        del session_id, cancel_scope
        result = await _aio_in_thread(
            lambda: self.harbor_env.exec(
                command=command,
                cwd=cwd,
                timeout_sec=int(timeout) if timeout else None,
            )
        )
        return CommandExecutionResult(
            returncode=result.return_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            cwd=cwd,
        )


# ── Environment ─────────────────────────────────────────────────────────────


@dataclass
class HarborEnvironmentSpec:
    """Serializable recipe for reconstructing a HarborEnvironment cold.

    Per the ownership model, env state = fold(effects, initial). The "initial"
    part is this spec — everything needed to start a fresh Harbor container
    that, once effects are replayed, matches the state at serialize time.
    """

    task_dir: str
    environment_name: str
    session_id: str
    working_dir: str = "/app"
    cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int = 10240
    docker_image: str | None = None
    # TODO(harbor-warm-restore): a warm path would add a field like
    # container_id / sandbox_handle so deserialize could reattach to a live
    # resource instead of cold-starting. Deferred until pooling is real.

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_kind": "harbor",
            "task_dir": self.task_dir,
            "environment_name": self.environment_name,
            "session_id": self.session_id,
            "working_dir": self.working_dir,
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "storage_mb": self.storage_mb,
            "docker_image": self.docker_image,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HarborEnvironmentSpec:
        assert data.get("env_kind") == "harbor", f"not a harbor env spec: {data!r}"
        return cls(
            task_dir=data["task_dir"],
            environment_name=data["environment_name"],
            session_id=data["session_id"],
            working_dir=data.get("working_dir", "/app"),
            cpus=data.get("cpus", 1),
            memory_mb=data.get("memory_mb", 2048),
            storage_mb=data.get("storage_mb", 10240),
            docker_image=data.get("docker_image"),
        )


class HarborEnvironment(CodingEnvironment):
    """Rollouts environment backed by a Harbor container.

    Subclasses CodingEnvironment so native rollouts agents see the same tool
    vocabulary (read/write/edit/bash/glob/grep/...) regardless of whether
    they're in a local dir, Modal sandbox, or Harbor Docker container.

    Construct via `create()` (async, starts the container). Call `close()` to
    stop and optionally delete.

    Serialize/deserialize implements the fold contract: serialize emits a
    `HarborEnvironmentSpec`; deserialize cold-restarts a fresh container from
    that spec. Replaying the session's effect log onto a cold-restarted env
    reconstructs state. Warm-restart (reattaching to a still-live container)
    is a v1 optimization on top; not implemented yet.
    """

    def __init__(
        self,
        *,
        spec: HarborEnvironmentSpec,
        harbor_env: Any,
        workspace: HarborWorkspaceResource,
        command_runner: HarborCommandRunner,
        tools: str | list[str] = "full",
    ) -> None:
        super().__init__(
            workspace=workspace,
            command_runner=command_runner,
            tools=tools,
            current_working_dir=spec.working_dir,
        )
        self.spec = spec
        self.harbor_env = harbor_env

    def get_name(self) -> str:
        return "harbor"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        *,
        task_dir: str | Path,
        environment_name: str,
        session_id: str,
        working_dir: str = "/app",
        cpus: int = 1,
        memory_mb: int = 2048,
        storage_mb: int = 10240,
        docker_image: str | None = None,
        trial_dir: str | Path | None = None,
        force_build: bool = False,
        tools: str | list[str] = "full",
    ) -> HarborEnvironment:
        """Start a Harbor DockerEnvironment from a task directory, return the
        composed HarborEnvironment ready for the rollouts runtime.

        Args:
            task_dir: Path to a directory containing a `Dockerfile` or
                `docker-compose.yaml`. This is what Harbor calls
                `environment_dir`.
            environment_name: Task name — Harbor uses this to tag resources.
            session_id: Per-run identifier, typically `<task_name>__<trial_id>`.
            working_dir: The agent's cwd inside the container. Defaults to
                `/app` which is the most common Harbor task convention.
            cpus/memory_mb/storage_mb: Container resource caps.
            docker_image: Optional prebuilt image. If set, skip build phase.
            trial_dir: Where Harbor writes its own bookkeeping (logs, etc.).
                Defaults to a fresh tempdir.
            force_build: If True, rebuild the image even if cached.
            tools: Toolset for CodingEnvironment — passed through as-is.
        """
        _require_harbor()

        task_dir = Path(task_dir)
        assert task_dir.is_dir(), f"task_dir not a directory: {task_dir}"

        if trial_dir is None:
            trial_dir = Path(tempfile.mkdtemp(prefix="rollouts-harbor-trial-"))
        else:
            trial_dir = Path(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        spec = HarborEnvironmentSpec(
            task_dir=str(task_dir),
            environment_name=environment_name,
            session_id=session_id,
            working_dir=working_dir,
            cpus=cpus,
            memory_mb=memory_mb,
            storage_mb=storage_mb,
            docker_image=docker_image,
        )

        env_config = _HarborEnvironmentConfig(
            cpus=cpus,
            memory_mb=memory_mb,
            storage_mb=storage_mb,
            docker_image=docker_image,
        )
        trial_paths = _HarborTrialPaths(trial_dir=trial_dir)
        # trial_paths.mkdir() creates its expected subdirs; do it pre-start
        # so Harbor doesn't find the layout half-built.
        if hasattr(trial_paths, "mkdir"):
            trial_paths.mkdir()

        harbor_env = _HarborDockerEnvironment(
            environment_dir=task_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=env_config,
        )

        logger.info(
            "HarborEnvironment.create: starting (task_dir=%s, name=%s, session=%s, force_build=%s)",
            task_dir,
            environment_name,
            session_id,
            force_build,
        )
        await _aio_in_thread(lambda: harbor_env.start(force_build=force_build))
        logger.info("HarborEnvironment.create: started")

        workspace = HarborWorkspaceResource(harbor_env=harbor_env, working_dir=working_dir)
        command_runner = HarborCommandRunner(harbor_env=harbor_env)
        return cls(
            spec=spec,
            harbor_env=harbor_env,
            workspace=workspace,
            command_runner=command_runner,
            tools=tools,
        )

    async def close(self, delete: bool = True) -> None:
        """Stop the Harbor container. `delete=True` also removes it."""
        logger.info("HarborEnvironment.close: stopping (delete=%s)", delete)
        try:
            await _aio_in_thread(lambda: self.harbor_env.stop(delete=delete))
        except Exception as exc:
            # Don't crash eval teardown because Harbor had trouble stopping.
            logger.warning("HarborEnvironment.close: harbor_env.stop raised: %s", exc)

    # ── Serialize / fold contract ─────────────────────────────────────────

    async def serialize(self) -> dict[str, Any]:
        return self.spec.to_dict()

    @staticmethod
    async def deserialize(data: dict[str, Any]) -> HarborEnvironment:
        """Cold-restore: spin up a fresh container from the spec.

        Does NOT replay the session's effect log — that's the caller's job
        (the fold lives at the session level, not on the environment). This
        method returns an environment at its initial state; feed it effects
        to reach the state that was serialized.

        Warm-restore (reattach to an existing live container) is not
        implemented; `deserialize` always takes the cold path.
        """
        spec = HarborEnvironmentSpec.from_dict(data)
        return await HarborEnvironment.create(
            task_dir=spec.task_dir,
            environment_name=spec.environment_name,
            session_id=spec.session_id,
            working_dir=spec.working_dir,
            cpus=spec.cpus,
            memory_mb=spec.memory_mb,
            storage_mb=spec.storage_mb,
            docker_image=spec.docker_image,
        )


# ── Factory (thin; no pool today) ───────────────────────────────────────────


async def make_harbor_environment(
    task_dir: str | Path,
    *,
    environment_name: str,
    session_id: str,
    **kwargs: Any,
) -> HarborEnvironment:
    """Thin factory. Same as HarborEnvironment.create; named for use as an
    EvalSpec.make_environment callable.

    When pooling / reuse becomes real, this is where a resource-manager
    object would earn itself. For v0 it's just a forwarder.
    """
    return await HarborEnvironment.create(
        task_dir=task_dir,
        environment_name=environment_name,
        session_id=session_id,
        **kwargs,
    )


# TODO(harbor-as-sandbox-workspace-resource): stub for the "agent runs
# INSIDE a Harbor container" shape.
#
# Today, native rollouts agents drive Harbor containers from outside by
# routing tool calls through HarborCommandRunner.exec. External runtimes
# (claude-code, codex, ACP variants in remote_runtime.py) instead *launch
# the agent inside a sandbox* and poll its session log. The two paths
# have complementary roles and together span the native/external parity
# story — but right now external runtimes can't target Harbor containers
# because there's no adapter presenting a Harbor DockerEnvironment as a
# SandboxWorkspaceResource (what trajectory_from_remote_* functions
# accept).
#
# What the adapter needs to do:
#   - Expose harbor_env.exec, .upload_file, .download_file behind
#     SandboxWorkspaceResource's interface (similar to what
#     HarborCommandRunner and HarborWorkspaceResource do, but adapted to
#     the SandboxWorkspaceResource protocol instead of CodingEnvironment's).
#   - Bootstrap the external CLI inside the container (install node,
#     install the CLI, set API keys in env). Today's non-Harbor remote
#     launchers in remote_runtime.py do this for Modal/Bifrost workspaces;
#     Harbor's DockerEnvironment would need its own install path.
#   - Return the adapter from HarborEnvironment.as_sandbox_workspace()
#     so callers write:
#       env = await HarborEnvironment.create(...)
#       artifact = await trajectory_from_remote_claude_code(
#           prompt, sample_id, sample_data,
#           workspace=env.as_sandbox_workspace(),
#           cwd=env.spec.working_dir,
#           ...,
#       )
#
# Scope: ~1 day. Blocked on nothing structurally — just infra wiring.
# Land when we want external/native parity on Harbor tasks.
#
# See also: rollouts/rollouts/eval/remote_runtime.py's run_external_agent
# entry point; it already takes ClaudeCodeEnvironment / CodexEnvironment,
# so the adapter surface needed here is the workspace field on those
# types, not a whole new code path.
