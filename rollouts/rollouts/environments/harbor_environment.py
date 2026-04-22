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
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, TypeVar

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
_LIVE_HARBOR_BACKENDS: dict[str, Any] = {}
_MODAL_FILE_IO_CHUNK_BYTES = 16 * 1024 * 1024


# ── Harbor availability ─────────────────────────────────────────────────────

_HARBOR_IMPORT_ERROR: str | None = None
_HARBOR_DOCKER_IMPORT_ERROR: str | None = None
_HARBOR_MODAL_IMPORT_ERROR: str | None = None
try:
    from harbor.environments.base import BaseEnvironment as _HarborBaseEnvironment  # noqa: F401
    from harbor.models.task.config import EnvironmentConfig as _HarborEnvironmentConfig
    from harbor.models.trial.paths import TrialPaths as _HarborTrialPaths

    HARBOR_AVAILABLE = True
except ImportError as exc:
    HARBOR_AVAILABLE = False
    _HARBOR_IMPORT_ERROR = str(exc)
    _HarborEnvironmentConfig = None  # type: ignore[assignment]
    _HarborTrialPaths = None  # type: ignore[assignment]

try:
    from harbor.environments.docker.docker import DockerEnvironment as _HarborDockerEnvironment
except ImportError as exc:
    _HarborDockerEnvironment = None  # type: ignore[assignment]
    _HARBOR_DOCKER_IMPORT_ERROR = str(exc)

try:
    from harbor.environments.modal import ModalEnvironment as _HarborModalEnvironment
except ImportError as exc:
    _HarborModalEnvironment = None  # type: ignore[assignment]
    _HARBOR_MODAL_IMPORT_ERROR = str(exc)


async def _modal_sandbox_upload_file(
    *,
    sandbox: Any,
    source_path: Path | str,
    target_path: str,
) -> None:
    source = Path(source_path)
    target_parent = str(PurePosixPath(target_path).parent)

    if target_parent not in ("", ".", "/"):
        await sandbox.mkdir.aio(target_parent, parents=True)

    remote_file = await sandbox.open.aio(target_path, "wb")
    try:
        with source.open("rb") as local_file:
            while True:
                chunk = local_file.read(_MODAL_FILE_IO_CHUNK_BYTES)
                if not chunk:
                    break
                await remote_file.write.aio(chunk)
    finally:
        await remote_file.close.aio()


async def _modal_sandbox_download_file(
    *,
    sandbox: Any,
    source_path: str,
    target_path: Path | str,
) -> None:
    local_path = Path(target_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    remote_file = await sandbox.open.aio(source_path, "rb")
    try:
        with local_path.open("wb") as local_file:
            while True:
                chunk = await remote_file.read.aio(_MODAL_FILE_IO_CHUNK_BYTES)
                if not chunk:
                    break
                local_file.write(chunk)
    finally:
        await remote_file.close.aio()


def _patch_harbor_modal_filesystem_compat(
    *,
    harbor_modal_environment_cls: type[Any],
    sandbox_cls: type[Any],
) -> None:
    if hasattr(sandbox_cls, "filesystem"):
        return
    if getattr(harbor_modal_environment_cls, "_rollouts_modal_fs_compat", False):
        return

    async def _sdk_upload_file(self: Any, source_path: Path | str, target_path: str) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        await _modal_sandbox_upload_file(
            sandbox=self._sandbox,
            source_path=source_path,
            target_path=target_path,
        )

    async def _sdk_download_file(
        self: Any,
        source_path: str,
        target_path: Path | str,
    ) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        await _modal_sandbox_download_file(
            sandbox=self._sandbox,
            source_path=source_path,
            target_path=target_path,
        )

    harbor_modal_environment_cls._sdk_upload_file = _sdk_upload_file
    harbor_modal_environment_cls._sdk_download_file = _sdk_download_file
    harbor_modal_environment_cls._rollouts_modal_fs_compat = True


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


def _require_harbor_docker() -> None:
    _require_harbor()
    if _HarborDockerEnvironment is not None:
        return
    raise ImportError(
        "Harbor DockerEnvironment is unavailable. Ensure Harbor is installed correctly.\n"
        f"Underlying import error: {_HARBOR_DOCKER_IMPORT_ERROR}"
    )


def _require_harbor_modal() -> None:
    _require_harbor()
    if _HarborModalEnvironment is not None:
        import modal

        _patch_harbor_modal_filesystem_compat(
            harbor_modal_environment_cls=_HarborModalEnvironment,
            sandbox_cls=modal.Sandbox,
        )
        return
    raise ImportError(
        "Harbor ModalEnvironment is unavailable. Install Harbor with Modal support.\n"
        "Example:\n"
        "    pip install 'harbor[modal]'\n"
        "    uv tool install 'harbor[modal]'\n"
        "\n"
        f"Underlying import error: {_HARBOR_MODAL_IMPORT_ERROR}"
    )


@dataclass(frozen=True)
class LocalHarborHost:
    kind: Literal["local"] = "local"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


@dataclass(frozen=True)
class ModalHarborHost:
    kind: Literal["modal"] = "modal"
    app_name: str = "__harbor__"
    secrets: tuple[str, ...] = ()
    registry_secret: str | None = None
    volumes: dict[str, str] | None = None
    sandbox_timeout_secs: int = 60 * 60 * 24
    sandbox_idle_timeout_secs: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "app_name": self.app_name,
            "secrets": list(self.secrets),
            "registry_secret": self.registry_secret,
            "volumes": self.volumes,
            "sandbox_timeout_secs": self.sandbox_timeout_secs,
            "sandbox_idle_timeout_secs": self.sandbox_idle_timeout_secs,
        }


@dataclass(frozen=True)
class SSHHarborHost:
    kind: Literal["ssh"] = "ssh"
    ssh: str = ""
    ssh_key_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ssh": self.ssh,
            "ssh_key_path": self.ssh_key_path,
        }


HarborHostConfig = LocalHarborHost | ModalHarborHost | SSHHarborHost


def parse_harbor_host_config(data: dict[str, Any] | None) -> HarborHostConfig:
    if data is None:
        return LocalHarborHost()

    kind = data.get("kind", "local")
    if kind == "local":
        return LocalHarborHost()
    if kind == "modal":
        return ModalHarborHost(
            app_name=data.get("app_name", "__harbor__"),
            secrets=tuple(data.get("secrets", ())),
            registry_secret=data.get("registry_secret"),
            volumes=data.get("volumes"),
            sandbox_timeout_secs=data.get("sandbox_timeout_secs", 60 * 60 * 24),
            sandbox_idle_timeout_secs=data.get("sandbox_idle_timeout_secs"),
        )
    if kind == "ssh":
        return SSHHarborHost(
            ssh=data.get("ssh", ""),
            ssh_key_path=data.get("ssh_key_path"),
        )
    raise ValueError(f"Unknown Harbor host kind: {kind!r}")


def _serialize_harbor_host_config(host: HarborHostConfig) -> dict[str, Any]:
    return host.to_dict()


def attach_harbor_host_to_tasks(
    tasks: list[dict[str, Any]],
    host: HarborHostConfig,
) -> list[dict[str, Any]]:
    host_dict = _serialize_harbor_host_config(host)
    return [{**task, "harbor_host": host_dict} for task in tasks]


def _make_harbor_backend(
    *,
    host: HarborHostConfig,
    task_dir: Path,
    environment_name: str,
    session_id: str,
    trial_paths: Any,
    task_env_config: Any,
) -> Any:
    # TODO(pooling): heterogeneous host pooling/lease management belongs above
    # HarborEnvironment.create. Keep per-sample host selection here as a pure
    # "which substrate realizes this task?" decision; do not smear pool policy
    # into this constructor boundary.
    if isinstance(host, LocalHarborHost):
        _require_harbor_docker()
        return _HarborDockerEnvironment(
            environment_dir=task_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
        )

    if isinstance(host, ModalHarborHost):
        _require_harbor_modal()
        return _HarborModalEnvironment(
            environment_dir=task_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            app_name=host.app_name,
            secrets=list(host.secrets),
            registry_secret=host.registry_secret,
            volumes=host.volumes,
            sandbox_timeout_secs=host.sandbox_timeout_secs,
            sandbox_idle_timeout_secs=host.sandbox_idle_timeout_secs,
        )

    if isinstance(host, SSHHarborHost):
        raise NotImplementedError(
            "SSHHarborHost is not implemented yet. Harbor does not expose a built-in "
            "SSH environment in the pinned commit we use today."
        )

    raise AssertionError(f"Unhandled Harbor host config: {host!r}")


def _register_live_harbor_backend(session_id: str, harbor_env: Any) -> None:
    _LIVE_HARBOR_BACKENDS[session_id] = harbor_env


def _lookup_live_harbor_backend(session_id: str) -> Any | None:
    return _LIVE_HARBOR_BACKENDS.get(session_id)


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
    host: HarborHostConfig = LocalHarborHost()
    # TODO(harbor-warm-restore): cross-process warm restore would add a field
    # like container_id / sandbox_handle so deserialize could reattach without
    # relying on an in-memory backend reference. Deferred until pooling is real.

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
            "host": _serialize_harbor_host_config(self.host),
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
            host=parse_harbor_host_config(data.get("host")),
        )


class HarborEnvironment(CodingEnvironment):
    """Rollouts environment backed by a Harbor container.

    Subclasses CodingEnvironment so native rollouts agents see the same tool
    vocabulary (read/write/edit/bash/glob/grep/...) regardless of whether
    they're in a local dir, Modal sandbox, or Harbor Docker container.

    Construct via `create()` (async, starts the container). Call `close()` to
    stop and optionally delete.

    Serialize/deserialize has two modes because the runtime uses it in two
    different ways:

    - Hot path: between tool calls in the same process, reattach to the
      existing Harbor substrate via an in-memory live-backend registry.
    - Cold path: if no live backend reference is present, fall back to the
      spec and cold-start a fresh Harbor backend.
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

    @classmethod
    def _from_live_backend(
        cls,
        *,
        spec: HarborEnvironmentSpec,
        harbor_env: Any,
        current_working_dir: str | None = None,
        tools: str | list[str] = "full",
    ) -> HarborEnvironment:
        working_dir = current_working_dir or spec.working_dir
        if spec.working_dir != working_dir:
            spec = replace(spec, working_dir=working_dir)
        _register_live_harbor_backend(spec.session_id, harbor_env)
        workspace = HarborWorkspaceResource(harbor_env=harbor_env, working_dir=working_dir)
        command_runner = HarborCommandRunner(harbor_env=harbor_env)
        return cls(
            spec=spec,
            harbor_env=harbor_env,
            workspace=workspace,
            command_runner=command_runner,
            tools=tools,
        )

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
        host: HarborHostConfig = LocalHarborHost(),
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
            host=host,
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

        harbor_env = _make_harbor_backend(
            host=host,
            task_dir=task_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=env_config,
        )

        logger.info(
            "HarborEnvironment.create: starting (task_dir=%s, name=%s, session=%s, host=%s, force_build=%s)",
            task_dir,
            environment_name,
            session_id,
            host.kind,
            force_build,
        )
        await _aio_in_thread(lambda: harbor_env.start(force_build=force_build))
        logger.info("HarborEnvironment.create: started")

        return cls._from_live_backend(
            spec=spec,
            harbor_env=harbor_env,
            current_working_dir=working_dir,
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
        finally:
            if _lookup_live_harbor_backend(self.spec.session_id) is self.harbor_env:
                _LIVE_HARBOR_BACKENDS.pop(self.spec.session_id, None)

    # ── Serialize / fold contract ─────────────────────────────────────────

    async def serialize(self) -> dict[str, Any]:
        data = self.spec.to_dict()
        # The runtime uses serialize/deserialize as a hot in-memory handoff
        # between tool calls. Carry the current cwd; deserialize reattaches to
        # the live backend via the in-process registry when available.
        data["working_dir"] = self.current_working_dir
        data["tools"] = self.tools
        return data

    @staticmethod
    async def deserialize(data: dict[str, Any]) -> HarborEnvironment:
        """Restore Harbor environment state.

        If the live backend is still registered for this session, this is a
        warm in-process reattach used by the runtime hot path. Otherwise fall
        back to a cold create from the serialized spec.
        """
        spec = HarborEnvironmentSpec.from_dict(data)
        tools = data.get("tools", "full")
        live_harbor_env = _lookup_live_harbor_backend(spec.session_id)
        if live_harbor_env is not None:
            return HarborEnvironment._from_live_backend(
                spec=spec,
                harbor_env=live_harbor_env,
                current_working_dir=spec.working_dir,
                tools=tools,
            )

        return await HarborEnvironment.create(
            task_dir=spec.task_dir,
            environment_name=spec.environment_name,
            session_id=spec.session_id,
            working_dir=spec.working_dir,
            cpus=spec.cpus,
            memory_mb=spec.memory_mb,
            storage_mb=spec.storage_mb,
            docker_image=spec.docker_image,
            host=spec.host,
            tools=tools,
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
