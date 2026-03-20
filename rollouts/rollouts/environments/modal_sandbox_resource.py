from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import trio

from ..infra_errors import WorkspaceInfraError
from .resources import CommandExecutionResult, SessionExecSpec
from .runtime_probe import build_gpu_runtime_probe_script

if TYPE_CHECKING:
    import modal


DEFAULT_WORKSPACE_DIR = "/workspace"
SANDBOX_COMMAND_TIMEOUT_RETRIES = 1
SANDBOX_COMMAND_RESET_RETRIES = 1

logger = logging.getLogger(__name__)
_event_logger = logging.getLogger("rollouts.eval.events")


def _command_preview(command: str, *, max_len: int = 160) -> str:
    compact = " ".join(command.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


@dataclass(frozen=True)
class ModalSandboxResourceConfig:
    app_name: str = "rollouts-sandbox"
    gpu: str = "A100"
    # Provider-side sandbox TTL. This is the hard upper bound on how long a
    # live sandbox may exist if local cleanup never runs.
    timeout_seconds: int = 1800
    python_version: str = "3.10"
    workspace_dir: str = DEFAULT_WORKSPACE_DIR
    image_registry: str = "nvidia/cuda:13.0.0-devel-ubuntu22.04"
    apt_packages: tuple[str, ...] = ("git", "gcc-10", "g++-10", "clang", "build-essential")
    pip_packages: tuple[str, ...] = (
        "torch",
        "numpy",
        "ninja",
        "triton",
        "cupy-cuda12x",
        "nvidia-cutlass-dsl",
        "tilelang",
    )
    env: dict[str, str] = field(
        default_factory=lambda: {
            "HF_HOME": "/root/.cache/huggingface",
            "PYTHONPATH": "/root:/root/src",
            "THUNDERKITTENS_ROOT": "/root/ThunderKittens",
        }
    )
    run_commands: tuple[str, ...] = (
        f"mkdir -p {DEFAULT_WORKSPACE_DIR}",
        "if [ ! -d /root/ThunderKittens ]; then git clone -b main https://github.com/HazyResearch/ThunderKittens.git /root/ThunderKittens; fi",
    )


@dataclass(frozen=True)
class ModalSandboxLease:
    resource_index: int
    acquired_at: float


@dataclass
class ModalSandboxResource:
    # TODO(session-first): this should eventually be split into a Modal-backed
    # `InspectableRemoteSession` plus a `SessionBackedWorkspaceHandle`. The
    # actual substrate here is modal exec + file movement, not the flattened
    # workspace protocol.
    config: ModalSandboxResourceConfig
    sample_data: dict[str, Any] = field(default_factory=dict)
    workspace_setup: Any | None = None
    working_dir: str = DEFAULT_WORKSPACE_DIR
    _sandbox: modal.Sandbox | None = field(default=None, repr=False)
    _sandbox_id: str | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)
    _start_attempts: int = field(default=0, repr=False)
    _start_failures: int = field(default=0, repr=False)
    _provision_duration_ms: float | None = field(default=None, repr=False)
    _last_error: str | None = field(default=None, repr=False)
    _runtime_description: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.working_dir == DEFAULT_WORKSPACE_DIR:
            self.working_dir = self.config.workspace_dir

    def for_sample(self, sample_data: dict[str, Any]) -> ModalSandboxResource:
        return ModalSandboxResource(
            config=self.config,
            sample_data=sample_data,
            workspace_setup=self.workspace_setup,
            working_dir=self.working_dir,
        )

    async def start(self) -> None:
        await self.prepare(self.sample_data)

    async def prepare(self, sample_data: dict[str, Any] | None = None) -> None:
        if sample_data is not None:
            self.sample_data = sample_data
        sandbox = await self._ensure_sandbox()
        if self.workspace_setup is not None:
            await trio.to_thread.run_sync(lambda: self.workspace_setup(sandbox, self.sample_data))

    async def close(self) -> None:
        if self._sandbox is None:
            return
        await self._terminate_sandbox()
        self._sandbox = None
        self._sandbox_id = None
        self._started = False

    async def reset(self) -> None:
        if self._sandbox is not None:
            try:
                await self._terminate_sandbox()
            except Exception as exc:
                self._last_error = str(exc)
        self._sandbox = None
        self._sandbox_id = None
        self._started = False
        self._runtime_description = None

    async def describe_runtime(self) -> dict[str, Any]:
        sandbox = await self._ensure_sandbox()
        runtime = await self._run_runtime_probe(sandbox)
        self._runtime_description = runtime
        return runtime

    def stats(self) -> dict[str, Any]:
        return {
            "kind": "modal_sandbox",
            "started": self._started,
            "start_attempts": self._start_attempts,
            "start_failures": self._start_failures,
            "sandbox_id": self._sandbox_id,
            "gpu": self.config.gpu,
            "app_name": self.config.app_name,
            "workspace_dir": self.working_dir,
            "provision_duration_ms": self._provision_duration_ms,
            "last_error": self._last_error,
            "runtime": self._runtime_description,
        }

    def serialize_state(self) -> dict[str, Any]:
        return {
            "kind": "modal_sandbox_resource",
            "config": asdict(self.config),
            "sample_data": self.sample_data,
            "working_dir": self.working_dir,
            "sandbox_id": self._sandbox_id,
            "started": self._started,
            "start_attempts": self._start_attempts,
            "start_failures": self._start_failures,
            "provision_duration_ms": self._provision_duration_ms,
            "last_error": self._last_error,
            "runtime": self._runtime_description,
        }

    @classmethod
    def deserialize_state(
        cls,
        data: dict[str, Any],
        *,
        workspace_setup: Any | None = None,
    ) -> ModalSandboxResource:
        resource = cls(
            config=ModalSandboxResourceConfig(**data["config"]),
            sample_data=data.get("sample_data", {}),
            workspace_setup=workspace_setup,
            working_dir=data.get("working_dir", DEFAULT_WORKSPACE_DIR),
        )
        resource._sandbox_id = data.get("sandbox_id")
        resource._started = data.get("started", False)
        resource._start_attempts = data.get("start_attempts", 0)
        resource._start_failures = data.get("start_failures", 0)
        resource._provision_duration_ms = data.get("provision_duration_ms")
        resource._last_error = data.get("last_error")
        resource._runtime_description = data.get("runtime")
        return resource

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        if not path:
            return current_working_dir
        pure = PurePosixPath(path)
        if pure.is_absolute():
            return str(pure)
        return str(PurePosixPath(current_working_dir) / pure)

    async def read_file(self, path: str) -> bytes:
        resolved = self.resolve_path(self.working_dir, path)
        result = await self.run(
            f"python - <<'PY'\nfrom pathlib import Path\nprint(Path({resolved!r}).read_text())\nPY",
            cwd=self.working_dir,
            timeout=30.0,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or f"Failed to read {resolved}")
        return result.stdout.encode()

    async def write_file(self, path: str, content: bytes) -> None:
        sandbox = await self._ensure_sandbox()
        resolved = self.resolve_path(self.working_dir, path)
        parent_dir = str(PurePosixPath(resolved).parent)
        payload = base64.b64encode(content).decode()

        def do_write() -> tuple[str, str, int]:
            proc = sandbox.exec(
                "bash",
                "-lc",
                f"mkdir -p {parent_dir} && echo '{payload}' | base64 -d > {resolved}",
                timeout=30,
            )
            proc.wait()
            return proc.stdout.read(), proc.stderr.read(), proc.returncode

        stdout, stderr, returncode = await trio.to_thread.run_sync(do_write)
        if returncode != 0:
            raise RuntimeError(stderr or stdout or f"Failed to write {resolved}")

    async def exec(self, spec: SessionExecSpec) -> CommandExecutionResult:
        return await self.run(
            spec.command,
            cwd=spec.cwd,
            timeout=spec.timeout,
            session_id=spec.session_id,
            cancel_scope=spec.cancel_scope,
        )

    async def upload_bytes(self, remote_path: str, content: bytes) -> None:
        await self.write_file(remote_path, content)

    async def download_bytes(self, remote_path: str) -> bytes:
        return await self.read_file(remote_path)

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
        sandbox = await self._ensure_sandbox()
        timeout_seconds = max(1, math.ceil(timeout))
        command_preview = _command_preview(command)

        def do_run() -> tuple[str, str, int]:
            proc = sandbox.exec(
                "bash",
                "-lc",
                f"cd {cwd} && {command}",
                timeout=timeout_seconds,
            )
            proc.wait()
            return proc.stdout.read(), proc.stderr.read(), proc.returncode

        total_attempts = 1 + SANDBOX_COMMAND_TIMEOUT_RETRIES + SANDBOX_COMMAND_RESET_RETRIES
        same_sandbox_attempts = 1 + SANDBOX_COMMAND_TIMEOUT_RETRIES

        for attempt in range(total_attempts):
            started_at = time.perf_counter()
            _event_logger.info(
                "sandbox_command_start",
                extra={
                    "sandbox_id": self._sandbox_id,
                    "cwd": cwd,
                    "timeout_seconds": timeout_seconds,
                    "attempt": attempt + 1,
                    "max_attempts": total_attempts,
                    "problem_id": self.sample_data.get("problem_id"),
                    "problem_name": self.sample_data.get("problem_name"),
                    "command_preview": command_preview,
                },
            )
            try:
                stdout, stderr, returncode = await trio.to_thread.run_sync(do_run)
                duration_ms = (time.perf_counter() - started_at) * 1000.0
                _event_logger.info(
                    "sandbox_command_end",
                    extra={
                        "sandbox_id": self._sandbox_id,
                        "cwd": cwd,
                        "timeout_seconds": timeout_seconds,
                        "attempt": attempt + 1,
                        "max_attempts": total_attempts,
                        "problem_id": self.sample_data.get("problem_id"),
                        "problem_name": self.sample_data.get("problem_name"),
                        "command_preview": command_preview,
                        "duration_ms": round(duration_ms, 1),
                        "returncode": returncode,
                        "status": "success" if returncode == 0 else "error",
                    },
                )
                return CommandExecutionResult(
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                    cwd=cwd,
                )
            except (concurrent.futures.CancelledError, TimeoutError) as exc:
                self._last_error = f"sandbox command timed out after {timeout_seconds}s"
                if attempt + 1 < same_sandbox_attempts:
                    logger.warning(
                        "Retrying sandbox command after timeout",
                        extra={
                            "timeout_seconds": timeout_seconds,
                            "attempt": attempt + 1,
                            "max_attempts": total_attempts,
                            "sandbox_id": self._sandbox_id,
                            "cwd": cwd,
                            "problem_id": self.sample_data.get("problem_id"),
                            "problem_name": self.sample_data.get("problem_name"),
                            "recovery_action": "same_sandbox_retry",
                        },
                    )
                    _event_logger.info(
                        "sandbox_command_retry",
                        extra={
                            "timeout_seconds": timeout_seconds,
                            "attempt": attempt + 1,
                            "max_attempts": total_attempts,
                            "sandbox_id": self._sandbox_id,
                            "cwd": cwd,
                            "problem_id": self.sample_data.get("problem_id"),
                            "problem_name": self.sample_data.get("problem_name"),
                            "command_preview": command_preview,
                            "recovery_action": "same_sandbox_retry",
                        },
                    )
                    continue
                if attempt + 1 < total_attempts:
                    timed_out_sandbox_id = self._sandbox_id
                    logger.warning(
                        "Resetting sandbox after repeated command timeout",
                        extra={
                            "timeout_seconds": timeout_seconds,
                            "attempt": attempt + 1,
                            "max_attempts": total_attempts,
                            "sandbox_id": timed_out_sandbox_id,
                            "cwd": cwd,
                            "problem_id": self.sample_data.get("problem_id"),
                            "problem_name": self.sample_data.get("problem_name"),
                            "recovery_action": "reset_sandbox",
                        },
                    )
                    _event_logger.info(
                        "sandbox_command_retry",
                        extra={
                            "timeout_seconds": timeout_seconds,
                            "attempt": attempt + 1,
                            "max_attempts": total_attempts,
                            "sandbox_id": timed_out_sandbox_id,
                            "cwd": cwd,
                            "problem_id": self.sample_data.get("problem_id"),
                            "problem_name": self.sample_data.get("problem_name"),
                            "command_preview": command_preview,
                            "recovery_action": "reset_sandbox",
                        },
                    )
                    await self.reset()
                    sandbox = await self._ensure_sandbox()
                    continue
                raise WorkspaceInfraError(
                    f"sandbox command timed out after {timeout_seconds}s while running "
                    f"command in {cwd}; exhausted {total_attempts} attempts",
                    kind="workspace_timeout",
                ) from exc

    async def _ensure_sandbox(self) -> modal.Sandbox:
        if self._sandbox is not None:
            return self._sandbox
        if self._sandbox_id is not None:
            try:
                import modal

                self._sandbox = modal.Sandbox.from_id(self._sandbox_id)
                self._started = True
                return self._sandbox
            except Exception as exc:
                self._last_error = str(exc)
                self._sandbox = None

        self._start_attempts += 1
        started_at = time.perf_counter()
        try:
            sandbox_id = await self._provision_sandbox_via_broker()
            self._sandbox_id = sandbox_id

            import modal

            self._sandbox = modal.Sandbox.from_id(sandbox_id)
            self._started = True
            self._last_error = None
            self._provision_duration_ms = (time.perf_counter() - started_at) * 1000.0
            return self._sandbox
        except Exception as exc:
            self._start_failures += 1
            self._started = False
            self._last_error = str(exc)
            self._provision_duration_ms = (time.perf_counter() - started_at) * 1000.0
            raise

    def _broker_deps(self) -> Any:
        return SimpleNamespace(
            source_type="registry",
            source_ref=self.config.image_registry,
            python_version=self.config.python_version,
            system_packages=tuple(self.config.apt_packages),
            pip_packages=tuple(self.config.pip_packages),
            pip_index_url=None,
            pip_extra_index_url=None,
            pip_prerelease=False,
            env=dict(self.config.env),
            bootstrap_commands=(
                tuple(self.config.run_commands) + (self._manifest_write_command(),)
            ),
        )

    def _manifest_write_command(self) -> str:
        payload = {
            "schema_version": 1,
            "source_type": "registry",
            "source_ref": self.config.image_registry,
            "resolved_image_ref": None,
            "image_name": None,
            "cuda_version": self.config.image_registry.split(":")[1].split("-")[0]
            if ":" in self.config.image_registry
            else None,
            "python_version": self.config.python_version,
            "env": self.config.env,
            "features": ["kernelbench-v3", "kernelbench-backend:cuda"],
            "installed_groups": ["kernelbench-v3-runtime"],
            "paths": {
                "workspace_dir": self.config.workspace_dir,
                "thunderkittens_root": self.config.env.get("THUNDERKITTENS_ROOT", ""),
            },
        }
        encoded = base64.b64encode(json.dumps(payload, indent=2, sort_keys=True).encode()).decode()
        return (
            'python3 -c "import base64; from pathlib import Path; '
            "path = Path('/etc/rollouts-image.json').expanduser(); "
            "path.parent.mkdir(parents=True, exist_ok=True); "
            f"path.write_text(base64.b64decode('{encoded}').decode('utf-8'))\""
        )

    async def _provision_sandbox_via_broker(self) -> str:
        from broker.providers import modal as broker_modal
        from broker.types import ProvisionRequest

        request = ProvisionRequest(
            gpu_type=self.config.gpu,
            gpu_count=1,
            provider="modal",
            name=self.config.app_name,
            max_lifetime_seconds=self.config.timeout_seconds,
            raw_data={"deps": self._broker_deps()},
        )
        instance = await broker_modal.provision_instance(request)
        if instance is None:
            raise RuntimeError(
                f"Broker failed to provision Modal sandbox for app={self.config.app_name!r}"
            )
        return instance.id

    async def _terminate_sandbox(self) -> None:
        if self._sandbox_id is None:
            return
        from broker.providers import modal as broker_modal

        terminated = await broker_modal.terminate_instance(self._sandbox_id)
        if not terminated and self._sandbox is not None:
            self._sandbox.terminate()

    async def _run_runtime_probe(self, sandbox: modal.Sandbox) -> dict[str, Any]:
        script = build_gpu_runtime_probe_script()

        def do_probe() -> tuple[str, str, int]:
            proc = sandbox.exec("python", "-c", script, timeout=30)
            proc.wait()
            return proc.stdout.read(), proc.stderr.read(), proc.returncode

        stdout, stderr, returncode = await trio.to_thread.run_sync(do_probe)
        if returncode != 0:
            return {
                "runtime_ok": False,
                "error": stderr or stdout or "runtime probe failed",
                "errors": [stderr or stdout or "runtime probe failed"],
            }
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {
                "runtime_ok": False,
                "error": f"invalid runtime probe output: {stdout}",
                "errors": [stderr] if stderr else [],
            }


@dataclass
class ManagedModalSandboxResource:
    manager: ModalSandboxManager
    sample_data: dict[str, Any] = field(default_factory=dict)
    _lease: ModalSandboxLease | None = field(default=None, repr=False)
    _resource: ModalSandboxResource | None = field(default=None, repr=False)

    @property
    def working_dir(self) -> str:
        resource = self._resource
        if resource is not None:
            return resource.working_dir
        return self.manager.config.workspace_dir

    async def start(self) -> None:
        await self._ensure_resource()

    async def close(self) -> None:
        if self._lease is None:
            return
        await self.manager.release(self._lease)
        self._lease = None
        self._resource = None

    async def describe_runtime(self) -> dict[str, Any]:
        resource = await self._ensure_resource()
        return await resource.describe_runtime()

    def stats(self) -> dict[str, Any]:
        stats = self.manager.stats()
        if self._resource is not None:
            stats["resource"] = self._resource.stats()
        return stats

    def serialize_state(self) -> dict[str, Any]:
        return {
            "kind": "managed_modal_sandbox_resource",
            "manager": self.manager.serialize_state(),
            "sample_data": self.sample_data,
            "lease_state": "leased" if self._resource is not None else "uninitialized",
            "resource": self._resource.serialize_state() if self._resource is not None else None,
        }

    @classmethod
    def deserialize_state(
        cls,
        data: dict[str, Any],
    ) -> ManagedModalSandboxResource:
        manager = ModalSandboxManager.deserialize_state(data["manager"])
        sample_data = data.get("sample_data", {})
        resource_data = data.get("resource")
        if resource_data is None:
            return cls(manager=manager, sample_data=sample_data)

        resource = ModalSandboxResource.deserialize_state(resource_data)
        manager._bootstrap_from_state([resource], leased_indices={0})
        return cls(
            manager=manager,
            sample_data=sample_data,
            _lease=ModalSandboxLease(resource_index=0, acquired_at=time.time()),
            _resource=resource,
        )

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        resource = self._resource
        if resource is not None:
            return resource.resolve_path(current_working_dir, path)
        if not path:
            return current_working_dir
        pure = PurePosixPath(path)
        if pure.is_absolute():
            return str(pure)
        return str(PurePosixPath(current_working_dir) / pure)

    async def read_file(self, path: str) -> bytes:
        resource = await self._ensure_resource()
        return await resource.read_file(path)

    async def write_file(self, path: str, content: bytes) -> None:
        resource = await self._ensure_resource()
        await resource.write_file(path, content)

    async def run(
        self,
        command: str,
        *,
        cwd: str,
        timeout: float,
        session_id: str | None = None,
        cancel_scope: Any | None = None,
    ) -> CommandExecutionResult:
        resource = await self._ensure_resource()
        return await resource.run(
            command,
            cwd=cwd,
            timeout=timeout,
            session_id=session_id,
            cancel_scope=cancel_scope,
        )

    async def exec(self, spec: SessionExecSpec) -> CommandExecutionResult:
        resource = await self._ensure_resource()
        return await resource.exec(spec)

    async def upload_bytes(self, remote_path: str, content: bytes) -> None:
        resource = await self._ensure_resource()
        await resource.upload_bytes(remote_path, content)

    async def download_bytes(self, remote_path: str) -> bytes:
        resource = await self._ensure_resource()
        return await resource.download_bytes(remote_path)

    async def _ensure_resource(self) -> ModalSandboxResource:
        if self._resource is not None:
            return self._resource
        lease, resource = await self.manager.acquire(self.sample_data)
        self._lease = lease
        self._resource = resource
        return resource


@dataclass
class ModalSandboxManager:
    config: ModalSandboxResourceConfig
    workspace_setup: Any | None = None
    max_sandboxes: int = 1
    # Default cold. Expensive GPU sandboxes should terminate on release unless a
    # caller explicitly opts into warm retention.
    keep_warm: bool = False
    # TODO(lifecycle): keep_warm currently means leaked-cost risk if the owning
    # process dies before `stop()` runs. We need a durable lease registry /
    # janitor state machine, not only in-process finally cleanup.
    resource_factory: Callable[[ModalSandboxResourceConfig, Any | None], ModalSandboxResource] = (
        lambda config, workspace_setup: ModalSandboxResource(
            config=config,
            workspace_setup=workspace_setup,
        )
    )
    _resources: list[ModalSandboxResource] = field(default_factory=list, repr=False)
    _available_send: trio.MemorySendChannel[int] | None = field(default=None, repr=False)
    _available_recv: trio.MemoryReceiveChannel[int] | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)
    _creation_lock: trio.Lock = field(default_factory=trio.Lock, repr=False)
    _acquire_count: int = field(default=0, repr=False)
    _release_count: int = field(default=0, repr=False)
    _wait_events: int = field(default=0, repr=False)
    _create_count: int = field(default=0, repr=False)
    _reuse_count: int = field(default=0, repr=False)
    _in_flight: int = field(default=0, repr=False)
    _max_in_flight: int = field(default=0, repr=False)
    _last_error: str | None = field(default=None, repr=False)

    async def start(self) -> None:
        if self._started:
            return
        if self.max_sandboxes <= 0:
            raise ValueError("max_sandboxes must be >= 1")
        send, recv = trio.open_memory_channel[int](self.max_sandboxes)
        self._available_send = send
        self._available_recv = recv
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        for resource in self._resources:
            try:
                await resource.close()
            except Exception as exc:
                self._last_error = str(exc)
        self._resources = []
        if self._available_send is not None:
            self._available_send.close()
        if self._available_recv is not None:
            self._available_recv.close()
        self._available_send = None
        self._available_recv = None
        self._started = False
        self._in_flight = 0

    async def acquire(
        self,
        sample_data: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> tuple[ModalSandboxLease, ModalSandboxResource]:
        await self.start()
        recv = self._require_recv()

        if recv.statistics().current_buffer_used == 0:
            async with self._creation_lock:
                if len(self._resources) < self.max_sandboxes:
                    resource_index = len(self._resources)
                    resource = self.resource_factory(self.config, self.workspace_setup)
                    self._resources.append(resource)
                    self._create_count += 1
                    await resource.prepare(sample_data)
                    lease = ModalSandboxLease(
                        resource_index=resource_index, acquired_at=time.time()
                    )
                    self._acquire_count += 1
                    self._in_flight += 1
                    if self._in_flight > self._max_in_flight:
                        self._max_in_flight = self._in_flight
                    return lease, resource

        if recv.statistics().current_buffer_used == 0:
            self._wait_events += 1

        if timeout is None:
            resource_index = await recv.receive()
        else:
            try:
                with trio.fail_after(timeout):
                    resource_index = await recv.receive()
            except trio.TooSlowError as exc:
                raise TimeoutError(f"Timed out acquiring modal sandbox after {timeout}s") from exc

        resource = self._resources[resource_index]
        try:
            await resource.prepare(sample_data)
        except Exception as exc:
            self._last_error = str(exc)
            self._require_send().send_nowait(resource_index)
            raise

        self._reuse_count += 1
        self._acquire_count += 1
        self._in_flight += 1
        if self._in_flight > self._max_in_flight:
            self._max_in_flight = self._in_flight
        return ModalSandboxLease(resource_index=resource_index, acquired_at=time.time()), resource

    async def release(self, lease: ModalSandboxLease) -> None:
        if not self._started:
            raise ValueError("Manager not started. Cannot release lease.")
        resource = self._resources[lease.resource_index]
        if not self.keep_warm:
            await resource.close()
        # TODO(lifecycle): release() is currently only an in-process transition.
        # We still need durable owner/lease records with TTL so another process
        # can reap retained sandboxes after crashes or abandoned runs.
        self._release_count += 1
        assert self._in_flight > 0, "cannot release modal sandbox when none are in flight"
        self._in_flight -= 1
        self._require_send().send_nowait(lease.resource_index)

    def make_resource(self, sample_data: dict[str, Any]) -> ManagedModalSandboxResource:
        return ManagedModalSandboxResource(manager=self, sample_data=sample_data)

    def stats(self) -> dict[str, Any]:
        return {
            "kind": "modal_sandbox_manager",
            "started": self._started,
            "max_sandboxes": self.max_sandboxes,
            "num_resources": len(self._resources),
            "available_resources": (
                self._available_recv.statistics().current_buffer_used
                if self._available_recv is not None
                else 0
            ),
            "in_flight": self._in_flight,
            "acquire_count": self._acquire_count,
            "release_count": self._release_count,
            "wait_events": self._wait_events,
            "create_count": self._create_count,
            "reuse_count": self._reuse_count,
            "max_in_flight": self._max_in_flight,
            "keep_warm": self.keep_warm,
            "last_error": self._last_error,
            "resources": [resource.stats() for resource in self._resources],
        }

    def serialize_state(self) -> dict[str, Any]:
        return {
            "kind": "modal_sandbox_manager",
            "config": asdict(self.config),
            "max_sandboxes": self.max_sandboxes,
            "keep_warm": self.keep_warm,
        }

    @classmethod
    def deserialize_state(cls, data: dict[str, Any]) -> ModalSandboxManager:
        manager = cls(
            config=ModalSandboxResourceConfig(**data["config"]),
            max_sandboxes=data.get("max_sandboxes", 1),
            keep_warm=data.get("keep_warm", False),
        )
        manager._bootstrap_from_state([], leased_indices=set())
        return manager

    def _require_recv(self) -> trio.MemoryReceiveChannel[int]:
        if self._available_recv is None:
            raise ValueError("Manager not started. Call start() first.")
        return self._available_recv

    def _require_send(self) -> trio.MemorySendChannel[int]:
        if self._available_send is None:
            raise ValueError("Manager not started. Call start() first.")
        return self._available_send

    def _bootstrap_from_state(
        self,
        resources: list[ModalSandboxResource],
        *,
        leased_indices: set[int],
    ) -> None:
        send, recv = trio.open_memory_channel[int](max(self.max_sandboxes, 1))
        self._available_send = send
        self._available_recv = recv
        self._resources = list(resources)
        self._started = True
        self._in_flight = len(leased_indices)
        self._max_in_flight = max(self._max_in_flight, self._in_flight)
        for idx in range(len(self._resources)):
            if idx not in leased_indices:
                self._available_send.send_nowait(idx)
