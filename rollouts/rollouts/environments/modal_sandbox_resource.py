from __future__ import annotations

import base64
import json
import math
import os
import time
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Callable

import trio

from .resources import CommandExecutionResult

if TYPE_CHECKING:
    import modal


DEFAULT_WORKSPACE_DIR = "/workspace"


@dataclass(frozen=True)
class ModalSandboxResourceConfig:
    app_name: str = "rollouts-sandbox"
    gpu: str = "A100"
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


@dataclass(frozen=True)
class ModalSandboxLease:
    resource_index: int
    acquired_at: float


@dataclass
class ModalSandboxResource:
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
        self._sandbox.terminate()
        self._sandbox = None
        self._sandbox_id = None
        self._started = False

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
        result = await self.run(f"python - <<'PY'\nfrom pathlib import Path\nprint(Path({resolved!r}).read_text())\nPY", cwd=self.working_dir, timeout=30.0)
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

        def do_run() -> tuple[str, str, int]:
            proc = sandbox.exec(
                "bash",
                "-lc",
                f"cd {cwd} && {command}",
                timeout=max(1, math.ceil(timeout)),
            )
            proc.wait()
            return proc.stdout.read(), proc.stderr.read(), proc.returncode

        stdout, stderr, returncode = await trio.to_thread.run_sync(do_run)
        return CommandExecutionResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            cwd=cwd,
        )

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
            sandbox_id = await trio.to_thread.run_sync(self._create_sandbox_subprocess)
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

    def _create_sandbox_subprocess(self) -> str:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c", self._build_sandbox_creation_script()],
            capture_output=True,
            text=True,
            env={**os.environ},
        )
        if result.returncode != 0:
            raise RuntimeError(f"Sandbox creation failed: {result.stderr}")
        return result.stdout.strip().splitlines()[-1]

    def _build_sandbox_creation_script(self) -> str:
        apt_packages = ", ".join(repr(pkg) for pkg in self.config.apt_packages)
        pip_packages = ", ".join(repr(pkg) for pkg in self.config.pip_packages)
        env_json = json.dumps(self.config.env)
        return f"""
import asyncio
import json
import os
import modal

async def create_sandbox() -> None:
    app = modal.App.lookup({self.config.app_name!r}, create_if_missing=True)
    image = (
        modal.Image.from_registry(
            {self.config.image_registry!r},
            add_python={self.config.python_version!r},
        )
        .apt_install({apt_packages})
        .pip_install({pip_packages})
        .env(json.loads({env_json!r}))
        .run_commands(
            {f"mkdir -p {self.config.workspace_dir}"!r},
            "if [ ! -d /root/ThunderKittens ]; then git clone -b main https://github.com/HazyResearch/ThunderKittens.git /root/ThunderKittens; fi",
        )
    )
    sandbox = modal.Sandbox.create(
        app=app,
        image=image,
        gpu={self.config.gpu!r},
        timeout={self.config.timeout_seconds},
    )
    print(sandbox.object_id)

asyncio.run(create_sandbox())
"""

    async def _run_runtime_probe(self, sandbox: modal.Sandbox) -> dict[str, Any]:
        script = """
import json
import socket
runtime = {"hostname": socket.gethostname(), "runtime_ok": True}
errors = []
try:
    import torch
    tk_root = os.environ.get("THUNDERKITTENS_ROOT", "/root/ThunderKittens")
    try:
        import ninja  # noqa: F401
        ninja_available = True
    except Exception:
        ninja_available = False
    torch_info = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "ninja_available": ninja_available,
        "thunderkittens_root_exists": os.path.isdir(tk_root),
        "thunderkittens_root": tk_root,
    }
    for package_name in ("triton", "cupy", "tilelang", "cutlass"):
        try:
            __import__(package_name)
            torch_info[f"{package_name}_available"] = True
        except Exception:
            torch_info[f"{package_name}_available"] = False
    if torch_info["cuda_available"]:
        torch_info["device_name"] = torch.cuda.get_device_name(0)
    runtime["torch"] = torch_info
    runtime["thunderkittens_root_exists"] = torch_info["thunderkittens_root_exists"]
    runtime["thunderkittens_root"] = tk_root
except Exception as exc:
    runtime["torch"] = {"available": False}
    runtime["runtime_ok"] = False
    runtime["error"] = f"import torch failed: {exc!r}"
    errors.append(runtime["error"])
runtime["errors"] = errors
print(json.dumps(runtime))
"""

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
    keep_warm: bool = False
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
                    lease = ModalSandboxLease(resource_index=resource_index, acquired_at=time.time())
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
