from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any

import trio

from .resources import CommandExecutionResult, WorkspaceInfraError
from .runtime_probe import build_gpu_runtime_probe_script

DEFAULT_BIFROST_WORKSPACE = "~/.bifrost/workspaces/rollouts"


@dataclass(frozen=True)
class BrokerBifrostWorkspaceResourceConfig:
    workspace_path: str = DEFAULT_BIFROST_WORKSPACE
    bootstrap_cmds: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    gpu_type: str = "A100"
    gpu_count: int = 1
    provider: str | None = "runpod"
    max_price: float | None = None
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    name_prefix: str = "rollouts-workspace"
    connect_timeout_seconds: int = 30
    provision_timeout_seconds: int = 900
    keep_instance: bool = False
    ssh_key_path: str | None = None


@dataclass(frozen=True)
class BrokerBifrostWorkspaceLease:
    gpu_id: int
    acquired_at: float


@dataclass(frozen=True)
class BrokerBifrostWorkspaceLeasePoolConfig:
    resource_config: BrokerBifrostWorkspaceResourceConfig
    gpu_ids: tuple[int, ...]
    node_gpu_count: int
    keep_warm: bool = False

    def __post_init__(self) -> None:
        if not self.gpu_ids:
            raise ValueError("gpu_ids must not be empty")
        if self.node_gpu_count < 1:
            raise ValueError("node_gpu_count must be >= 1")
        if any(gpu_id < 0 for gpu_id in self.gpu_ids):
            raise ValueError("gpu_ids must be >= 0")
        max_gpu_id = max(self.gpu_ids)
        if max_gpu_id >= self.node_gpu_count:
            raise ValueError(
                f"gpu_ids contain {max_gpu_id}, but node_gpu_count is only {self.node_gpu_count}"
            )


@dataclass
class BrokerBifrostWorkspaceResource:
    config: BrokerBifrostWorkspaceResourceConfig
    sample_data: dict[str, Any] = field(default_factory=dict)
    _instance_id: str | None = field(default=None, repr=False)
    _provider: str | None = field(default=None, repr=False)
    _ssh_connection: str | None = field(default=None, repr=False)
    _ssh_key_path: str | None = field(default=None, repr=False)
    _workspace_path: str | None = field(default=None, repr=False)
    _runtime_description: dict[str, Any] | None = field(default=None, repr=False)
    _last_error: str | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)
    _provision_duration_ms: float | None = field(default=None, repr=False)
    _client: Any | None = field(default=None, repr=False)
    _instance: Any | None = field(default=None, repr=False)
    _should_terminate_instance: bool = field(default=True, repr=False)

    @property
    def working_dir(self) -> str:
        return self._workspace_path or self.config.workspace_path

    async def start(self) -> None:
        await self.prepare(self.sample_data)

    async def prepare(self, sample_data: dict[str, Any] | None = None) -> None:
        if sample_data is not None:
            self.sample_data = sample_data
        await self._ensure_client()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        if (
            self._instance is not None
            and self._should_terminate_instance
            and not self.config.keep_instance
        ):
            await self._instance.terminate()
            self._instance = None
            self._instance_id = None
            self._provider = None
            self._ssh_connection = None
            self._ssh_key_path = None
        self._started = False

    async def describe_runtime(self) -> dict[str, Any]:
        client = await self._ensure_client()
        with trio.fail_after(60):
            result = await client.exec(self._runtime_probe_script(), working_dir=self.working_dir)
        if result.exit_code != 0:
            self._runtime_description = {
                "runtime_ok": False,
                "error": result.stderr or result.stdout or "runtime probe failed",
                "errors": [result.stderr or result.stdout or "runtime probe failed"],
            }
            return self._runtime_description
        import json

        try:
            self._runtime_description = json.loads(result.stdout)
        except json.JSONDecodeError:
            self._runtime_description = {
                "runtime_ok": False,
                "error": f"invalid runtime probe output: {result.stdout}",
                "errors": [result.stderr] if result.stderr else [],
            }
        return self._runtime_description

    def stats(self) -> dict[str, Any]:
        return {
            "kind": "broker_bifrost_workspace_resource",
            "started": self._started,
            "instance_id": self._instance_id,
            "provider": self._provider,
            "ssh_connection": self._ssh_connection,
            "workspace_path": self._workspace_path or self.config.workspace_path,
            "gpu_type": self.config.gpu_type,
            "last_error": self._last_error,
            "provision_duration_ms": self._provision_duration_ms,
            "runtime": self._runtime_description,
        }

    def serialize_state(self) -> dict[str, Any]:
        return {
            "kind": "broker_bifrost_workspace_resource",
            "config": asdict(self.config),
            "sample_data": self.sample_data,
            "instance_id": self._instance_id,
            "provider": self._provider,
            "ssh_connection": self._ssh_connection,
            "ssh_key_path": self._ssh_key_path,
            "workspace_path": self._workspace_path,
            "runtime": self._runtime_description,
            "last_error": self._last_error,
            "started": self._started,
        }

    @classmethod
    def deserialize_state(cls, data: dict[str, Any]) -> BrokerBifrostWorkspaceResource:
        resource = cls(
            config=BrokerBifrostWorkspaceResourceConfig(**data["config"]),
            sample_data=data.get("sample_data", {}),
        )
        resource._instance_id = data.get("instance_id")
        resource._provider = data.get("provider")
        resource._ssh_connection = data.get("ssh_connection")
        resource._ssh_key_path = data.get("ssh_key_path")
        resource._workspace_path = data.get("workspace_path")
        resource._runtime_description = data.get("runtime")
        resource._last_error = data.get("last_error")
        resource._started = data.get("started", False)
        resource._should_terminate_instance = False
        return resource

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        if not path:
            return current_working_dir
        pure = PurePosixPath(path)
        if pure.is_absolute():
            return str(pure)
        return str(PurePosixPath(current_working_dir) / pure)

    async def read_file(self, path: str) -> bytes:
        client = await self._ensure_client()
        resolved = self.resolve_path(self.working_dir, path)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            local_path = tmp.name
        try:
            with trio.fail_after(60):
                result = await client.download_files(resolved, local_path, recursive=False)
            if not result.success:
                raise RuntimeError(result.error_message or f"Failed to read {resolved}")
            return await trio.Path(local_path).read_bytes()
        finally:
            local_tmp = trio.Path(local_path)
            if await local_tmp.exists():
                await local_tmp.unlink()

    async def write_file(self, path: str, content: bytes) -> None:
        client = await self._ensure_client()
        resolved = self.resolve_path(self.working_dir, path)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            local_path = tmp.name
        try:
            await trio.Path(local_path).write_bytes(content)
            with trio.fail_after(60):
                result = await client.upload_files(local_path, resolved, recursive=False)
            if not result.success:
                raise RuntimeError(result.error_message or f"Failed to write {resolved}")
        finally:
            local_tmp = trio.Path(local_path)
            if await local_tmp.exists():
                await local_tmp.unlink()

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
        client = await self._ensure_client()
        try:
            with trio.fail_after(timeout):
                result = await client.exec(
                    command,
                    env=self.config.env,
                    working_dir=cwd,
                )
        except trio.TooSlowError as exc:
            raise WorkspaceInfraError(
                f"workspace command timed out after {timeout}s while running command in {cwd}",
                kind="workspace_timeout",
            ) from exc
        return CommandExecutionResult(
            returncode=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            cwd=cwd,
        )

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        started_at = time.perf_counter()
        instance, ssh_key_path = await self._lookup_or_create_instance()
        self._instance = instance
        self._instance_id = instance.id
        self._provider = instance.provider
        self._ssh_connection = instance.ssh_connection_string()
        self._ssh_key_path = ssh_key_path

        client = self._make_bifrost_client(
            ssh_connection=self._ssh_connection,
            ssh_key_path=ssh_key_path,
            timeout=self.config.connect_timeout_seconds,
        )
        bootstrap_cmd = list(self.config.bootstrap_cmds) if self.config.bootstrap_cmds else None
        self._workspace_path = await client.push(
            workspace_path=self.config.workspace_path,
            bootstrap_cmd=bootstrap_cmd,
        )
        self._client = client
        self._started = True
        self._last_error = None
        self._provision_duration_ms = (time.perf_counter() - started_at) * 1000.0
        return client

    async def _lookup_or_create_instance(self) -> tuple[Any, str]:
        client = self._make_gpu_client()
        if self._instance_id is not None and self._provider is not None:
            instance = await client.get_instance(self._instance_id, self._provider)
            if instance is None:
                raise WorkspaceInfraError(
                    f"Could not reconnect to remote instance {self._provider}:{self._instance_id}",
                    kind="workspace_reconnect_failed",
                )
        else:
            query = client.gpu_type.contains(self.config.gpu_type)
            if self.config.provider:
                query = query & (client.provider == self.config.provider)
            if self.config.max_price is not None:
                query = query & (client.price_per_hour <= self.config.max_price)

            name_suffix = self.sample_data.get("problem_name") or self.sample_data.get("problem_id")
            instance = await client.create(
                query,
                image=self.config.image,
                name=(
                    f"{self.config.name_prefix}-{name_suffix}"
                    if name_suffix is not None
                    else self.config.name_prefix
                ),
                gpu_count=self.config.gpu_count,
            )
            assert instance is not None, "broker create must return an instance"
            ssh_ready = await instance.wait_until_ssh_ready(
                timeout=self.config.provision_timeout_seconds
            )
            if not ssh_ready:
                raise WorkspaceInfraError(
                    f"remote instance {instance.id} did not become SSH-ready within "
                    f"{self.config.provision_timeout_seconds}s",
                    kind="workspace_provision_timeout",
                )

        ssh_key_path = self._resolve_ssh_key_path(client=client, provider=instance.provider)
        if ssh_key_path is None:
            raise ValueError(
                f"No SSH key configured for provider {instance.provider!r}; "
                "set broker SSH credentials or ssh_key_path explicitly."
            )
        return instance, ssh_key_path

    def _resolve_ssh_key_path(self, *, client: Any, provider: str) -> str | None:
        if self.config.ssh_key_path is not None:
            return self.config.ssh_key_path

        ssh_key_path = client.get_ssh_key_path(provider=provider)
        if ssh_key_path is not None:
            return ssh_key_path

        from infra_utils.config import discover_ssh_keys, get_ssh_key_path

        configured_key = get_ssh_key_path()
        if configured_key is not None:
            return configured_key

        discovered_keys = discover_ssh_keys()
        if discovered_keys:
            return discovered_keys[0]
        return None

    def _make_gpu_client(self) -> Any:
        from broker.client import GPUClient
        from broker.credentials import get_credentials

        return GPUClient(
            credentials=get_credentials(),
            ssh_key_path=self.config.ssh_key_path,
        )

    def _make_bifrost_client(
        self,
        *,
        ssh_connection: str,
        ssh_key_path: str,
        timeout: int,
    ) -> Any:
        from bifrost.async_client import AsyncBifrostClient

        return AsyncBifrostClient(
            ssh_connection=ssh_connection,
            ssh_key_path=ssh_key_path,
            timeout=timeout,
        )

    def _runtime_probe_script(self) -> str:
        return f"""
python3 <<'PY'
{build_gpu_runtime_probe_script()}
PY
"""


@dataclass
class ManagedBrokerBifrostWorkspaceResource:
    manager: BrokerBifrostWorkspaceManager
    sample_data: dict[str, Any] = field(default_factory=dict)
    _lease: BrokerBifrostWorkspaceLease | None = field(default=None, repr=False)
    _resource: BrokerBifrostWorkspaceResource | None = field(default=None, repr=False)

    @property
    def working_dir(self) -> str:
        if self._resource is not None:
            return self._resource.working_dir
        return self.manager.working_dir_for_gpu(self.manager.config.gpu_ids[0])

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
            "kind": "managed_broker_bifrost_workspace_resource",
            "manager": self.manager.serialize_state(),
            "sample_data": self.sample_data,
            "lease_state": "leased" if self._resource is not None else "uninitialized",
            "resource": self._resource.serialize_state() if self._resource is not None else None,
        }

    @classmethod
    def deserialize_state(cls, data: dict[str, Any]) -> ManagedBrokerBifrostWorkspaceResource:
        manager = BrokerBifrostWorkspaceManager.deserialize_state(data["manager"])
        sample_data = data.get("sample_data", {})
        resource_data = data.get("resource")
        if resource_data is None:
            return cls(manager=manager, sample_data=sample_data)

        resource = BrokerBifrostWorkspaceResource.deserialize_state(resource_data)
        gpu_id = resource.config.env.get("CUDA_VISIBLE_DEVICES")
        leased_gpu_ids = {int(gpu_id)} if gpu_id is not None else set()
        manager._bootstrap_from_state(
            {int(gpu_id): resource} if gpu_id is not None else {}, leased_gpu_ids=leased_gpu_ids
        )
        return cls(
            manager=manager,
            sample_data=sample_data,
            _lease=(
                BrokerBifrostWorkspaceLease(gpu_id=int(gpu_id), acquired_at=time.time())
                if gpu_id is not None
                else None
            ),
            _resource=resource,
        )

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        if self._resource is not None:
            return self._resource.resolve_path(current_working_dir, path)
        return BrokerBifrostWorkspaceResource(
            config=self.manager._resource_config_for_gpu(self.manager.config.gpu_ids[0]),
            sample_data=self.sample_data,
        ).resolve_path(current_working_dir, path)

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

    async def _ensure_resource(self) -> BrokerBifrostWorkspaceResource:
        if self._resource is not None:
            return self._resource
        lease, resource = await self.manager.acquire(self.sample_data)
        self._lease = lease
        self._resource = resource
        return resource


@dataclass
class BrokerBifrostWorkspaceManager:
    config: BrokerBifrostWorkspaceLeasePoolConfig
    _resources: dict[int, BrokerBifrostWorkspaceResource] = field(default_factory=dict, repr=False)
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
    _instance_id: str | None = field(default=None, repr=False)
    _provider: str | None = field(default=None, repr=False)
    _ssh_connection: str | None = field(default=None, repr=False)
    _ssh_key_path: str | None = field(default=None, repr=False)
    _instance: Any | None = field(default=None, repr=False)

    async def start(self) -> None:
        if self._started:
            return
        send, recv = trio.open_memory_channel[int](len(self.config.gpu_ids))
        self._available_send = send
        self._available_recv = recv
        for gpu_id in self.config.gpu_ids:
            send.send_nowait(gpu_id)
        self._started = True
        await self._ensure_instance()

    async def stop(self) -> None:
        if not self._started:
            return
        for resource in self._resources.values():
            try:
                await resource.close()
            except Exception as exc:
                self._last_error = str(exc)
        self._resources = {}
        if self._instance is not None and not self.config.resource_config.keep_instance:
            await self._instance.terminate()
        self._instance = None
        self._instance_id = None
        self._provider = None
        self._ssh_connection = None
        self._ssh_key_path = None
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
    ) -> tuple[BrokerBifrostWorkspaceLease, BrokerBifrostWorkspaceResource]:
        await self.start()
        recv = self._require_recv()
        if recv.statistics().current_buffer_used == 0:
            self._wait_events += 1

        if timeout is None:
            gpu_id = await recv.receive()
        else:
            try:
                with trio.fail_after(timeout):
                    gpu_id = await recv.receive()
            except trio.TooSlowError as exc:
                raise TimeoutError(f"Timed out acquiring workspace lease after {timeout}s") from exc

        async with self._creation_lock:
            resource = self._resources.get(gpu_id)
            created = resource is None
            if resource is None:
                resource = self._build_resource(gpu_id, sample_data)
                self._resources[gpu_id] = resource
                self._create_count += 1
            resource.sample_data = sample_data

        try:
            await resource.prepare(sample_data)
        except Exception as exc:
            self._last_error = str(exc)
            self._require_send().send_nowait(gpu_id)
            raise

        if not created:
            self._reuse_count += 1
        self._acquire_count += 1
        self._in_flight += 1
        if self._in_flight > self._max_in_flight:
            self._max_in_flight = self._in_flight
        return BrokerBifrostWorkspaceLease(gpu_id=gpu_id, acquired_at=time.time()), resource

    async def release(self, lease: BrokerBifrostWorkspaceLease) -> None:
        if not self._started:
            raise ValueError("Manager not started. Cannot release lease.")
        resource = self._resources[lease.gpu_id]
        if not self.config.keep_warm:
            await resource.close()
        self._release_count += 1
        assert self._in_flight > 0, "cannot release workspace lease when none are in flight"
        self._in_flight -= 1
        self._require_send().send_nowait(lease.gpu_id)

    def make_resource(self, sample_data: dict[str, Any]) -> ManagedBrokerBifrostWorkspaceResource:
        return ManagedBrokerBifrostWorkspaceResource(manager=self, sample_data=sample_data)

    def working_dir_for_gpu(self, gpu_id: int) -> str:
        return f"{self.config.resource_config.workspace_path.rstrip('/')}/gpu-{gpu_id}"

    def stats(self) -> dict[str, Any]:
        return {
            "kind": "broker_bifrost_workspace_manager",
            "started": self._started,
            "gpu_ids": self.config.gpu_ids,
            "node_gpu_count": self.config.node_gpu_count,
            "instance_id": self._instance_id,
            "provider": self._provider,
            "ssh_connection": self._ssh_connection,
            "num_resources": len(self._resources),
            "in_flight": self._in_flight,
            "acquire_count": self._acquire_count,
            "release_count": self._release_count,
            "wait_events": self._wait_events,
            "create_count": self._create_count,
            "reuse_count": self._reuse_count,
            "max_in_flight": self._max_in_flight,
            "keep_warm": self.config.keep_warm,
            "last_error": self._last_error,
            "resources": {gpu_id: resource.stats() for gpu_id, resource in self._resources.items()},
        }

    def serialize_state(self) -> dict[str, Any]:
        return {
            "kind": "broker_bifrost_workspace_manager",
            "config": asdict(self.config),
            "instance_id": self._instance_id,
            "provider": self._provider,
            "ssh_connection": self._ssh_connection,
            "ssh_key_path": self._ssh_key_path,
        }

    @classmethod
    def deserialize_state(cls, data: dict[str, Any]) -> BrokerBifrostWorkspaceManager:
        manager = cls(
            config=BrokerBifrostWorkspaceLeasePoolConfig(
                resource_config=BrokerBifrostWorkspaceResourceConfig(
                    **data["config"]["resource_config"]
                ),
                gpu_ids=tuple(data["config"]["gpu_ids"]),
                node_gpu_count=data["config"]["node_gpu_count"],
                keep_warm=data["config"].get("keep_warm", False),
            )
        )
        manager._instance_id = data.get("instance_id")
        manager._provider = data.get("provider")
        manager._ssh_connection = data.get("ssh_connection")
        manager._ssh_key_path = data.get("ssh_key_path")
        manager._bootstrap_from_state({}, leased_gpu_ids=set())
        return manager

    def _bootstrap_from_state(
        self,
        resources: dict[int, BrokerBifrostWorkspaceResource],
        *,
        leased_gpu_ids: set[int],
    ) -> None:
        send, recv = trio.open_memory_channel[int](max(len(self.config.gpu_ids), 1))
        self._available_send = send
        self._available_recv = recv
        for gpu_id in self.config.gpu_ids:
            if gpu_id not in leased_gpu_ids:
                send.send_nowait(gpu_id)
        self._resources = resources
        self._started = True
        self._in_flight = len(leased_gpu_ids)

    async def _ensure_instance(self) -> None:
        if self._instance is not None:
            return
        if self._instance_id is not None and self._provider is not None:
            probe = BrokerBifrostWorkspaceResource(config=self.config.resource_config)
            client = probe._make_gpu_client()
            instance = await client.get_instance(self._instance_id, self._provider)
            if instance is None:
                raise WorkspaceInfraError(
                    f"Could not reconnect to remote instance {self._provider}:{self._instance_id}",
                    kind="workspace_reconnect_failed",
                )
            self._instance = instance
            if self._ssh_key_path is None:
                self._ssh_key_path = probe._resolve_ssh_key_path(
                    client=client, provider=instance.provider
                )
            if self._ssh_connection is None:
                self._ssh_connection = instance.ssh_connection_string()
            return

        probe = BrokerBifrostWorkspaceResource(
            config=BrokerBifrostWorkspaceResourceConfig(**{
                **asdict(self.config.resource_config),
                "gpu_count": self.config.node_gpu_count,
            })
        )
        instance, ssh_key_path = await probe._lookup_or_create_instance()
        self._instance = instance
        self._instance_id = instance.id
        self._provider = instance.provider
        self._ssh_connection = instance.ssh_connection_string()
        self._ssh_key_path = ssh_key_path

    def _build_resource(
        self,
        gpu_id: int,
        sample_data: dict[str, Any],
    ) -> BrokerBifrostWorkspaceResource:
        env = {**self.config.resource_config.env, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        resource_config = self._resource_config_for_gpu(gpu_id)
        resource = BrokerBifrostWorkspaceResource(
            config=resource_config,
            sample_data=sample_data,
        )
        resource._instance = self._instance
        resource._instance_id = self._instance_id
        resource._provider = self._provider
        resource._ssh_connection = self._ssh_connection
        resource._ssh_key_path = self._ssh_key_path
        resource._should_terminate_instance = False
        return resource

    def _resource_config_for_gpu(self, gpu_id: int) -> BrokerBifrostWorkspaceResourceConfig:
        env = {**self.config.resource_config.env, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        return BrokerBifrostWorkspaceResourceConfig(
            workspace_path=self.working_dir_for_gpu(gpu_id),
            bootstrap_cmds=self.config.resource_config.bootstrap_cmds,
            env=env,
            gpu_type=self.config.resource_config.gpu_type,
            gpu_count=self.config.resource_config.gpu_count,
            provider=self.config.resource_config.provider,
            max_price=self.config.resource_config.max_price,
            image=self.config.resource_config.image,
            name_prefix=self.config.resource_config.name_prefix,
            connect_timeout_seconds=self.config.resource_config.connect_timeout_seconds,
            provision_timeout_seconds=self.config.resource_config.provision_timeout_seconds,
            keep_instance=self.config.resource_config.keep_instance,
            ssh_key_path=self.config.resource_config.ssh_key_path,
        )

    def _require_recv(self) -> trio.MemoryReceiveChannel[int]:
        if self._available_recv is None:
            raise ValueError("Manager not started. Call start() first.")
        return self._available_recv

    def _require_send(self) -> trio.MemorySendChannel[int]:
        if self._available_send is None:
            raise ValueError("Manager not started. Call start() first.")
        return self._available_send
