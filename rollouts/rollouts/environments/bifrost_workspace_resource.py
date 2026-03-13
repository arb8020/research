from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any

import trio

from .resources import CommandExecutionResult


DEFAULT_BIFROST_WORKSPACE = "~/.bifrost/workspaces/rollouts"


@dataclass(frozen=True)
class BrokerBifrostWorkspaceResourceConfig:
    workspace_path: str = DEFAULT_BIFROST_WORKSPACE
    bootstrap_cmds: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    gpu_type: str = "A100"
    provider: str | None = "runpod"
    max_price: float | None = None
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    name_prefix: str = "rollouts-workspace"
    connect_timeout_seconds: int = 30
    provision_timeout_seconds: int = 900
    keep_instance: bool = False
    ssh_key_path: str | None = None


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
        if self._instance is not None and not self.config.keep_instance:
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
            raise RuntimeError(
                f"workspace command timed out after {timeout}s while running command in {cwd}"
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
                raise RuntimeError(
                    f"Could not reconnect to remote instance {self._provider}:{self._instance_id}"
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
                gpu_count=1,
            )
            assert instance is not None, "broker create must return an instance"
            ssh_ready = await instance.wait_until_ssh_ready(
                timeout=self.config.provision_timeout_seconds
            )
            if not ssh_ready:
                raise RuntimeError(
                    f"remote instance {instance.id} did not become SSH-ready within "
                    f"{self.config.provision_timeout_seconds}s"
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
        return """
python3 <<'PY'
import json
import os
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
PY
"""
