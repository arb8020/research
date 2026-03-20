from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rollouts.environments.resource_pools import (
    AcquiredWorkspaceResource,
    ModalSandboxWorkspacePool,
)


@dataclass
class _FakeResource:
    working_dir: str = "/workspace"

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def describe_runtime(self) -> dict[str, Any]:
        return {"runtime_ok": True}

    def stats(self) -> dict[str, Any]:
        return {}

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        return f"{current_working_dir}/{path}"

    async def read_file(self, path: str) -> bytes:
        return path.encode()

    async def write_file(self, path: str, content: bytes) -> None:
        del path, content
        return None

    async def run(
        self,
        command: str,
        *,
        cwd: str,
        timeout: float,
        session_id: str | None = None,
        cancel_scope: Any | None = None,
    ) -> Any:
        del command, cwd, timeout, session_id, cancel_scope
        return None


@pytest.mark.trio
async def test_acquired_workspace_resource_releases_only_once() -> None:
    released = {"count": 0}

    async def release() -> None:
        released["count"] += 1

    lease = AcquiredWorkspaceResource(resource=_FakeResource(), _release=release)

    await lease.close()
    await lease.close()

    assert released["count"] == 1


@pytest.mark.trio
async def test_modal_sandbox_workspace_pool_wraps_manager_acquire_and_release() -> None:
    calls: dict[str, Any] = {}

    class _FakeManager:
        async def acquire(
            self,
            sample_data: dict[str, object],
            *,
            timeout: float | None = None,
        ) -> tuple[str, _FakeResource]:
            calls["sample_data"] = sample_data
            calls["timeout"] = timeout
            return ("lease-token", _FakeResource())

        async def release(self, lease: object) -> None:
            calls["released"] = lease

    pool = ModalSandboxWorkspacePool(manager=_FakeManager())  # type: ignore[arg-type]

    leased = await pool.acquire({"problem_id": 7}, timeout=12.5)
    assert leased.working_dir == "/workspace"
    assert calls["sample_data"] == {"problem_id": 7}
    assert calls["timeout"] == 12.5

    await leased.close()
    assert calls["released"] == "lease-token"
