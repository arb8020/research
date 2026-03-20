from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .modal_sandbox_resource import ManagedModalSandboxResource, ModalSandboxManager
from .resources import SandboxWorkspaceResource


@runtime_checkable
class WorkspaceResourcePool(Protocol):
    """Pool of persistent workspace resources.

    Environments can acquire one of these during deserialize/initialize and hold
    it across turns. This is the persistent-resource side of the future pool
    model; transient per-tool-call resources can reuse the same acquire/release
    algebra later without changing the environment-facing shape.
    """

    async def acquire(
        self,
        sample_data: dict[str, object],
        *,
        timeout: float | None = None,
    ) -> AcquiredWorkspaceResource: ...


@dataclass
class AcquiredWorkspaceResource:
    """Leased workspace resource with explicit release semantics."""

    resource: SandboxWorkspaceResource
    _release: Callable[[], Awaitable[None]]
    _released: bool = False

    @property
    def working_dir(self) -> str:
        return self.resource.working_dir

    async def close(self) -> None:
        if self._released:
            return
        self._released = True
        await self._release()

    async def __aenter__(self) -> AcquiredWorkspaceResource:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        del exc_type, exc, tb
        await self.close()


@dataclass
class ModalSandboxWorkspacePool:
    """Workspace pool adapter backed by ModalSandboxManager.

    This is intentionally thin. It gives environments an explicit `acquire()`
    shape now, while keeping the current implementation on the existing Modal
    sandbox manager. A future separate GPU-eval pool can implement the same
    environment-facing lease model.
    """

    manager: ModalSandboxManager

    async def acquire(
        self,
        sample_data: dict[str, object],
        *,
        timeout: float | None = None,
    ) -> AcquiredWorkspaceResource:
        lease, resource = await self.manager.acquire(sample_data, timeout=timeout)

        async def release() -> None:
            await self.manager.release(lease)

        return AcquiredWorkspaceResource(resource=resource, _release=release)

    def make_managed_resource(
        self,
        sample_data: dict[str, object],
    ) -> ManagedModalSandboxResource:
        """Compatibility helper for older environment code.

        Existing code that wants a self-managing resource can keep using the
        `ManagedModalSandboxResource` wrapper until environments are updated to
        hold explicit leases directly.
        """

        return self.manager.make_resource(sample_data)
