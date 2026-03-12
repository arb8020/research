"""SandboxPool: explicit worker leases for GPU sandbox scoring."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rollouts.gpu_sandbox.config import AnySandboxConfig
    from rollouts.gpu_sandbox.worker import SandboxWorker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SandboxLease:
    """Lease for exclusive access to a sandbox worker."""

    worker_index: int
    acquired_at: float


@dataclass(frozen=True)
class SandboxPoolStats:
    """Explicit observability snapshot for sandbox resources."""

    started: bool
    num_workers: int
    available_workers: int
    in_flight: int
    acquire_count: int
    release_count: int
    score_requests: int
    score_failures: int
    wait_events: int
    max_in_flight: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "num_workers": self.num_workers,
            "available_workers": self.available_workers,
            "in_flight": self.in_flight,
            "acquire_count": self.acquire_count,
            "release_count": self.release_count,
            "score_requests": self.score_requests,
            "score_failures": self.score_failures,
            "wait_events": self.wait_events,
            "max_in_flight": self.max_in_flight,
        }


@dataclass
class SandboxPool:
    """Pool of GPU sandboxes with explicit acquire/release semantics."""

    configs: list[AnySandboxConfig] = field(default_factory=list)

    _workers: list[SandboxWorker] = field(default_factory=list, repr=False)
    _available_worker_ids: asyncio.Queue[int] = field(default_factory=asyncio.Queue, repr=False)
    _started: bool = field(default=False, repr=False)
    _in_flight: int = field(default=0, repr=False)
    _acquire_count: int = field(default=0, repr=False)
    _release_count: int = field(default=0, repr=False)
    _score_requests: int = field(default=0, repr=False)
    _score_failures: int = field(default=0, repr=False)
    _wait_events: int = field(default=0, repr=False)
    _max_in_flight: int = field(default=0, repr=False)

    @property
    def is_local(self) -> bool:
        """True if pool has no remote sandboxes (uses local subprocess)."""
        from rollouts.gpu_sandbox.config import LocalSandboxConfig

        return not self.configs or all(isinstance(c, LocalSandboxConfig) for c in self.configs)

    @property
    def num_workers(self) -> int:
        """Total number of workers across all configs."""
        return sum(c.count for c in self.configs) if self.configs else 1

    async def start(self) -> None:
        """Provision sandboxes and populate the lease queue."""
        if self._started:
            return

        if not self.configs:
            logger.info("SandboxPool: no configs, using local subprocess scoring")
            workers = [self._create_local_worker()]
        else:
            logger.info(
                "SandboxPool: starting %s workers from %s configs",
                self.num_workers,
                len(self.configs),
            )
            workers = await self._provision_all()

        self._workers = workers
        self._available_worker_ids = asyncio.Queue()
        for worker_index in range(len(self._workers)):
            self._available_worker_ids.put_nowait(worker_index)

        self._started = True
        logger.info("SandboxPool: %s workers ready", len(self._workers))

    async def stop(self) -> None:
        """Stop all workers and reset lease state."""
        if not self._started:
            return

        logger.info("SandboxPool: stopping %s workers", len(self._workers))
        for worker in self._workers:
            try:
                await worker.close()
            except Exception as e:
                logger.warning("Error closing worker: %s", e)

        self._workers = []
        self._available_worker_ids = asyncio.Queue()
        self._started = False
        self._in_flight = 0

    async def ensure_capacity(self, target_idle: int | None = None) -> None:
        """Ensure the pool is running and optionally validate idle capacity."""
        await self.start()
        if target_idle is None:
            return
        if target_idle < 0:
            raise ValueError(f"target_idle must be >= 0, got {target_idle}")
        if target_idle > len(self._workers):
            raise ValueError(
                f"target_idle={target_idle} exceeds provisioned workers={len(self._workers)}"
            )

    async def acquire(self, timeout: float | None = None) -> SandboxLease:
        """Acquire a worker lease."""
        if not self._started:
            raise ValueError("Pool not started. Call start() first.")

        should_wait = self._available_worker_ids.empty()
        if should_wait:
            self._wait_events += 1

        if timeout is None:
            worker_index = await self._available_worker_ids.get()
        else:
            try:
                worker_index = await asyncio.wait_for(
                    self._available_worker_ids.get(), timeout=timeout
                )
            except TimeoutError as e:
                raise TimeoutError(f"Timed out acquiring sandbox lease after {timeout}s") from e

        self._acquire_count += 1
        self._in_flight += 1
        if self._in_flight > self._max_in_flight:
            self._max_in_flight = self._in_flight
        return SandboxLease(worker_index=worker_index, acquired_at=time.time())

    async def release(self, lease: SandboxLease) -> None:
        """Release a previously acquired worker lease."""
        if not self._started:
            raise ValueError("Pool not started. Cannot release lease.")
        if lease.worker_index < 0:
            raise ValueError(f"worker_index must be >= 0, got {lease.worker_index}")
        if lease.worker_index >= len(self._workers):
            raise ValueError(
                f"lease worker_index {lease.worker_index} out of range for {len(self._workers)} workers"
            )
        assert self._in_flight > 0, "cannot release lease when no workers are in flight"

        self._release_count += 1
        self._in_flight -= 1
        self._available_worker_ids.put_nowait(lease.worker_index)

    def stats(self) -> dict[str, Any]:
        """Return explicit resource/lease statistics."""
        snapshot = SandboxPoolStats(
            started=self._started,
            num_workers=len(self._workers),
            available_workers=self._available_worker_ids.qsize(),
            in_flight=self._in_flight,
            acquire_count=self._acquire_count,
            release_count=self._release_count,
            score_requests=self._score_requests,
            score_failures=self._score_failures,
            wait_events=self._wait_events,
            max_in_flight=self._max_in_flight,
        )
        return snapshot.to_dict()

    async def score_one(
        self,
        kernel_code: str,
        ref_code: str,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Score a single kernel using an explicit worker lease."""
        self._score_requests += 1
        lease = await self.acquire(timeout=timeout)
        try:
            worker = self._workers[lease.worker_index]
            return await worker.score(kernel_code, ref_code, timeout)
        except Exception as e:
            self._score_failures += 1
            raise
        finally:
            await self.release(lease)

    async def score_batch(
        self,
        samples: list[dict[str, Any]],
        timeout: float = 120.0,
    ) -> list[dict[str, Any]]:
        """Score a batch by acquiring explicit leases per request."""
        if not self._started:
            raise ValueError("Pool not started. Call start() first.")

        async def score_request(sample: dict[str, Any]) -> dict[str, Any]:
            try:
                return await self.score_one(
                    sample["kernel_code"],
                    sample["ref_code"],
                    timeout,
                )
            except Exception as e:
                return {
                    "compiled": 0.0,
                    "correct": 0.0,
                    "speedup": 0.0,
                    "reward": 0.0,
                    "error": str(e),
                }

        return await asyncio.gather(*[score_request(sample) for sample in samples])

    async def _provision_all(self) -> list[SandboxWorker]:
        """Provision all sandboxes from configs."""
        from rollouts.gpu_sandbox.config import (
            BrokerSandboxConfig,
            ExistingInstanceConfig,
            LocalSandboxConfig,
        )

        workers = []
        for config in self.configs:
            if isinstance(config, LocalSandboxConfig):
                for _ in range(config.count):
                    workers.append(self._create_local_worker())
                continue

            if isinstance(config, BrokerSandboxConfig):
                workers.extend(await self._provision_broker(config))
                continue

            if isinstance(config, ExistingInstanceConfig):
                workers.extend(await self._connect_existing(config))
                continue

            raise ValueError(f"Unknown config type: {type(config)}")

        return workers

    def _create_local_worker(self) -> SandboxWorker:
        """Create a local subprocess worker."""
        from rollouts.gpu_sandbox.worker import LocalSandboxWorker

        return LocalSandboxWorker()

    async def _provision_broker(self, config: Any) -> list[SandboxWorker]:
        """Provision sandboxes via broker."""
        from broker.client import GPUClient
        from broker.credentials import get_credentials
        from rollouts.gpu_sandbox.worker import BrokerSandboxWorker

        client = GPUClient(credentials=get_credentials())

        query = client.gpu_type.contains(config.gpu_type)
        if config.provider:
            query = query & (client.provider == config.provider)
        if config.max_price:
            query = query & (client.price_per_hour <= config.max_price)

        workers = []
        instances = []
        for i in range(config.count):
            logger.info(
                "Provisioning sandbox %s/%s (gpu=%s, provider=%s)",
                i + 1,
                config.count,
                config.gpu_type,
                config.provider or "any",
            )
            instance = await client.create(
                query,
                gpu_count=1,
                name=f"kernel-scorer-{i}",
                exposed_ports=list(config.exposed_ports),
                docker_image=config.docker_image,
            )
            assert instance is not None, "Broker returned no instance"
            logger.info("Waiting for SSH ready on %s...", instance.id)
            await instance.wait_until_ssh_ready(timeout=config.timeout_seconds)
            instances.append(instance)
            workers.append(BrokerSandboxWorker(instance=instance, keep_alive=config.keep_alive))

        self._broker_instances = getattr(self, "_broker_instances", [])
        self._broker_instances.extend(instances)
        return workers

    async def _connect_existing(self, config: Any) -> list[SandboxWorker]:
        """Connect to existing broker instances."""
        from rollouts.gpu_sandbox.worker import BrokerSandboxWorker

        workers = []
        for instance in config.instances:
            workers.append(
                BrokerSandboxWorker(
                    instance=instance,
                    keep_alive=True,
                )
            )
        return workers

    async def __aenter__(self) -> SandboxPool:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.stop()
