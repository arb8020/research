"""SandboxPool: manages a pool of GPU sandboxes for kernel scoring.

The pool handles:
1. Provisioning sandboxes via broker (RunPod, Modal, etc.)
2. Starting scoring worker servers on each sandbox
3. Distributing scoring work across available workers
4. Health checks and worker recovery

Uses broker (~/research/broker) for GPU provisioning.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rollouts.gpu_sandbox.config import AnySandboxConfig
    from rollouts.gpu_sandbox.worker import SandboxWorker

logger = logging.getLogger(__name__)


@dataclass
class SandboxPool:
    """Pool of GPU sandboxes for distributed kernel scoring.

    Supports heterogeneous providers (Modal, RunPod, SSH) in the same pool.
    All workers run the same scoring protocol, so provider doesn't matter
    for the scoring interface.

    Usage:
        pool = SandboxPool([
            ModalSandboxConfig(gpu="A100", count=2),
            SSHSandboxConfig(hosts=("user@gpu-server:22",)),
        ])
        await pool.start()
        scores = await pool.score_batch(samples)
        await pool.stop()
    """

    configs: list[AnySandboxConfig] = field(default_factory=list)

    # Internal state
    _workers: list[SandboxWorker] = field(default_factory=list, repr=False)
    _started: bool = field(default=False, repr=False)
    _next_worker: int = field(default=0, repr=False)  # Round-robin index

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
        """Provision sandboxes and connect workers.

        Idempotent - safe to call multiple times.
        """
        if self._started:
            return

        if not self.configs:
            logger.info("SandboxPool: no configs, using local subprocess scoring")
            self._workers = [self._create_local_worker()]
        else:
            logger.info(
                f"SandboxPool: starting {self.num_workers} workers from {len(self.configs)} configs"
            )
            self._workers = await self._provision_all()

        self._started = True
        logger.info(f"SandboxPool: {len(self._workers)} workers ready")

    async def stop(self) -> None:
        """Stop all workers and cleanup sandboxes."""
        if not self._started:
            return

        logger.info(f"SandboxPool: stopping {len(self._workers)} workers")
        for worker in self._workers:
            try:
                await worker.close()
            except Exception as e:
                logger.warning(f"Error closing worker: {e}")

        self._workers = []
        self._started = False

    async def score_one(
        self,
        kernel_code: str,
        ref_code: str,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Score a single kernel on the next available worker.

        Args:
            kernel_code: Generated kernel code (ModelNew class)
            ref_code: Reference problem code (Model, get_inputs, get_init_inputs)
            timeout: Scoring timeout in seconds

        Returns:
            Dict with: compiled, correct, speedup, reward, error (if any)
        """
        assert self._started, "Pool not started. Call start() first."
        worker = self._get_next_worker()
        return await worker.score(kernel_code, ref_code, timeout)

    async def score_batch(
        self,
        samples: list[dict[str, Any]],
        timeout: float = 120.0,
    ) -> list[dict[str, Any]]:
        """Score a batch of samples across all workers.

        Distributes work round-robin across workers for load balancing.

        Args:
            samples: List of dicts with 'kernel_code' and 'ref_code' keys
            timeout: Per-sample scoring timeout

        Returns:
            List of score dicts (same order as input samples)
        """
        assert self._started, "Pool not started. Call start() first."

        # Create tasks for each sample
        tasks = []
        for sample in samples:
            worker = self._get_next_worker()
            task = worker.score(
                sample["kernel_code"],
                sample["ref_code"],
                timeout,
            )
            tasks.append(task)

        # Run all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        scores = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                scores.append({
                    "compiled": 0.0,
                    "correct": 0.0,
                    "speedup": 0.0,
                    "reward": 0.0,
                    "error": str(result),
                })
            else:
                scores.append(result)

        return scores

    def _get_next_worker(self) -> SandboxWorker:
        """Get next worker using round-robin."""
        worker = self._workers[self._next_worker]
        self._next_worker = (self._next_worker + 1) % len(self._workers)
        return worker

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
            elif isinstance(config, BrokerSandboxConfig):
                new_workers = await self._provision_broker(config)
                workers.extend(new_workers)
            elif isinstance(config, ExistingInstanceConfig):
                new_workers = await self._connect_existing(config)
                workers.extend(new_workers)
            else:
                raise ValueError(f"Unknown config type: {type(config)}")

        return workers

    def _create_local_worker(self) -> SandboxWorker:
        """Create a local subprocess worker."""
        from rollouts.gpu_sandbox.worker import LocalSandboxWorker

        return LocalSandboxWorker()

    async def _provision_broker(self, config: BrokerSandboxConfig) -> list[SandboxWorker]:
        """Provision sandboxes via broker."""
        from broker.client import GPUClient
        from broker.credentials import get_credentials
        from rollouts.gpu_sandbox.worker import BrokerSandboxWorker

        client = GPUClient(credentials=get_credentials())

        # Build query
        query = client.gpu_type.contains(config.gpu_type)
        if config.provider:
            query = query & (client.provider == config.provider)
        if config.max_price:
            query = query & (client.price_per_hour <= config.max_price)

        workers = []
        instances = []

        for i in range(config.count):
            logger.info(
                f"Provisioning sandbox {i + 1}/{config.count} "
                f"(gpu={config.gpu_type}, provider={config.provider or 'any'})"
            )

            instance = await client.create(
                query,
                gpu_count=1,
                name=f"kernel-scorer-{i}",
                exposed_ports=list(config.exposed_ports),
                docker_image=config.docker_image,
            )

            logger.info(f"Waiting for SSH ready on {instance.id}...")
            await instance.wait_until_ssh_ready(timeout=config.timeout_seconds)

            instances.append(instance)

            # Create worker wrapping the instance
            worker = BrokerSandboxWorker(
                instance=instance,
                keep_alive=config.keep_alive,
            )
            workers.append(worker)

        # Store instances for cleanup
        self._broker_instances = getattr(self, "_broker_instances", [])
        self._broker_instances.extend(instances)

        return workers

    async def _connect_existing(self, config: ExistingInstanceConfig) -> list[SandboxWorker]:
        """Connect to existing broker instances."""
        from rollouts.gpu_sandbox.worker import BrokerSandboxWorker

        workers = []
        for instance in config.instances:
            worker = BrokerSandboxWorker(
                instance=instance,
                keep_alive=True,  # Don't terminate instances we didn't provision
            )
            workers.append(worker)

        return workers

    async def __aenter__(self) -> SandboxPool:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
