from __future__ import annotations

from typing import Any, cast

import pytest
import trio

from rollouts.gpu_sandbox.pool import SandboxPool


class FakeWorker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    async def score(self, kernel_code: str, ref_code: str, timeout: float) -> dict:
        return {
            "compiled": 1.0,
            "correct": 1.0,
            "speedup": float(len(kernel_code) + len(ref_code)),
            "pass_rate": 1.0,
        }

    async def close(self) -> None:
        self.closed = True


def test_sandbox_pool_explicit_leases_and_stats() -> None:
    async def run() -> None:
        pool = SandboxPool()
        pool._workers = cast(Any, [FakeWorker("w0"), FakeWorker("w1")])
        send, recv = trio.open_memory_channel[int](len(pool._workers))
        pool._available_worker_ids_send = send
        pool._available_worker_ids_recv = recv
        pool._started = True
        for worker_index in range(len(pool._workers)):
            send.send_nowait(worker_index)

        lease = await pool.acquire()
        stats_during_lease = pool.stats()
        assert stats_during_lease["in_flight"] == 1
        assert stats_during_lease["available_workers"] == 1

        await pool.release(lease)
        stats_after_release = pool.stats()
        assert stats_after_release["in_flight"] == 0
        assert stats_after_release["available_workers"] == 2

        result = await pool.score_one("abc", "def")
        assert result["compiled"] == 1.0
        assert pool.stats()["score_requests"] == 1

    trio.run(run)


def test_sandbox_pool_score_batch_respects_started_boundary() -> None:
    async def run() -> None:
        pool = SandboxPool()

        with pytest.raises(ValueError, match="Pool not started"):
            await pool.score_batch([{"kernel_code": "a", "ref_code": "b"}])

    trio.run(run)
