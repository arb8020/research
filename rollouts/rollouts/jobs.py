"""Job registry for rollouts — queries broker for live pods.

The cloud provider (RunPod, etc.) is the single source of truth.
Jobs are identified by pod name prefix "rollouts/".

Tiger Style:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
"""

from __future__ import annotations

from dataclasses import dataclass

# Naming convention for rollouts pods
ROLLOUTS_POD_PREFIX = "rollouts/"


@dataclass(frozen=True)
class JobNode:
    """A node participating in a job."""

    provider: str  # e.g. "runpod", "modal"
    node_id: str  # e.g. "leniwdl4iqbujm"


@dataclass(frozen=True)
class Job:
    """A rollouts job with its nodes."""

    job_id: str
    nodes: tuple[JobNode, ...]

    @property
    def node_ids(self) -> list[str]:
        """Return provider:node_id strings for all nodes."""
        return [f"{n.provider}:{n.node_id}" for n in self.nodes]


def _get_credentials() -> dict[str, str]:
    """Get broker credentials."""
    from broker.credentials import get_credentials

    return get_credentials()


def _list_instances_sync() -> list:
    """Query broker for all instances (sync wrapper)."""
    import trio

    from broker.client import GPUClient

    credentials = _get_credentials()
    if not credentials:
        return []

    async def _fetch() -> list:
        client = GPUClient(credentials=credentials)
        return await client.list_instances()

    return trio.run(_fetch)


def _instances_to_jobs(instances: list) -> list[Job]:
    """Convert broker instances to Job objects, filtering by naming convention."""
    jobs = []
    for inst in instances:
        name = inst.name or ""
        if not name.startswith(ROLLOUTS_POD_PREFIX):
            continue

        job_id = name[len(ROLLOUTS_POD_PREFIX) :]
        if not job_id:
            continue

        node = JobNode(provider=inst.provider, node_id=inst.id)
        jobs.append(Job(job_id=job_id, nodes=(node,)))

    # Sort by job_id descending (job_id contains timestamp like run_20250127-143052)
    jobs.sort(key=lambda j: j.job_id, reverse=True)
    return jobs


def list_jobs() -> list[Job]:
    """List all rollouts jobs from broker, most recent first."""
    instances = _list_instances_sync()
    return _instances_to_jobs(instances)


def get_job(job_id: str) -> Job:
    """Get a job by ID. Raises AssertionError if not found."""
    jobs = list_jobs()
    for job in jobs:
        if job.job_id == job_id:
            return job
    raise AssertionError(f"Job not found: {job_id}")


def get_latest_job() -> Job:
    """Get the most recently started job. Raises AssertionError if none."""
    jobs = list_jobs()
    assert jobs, "No rollouts jobs found"
    return jobs[0]
