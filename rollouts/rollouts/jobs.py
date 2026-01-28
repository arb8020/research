"""Local job registry for rollouts.

Maps job_id → nodes + metadata. Stored at ~/.rollouts/jobs.json.

This is the *only* source of truth for job→node relationships.
Live instance state (ports, IPs, status) always comes from broker.

Invariants:
- Provisioner writes at launch (single writer per job)
- --attach reads for node mapping, queries broker for live data
- --runs reads + reconciles against broker (prunes dead jobs)
- Never stores ports/IPs — always queries broker for those

Tiger Style:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

JOBS_PATH = Path.home() / ".rollouts" / "jobs.json"


@dataclass(frozen=True)
class JobNode:
    """A node participating in a job."""

    provider: str  # e.g. "runpod", "modal"
    node_id: str  # e.g. "leniwdl4iqbujm"
    role: str  # e.g. "training", "inference"


@dataclass(frozen=True)
class Job:
    """A rollouts job with its nodes and metadata."""

    job_id: str
    nodes: tuple[JobNode, ...]
    script: str
    started_at: str  # ISO 8601

    @property
    def node_ids(self) -> list[str]:
        """Return provider:node_id strings for all nodes."""
        return [f"{n.provider}:{n.node_id}" for n in self.nodes]


def _read_jobs_file() -> dict:
    """Read jobs.json, return empty dict if missing."""
    if not JOBS_PATH.exists():
        return {}
    return json.loads(JOBS_PATH.read_text())


def _write_jobs_file(data: dict) -> None:
    """Write jobs.json atomically."""
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOBS_PATH.write_text(json.dumps(data, indent=2) + "\n")


def save_job(job: Job) -> None:
    """Save a job entry. Overwrites if job_id already exists."""
    data = _read_jobs_file()
    data[job.job_id] = {
        "nodes": [asdict(n) for n in job.nodes],
        "script": job.script,
        "started_at": job.started_at,
    }
    _write_jobs_file(data)


def get_job(job_id: str) -> Job:
    """Get a job by ID. Raises AssertionError if not found."""
    data = _read_jobs_file()
    assert job_id in data, f"Job not found: {job_id}"
    return _parse_job(job_id, data[job_id])


def get_latest_job() -> Job:
    """Get the most recently started job. Raises AssertionError if none."""
    data = _read_jobs_file()
    assert data, f"No jobs found in {JOBS_PATH}"
    # Sort by started_at descending
    latest_id = max(data, key=lambda k: data[k].get("started_at", ""))
    return _parse_job(latest_id, data[latest_id])


def list_jobs() -> list[Job]:
    """List all jobs, most recent first."""
    data = _read_jobs_file()
    jobs = [_parse_job(k, v) for k, v in data.items()]
    jobs.sort(key=lambda j: j.started_at, reverse=True)
    return jobs


def prune_jobs(live_node_ids: set[str]) -> int:
    """Remove jobs whose nodes are all dead.

    Args:
        live_node_ids: Set of node IDs currently alive (from broker).

    Returns:
        Number of jobs pruned.
    """
    data = _read_jobs_file()
    alive = {}
    pruned = 0

    for job_id, entry in data.items():
        nodes = entry.get("nodes", [])
        has_live_node = any(n["node_id"] in live_node_ids for n in nodes)
        if has_live_node:
            alive[job_id] = entry
        else:
            pruned += 1

    if pruned > 0:
        _write_jobs_file(alive)

    return pruned


def _parse_job(job_id: str, entry: dict) -> Job:
    """Parse a job entry from the JSON structure."""
    nodes = tuple(
        JobNode(
            provider=n["provider"],
            node_id=n["node_id"],
            role=n["role"],
        )
        for n in entry.get("nodes", [])
    )
    return Job(
        job_id=job_id,
        nodes=nodes,
        script=entry.get("script", "?"),
        started_at=entry.get("started_at", "?"),
    )


def make_job(
    job_id: str,
    provider: str,
    node_id: str,
    script: str,
    role: str = "training",
) -> Job:
    """Convenience: create a single-node Job and save it."""
    job = Job(
        job_id=job_id,
        nodes=(JobNode(provider=provider, node_id=node_id, role=role),),
        script=script,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    save_job(job)
    return job
