"""Job registry for rollouts.

Jobs are tracked in ~/.rollouts/jobs.json (local registry).
The local registry is the source of truth for job metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

JOBS_DIR = Path.home() / ".rollouts"
JOBS_FILE = JOBS_DIR / "jobs.json"

JobStatus = Literal["starting", "running", "completed", "failed", "unknown"]


@dataclass
class JobNode:
    """A node participating in a job."""

    provider: str  # e.g. "runpod", "modal"
    node_id: str  # e.g. "leniwdl4iqbujm"


@dataclass
class Job:
    """A rollouts job with its metadata."""

    job_id: str
    nodes: list[JobNode]
    status: JobStatus = "starting"
    config_path: str | None = None
    started_at: str | None = None
    log_path: str | None = None

    @property
    def node_ids(self) -> list[str]:
        return [f"{n.provider}:{n.node_id}" for n in self.nodes]

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "nodes": [{"provider": n.provider, "node_id": n.node_id} for n in self.nodes],
            "status": self.status,
            "config_path": self.config_path,
            "started_at": self.started_at,
            "log_path": self.log_path,
        }

    @classmethod
    def from_dict(cls, job_id: str, data: dict) -> Job:
        nodes = [
            JobNode(provider=n["provider"], node_id=n["node_id"]) for n in data.get("nodes", [])
        ]
        return cls(
            job_id=job_id,
            nodes=nodes,
            status=data.get("status", "unknown"),
            config_path=data.get("config_path"),
            started_at=data.get("started_at"),
            log_path=data.get("log_path"),
        )


def _load_registry() -> dict[str, dict]:
    if not JOBS_FILE.exists():
        return {}
    try:
        with open(JOBS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_registry(registry: dict[str, dict]) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JOBS_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def register_job(
    job_id: str,
    provider: str,
    node_id: str,
    config_path: str | None = None,
    log_path: str | None = None,
) -> Job:
    """Register a new job. Called immediately after generating job_id."""
    job = Job(
        job_id=job_id,
        nodes=[JobNode(provider=provider, node_id=node_id)],
        status="starting",
        config_path=config_path,
        started_at=datetime.now(timezone.utc).isoformat(),
        log_path=log_path,
    )
    registry = _load_registry()
    registry[job_id] = job.to_dict()
    _save_registry(registry)
    return job


def update_job_status(job_id: str, status: JobStatus) -> None:
    registry = _load_registry()
    if job_id in registry:
        registry[job_id]["status"] = status
        _save_registry(registry)


def update_job_node(job_id: str, provider: str, node_id: str) -> None:
    """Update node info after provisioning completes."""
    registry = _load_registry()
    if job_id in registry:
        registry[job_id]["nodes"] = [{"provider": provider, "node_id": node_id}]
        _save_registry(registry)


def list_jobs(limit: int = 50) -> list[Job]:
    """List jobs, most recent first."""
    registry = _load_registry()
    jobs = [Job.from_dict(job_id, data) for job_id, data in registry.items()]
    jobs.sort(key=lambda j: j.started_at or "", reverse=True)
    return jobs[:limit]


def get_job(job_id: str) -> Job:
    registry = _load_registry()
    assert job_id in registry, f"Job not found: {job_id}"
    return Job.from_dict(job_id, registry[job_id])


def get_latest_job() -> Job:
    jobs = list_jobs(limit=1)
    assert jobs, "No rollouts jobs found"
    return jobs[0]
