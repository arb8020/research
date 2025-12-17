# MiniRay Jobs: Detached Work

> **Status:** Design doc (not implemented)

## Problem

Training jobs block your terminal. You want to:
1. Kick off a training run
2. Get your terminal back
3. Check on it later

## Non-goals

This is NOT a job queue. No:
- Persistence across reboots
- Retry logic
- Scheduling / priority
- Distributed coordination

Just "run this in the background, let me check on it."

## Design

### Why tmux?

Unix processes attach to their parent. When parent dies, children get SIGHUP. You need *something* to keep them alive.

Options considered:
1. **Double fork (daemon pattern)** - No way to reconnect or get output
2. **Double fork + Unix socket** - Reimplementing tmux poorly
3. **tmux** - Already solves session persistence, on every machine

tmux is the right primitive. We just wrap it minimally.

### Data structure

```python
@dataclass(frozen=True)
class Job:
    """Job info - just data, not a resource."""
    name: str
    log_path: Path
```

Job is a frozen dataclass, not a class with methods. It doesn't own the tmux session - it's just info about one.

### Functions

```python
def submit(cmd: str, name: str | None = None) -> Job:
    """Submit job to tmux, return job info."""

def is_alive(job: Job) -> bool:
    """Check if job's tmux session exists."""

def logs(job: Job, tail: int = 50) -> list[str]:
    """Read last N lines of job log."""

def kill(job: Job) -> None:
    """Kill job's tmux session."""

def attach(job: Job) -> None:
    """Attach to job's tmux session (replaces current process)."""

def list_jobs() -> list[Job]:
    """List all running miniray jobs."""
```

Functions operate on data. No fake OOP.

### Implementation sketch

```python
# miniray/jobs.py

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import subprocess
import os


JOBS_DIR = Path("/tmp/miniray/jobs")


@dataclass(frozen=True)
class Job:
    name: str
    log_path: Path


def submit(cmd: str, name: str | None = None) -> Job:
    name = name or f"job-{uuid4().hex[:6]}"
    log_path = JOBS_DIR / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "tmux", "new-session", "-d", "-s", name,
        f"{cmd} 2>&1 | tee {log_path}"
    ], check=True)

    return Job(name=name, log_path=log_path)


def is_alive(job: Job) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", job.name],
        capture_output=True
    )
    return result.returncode == 0


def logs(job: Job, tail: int = 50) -> list[str]:
    if not job.log_path.exists():
        return []
    return job.log_path.read_text().splitlines()[-tail:]


def kill(job: Job) -> None:
    subprocess.run(["tmux", "kill-session", "-t", job.name], capture_output=True)


def attach(job: Job) -> None:
    os.execvp("tmux", ["tmux", "attach", "-t", job.name])


def list_jobs() -> list[Job]:
    if not JOBS_DIR.exists():
        return []

    jobs = []
    for log_file in JOBS_DIR.glob("*.log"):
        name = log_file.stem
        job = Job(name=name, log_path=log_file)
        if is_alive(job):
            jobs.append(job)
    return jobs
```

## Usage

```python
from miniray.jobs import submit, is_alive, logs, list_jobs, attach

# Submit a training job
job = submit("python examples/rl/calculator/grpo_01_01.py")
print(f"Started: {job.name}")  # job-a3f2b1

# Check on it later
if is_alive(job):
    print(logs(job, tail=5))

# List all running jobs
for j in list_jobs():
    print(j.name)

# Attach to see live output
attach(job)  # Takes over terminal
```

Or from shell:
```bash
# Submit
python -c "from miniray.jobs import submit; print(submit('python train.py').name)"

# Attach directly
tmux attach -t job-a3f2b1
```

## Why this fits MiniRay

MiniRay's philosophy (from Heinrich):
- Simple primitives over abstractions
- You see and control everything
- Use Unix correctly, don't reinvent it

Jobs follows this:
- tmux is the primitive (we don't reimplement it)
- Job is just data (frozen dataclass)
- Functions are explicit (no hidden state)
- ~50 lines total

## What this is NOT

Per the classes vs functional doc: Job doesn't own a resource. The tmux session exists independently. Job is just info about it - so it's a frozen dataclass, not a class with methods.

This is also not part of MiniRay's core (Worker, RemoteWorker, Cluster). Those are process primitives. Jobs is a convenience layer on top, using tmux for persistence.
