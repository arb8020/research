"""JIT-fetch Terminal-Bench 2.0 task definitions.

TB2 task definitions live in github.com/harbor-framework/terminal-bench-2.
Each task is a directory containing task.toml, instruction.md, environment/
(with a Dockerfile), tests/ (with test.sh), and solution/. We clone the repo
once into a cache dir, pinned to a specific commit for reproducibility, and
hand out task directories to the eval runner.

Nothing here depends on Harbor — cloning is pure git. Harbor becomes a
dependency only when eval.py actually constructs a HarborEnvironment.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Pinned commit of the TB2 dataset repo. Bumping this is a deliberate act —
# task definitions may change behavior between commits. Picked this commit
# on 2026-04-17 after the "Various task fixes for TB2.1" commit landed;
# represents the current TB2 set.
TB2_REPO = "https://github.com/harbor-framework/terminal-bench-2.git"
TB2_COMMIT = "53ff2b8"
TB2_CACHE_ROOT = Path.home() / ".rollouts" / "cache" / "terminal-bench-2"


@dataclass(frozen=True)
class TB2TaskRef:
    """Handle to a cloned TB2 task directory."""

    task_id: str
    task_dir: Path

    @property
    def environment_dir(self) -> Path:
        """Harbor's DockerEnvironment wants this (Dockerfile + friends)."""
        return self.task_dir / "environment"

    @property
    def tests_dir(self) -> Path:
        """Harbor's Verifier uploads this to /tests in the container."""
        return self.task_dir / "tests"

    @property
    def task_toml(self) -> Path:
        return self.task_dir / "task.toml"

    @property
    def instruction_md(self) -> Path:
        return self.task_dir / "instruction.md"


def ensure_tb2_checkout(commit: str = TB2_COMMIT) -> Path:
    """Clone TB2 at `commit` into the cache if not already present.

    Returns the path to the checkout. Idempotent — a second call with the
    same commit returns the existing checkout without touching the network.
    Different commits share the cache root but live in sibling directories.
    """
    checkout = TB2_CACHE_ROOT / commit
    marker = checkout / ".rollouts-cloned"
    if marker.exists():
        return checkout

    TB2_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if checkout.exists():
        # Partially-cloned leftover from a prior failed attempt. Wipe.
        import shutil

        shutil.rmtree(checkout)

    # Clone shallow-at-commit. Git doesn't natively shallow-clone an
    # arbitrary commit with --depth 1, so we do init + fetch + checkout.
    logger.info("cloning TB2 @ %s into %s", commit, checkout)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", TB2_REPO], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", commit],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "checkout", "-q", "FETCH_HEAD"], check=True)
    marker.touch()
    logger.info("cloned TB2 @ %s", commit)
    return checkout


def get_task(task_id: str, *, commit: str = TB2_COMMIT) -> TB2TaskRef:
    """Return a TB2TaskRef for `task_id`, cloning the dataset on first call."""
    checkout = ensure_tb2_checkout(commit=commit)
    task_dir = checkout / task_id
    if not task_dir.is_dir():
        raise FileNotFoundError(
            f"TB2 task {task_id!r} not found at {task_dir}. "
            f"Check the spelling or bump TB2_COMMIT to a newer revision."
        )
    for required in ("task.toml", "instruction.md", "environment", "tests"):
        if not (task_dir / required).exists():
            raise RuntimeError(
                f"TB2 task {task_id!r} missing expected {required!r}. "
                f"Dataset shape may have changed; check task_dir={task_dir}"
            )
    return TB2TaskRef(task_id=task_id, task_dir=task_dir)


def list_available_tasks(*, commit: str = TB2_COMMIT) -> list[str]:
    """List task IDs in the cached TB2 checkout."""
    checkout = ensure_tb2_checkout(commit=commit)
    return sorted(
        p.name
        for p in checkout.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (p / "task.toml").exists()
    )
