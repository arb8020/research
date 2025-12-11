"""Job submission and monitoring for remote execution.

This module provides a high-level API for running jobs on remote machines.
Composes the lower-level primitives (tmux, job_monitor, python_env, gpu).

Tiger Style:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
- Dataclasses for state

Usage:
    from bifrost import BifrostClient
    from kerbal import submit, DependencyConfig

    client = BifrostClient("root@gpu:22")
    workspace = client.push()

    job = submit(
        client,
        command="python train.py",
        workspace=workspace,
        gpu_ids=[0, 1, 2, 3],
        deps=DependencyConfig(dependencies=["torch", "transformers"]),
    )

    # Option 1: Stream logs (blocking)
    success, exit_code = job.stream()

    # Option 2: Detach and check later
    print(f"Job running: {job.session_name}")
    # ... later ...
    status = job.status()
    if status == "success":
        print("Done!")
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from kerbal.gpu import check_gpus_available
from kerbal.job_monitor import LogStreamConfig, stream_log_until_complete
from kerbal.tmux import start_tmux_session

if TYPE_CHECKING:
    from bifrost import BifrostClient

    from kerbal.protocol import DependencyConfig

logger = logging.getLogger(__name__)


@dataclass
class JobHandle:
    """Handle to a running or completed job.

    Tiger Style: Immutable reference to job state on remote.
    All methods query remote state - no local caching.
    """

    client: "BifrostClient"
    session_name: str
    log_file: str
    workspace: str

    def stream(self, timeout_sec: int = 3600) -> tuple[bool, int | None]:
        """Stream logs until job completes.

        Blocks and prints logs in real-time. Returns when job finishes.

        Args:
            timeout_sec: Maximum time to wait (default: 1 hour)

        Returns:
            (success, exit_code) - success=True if exit_code==0
        """
        config = LogStreamConfig(
            session_name=self.session_name,
            log_file=self.log_file,
            timeout_sec=timeout_sec,
        )
        success, exit_code, err = stream_log_until_complete(self.client, config)
        if err:
            logger.error(f"Job failed: {err}")
        return success, exit_code

    def wait(self, timeout_sec: int = 3600, poll_interval: float = 5.0) -> int | None:
        """Wait for job completion without streaming logs.

        Args:
            timeout_sec: Maximum time to wait
            poll_interval: How often to check status

        Returns:
            Exit code, or None if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            status = self.status()
            if status == "success":
                return 0
            elif status == "failed":
                return self._get_exit_code()
            time.sleep(poll_interval)
        return None

    def status(self) -> Literal["running", "success", "failed", "unknown"]:
        """Check job status.

        Returns:
            "running" - job still executing
            "success" - job completed with exit code 0
            "failed" - job completed with non-zero exit code
            "unknown" - cannot determine status
        """
        # Check if tmux session alive
        result = self.client.exec(f"tmux has-session -t {self.session_name} 2>&1")
        if result.exit_code == 0:
            return "running"

        # Session dead - check exit code
        exit_code = self._get_exit_code()
        if exit_code is None:
            return "unknown"
        elif exit_code == 0:
            return "success"
        else:
            return "failed"

    def logs(self, tail: int | None = None) -> str:
        """Get log content.

        Args:
            tail: If set, only return last N lines

        Returns:
            Log content as string
        """
        if tail:
            cmd = f"tail -n {tail} {self.log_file} 2>/dev/null || true"
        else:
            cmd = f"cat {self.log_file} 2>/dev/null || true"
        result = self.client.exec(cmd)
        return result.stdout

    def kill(self) -> None:
        """Kill the job if still running."""
        self.client.exec(f"tmux kill-session -t {self.session_name} 2>/dev/null || true")
        logger.info(f"Killed job: {self.session_name}")

    def _get_exit_code(self) -> int | None:
        """Extract exit code from log file."""
        cmd = (
            f"grep -E 'EXIT_CODE:' {self.log_file} 2>/dev/null | "
            f"tail -1 | awk '{{print $NF}}'"
        )
        result = self.client.exec(cmd)
        if result.stdout and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
        return None


def submit(
    client: "BifrostClient",
    command: str,
    workspace: str,
    gpu_ids: list[int] | None = None,
    deps: "DependencyConfig | None" = None,
    env_vars: dict[str, str] | None = None,
    job_name: str | None = None,
    check_gpus: bool = True,
) -> JobHandle:
    """Submit a job to run on remote.

    Deploys dependencies (if specified), starts job in tmux, returns handle.
    Job runs detached - use handle.stream() to watch logs.

    Args:
        client: BifrostClient instance
        command: Command to run
        workspace: Remote workspace path (from client.push())
        gpu_ids: GPUs to use (sets CUDA_VISIBLE_DEVICES)
        deps: Dependencies to install (optional)
        env_vars: Additional environment variables
        job_name: Tmux session name (default: "job-{timestamp}")
        check_gpus: Whether to verify GPU availability (default: True)

    Returns:
        JobHandle for monitoring the job

    Raises:
        AssertionError: If preconditions not met (GPUs unavailable, etc.)

    Example:
        job = submit(client, "python train.py", workspace, gpu_ids=[0,1])
        success, exit_code = job.stream()  # Watch logs
    """
    assert client is not None, "BifrostClient required"
    assert command, "command required"
    assert workspace, "workspace required"

    # Generate job name if not provided
    if job_name is None:
        import time
        job_name = f"job-{int(time.time())}"

    logger.info(f"Submitting job: {job_name}")
    logger.info(f"  Command: {command[:80]}{'...' if len(command) > 80 else ''}")
    logger.info(f"  Workspace: {workspace}")

    # Check GPU availability
    if check_gpus and gpu_ids:
        logger.info(f"  Checking GPUs: {gpu_ids}")
        available, err = check_gpus_available(client, gpu_ids)
        assert available, f"GPUs not available: {err}"
        logger.info(f"  GPUs available: {gpu_ids}")

    # Install dependencies if specified
    if deps is not None:
        logger.info("  Installing dependencies...")
        from kerbal.python_env import setup_script_deps
        setup_script_deps(client, workspace, deps)
        logger.info("  Dependencies installed")

    # Build environment variables
    final_env_vars = dict(env_vars) if env_vars else {}
    if gpu_ids:
        final_env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    # Setup log file
    log_file = f"{workspace}/{job_name}.log"

    # Start job in tmux
    session_name, err = start_tmux_session(
        client,
        session_name=job_name,
        command=command,
        workspace=workspace,
        log_file=log_file,
        env_vars=final_env_vars if final_env_vars else None,
    )
    assert err is None, f"Failed to start job: {err}"

    logger.info(f"  Job started: {session_name}")
    logger.info(f"  Log file: {log_file}")

    return JobHandle(
        client=client,
        session_name=session_name,
        log_file=log_file,
        workspace=workspace,
    )
