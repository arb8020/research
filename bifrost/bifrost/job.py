"""Job operations for bifrost (functions-over-classes pattern).

Pure functions that operate on JobInfo - no methods on the dataclass itself.
Session is passed explicitly to every function.

This absorbs functionality from kerbal.job_monitor into bifrost.

Tiger Style:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
- Tuple returns for errors

Example:
    from bifrost import BifrostClient
    from bifrost.types import ProcessSpec, JobInfo
    from bifrost.job import job_status, job_wait, job_stream_logs, job_kill

    client = BifrostClient("root@gpu:22")
    job = client.submit(ProcessSpec(command="python", args=("train.py",)), name="training")

    # Check status
    status = job_status(client, job)  # "running" | "completed" | "failed"

    # Stream logs
    for line in job_stream_logs(client, job):
        print(line)

    # Wait for completion
    exit_code = job_wait(client, job, timeout=3600)

    # Or kill the job
    job_kill(client, job)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import BifrostClient
    from .types import JobInfo

logger = logging.getLogger(__name__)


def job_status(session: BifrostClient, job: JobInfo) -> str:
    """Check job status.

    Args:
        session: BifrostClient instance (owns SSH connection)
        job: JobInfo identifier

    Returns:
        "running" if tmux session is alive
        "completed" if session has exited
    """
    result = session.exec(f"tmux has-session -t {job.tmux_session} 2>/dev/null")
    return "running" if result.exit_code == 0 else "completed"


def job_exit_code(session: BifrostClient, job: JobInfo) -> int | None:
    """Get job exit code from log file.

    Looks for "EXIT_CODE: N" marker written by tmux wrapper.

    Args:
        session: BifrostClient instance
        job: JobInfo identifier

    Returns:
        Exit code as int, or None if not found (job still running or marker missing)
    """
    if not job.log_file:
        return None

    # Look for exit code marker
    cmd = f"grep 'EXIT_CODE:' {job.log_file} 2>/dev/null | tail -1 | awk '{{print $NF}}'"
    result = session.exec(cmd)

    if result.stdout:
        code_str = result.stdout.strip()
        if code_str.lstrip("-").isdigit():
            return int(code_str)

    return None


def job_wait(
    session: BifrostClient,
    job: JobInfo,
    timeout: float | None = None,
    poll_interval: float = 2.0,
) -> int:
    """Wait for job completion and return exit code.

    Polls tmux session status until job exits or timeout.

    Args:
        session: BifrostClient instance
        job: JobInfo identifier
        timeout: Maximum wait time in seconds (None = wait forever)
        poll_interval: How often to poll status

    Returns:
        Exit code (0 = success)

    Raises:
        TimeoutError: If timeout exceeded
    """
    start_time = time.time()

    while True:
        status = job_status(session, job)
        if status == "completed":
            # Job finished, get exit code
            exit_code = job_exit_code(session, job)
            return exit_code if exit_code is not None else 0

        # Check timeout
        if timeout is not None:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Job {job.name} did not complete within {timeout}s")

        time.sleep(poll_interval)


def job_logs(session: BifrostClient, job: JobInfo, tail: int = 100) -> str:
    """Get recent job logs.

    Args:
        session: BifrostClient instance
        job: JobInfo identifier
        tail: Number of lines to return (default: 100)

    Returns:
        Log content as string, or empty string if no log file
    """
    if not job.log_file:
        return ""

    result = session.exec(f"tail -n {tail} {job.log_file} 2>/dev/null || true")
    return result.stdout


def job_stream_logs(
    session: BifrostClient,
    job: JobInfo,
    timeout: float | None = None,
    poll_interval: float = 2.0,
) -> Iterator[str]:
    """Stream job logs in real-time.

    Yields log lines as they appear. Stops when job completes or timeout.
    Uses position-tracked tailing for efficiency.

    Args:
        session: BifrostClient instance
        job: JobInfo identifier
        timeout: Maximum stream time in seconds (None = until job completes)
        poll_interval: How often to poll for new content

    Yields:
        Log lines as strings

    Example:
        for line in job_stream_logs(client, job):
            print(line)
            if "ERROR" in line:
                job_kill(client, job)
                break
    """
    if not job.log_file:
        return

    last_position = 0
    start_time = time.time()

    while True:
        # Check if job still running
        status = job_status(session, job)

        # Tail log from last position
        content, new_pos = _tail_log_from_position(session, job.log_file, last_position)

        if content:
            last_position = new_pos
            for line in content.splitlines():
                yield line

        # Job finished - yield any remaining content and stop
        if status == "completed":
            # Final tail to catch anything we missed
            content, _ = _tail_log_from_position(session, job.log_file, last_position)
            if content:
                for line in content.splitlines():
                    yield line
            return

        # Check timeout
        if timeout is not None:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                return

        time.sleep(poll_interval)


def job_stream_until_complete(
    session: BifrostClient,
    job: JobInfo,
    on_line: Callable[[str], None] | None = None,
    timeout: float = 3600,
    poll_interval: float = 2.0,
) -> tuple[bool, int | None, str | None]:
    """Stream job logs until completion.

    This is a convenience wrapper around job_stream_logs that matches
    the kerbal.job_monitor.stream_log_until_complete interface.

    Args:
        session: BifrostClient instance
        job: JobInfo identifier
        on_line: Callback for each line (default: print to stdout)
        timeout: Maximum wait time in seconds
        poll_interval: How often to poll

    Returns:
        (success, exit_code, error_message)
        - success=True if job completed successfully (exit code 0)
        - exit_code=int if job exited
        - error_message=str if failed or timeout
    """
    start_time = time.time()

    try:
        for line in job_stream_logs(session, job, timeout=timeout, poll_interval=poll_interval):
            if on_line is not None:
                on_line(line)
            else:
                print(line)
    except TimeoutError:
        job_kill(session, job)
        return False, None, f"Timeout after {timeout}s"

    # Check if we timed out
    elapsed = time.time() - start_time
    if elapsed >= timeout:
        job_kill(session, job)
        return False, None, f"Timeout after {timeout}s"

    # Get exit code
    exit_code = job_exit_code(session, job)

    if exit_code is None:
        return False, None, "Job completed but no exit code found"
    elif exit_code == 0:
        return True, exit_code, None
    else:
        return False, exit_code, f"Exit code {exit_code}"


def job_kill(session: BifrostClient, job: JobInfo) -> None:
    """Kill a running job.

    Terminates the tmux session for this job.

    Args:
        session: BifrostClient instance
        job: JobInfo identifier
    """
    session.exec(f"tmux kill-session -t {job.tmux_session} 2>/dev/null || true")
    logger.info(f"Killed job: {job.name}")


def _tail_log_from_position(
    session: BifrostClient,
    log_file: str,
    last_position: int,
) -> tuple[str, int]:
    """Tail log file from byte position.

    Position tracking enables efficient streaming - only fetch new content.

    Returns:
        (content, new_position)
    """
    # tail -c +N reads from byte N (1-indexed)
    cmd = f"tail -c +{last_position + 1} {log_file} 2>/dev/null || true"
    result = session.exec(cmd)

    if result.stdout:
        new_content = result.stdout
        new_position = last_position + len(new_content.encode("utf-8"))
        return new_content, new_position
    else:
        return "", last_position
