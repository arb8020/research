"""Async subprocess utilities with proper cancellation support.

Provides helpers for running subprocesses that can be interrupted via
Escape (soft cancel) or Ctrl+C (hard cancel) in the TUI.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import TYPE_CHECKING

import trio

if TYPE_CHECKING:
    from trio import Process
    from trio.abc import ReceiveStream


async def read_process_output(process: Process) -> tuple[bytes, bytes]:
    """Read stdout and stderr concurrently from a trio process."""
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    async def read_stream(stream: ReceiveStream | None, chunks: list[bytes]) -> None:
        if stream is None:
            return
        try:
            async for chunk in stream:
                chunks.append(chunk)
        except trio.ClosedResourceError:
            pass

    async with trio.open_nursery() as nursery:
        nursery.start_soon(read_stream, process.stdout, stdout_chunks)
        nursery.start_soon(read_stream, process.stderr, stderr_chunks)

    return b"".join(stdout_chunks), b"".join(stderr_chunks)


async def kill_process_tree(process: Process, graceful_timeout: float = 5.0) -> None:
    """Kill process and all children. SIGTERM first, SIGKILL after timeout.

    Assumes process was started with start_new_session=True, so we can
    kill the entire process group using the PID as the PGID.
    """
    pid = process.pid
    if pid is None:
        return

    try:
        # With start_new_session=True, the process IS the session/group leader
        # so its PID equals its PGID - kill the entire group
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        # Process already dead or we can't kill the group, try direct kill
        try:
            process.terminate()
        except ProcessLookupError:
            return

    # Wait briefly for graceful exit
    with trio.move_on_after(graceful_timeout):
        await process.wait()
        return

    # Still alive - force kill
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except ProcessLookupError:
            pass


class _ProcessHolder:
    """Holds a reference to a subprocess so it can be killed from outside the thread."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None


def _run_command_sync(
    command: str, cwd: str, timeout: float, holder: _ProcessHolder
) -> tuple[int, str, str]:
    """Synchronous subprocess execution (runs in thread)."""
    process = subprocess.Popen(
        ["sh", "-c", command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    holder.process = process

    try:
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired as e:
        # Kill on timeout
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
        process.wait()
        raise TimeoutError(f"Command timed out after {timeout} seconds") from e


async def run_command(
    command: str,
    cwd: str,
    timeout: float = 120,  # noqa: ASYNC109
) -> tuple[int, str, str]:
    """Run a shell command with cancellation support.

    Uses a thread with cancellable=True so trio can abandon it on cancellation.
    When cancelled, the thread is abandoned and the process is killed.

    Args:
        command: Shell command to run
        cwd: Working directory
        timeout: Timeout in seconds

    Returns:
        Tuple of (returncode, stdout, stderr)

    Raises:
        trio.Cancelled: If cancelled via Escape/Ctrl+C (process is killed)
        TimeoutError: If command exceeds timeout
    """
    holder = _ProcessHolder()

    try:
        # Run in thread with abandon_on_cancel=True - allows trio to abandon the thread
        # when the cancel scope is cancelled
        return await trio.to_thread.run_sync(
            lambda: _run_command_sync(command, cwd, timeout, holder),
            abandon_on_cancel=True,
        )
    except trio.Cancelled:
        # Kill the process if it exists
        if holder.process and holder.process.pid:
            try:
                os.killpg(holder.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    holder.process.kill()
                except (ProcessLookupError, OSError):
                    pass
        raise
