"""Async subprocess utilities with proper cancellation support.

Provides helpers for running subprocesses that can be interrupted via
Escape (soft cancel) or Ctrl+C (hard cancel) in the TUI.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import TYPE_CHECKING, Any

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
        async for chunk in stream:
            chunks.append(chunk)

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


async def run_command(
    command: str,
    cwd: str,
    timeout: float = 120,  # noqa: ASYNC109
) -> tuple[int, str, str]:
    """Run a shell command with cancellation support.

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
    process = await trio.lowlevel.open_process(
        ["sh", "-c", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        start_new_session=True,
    )

    try:
        with trio.move_on_after(timeout) as timeout_scope:
            stdout_data, stderr_data = await read_process_output(process)
            returncode = await process.wait()

        if timeout_scope.cancelled_caught:
            await kill_process_tree(process)
            raise TimeoutError(f"Command timed out after {timeout} seconds")

        stdout = stdout_data.decode("utf-8", errors="replace") if stdout_data else ""
        stderr = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""

        return returncode, stdout, stderr

    except trio.Cancelled:
        await kill_process_tree(process)
        raise
