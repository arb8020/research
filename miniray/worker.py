"""Lightweight worker pattern for multi-GPU training (Heinrich-inspired).

Simple process management using os.fork() and socketpair for IPC.
No multiprocessing module, no Ray - just clean primitives.

Tiger Style: Explicit state, clear error messages.
Casey Muratori: Minimal coupling, simple primitives.
Heinrich Kuttler: os.fork() + socketpair > multiprocessing.

References:
- /Users/chiraagbalu/research/docs/code_style/multiprocessing_heinrich.md

TODO (Heinrich parity):
- [x] memfd_create + mmap for shared memory (large tensor passing without copies)
- [x] PDEATHSIG - child dies when parent dies
- [x] select.select() for multiplexing - wait_any() function and fileno() method
- [x] marshal as alternative to JSON (faster for simple Python types)
"""

from __future__ import annotations

import base64
import ctypes
import json
import marshal
import mmap
import os
import signal
import socket
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

# ============================================================================
# PDEATHSIG - Child dies when parent dies (Linux only)
# ============================================================================


def _set_pdeathsig() -> None:
    """Set PDEATHSIG so child dies when parent dies (Linux only).

    Called in child process after fork. On non-Linux, this is a no-op.
    """
    if sys.platform != "linux":
        return

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        result = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        if result != 0:
            errno = ctypes.get_errno()
            print(f"Warning: prctl(PR_SET_PDEATHSIG) failed with errno {errno}", flush=True)
    except OSError as e:
        print(f"Warning: Could not set PDEATHSIG: {e}", flush=True)


# ============================================================================
# Serialization - JSON (default) or marshal (faster)
# ============================================================================

SerializationFormat = Literal["json", "marshal"]


def _serialize(msg: Any, fmt: SerializationFormat) -> str:
    """Serialize message to text (newline-safe)."""
    if fmt == "json":
        return json.dumps(msg)
    else:
        # marshal -> bytes -> base64 -> text (keeps newline protocol!)
        return base64.b64encode(marshal.dumps(msg)).decode("ascii")


def _deserialize(text: str, fmt: SerializationFormat) -> Any:
    """Deserialize text to message."""
    if fmt == "json":
        return json.loads(text)
    else:
        return marshal.loads(base64.b64decode(text))


# ============================================================================
# SharedMemory - memfd (Linux) or tempfile (macOS)
# ============================================================================


@dataclass
class SharedMemory:
    """Shared memory region using memfd (Linux) or tempfile fallback (macOS).

    Use for passing large data (tensors) between processes without copying.
    The fd can be passed to child processes via fork.

    Example:
        >>> shm = SharedMemory.create("weights", 1024 * 1024)  # 1MB
        >>> buf = shm.map()
        >>> buf[:4] = b"test"
        >>> # Child can mmap same fd after fork
        >>> shm.close()
    """

    fd: int
    size: int
    _mmap: mmap.mmap | None = field(default=None, repr=False)

    @classmethod
    def create(cls, name: str, size: int) -> SharedMemory:
        """Create shared memory region.

        Args:
            name: Name for the region (used in memfd_create)
            size: Size in bytes

        Returns:
            SharedMemory instance with allocated region
        """
        assert size > 0, f"size must be > 0, got {size}"
        assert size <= 1024 * 1024 * 1024, f"size too large: {size} (max 1GB)"

        if sys.platform == "linux":
            fd = os.memfd_create(name)
        else:
            # macOS fallback - use anonymous tempfile
            f = tempfile.NamedTemporaryFile(delete=False, prefix=f"miniray_{name}_")
            fd = os.dup(f.fileno())  # Dup fd so it survives file close
            fname = f.name
            f.close()
            os.unlink(fname)  # Unlink file, but fd keeps data alive

        os.ftruncate(fd, size)
        return cls(fd=fd, size=size)

    def map(self) -> mmap.mmap:
        """Memory-map the region.

        Returns:
            mmap object that can be used like a bytearray
        """
        if self._mmap is None:
            self._mmap = mmap.mmap(self.fd, self.size)
        return self._mmap

    def close(self) -> None:
        """Close the shared memory region."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        os.close(self.fd)


@dataclass
class Worker:
    """Single worker process (Heinrich pattern).

    Uses os.fork() + socketpair for clean process isolation and communication.
    Simpler than multiprocessing, no pickle overhead, full control.

    Example:
        >>> def work(handle):
        ...     rank, world_size = handle.recv()
        ...     print(f"Worker {rank}/{world_size}")
        ...     result = do_training(rank)
        ...     handle.send(result)
        >>>
        >>> worker = Worker(work)
        >>> worker.send({"rank": 0, "world_size": 4})
        >>> result = worker.recv()
        >>> worker.wait()

    Attributes:
        pid: Child process ID
        r: Read file handle (for receiving messages)
        w: Write file handle (for sending messages)
        _format: Serialization format ("json" or "marshal")
    """

    pid: int
    _sock: socket.socket
    r: Any  # file-like object
    w: Any  # file-like object
    _format: SerializationFormat = "json"

    def __init__(
        self,
        work_fn: Callable[[Any], None],
        format: SerializationFormat = "json",
    ):
        """Spawn worker process.

        Args:
            work_fn: Function to execute in child process.
                     Takes a handle (self) as argument for communication.
            format: Serialization format - "json" (default, readable) or
                    "marshal" (faster, ~5-10x for large messages)

        Side effects:
            - Forks process
            - Creates socketpair for IPC
            - Child process executes work_fn and exits
        """
        self._format = format

        # Create bidirectional socket pair
        sock, sock0 = socket.socketpair()

        # Fork process
        pid = os.fork()

        if not pid:
            # === Child process ===
            _set_pdeathsig()  # Die when parent dies (Linux only)

            sock0, sock = sock, sock0  # Swap sockets
            sock0.close()

            # Create file handles for communication
            r = sock.makefile("r")
            w = sock.makefile("w")

            # Create handle for child to use
            child_handle = Worker.__new__(Worker)
            child_handle._sock = sock
            child_handle.r = r
            child_handle.w = w
            child_handle.pid = os.getpid()
            child_handle._format = format

            # Execute work function
            try:
                work_fn(child_handle)
            except Exception as e:
                # Log error and exit with failure code
                import traceback

                print(f"Worker {os.getpid()} failed: {e}", flush=True)
                traceback.print_exc()
                os._exit(1)

            # Clean exit
            os._exit(0)

        # === Parent process ===
        sock0.close()

        # Store communication handles
        self.pid = pid
        self._sock = sock
        self.r = sock.makefile("r")
        self.w = sock.makefile("w")

    def recv(self, max_size: int, timeout: float | None = None) -> Any:
        """Receive message from worker (blocking).

        Args:
            max_size: Maximum message size in bytes (required for safety)
            timeout: Optional timeout in seconds (TODO: implement with select)

        Returns:
            Deserialized message (Python object)

        Raises:
            EOFError: If worker closed connection
            json.JSONDecodeError: If message is malformed
            AssertionError: If message exceeds max_size

        Example:
            >>> result = worker.recv(max_size=1024 * 1024)  # 1MB
            >>> print(result["loss"])
        """
        # Tiger Style: Assert preconditions
        assert self.r is not None, "Worker not initialized properly"
        assert max_size > 0, f"max_size must be > 0, got {max_size}"
        assert max_size <= 100 * 1024 * 1024, f"max_size too large: {max_size} (max 100MB)"

        # TODO: Add timeout support with select.select()
        line = self.r.readline(max_size)

        # Tiger Style: Assert postconditions
        assert len(line) <= max_size, f"Message too large: {len(line)} > {max_size}"

        if not line:
            raise EOFError(f"Worker {self.pid} closed connection")

        return _deserialize(line.rstrip("\n"), self._format)

    def send(self, msg: Any) -> None:
        """Send message to worker (non-blocking).

        Args:
            msg: Message to send (must be serializable)

        Side effects:
            - Writes serialized message to socket
            - Flushes immediately

        Example:
            >>> worker.send({"cmd": "train", "batch": batch_data})
        """
        # Tiger Style: Assert we have a valid handle
        assert self.w is not None, "Worker not initialized properly"

        text = _serialize(msg, self._format)
        self.w.write(text)
        self.w.write("\n")
        self.w.flush()

    def fileno(self) -> int:
        """Return socket file descriptor for use with select().

        This allows Worker to be used directly with select.select():
            ready, _, _ = select.select(workers, [], [], timeout)

        Returns:
            Socket file descriptor
        """
        return self._sock.fileno()

    def is_alive(self) -> bool:
        """Check if worker process is still alive.

        Returns:
            True if worker is alive, False if dead/zombie

        Example:
            >>> if not worker.is_alive():
            ...     print("Worker died, restarting...")
            ...     worker = Worker(work_fn)
        """
        try:
            # Check if process exists (non-blocking)
            # WNOHANG returns immediately
            pid, status = os.waitpid(self.pid, os.WNOHANG)

            if pid == 0:
                # Process still running
                return True
            else:
                # Process exited
                return False
        except ChildProcessError:
            # Process doesn't exist
            return False

    def wait(self) -> None:
        """Wait for worker to exit (blocking).

        Raises:
            AssertionError: If worker exits with non-zero status

        Side effects:
            - Blocks until child exits
            - Cleans up zombie process
        """
        _, status = os.waitpid(self.pid, 0)
        rc = os.waitstatus_to_exitcode(status)

        # Tiger Style: Assert clean exit
        assert rc == 0, f"Worker {self.pid} exited with code {rc}"


def wait_any(
    workers: list[Worker],
    timeout: float | None = None,
) -> list[Worker]:
    """Wait for any workers to have data ready for recv().

    Uses select() to efficiently wait on multiple workers.
    This enables async-style coordination without threads:

        # Fire off work to multiple workers
        for worker in workers:
            worker.send({"cmd": "train", "batch": batch})

        # Wait for any to complete
        pending = list(workers)
        while pending:
            ready = wait_any(pending, timeout=1.0)
            for worker in ready:
                result = worker.recv()
                handle(result)
                pending.remove(worker)

    Args:
        workers: List of workers to wait on
        timeout: Max seconds to wait
            - None = block forever until at least one ready
            - 0 = poll (return immediately)
            - >0 = wait up to timeout seconds

    Returns:
        List of workers that have data ready (may be empty if timeout)

    Example:
        >>> ready = wait_any(workers, timeout=1.0)
        >>> for worker in ready:
        ...     msg = worker.recv()  # Won't block - data is ready
    """
    import select

    # Tiger Style: Assert preconditions
    assert len(workers) > 0, "workers list cannot be empty"
    assert timeout is None or timeout >= 0, f"timeout must be >= 0, got {timeout}"

    readable, _, _ = select.select(workers, [], [], timeout)
    return readable


def spawn_training_workers(
    num_workers: int,
    work_fn: Callable[[Any, int, int], None],
    with_gpus: bool = True,
) -> list[Worker]:
    """Spawn multiple training workers (convenience function).

    Args:
        num_workers: Number of workers to spawn
        work_fn: Work function taking (handle, rank, world_size)
        with_gpus: Whether to set CUDA_VISIBLE_DEVICES per worker

    Returns:
        List of Worker handles

    Example:
        >>> def train_worker(handle, rank, world_size):
        ...     # Receive config
        ...     config = handle.recv()
        ...
        ...     # Initialize torch.distributed
        ...     import torch.distributed as dist
        ...     dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        ...
        ...     # Training loop
        ...     while True:
        ...         msg = handle.recv()
        ...         if msg["cmd"] == "shutdown":
        ...             break
        ...         # ... do training ...
        ...         handle.send({"loss": 0.5})
        >>>
        >>> workers = spawn_training_workers(4, train_worker)
        >>> for worker in workers:
        ...     worker.send({"config": {...}})
    """
    # Tiger Style: Assert preconditions
    assert num_workers > 0, "num_workers must be > 0"
    assert num_workers < 1000, f"num_workers too large: {num_workers} (max 1000)"

    workers = []
    for rank in range(num_workers):
        # Create worker with rank and world_size bound
        # Tiger Style: Fix closure bug - bind rank in default argument
        def wrapped_work(handle, r=rank):
            # Set GPU affinity if requested
            if with_gpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(r)

            # Call user work function with rank/world_size
            work_fn(handle, r, num_workers)

        worker = Worker(wrapped_work)
        workers.append(worker)

    return workers


def shutdown_workers(workers: list[Worker]) -> None:
    """Clean shutdown of all workers.

    Args:
        workers: List of Worker handles

    Side effects:
        - Sends shutdown command to each worker
        - Waits for all workers to exit
        - Cleans up resources

    Example:
        >>> shutdown_workers(workers)
    """
    # Send shutdown to all workers
    for worker in workers:
        try:
            worker.send({"cmd": "shutdown"})
        except Exception as e:
            print(f"Warning: Failed to send shutdown to worker {worker.pid}: {e}")

    # Wait for all to exit
    for worker in workers:
        try:
            worker.wait()
        except AssertionError as e:
            print(f"Warning: Worker {worker.pid} had non-clean exit: {e}")


# ============================================================================
# Example usage (can be run as script for testing)
# ============================================================================

if __name__ == "__main__":
    """Test the Worker pattern"""

    def example_work(handle):
        """Example work function"""
        rank, world_size = handle.recv(max_size=1024)
        print(f"Hello from worker {rank}/{world_size}!", flush=True)

        # Simulate some work
        result = 2 * rank
        handle.send(result)

    # Spawn workers
    num_workers = 4
    workers = [Worker(example_work) for _ in range(num_workers)]

    # Send initial config
    for i, worker in enumerate(workers):
        worker.send((i, len(workers)))

    # Receive results
    for i, worker in enumerate(workers):
        result = worker.recv(max_size=1024)
        print(f"Worker {i} returned: {result}")

    # Wait for completion
    for worker in workers:
        worker.wait()

    print("All workers completed successfully!")
