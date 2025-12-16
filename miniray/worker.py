"""Lightweight worker pattern for multi-GPU training (Heinrich-inspired).

Simple process management using os.fork() and socketpair for IPC.
No multiprocessing module, no Ray - just clean primitives.

Tiger Style: Explicit state, clear error messages.
Casey Muratori: Minimal coupling, simple primitives.
Heinrich Kuttler: os.fork() + socketpair > multiprocessing.

References:
- /Users/chiraagbalu/research/docs/code_style/multiprocessing_heinrich.md
"""

import json
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


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
    """

    pid: int
    _sock: socket.socket
    r: Any  # file-like object
    w: Any  # file-like object

    def __init__(self, work_fn: Callable[[Any], None]):
        """Spawn worker process.

        Args:
            work_fn: Function to execute in child process.
                     Takes a handle (self) as argument for communication.

        Side effects:
            - Forks process
            - Creates socketpair for IPC
            - Child process executes work_fn and exits
        """
        # Create bidirectional socket pair
        sock, sock0 = socket.socketpair()

        # Fork process
        pid = os.fork()

        if not pid:
            # === Child process ===
            sock0, sock = sock, sock0  # Swap sockets
            sock0.close()

            # Create file handles for JSON communication
            r = sock.makefile("r")
            w = sock.makefile("w")

            # Create handle for child to use
            child_handle = Worker.__new__(Worker)
            child_handle._sock = sock
            child_handle.r = r
            child_handle.w = w
            child_handle.pid = os.getpid()

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

        return json.loads(line)

    def send(self, msg: Any) -> None:
        """Send message to worker (non-blocking).

        Args:
            msg: Message to send (must be JSON-serializable)

        Side effects:
            - Writes JSON to socket
            - Flushes immediately

        Example:
            >>> worker.send({"cmd": "train", "batch": batch_data})
        """
        # Tiger Style: Assert we have a valid handle
        assert self.w is not None, "Worker not initialized properly"

        json.dump(msg, self.w)
        self.w.write("\n")
        self.w.flush()

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
        rank, world_size = handle.recv()
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
        result = worker.recv()
        print(f"Worker {i} returned: {result}")

    # Wait for completion
    for worker in workers:
        worker.wait()

    print("All workers completed successfully!")
