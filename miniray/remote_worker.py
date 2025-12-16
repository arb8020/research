"""RemoteWorker: TCP-based distributed worker (Heinrich pattern over network).

Replaces UDS (Unix Domain Socket) with TCP for multi-node communication.
Same interface as Worker, but connects to remote process.

Tiger Style: Explicit network operations, clear error handling.
Heinrich Kuttler: Simple primitives, no pickle overhead.
"""

import json
import socket
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RemoteWorker:
    """TCP-based remote worker.

    Connects to a WorkerServer on another machine and communicates
    via JSON messages over TCP. Same API as local Worker pattern.

    Example:
        >>> # Worker server running on node2:10000
        >>> worker = RemoteWorker("node2.local", 10000)
        >>> worker.connect()  # Explicit connection
        >>> worker.send({"cmd": "train", "batch": batch_data})
        >>> result = worker.recv()
        >>> worker.close()

    Attributes:
        host: Hostname or IP of remote worker server
        port: Port number of remote worker server
    """

    host: str
    port: int
    _sock: socket.socket | None = field(default=None, init=False, repr=False)
    r: Any = field(default=None, init=False, repr=False)
    w: Any = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate configuration.

        Casey Muratori: Split construction from connection for better control.
        """
        # Tiger Style: Assert preconditions
        assert self.host, "host cannot be empty"
        assert self.port > 0, f"port must be > 0, got {self.port}"
        assert self.port <= 65535, f"port must be <= 65535, got {self.port}"

    def connect(self, timeout: float = 10.0) -> None:
        """Connect to remote worker server.

        Args:
            timeout: Connection timeout in seconds

        Side effects:
            - Creates TCP socket
            - Connects to remote host:port
            - Creates file handles for JSON communication

        Raises:
            ConnectionRefusedError: If server not running
            socket.timeout: If connection times out
            AssertionError: If already connected
        """
        # Tiger Style: Assert preconditions
        assert not self._connected, "Already connected"
        assert timeout > 0, f"timeout must be > 0, got {timeout}"

        # Create TCP socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        assert self._sock is not None, "Failed to create socket"

        # Set timeout for connection
        self._sock.settimeout(timeout)

        try:
            # Connect to remote worker server
            self._sock.connect((self.host, self.port))

            # Verify connection
            peername = self._sock.getpeername()
            assert peername is not None, "Connection failed - no peer name"

            # Remove timeout after connection
            self._sock.settimeout(None)

            # Create file handles for JSON I/O (same as Heinrich)
            self.r = self._sock.makefile("r")
            self.w = self._sock.makefile("w")

            # Tiger Style: Assert postconditions
            assert self.r is not None, "Failed to create read handle"
            assert self.w is not None, "Failed to create write handle"

            self._connected = True

        except ConnectionRefusedError:
            raise ConnectionRefusedError(
                f"Cannot connect to worker server at {self.host}:{self.port}. "
                "Is WorkerServer running on the remote node?"
            )
        except TimeoutError:
            raise TimeoutError(
                f"Timeout connecting to {self.host}:{self.port}. Check network connectivity."
            )

    def recv(self, max_size: int, timeout: float | None = None) -> Any:
        """Receive message from remote worker (blocking).

        Args:
            max_size: Maximum message size in bytes (required for safety)
            timeout: Optional timeout in seconds

        Returns:
            Deserialized message (Python object)

        Raises:
            EOFError: If remote worker closed connection
            json.JSONDecodeError: If message is malformed
            socket.timeout: If timeout expires
            AssertionError: If message exceeds max_size

        Example:
            >>> result = worker.recv(max_size=1024 * 1024)  # 1MB
            >>> print(result["loss"])
            0.42
        """
        # Tiger Style: Assert preconditions
        assert self._connected, "Not connected - call connect() first"
        assert self.r is not None, "RemoteWorker not initialized"
        assert self._sock is not None, "Socket is None"
        assert max_size > 0, f"max_size must be > 0, got {max_size}"
        assert max_size <= 100 * 1024 * 1024, f"max_size too large: {max_size} (max 100MB)"

        # Set timeout if requested
        if timeout is not None:
            assert timeout > 0, f"timeout must be > 0, got {timeout}"
            self._sock.settimeout(timeout)

        try:
            line = self.r.readline(max_size)

            # Tiger Style: Assert postcondition
            assert len(line) <= max_size, f"Message too large: {len(line)} > {max_size}"

            if not line:
                raise EOFError(f"Remote worker {self.host}:{self.port} closed connection")

            return json.loads(line)

        except TimeoutError:
            raise TimeoutError(f"Timeout waiting for response from {self.host}:{self.port}")
        finally:
            # Restore no timeout
            if timeout is not None:
                self._sock.settimeout(None)

    def send(self, msg: Any) -> None:
        """Send message to remote worker (non-blocking).

        Args:
            msg: Message to send (must be JSON-serializable)

        Side effects:
            - Writes JSON to TCP socket
            - Flushes immediately

        Raises:
            BrokenPipeError: If connection is broken

        Example:
            >>> worker.send({"cmd": "train", "batch": batch_data})
        """
        # Tiger Style: Assert preconditions
        assert self._connected, "Not connected - call connect() first"
        assert self.w is not None, "RemoteWorker not initialized"

        try:
            json.dump(msg, self.w)
            self.w.write("\n")
            self.w.flush()

        except BrokenPipeError:
            raise BrokenPipeError(
                f"Connection to {self.host}:{self.port} is broken. Remote worker may have crashed."
            )

    def fileno(self) -> int:
        """Return socket file descriptor for use with select().

        This allows RemoteWorker to be used with wait_any():
            ready = wait_any(workers, timeout=1.0)

        Returns:
            Socket file descriptor

        Raises:
            AssertionError: If not connected
        """
        assert self._connected, "Not connected - call connect() first"
        assert self._sock is not None, "Socket is None"
        return self._sock.fileno()

    def is_alive(self) -> bool:
        """Check if remote worker connection is alive.

        Returns:
            True if connected and socket is valid, False otherwise

        Example:
            >>> if not worker.is_alive():
            ...     print("Worker connection lost")
            ...     worker.connect()  # Reconnect
        """
        if not self._connected:
            return False

        try:
            # Check if socket is still connected
            # getpeername() raises if disconnected
            self._sock.getpeername()
            return True
        except (OSError, AttributeError):
            # Socket disconnected or doesn't exist
            return False

    def health_check(self, timeout: float = 5.0) -> bool:
        """Send ping and check if worker responds.

        Args:
            timeout: Timeout in seconds for health check

        Returns:
            True if worker responds with "alive", False otherwise

        Example:
            >>> if not worker.health_check(timeout=10.0):
            ...     print("Worker unresponsive, restarting...")
            ...     cluster.restart_worker(worker)

        Note:
            Worker must handle {"cmd": "ping"} and respond with {"status": "alive"}
        """
        if not self.is_alive():
            return False

        try:
            # Send ping
            self.send({"cmd": "ping"})

            # Wait for pong (with timeout)
            response = self.recv(max_size=1024, timeout=timeout)

            # Check response
            return response.get("status") == "alive"
        except Exception:
            # Any error means unhealthy
            return False

    def close(self) -> None:
        """Close connection to remote worker.

        Side effects:
            - Closes TCP socket
            - Closes file handles

        Example:
            >>> worker.close()
        """
        if self.r:
            self.r.close()
        if self.w:
            self.w.close()
        if self._sock:
            self._sock.close()
        self._connected = False

    def __enter__(self):
        """Context manager support.

        Example:
            >>> with RemoteWorker("node2", 10000) as worker:
            ...     worker.send({"cmd": "train"})
            ...     result = worker.recv()
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()


def create_remote_workers(
    hosts: list[tuple[str, int]],
    timeout: float = 10.0,
) -> list[RemoteWorker]:
    """Create multiple remote workers (convenience function).

    Args:
        hosts: List of (host, port) tuples
        timeout: Connection timeout per worker

    Returns:
        List of connected RemoteWorker instances

    Example:
        >>> workers = create_remote_workers([
        ...     ("node1", 10000),
        ...     ("node1", 10001),
        ...     ("node2", 10000),
        ...     ("node2", 10001),
        ... ])
    """
    workers = []
    for host, port in hosts:
        worker = RemoteWorker(host, port)
        workers.append(worker)

    return workers


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    """Test RemoteWorker (requires worker_server.py running)"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python remote_worker.py <host> <port>")
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    print(f"Connecting to worker at {host}:{port}...")
    worker = RemoteWorker(host, port)

    print("Sending test message...")
    worker.send({"cmd": "echo", "data": "Hello from RemoteWorker!"})

    print("Waiting for response...")
    response = worker.recv()
    print(f"Received: {response}")

    worker.close()
    print("Done!")
