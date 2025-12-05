"""WorkerServer: TCP server that spawns Heinrich-style workers.

Listens on a port, accepts connections, and spawns a local worker process
for each connection using os.fork(). Each worker communicates via the
accepted TCP socket using Heinrich's pattern.

Tiger Style: Explicit server lifecycle, clear error messages.
Heinrich Kuttler: fork + TCP = distributed workers.
"""

import os
import signal
import socket
import sys
from collections.abc import Callable


class WorkerServer:
    """TCP server that spawns workers on connection.

    Each incoming connection spawns a new worker process via os.fork().
    The worker inherits the TCP socket and uses it for JSON communication.

    Example:
        >>> def work_fn(handle):
        ...     while True:
        ...         msg = handle.recv()
        ...         if msg["cmd"] == "shutdown":
        ...             break
        ...         handle.send({"result": "ok"})
        >>>
        >>> server = WorkerServer(
        ...     host="0.0.0.0",
        ...     port=10000,
        ...     work_fn=work_fn,
        ...     num_workers=4,
        ... )
        >>> server.serve_forever()

    Attributes:
        host: Host to bind (0.0.0.0 for all interfaces)
        port: Port to bind
        work_fn: Function to execute in each worker
        num_workers: Maximum number of workers to spawn
    """

    def __init__(
        self,
        host: str,
        port: int,
        work_fn: Callable,
        num_workers: int = 4,
    ):
        """Initialize WorkerServer.

        Args:
            host: Host to bind (0.0.0.0 = all interfaces)
            port: Port to bind
            work_fn: Function taking (handle) as argument
            num_workers: Max workers to spawn
        """
        # Tiger Style: Assert preconditions
        assert port > 0, f"port must be > 0, got {port}"
        assert num_workers > 0, f"num_workers must be > 0, got {num_workers}"
        assert callable(work_fn), "work_fn must be callable"

        self.host = host
        self.port = port
        self.work_fn = work_fn
        self.num_workers = num_workers
        self.workers = []  # List of (pid, client_addr)
        self.listen_sock = None

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown.

        Catches SIGTERM and SIGINT to clean up workers.
        """

        def signal_handler(signum, frame):
            print(f"\n[WorkerServer] Received signal {signum}, shutting down...")
            self._shutdown()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _shutdown(self):
        """Shut down all workers and close server socket."""
        print(f"[WorkerServer] Shutting down {len(self.workers)} workers...")

        # Send SIGTERM to all worker processes
        for pid, addr in self.workers:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # Already dead

        # Wait for workers to exit
        for pid, addr in self.workers:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass  # Already reaped

        # Close listening socket
        if self.listen_sock:
            self.listen_sock.close()

        print("[WorkerServer] Shutdown complete")

    def serve_forever(self):
        """Start server and spawn workers.

        Listens on host:port and spawns a worker for each connection.
        Blocks until num_workers connections are accepted.

        Side effects:
            - Binds to host:port
            - Spawns worker processes
            - Blocks until all workers connected

        Raises:
            OSError: If port already in use
        """
        # Set up signal handlers
        self._setup_signal_handlers()

        # Create listening socket
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.listen_sock.bind((self.host, self.port))
        except OSError as e:
            if e.errno == 98:  # Address already in use
                raise OSError(
                    f"Port {self.port} already in use. "
                    "Another WorkerServer may be running."
                ) from e
            raise

        self.listen_sock.listen(self.num_workers)

        print(f"[WorkerServer] Listening on {self.host}:{self.port}")
        print(f"[WorkerServer] Ready to spawn {self.num_workers} workers")

        # Accept connections and spawn workers
        while len(self.workers) < self.num_workers:
            try:
                # Accept connection
                client_sock, client_addr = self.listen_sock.accept()
                print(f"[WorkerServer] Connection from {client_addr}")

                # Spawn worker process (Heinrich pattern)
                pid = os.fork()

                if not pid:
                    # === Child process (worker) ===
                    self._run_worker(client_sock, client_addr)
                    # Should never return
                    os._exit(1)

                # === Parent process (server) ===
                # Close client socket in parent (child owns it)
                client_sock.close()

                # Track worker
                self.workers.append((pid, client_addr))
                print(
                    f"[WorkerServer] Spawned worker {pid} "
                    f"(total: {len(self.workers)}/{self.num_workers})"
                )

            except KeyboardInterrupt:
                print("\n[WorkerServer] Interrupted")
                self._shutdown()
                break

        print(f"[WorkerServer] All {self.num_workers} workers spawned")

        # Wait for all workers to exit
        self._wait_for_workers()

    def _run_worker(self, client_sock: socket.socket, client_addr):
        """Run worker process (child process after fork).

        Args:
            client_sock: TCP socket to communicate with client
            client_addr: Address of client

        Side effects:
            - Closes listening socket (not needed in worker)
            - Calls work_fn with handle
            - Exits with code 0 on success, 1 on failure
        """
        # Close listening socket in child (don't need it)
        if self.listen_sock:
            self.listen_sock.close()

        # Set up file handles for JSON communication
        r = client_sock.makefile('r')
        w = client_sock.makefile('w')

        # Create handle for work function (same API as Worker)
        from rollouts.training.worker import Worker

        handle = Worker.__new__(Worker)
        handle._sock = client_sock
        handle.r = r
        handle.w = w
        handle.pid = os.getpid()

        print(f"[Worker {os.getpid()}] Started, serving {client_addr}")

        try:
            # Run work function
            self.work_fn(handle)
            print(f"[Worker {os.getpid()}] Completed successfully")
            os._exit(0)

        except Exception as e:
            import traceback

            print(f"[Worker {os.getpid()}] Failed: {e}", file=sys.stderr)
            traceback.print_exc()
            os._exit(1)

    def _wait_for_workers(self):
        """Wait for all worker processes to exit."""
        print("[WorkerServer] Waiting for workers to complete...")

        for pid, addr in self.workers:
            try:
                _, status = os.waitpid(pid, 0)
                exit_code = os.waitstatus_to_exitcode(status)

                if exit_code != 0:
                    print(
                        f"[WorkerServer] Worker {pid} exited with code {exit_code}",
                        file=sys.stderr,
                    )

            except ChildProcessError:
                # Already reaped (e.g., by signal handler)
                pass

        print("[WorkerServer] All workers exited")


# ============================================================================
# CLI for launching worker servers on compute nodes
# ============================================================================

def main():
    """CLI entry point for launching worker server.

    Usage:
        python -m rollouts.training.miniray.worker_server \\
            --port 10000 \\
            --workers 4 \\
            --work-fn my_module.train_worker_fn
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="MiniRay WorkerServer - spawn workers on TCP connections"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0 = all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port to bind",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers to spawn (default: 4)",
    )
    parser.add_argument(
        "--work-fn",
        required=True,
        help="Python path to work function (e.g., my_module.train_fn)",
    )
    args = parser.parse_args()

    # Import work function
    module_path, fn_name = args.work_fn.rsplit(".", 1)

    try:
        import importlib

        module = importlib.import_module(module_path)
        work_fn = getattr(module, fn_name)
    except (ImportError, AttributeError) as e:
        print(f"Error: Cannot import {args.work_fn}: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate work function
    if not callable(work_fn):
        print(f"Error: {args.work_fn} is not callable", file=sys.stderr)
        sys.exit(1)

    # Create and start server
    print(f"[WorkerServer] Starting with work_fn: {args.work_fn}")
    server = WorkerServer(
        host=args.host,
        port=args.port,
        work_fn=work_fn,
        num_workers=args.workers,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
