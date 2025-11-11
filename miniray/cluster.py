"""Cluster: Multi-node worker cluster management.

Handles SSH-ing to compute nodes, launching WorkerServers, and
connecting RemoteWorkers. Provides high-level API for distributed
computing with MiniRay.

Tiger Style: Explicit cluster operations, clear lifecycle.
Heinrich Kuttler: Simple distributed workers, no magic.
"""

import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

from miniray.remote_worker import RemoteWorker


@dataclass
class NodeConfig:
    """Configuration for a compute node.

    Example:
        >>> node = NodeConfig(
        ...     host="node1.cluster.edu",
        ...     num_workers=4,
        ...     base_port=10000,
        ... )
    """

    host: str
    """Hostname or IP address"""

    num_workers: int = 4
    """Number of workers to spawn on this node"""

    base_port: int = 10000
    """Starting port (workers use base_port, base_port+1, ...)"""

    gpus_per_worker: int = 1
    """GPUs per worker (for GPU affinity)"""

    ssh_key: Optional[str] = None
    """SSH key path (None = default)"""

    python_bin: str = "python"
    """Python binary path on remote node"""


@dataclass
class Cluster:
    """Manages a cluster of remote workers across compute nodes.

    Handles:
    - SSH-ing to nodes
    - Launching WorkerServer on each node
    - Connecting RemoteWorkers
    - Cleanup on shutdown

    Example:
        >>> cluster = Cluster(nodes=[
        ...     NodeConfig("node1", num_workers=4),
        ...     NodeConfig("node2", num_workers=4),
        ... ])
        >>>
        >>> # Launch workers
        >>> workers = cluster.start(work_fn="my_module.train_fn")
        >>>
        >>> # Use them
        >>> for worker in workers:
        ...     worker.send({"cmd": "train", "batch": batch})
        >>> results = [w.recv() for w in workers]
        >>>
        >>> # Cleanup
        >>> cluster.stop()

    Attributes:
        nodes: List of node configurations
    """

    nodes: List[NodeConfig]
    processes: List[subprocess.Popen] = field(default_factory=list, init=False)
    workers: List[RemoteWorker] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validate cluster configuration."""
        # Tiger Style: Assert preconditions
        assert len(self.nodes) > 0, "Must have at least one node"
        assert len(self.nodes) < 1000, f"Too many nodes: {len(self.nodes)} (max 1000)"

        total_workers = 0
        for node in self.nodes:
            assert node.num_workers > 0, f"num_workers must be > 0 for {node.host}"
            assert node.num_workers <= 16, f"num_workers too large for {node.host}: {node.num_workers} (max 16)"
            assert node.base_port > 0, f"base_port must be > 0 for {node.host}"
            assert node.base_port <= 65535, f"base_port out of range for {node.host}: {node.base_port}"
            assert node.base_port + node.num_workers <= 65536, \
                f"Port range exceeds 65535 for {node.host}: {node.base_port}+{node.num_workers}"
            total_workers += node.num_workers

        assert total_workers > 0, "Total workers must be > 0"
        assert total_workers < 10000, f"Total workers too large: {total_workers} (max 10000)"

    def launch_servers(
        self,
        work_fn: str,
        verbose: bool = True,
    ) -> int:
        """Launch worker servers on all nodes via SSH.

        Args:
            work_fn: Python path to work function (e.g., "module.function")
            verbose: Print progress messages

        Returns:
            Number of SSH processes launched

        Side effects:
            - SSH-es to each node
            - Launches WorkerServer processes in background

        Example:
            >>> num_launched = cluster.launch_servers(work_fn="train_module.worker_fn")
            >>> print(f"Launched {num_launched} server processes")
        """
        # Tiger Style: Assert preconditions
        assert work_fn, "work_fn cannot be empty"
        assert "." in work_fn, f"work_fn must be module.function format, got: {work_fn}"

        if verbose:
            total_workers = sum(node.num_workers for node in self.nodes)
            print(f"[MiniRay] Launching {total_workers} servers on {len(self.nodes)} nodes")

        self._launch_worker_servers(work_fn, verbose)

        if verbose:
            print(f"[MiniRay] Launched {len(self.processes)} server processes")

        return len(self.processes)

    def connect_to_servers(
        self,
        timeout: float = 10.0,
        verbose: bool = True,
    ) -> List[RemoteWorker]:
        """Connect to launched worker servers.

        Args:
            timeout: Connection timeout per worker
            verbose: Print progress messages

        Returns:
            List of connected RemoteWorker instances

        Side effects:
            - Creates RemoteWorker for each server
            - Connects via TCP

        Example:
            >>> cluster.launch_servers(work_fn="module.fn")
            >>> time.sleep(3)  # Wait for servers to start
            >>> workers = cluster.connect_to_servers()
            >>> print(f"Connected to {len(workers)} workers")

        Raises:
            ConnectionRefusedError: If cannot connect to worker server
        """
        # Tiger Style: Assert preconditions
        assert timeout > 0, f"timeout must be > 0, got {timeout}"
        assert len(self.processes) > 0, "No servers launched - call launch_servers() first"

        if verbose:
            print(f"[MiniRay] Connecting to workers...")

        self._connect_workers(verbose)

        if verbose:
            print(f"[MiniRay] Connected to {len(self.workers)} workers")

        return self.workers

    def start(
        self,
        work_fn: str,
        wait_time: float = 3.0,
        verbose: bool = True,
    ) -> List[RemoteWorker]:
        """Launch worker servers and connect (convenience method).

        This is a convenience wrapper that calls:
        1. launch_servers()
        2. time.sleep(wait_time)
        3. connect_to_servers()

        For more control, call those methods separately.

        Args:
            work_fn: Python path to work function (e.g., "module.function")
            wait_time: Seconds to wait for servers to start
            verbose: Print progress messages

        Returns:
            List of connected RemoteWorker instances

        Example:
            >>> workers = cluster.start(work_fn="train_module.worker_fn")
            >>> print(f"Cluster ready: {len(workers)} workers")

        Raises:
            ConnectionRefusedError: If cannot connect to worker server
        """
        # Tiger Style: Assert preconditions
        assert wait_time > 0, f"wait_time must be > 0, got {wait_time}"
        assert wait_time < 60, f"wait_time too large: {wait_time} (max 60s)"

        if verbose:
            total_workers = sum(node.num_workers for node in self.nodes)
            print(f"[MiniRay] Starting cluster: {len(self.nodes)} nodes, {total_workers} workers")

        # Phase 1: Launch
        self.launch_servers(work_fn, verbose)

        # Phase 2: Wait
        if verbose:
            print(f"[MiniRay] Waiting {wait_time}s for servers to start...")
        time.sleep(wait_time)

        # Phase 3: Connect
        self.connect_to_servers(verbose=verbose)

        if verbose:
            print(f"[MiniRay] Cluster ready: {len(self.workers)} workers")

        # Tiger Style: Assert postcondition
        assert len(self.workers) > 0, "No workers connected"

        return self.workers

    def _launch_worker_servers(self, work_fn: str, verbose: bool):
        """Launch WorkerServer on each node via SSH.

        Args:
            work_fn: Python path to work function
            verbose: Print progress
        """
        for node in self.nodes:
            for worker_idx in range(node.num_workers):
                port = node.base_port + worker_idx

                # Build SSH command to launch worker_server.py
                ssh_cmd = ["ssh"]

                # Add SSH key if specified
                if node.ssh_key:
                    ssh_cmd.extend(["-i", node.ssh_key])

                # Remote command to launch worker server
                remote_cmd = [
                    node.python_bin,
                    "-m",
                    "rollouts.training.miniray.worker_server",
                    "--port",
                    str(port),
                    "--workers",
                    "1",  # One worker per server
                    "--work-fn",
                    work_fn,
                ]

                ssh_cmd.append(node.host)
                ssh_cmd.extend(remote_cmd)

                if verbose:
                    print(f"[MiniRay]   Launching worker on {node.host}:{port}")

                # Launch via SSH (runs in background)
                proc = subprocess.Popen(
                    ssh_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.processes.append(proc)

    def _connect_workers(self, verbose: bool):
        """Connect RemoteWorker to each launched server.

        Args:
            verbose: Print progress
        """
        for node in self.nodes:
            for worker_idx in range(node.num_workers):
                port = node.base_port + worker_idx

                if verbose:
                    print(f"[MiniRay]   Connecting to {node.host}:{port}")

                try:
                    worker = RemoteWorker(node.host, port)
                    worker.connect()  # Explicit connection
                    self.workers.append(worker)
                except (ConnectionRefusedError, TimeoutError) as e:
                    print(f"[MiniRay] ERROR: Cannot connect to {node.host}:{port}")
                    print(f"[MiniRay]        {e}")
                    print(f"[MiniRay]        Check that worker_server.py is running")
                    # Continue connecting to other workers
                    continue

    def stop(self, verbose: bool = True):
        """Stop all worker servers and clean up.

        Args:
            verbose: Print progress messages

        Side effects:
            - Sends shutdown to all workers
            - Terminates SSH processes
            - Closes RemoteWorker connections
        """
        if verbose:
            print(f"[MiniRay] Stopping cluster ({len(self.workers)} workers)...")

        # Phase 1: Send shutdown command to all workers
        for worker in self.workers:
            try:
                worker.send({"cmd": "shutdown"})
            except Exception as e:
                # Worker may have already died
                if verbose:
                    print(f"[MiniRay]   Warning: Failed to shutdown worker: {e}")

        # Phase 2: Close all RemoteWorker connections
        for worker in self.workers:
            try:
                worker.close()
            except:
                pass

        # Phase 3: Terminate SSH processes
        for proc in self.processes:
            if proc.poll() is None:  # Still running
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

        if verbose:
            print("[MiniRay] Cluster stopped")

        # Clear state
        self.workers.clear()
        self.processes.clear()

    def __enter__(self):
        """Context manager support.

        Example:
            >>> with Cluster(nodes=[...]) as cluster:
            ...     workers = cluster.start(work_fn="...")
            ...     # ... use workers ...
            ...     # Automatic cleanup on exit
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.stop(verbose=False)


def spawn_distributed_workers(
    nodes: List[NodeConfig],
    work_fn: str,
    wait_time: float = 3.0,
    verbose: bool = True,
) -> List[RemoteWorker]:
    """Spawn distributed workers (convenience function).

    Args:
        nodes: List of node configurations
        work_fn: Python path to work function
        wait_time: Seconds to wait for servers to start
        verbose: Print progress

    Returns:
        List of connected RemoteWorker instances

    Example:
        >>> workers = spawn_distributed_workers(
        ...     nodes=[
        ...         NodeConfig("node1", num_workers=4),
        ...         NodeConfig("node2", num_workers=4),
        ...     ],
        ...     work_fn="my_module.train_fn",
        ... )
        >>> # ... use workers ...
    """
    cluster = Cluster(nodes=nodes)
    return cluster.start(work_fn=work_fn, wait_time=wait_time, verbose=verbose)


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    """Test cluster management"""
    import sys

    # Example work function for testing
    def echo_worker_fn(handle):
        """Echo server for testing"""
        while True:
            msg = handle.recv()
            if msg.get("cmd") == "shutdown":
                break
            # Echo back
            handle.send({"echo": msg})

    # For testing, you can do:
    # 1. On node1: python -m rollouts.training.miniray.worker_server --port 10000 --work-fn __main__.echo_worker_fn
    # 2. On node2: python -m rollouts.training.miniray.worker_server --port 10000 --work-fn __main__.echo_worker_fn
    # 3. Run this script

    print("Example cluster configuration:")
    cluster = Cluster(
        nodes=[
            NodeConfig("localhost", num_workers=2, base_port=10000),
        ]
    )

    print("\nTo test, first launch worker servers:")
    print("  Terminal 1: python -m rollouts.training.miniray.worker_server --port 10000 --work-fn __main__.echo_worker_fn")
    print("  Terminal 2: python -m rollouts.training.miniray.worker_server --port 10001 --work-fn __main__.echo_worker_fn")
    print("\nThen run: python -m rollouts.training.miniray.cluster")
