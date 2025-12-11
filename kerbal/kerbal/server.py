"""Server deployment and management for remote execution.

This module provides a high-level API for running long-lived servers on remote machines.
Composes the lower-level primitives (tmux, job_monitor, python_env, gpu).

Tiger Style:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
- Dataclasses for state

Usage:
    from bifrost import BifrostClient
    from kerbal import serve, DependencyConfig

    client = BifrostClient("root@gpu:22")
    workspace = client.push()

    server = serve(
        client,
        command="python -m sglang.launch_server --model Qwen/Qwen2.5-7B --port 30000",
        workspace=workspace,
        port=30000,
        gpu_ids=[0, 1],
        health_endpoint="/health",
    )

    print(f"Server ready: {server.url}")
    # ... use server ...
    server.stop()
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kerbal.gpu import check_gpus_available
from kerbal.tmux import start_tmux_session

if TYPE_CHECKING:
    from bifrost import BifrostClient

    from kerbal.protocol import DependencyConfig

logger = logging.getLogger(__name__)


@dataclass
class ServerHandle:
    """Handle to a running server.

    Tiger Style: Immutable reference to server state on remote.
    """

    client: "BifrostClient"
    session_name: str
    log_file: str
    workspace: str
    url: str
    port: int

    def stop(self) -> None:
        """Stop the server and release GPU memory."""
        # Kill the tmux session first
        self.client.exec(f"tmux kill-session -t {self.session_name} 2>/dev/null || true")
        # Then kill any remaining process using the port
        # Use lsof (more commonly available than fuser) to find and kill
        self.client.exec(
            f"lsof -ti tcp:{self.port} | xargs -r kill -9 2>/dev/null || true"
        )
        # Give GPU memory time to release
        import time
        time.sleep(2)
        logger.info(f"Stopped server: {self.session_name}")

    def is_alive(self) -> bool:
        """Check if server is still running."""
        result = self.client.exec(f"tmux has-session -t {self.session_name} 2>&1")
        return result.exit_code == 0

    def is_healthy(self) -> bool:
        """Check if server responds to health check."""
        result = self.client.exec(
            f"curl -sf {self.url} >/dev/null 2>&1 && echo 'OK' || echo 'FAIL'"
        )
        return "OK" in result.stdout

    def logs(self, tail: int | None = None) -> str:
        """Get server log content.

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

    def wait_until_healthy(self, timeout_sec: int = 300, poll_interval: float = 2.0) -> bool:
        """Wait for server to become healthy.

        Args:
            timeout_sec: Maximum time to wait
            poll_interval: How often to check

        Returns:
            True if healthy, False if timeout or server died
        """
        return _wait_for_health(
            self.client,
            self.url,
            self.session_name,
            timeout_sec,
            poll_interval,
        )


def serve(
    client: "BifrostClient",
    command: str,
    workspace: str,
    port: int,
    gpu_ids: list[int] | None = None,
    deps: "DependencyConfig | None" = None,
    env_vars: dict[str, str] | None = None,
    server_name: str | None = None,
    host: str = "0.0.0.0",
    health_endpoint: str = "/health",
    health_timeout: int = 300,
    check_gpus: bool = True,
    check_port: bool = True,
) -> ServerHandle:
    """Start a server on remote.

    Deploys dependencies (if specified), starts server in tmux,
    waits for health check, returns handle.

    Args:
        client: BifrostClient instance
        command: Server command to run
        workspace: Remote workspace path (from client.push())
        port: Port the server listens on
        gpu_ids: GPUs to use (sets CUDA_VISIBLE_DEVICES)
        deps: Dependencies to install (optional)
        env_vars: Additional environment variables
        server_name: Tmux session name (default: "server-{port}")
        host: Host to bind (default: 0.0.0.0)
        health_endpoint: Health check endpoint (default: /health)
        health_timeout: Seconds to wait for health (default: 300)
        check_gpus: Whether to verify GPU availability (default: True)
        check_port: Whether to verify port is free (default: True)

    Returns:
        ServerHandle for interacting with the server

    Raises:
        AssertionError: If preconditions not met (GPUs unavailable, port in use, etc.)
        TimeoutError: If server doesn't become healthy in time

    Example:
        server = serve(client, "python -m sglang.launch_server ...", workspace, port=30000)
        print(f"Ready: {server.url}")
        server.stop()
    """
    assert client is not None, "BifrostClient required"
    assert command, "command required"
    assert workspace, "workspace required"
    assert port > 0, "port must be positive"

    # Generate server name if not provided
    if server_name is None:
        server_name = f"server-{port}"

    logger.info(f"Starting server: {server_name}")
    logger.info(f"  Command: {command[:80]}{'...' if len(command) > 80 else ''}")
    logger.info(f"  Port: {port}")

    # Check GPU availability
    if check_gpus and gpu_ids:
        logger.info(f"  Checking GPUs: {gpu_ids}")
        available, err = check_gpus_available(client, gpu_ids)
        assert available, f"GPUs not available: {err}"
        logger.info(f"  GPUs available: {gpu_ids}")

    # Check port availability
    if check_port:
        logger.info(f"  Checking port {port}...")
        port_free, err = _check_port_free(client, port)
        assert port_free, f"Port {port} in use: {err}"
        logger.info(f"  Port {port} is free")

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
    log_file = f"{workspace}/{server_name}.log"

    # If deps were installed, activate the venv before running command
    final_command = command
    if deps is not None:
        # Prefix command with venv activation
        final_command = f"source {workspace}/.venv/bin/activate && {command}"

    # Start server in tmux
    session_name, err = start_tmux_session(
        client,
        session_name=server_name,
        command=final_command,
        workspace=workspace,
        log_file=log_file,
        env_vars=final_env_vars if final_env_vars else None,
        capture_exit_code=False,  # Servers don't exit normally
    )
    assert err is None, f"Failed to start server: {err}"

    logger.info(f"  Server started: {session_name}")
    logger.info(f"  Log file: {log_file}")

    # Build URL
    # Use localhost for health checks (server binds to 0.0.0.0 but we curl from same machine)
    health_url = f"http://localhost:{port}{health_endpoint}"
    public_url = f"http://localhost:{port}"

    # Wait for health
    logger.info(f"  Waiting for health check: {health_url}")
    healthy = _wait_for_health(
        client,
        health_url,
        session_name,
        health_timeout,
        poll_interval=2.0,
    )

    if not healthy:
        # Server failed to start - get logs for debugging
        logs = client.exec(f"tail -50 {log_file} 2>/dev/null || true").stdout
        # Kill the failed session
        client.exec(f"tmux kill-session -t {session_name} 2>/dev/null || true")
        raise TimeoutError(
            f"Server failed to become healthy within {health_timeout}s.\n"
            f"Last 50 lines of log:\n{logs}"
        )

    logger.info(f"  Server healthy: {public_url}")

    return ServerHandle(
        client=client,
        session_name=session_name,
        log_file=log_file,
        workspace=workspace,
        url=public_url,
        port=port,
    )


def _check_port_free(client: "BifrostClient", port: int) -> tuple[bool, str | None]:
    """Check if port is free on remote.

    Returns:
        (is_free, error_message)
    """
    result = client.exec(f"lsof -i :{port} 2>/dev/null || echo 'PORT_FREE'")
    if "PORT_FREE" in result.stdout:
        return True, None
    else:
        return False, f"Port {port} in use:\n{result.stdout}"


def _wait_for_health(
    client: "BifrostClient",
    health_url: str,
    session_name: str,
    timeout_sec: int,
    poll_interval: float,
) -> bool:
    """Wait for server health check to pass.

    Also monitors tmux session - returns False if server dies.

    Returns:
        True if healthy, False if timeout or server died
    """
    start_time = time.time()

    while time.time() - start_time < timeout_sec:
        # Check if server process still alive
        alive = client.exec(f"tmux has-session -t {session_name} 2>&1")
        if alive.exit_code != 0:
            logger.error("Server process died")
            return False

        # Check health endpoint
        health = client.exec(f"curl -sf {health_url} >/dev/null 2>&1 && echo 'OK'")
        if "OK" in health.stdout:
            return True

        # Also try /v1/models (common for LLM servers)
        if "/health" in health_url:
            alt_url = health_url.replace("/health", "/v1/models")
            health_alt = client.exec(f"curl -sf {alt_url} >/dev/null 2>&1 && echo 'OK'")
            if "OK" in health_alt.stdout:
                return True

        time.sleep(poll_interval)

    logger.error(f"Health check timeout after {timeout_sec}s")
    return False
