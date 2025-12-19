"""Server operations for bifrost (functions-over-classes pattern).

Pure functions that operate on ServerInfo - no methods on the dataclass itself.
Session is passed explicitly to every function.

Tiger Style:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
- Tuple returns for errors

Example:
    from bifrost import BifrostClient
    from bifrost.types import ProcessSpec, ServerInfo
    from bifrost.server import server_is_healthy, server_wait_until_healthy, server_stop

    client = BifrostClient("root@gpu:22")
    server = client.serve(
        ProcessSpec(command="python", args=("-m", "sglang.launch_server", "--model", "meta-llama/Llama-3.1-8B")),
        name="sglang",
        port=30000,
        health_endpoint="/health",
    )

    # Wait for server to be healthy
    if server_wait_until_healthy(client, server, timeout=300):
        print(f"Server ready at {server.url}")
    else:
        print("Server failed to start")
        server_stop(client, server)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import BifrostClient
    from .types import ServerInfo

logger = logging.getLogger(__name__)


def server_is_healthy(session: "BifrostClient", server: "ServerInfo") -> bool:
    """Check if server is responding to health checks.

    If health_endpoint is configured, makes HTTP request to check.
    Otherwise, just checks if tmux session is alive.

    Args:
        session: BifrostClient instance (owns SSH connection)
        server: ServerInfo identifier

    Returns:
        True if server is healthy, False otherwise
    """
    # First check if tmux session is alive
    result = session.exec(f"tmux has-session -t {server.tmux_session} 2>/dev/null")
    if result.exit_code != 0:
        return False  # Server process not running

    # If no health endpoint, just check session is alive
    if not server.health_endpoint or not server.port:
        return True

    # Make health check request
    url = f"http://localhost:{server.port}{server.health_endpoint}"
    cmd = f"curl -s -o /dev/null -w '%{{http_code}}' {url} 2>/dev/null || echo 000"
    result = session.exec(cmd)

    status_code = result.stdout.strip()
    return status_code == "200"


def server_wait_until_healthy(
    session: "BifrostClient",
    server: "ServerInfo",
    timeout: float = 300,
    poll_interval: float = 5.0,
) -> bool:
    """Wait for server to become healthy.

    Polls health check until server responds or timeout.

    Args:
        session: BifrostClient instance
        server: ServerInfo identifier
        timeout: Maximum wait time in seconds (default: 5 minutes)
        poll_interval: How often to check health

    Returns:
        True if server became healthy, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        if server_is_healthy(session, server):
            logger.info(f"Server {server.name} is healthy")
            return True

        # Check if process has crashed
        result = session.exec(f"tmux has-session -t {server.tmux_session} 2>/dev/null")
        if result.exit_code != 0:
            logger.error(f"Server {server.name} process has exited")
            return False

        time.sleep(poll_interval)

    logger.warning(f"Server {server.name} did not become healthy within {timeout}s")
    return False


def server_logs(session: "BifrostClient", server: "ServerInfo", tail: int = 100) -> str:
    """Get recent server logs.

    Args:
        session: BifrostClient instance
        server: ServerInfo identifier
        tail: Number of lines to return (default: 100)

    Returns:
        Log content as string, or empty string if no log file
    """
    if not server.log_file:
        return ""

    result = session.exec(f"tail -n {tail} {server.log_file} 2>/dev/null || true")
    return result.stdout


def server_stop(session: "BifrostClient", server: "ServerInfo") -> None:
    """Stop a running server.

    Terminates the tmux session for this server.

    Args:
        session: BifrostClient instance
        server: ServerInfo identifier
    """
    session.exec(f"tmux kill-session -t {server.tmux_session} 2>/dev/null || true")
    logger.info(f"Stopped server: {server.name}")


def server_is_running(session: "BifrostClient", server: "ServerInfo") -> bool:
    """Check if server process is running (tmux session alive).

    Different from server_is_healthy - this just checks if process exists,
    not if it's responding to requests.

    Args:
        session: BifrostClient instance
        server: ServerInfo identifier

    Returns:
        True if tmux session is alive
    """
    result = session.exec(f"tmux has-session -t {server.tmux_session} 2>/dev/null")
    return result.exit_code == 0
