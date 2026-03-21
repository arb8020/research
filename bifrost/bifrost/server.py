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

import asyncio
import inspect
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import BifrostClient
    from .types import ServerInfo

logger = logging.getLogger(__name__)


def _run_handle_method(method: object, *args: object) -> object:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "bifrost.server sync helpers cannot be called from a running asyncio loop; "
            "await the ServiceHandle methods directly"
        )

    result = method(*args)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def server_is_healthy(session: BifrostClient, server: ServerInfo) -> bool:
    """Check if server is responding to health checks.

    If health_endpoint is configured, makes HTTP request to check.
    Otherwise, just checks if the detached service is still running.

    Args:
        session: BifrostClient instance (owns SSH connection)
        server: ServerInfo identifier

    Returns:
        True if server is healthy, False otherwise
    """
    del session
    return bool(_run_handle_method(server.is_healthy))


def server_wait_until_healthy(
    session: BifrostClient,
    server: ServerInfo,
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

        if not server_is_running(session, server):
            logger.error(f"Server {server.name} process has exited")
            return False

        time.sleep(poll_interval)

    logger.warning(f"Server {server.name} did not become healthy within {timeout}s")
    return False


def server_logs(session: BifrostClient, server: ServerInfo, tail: int = 100) -> str:
    """Get recent server logs.

    Args:
        session: BifrostClient instance
        server: ServerInfo identifier
        tail: Number of lines to return (default: 100)

    Returns:
        Log content as string, or empty string if no log file
    """
    del session
    return str(_run_handle_method(server.logs, tail))


def server_stop(session: BifrostClient, server: ServerInfo) -> None:
    """Stop a running server.

    Terminates the detached service for this server.

    Args:
        session: BifrostClient instance
        server: ServerInfo identifier
    """
    del session
    _run_handle_method(server.stop)
    logger.info(f"Stopped server: {server.name}")


def server_is_running(session: BifrostClient, server: ServerInfo) -> bool:
    """Check if server process is running.

    Different from server_is_healthy - this just checks if process exists,
    not if it's responding to requests.

    Args:
        session: BifrostClient instance
        server: ServerInfo identifier

    Returns:
        True if the detached service is still alive
    """
    del session
    return bool(_run_handle_method(server.is_running))
