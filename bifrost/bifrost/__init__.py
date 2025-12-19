"""Bifrost SDK - Python client for remote GPU execution and job management."""

from .async_client import AsyncBifrostClient
from .client import BifrostClient

# Job operations (functions-over-classes pattern)
from .job import (
    job_exit_code,
    job_kill,
    job_logs,
    job_status,
    job_stream_logs,
    job_stream_until_complete,
    job_wait,
)

# Node provisioning
from .provision import GPUQuery, acquire_node
from .remote_fs import (
    ensure_dir,
    path_exists,
    read_file,
    remove_file,
    write_file_safe,
)

# Server operations (functions-over-classes pattern)
from .server import (
    server_is_healthy,
    server_is_running,
    server_logs,
    server_stop,
    server_wait_until_healthy,
)
from .types import (
    EnvironmentVariables,
    JobInfo,
    JobMetadata,
    JobStatus,
    ProcessSpec,
    ServerInfo,
    SessionInfo,
    SSHConnection,
)

__all__ = [
    # Clients
    "BifrostClient",
    "AsyncBifrostClient",
    # New v2 types (frozen dataclasses)
    "ProcessSpec",
    "JobInfo",
    "ServerInfo",
    "GPUQuery",
    # Job functions
    "job_status",
    "job_wait",
    "job_logs",
    "job_stream_logs",
    "job_stream_until_complete",
    "job_exit_code",
    "job_kill",
    # Server functions
    "server_is_healthy",
    "server_wait_until_healthy",
    "server_logs",
    "server_stop",
    "server_is_running",
    # Provisioning
    "acquire_node",
    # Legacy types (for backwards compatibility)
    "JobStatus",
    "SSHConnection",
    "EnvironmentVariables",
    "SessionInfo",
    "JobMetadata",
    # Remote filesystem helpers
    "write_file_safe",
    "ensure_dir",
    "path_exists",
    "read_file",
    "remove_file",
]
