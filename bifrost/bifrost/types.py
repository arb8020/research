"""Bifrost SDK data types and structures."""

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SSHConnection:
    """SSH connection information."""

    user: str
    host: str
    port: int

    @classmethod
    def from_string(cls, ssh_string: str) -> "SSHConnection":
        """Parse SSH string in multiple formats.

        Supports:
        - 'user@host:port' format
        - 'ssh -p port user@host' format (standard SSH command)
        """
        # Handle standard SSH command format: "ssh -p port user@host"
        if ssh_string.startswith("ssh "):
            return cls._parse_ssh_command(ssh_string)

        # Handle user@host:port format
        if "@" not in ssh_string or ":" not in ssh_string:
            raise ValueError(
                f"Invalid SSH format: {ssh_string}. Expected: user@host:port or ssh -p port user@host"
            )

        user_host, port_str = ssh_string.rsplit(":", 1)
        user, host = user_host.split("@", 1)

        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port: {port_str}")

        return cls(user=user, host=host, port=port)

    @classmethod
    def _parse_ssh_command(cls, ssh_cmd: str) -> "SSHConnection":
        """Parse SSH command format: 'ssh -p port user@host'"""
        import shlex

        try:
            parts = shlex.split(ssh_cmd)
        except ValueError as e:
            raise ValueError(f"Failed to parse SSH command: {e}")

        if not parts or parts[0] != "ssh":
            raise ValueError(f"Invalid SSH command: {ssh_cmd}")

        port = 22  # default SSH port
        user_host = None

        # Parse SSH command arguments
        i = 1
        while i < len(parts):
            if parts[i] == "-p" and i + 1 < len(parts):
                try:
                    port = int(parts[i + 1])
                    i += 2
                except ValueError:
                    raise ValueError(f"Invalid port in SSH command: {parts[i + 1]}")
            elif "@" in parts[i]:
                user_host = parts[i]
                break
            else:
                i += 1

        if not user_host or "@" not in user_host:
            raise ValueError(f"No user@host found in SSH command: {ssh_cmd}")

        user, host = user_host.split("@", 1)

        return cls(user=user, host=host, port=port)

    def __str__(self) -> str:
        return f"{self.user}@{self.host}:{self.port}"


@dataclass
class RemoteConfig:
    """Configuration for connecting to remote GPU instance."""

    host: str
    port: int
    user: str
    key_path: str

    def __post_init__(self):
        # Tiger Style assertions
        assert isinstance(self.host, str) and len(self.host) > 0, "host must be non-empty string"
        assert isinstance(self.port, int) and 0 < self.port < 65536, (
            f"port must be between 1-65535, got {self.port}"
        )
        assert isinstance(self.user, str) and len(self.user) > 0, "user must be non-empty string"
        assert isinstance(self.key_path, str) and len(self.key_path) > 0, (
            "key_path must be non-empty string"
        )


@dataclass
class ExecResult:
    """Result from executing a command via SSH."""

    stdout: str
    stderr: str
    exit_code: int

    def __post_init__(self):
        # Tiger Style assertions
        assert isinstance(self.stdout, str), "stdout must be string"
        assert isinstance(self.stderr, str), "stderr must be string"
        assert isinstance(self.exit_code, int), "exit_code must be int"

    @property
    def success(self) -> bool:
        """Returns True if command exited with code 0."""
        return self.exit_code == 0


@dataclass
class CopyResult:
    """Result of a file copy operation."""

    success: bool
    files_copied: int
    total_bytes: int
    duration_seconds: float
    error_message: str | None = None

    @property
    def throughput_mbps(self) -> float:
        """Calculate transfer throughput in MB/s."""
        if self.duration_seconds > 0:
            return (self.total_bytes / (1024 * 1024)) / self.duration_seconds
        return 0.0


@dataclass
class RemotePath:
    """Represents a remote file path."""

    path: str

    def __post_init__(self):
        # Ensure path is absolute for consistency
        if not self.path.startswith("/") and not self.path.startswith("~"):
            self.path = f"./{self.path}"


class BifrostError(Exception):
    """Base exception for Bifrost SDK errors."""

    pass


class ConnectionError(BifrostError):
    """SSH connection related errors."""

    pass


class JobError(BifrostError):
    """Job execution related errors."""

    pass


class TransferError(BifrostError):
    """File transfer related errors."""

    pass


# ============================================================================
# New Frozen Dataclasses (Type Improvements)
# ============================================================================


@dataclass(frozen=True)
class EnvironmentVariables:
    """Environment variables for remote command execution.

    Validates variable names follow shell naming rules.
    Immutable to prevent accidental modification during execution.
    """

    variables: dict[str, str]

    def __post_init__(self):
        # Tiger Style: assert all inputs
        assert isinstance(self.variables, dict), "variables must be dict"

        for key, value in self.variables.items():
            # Shell variable name rules: [A-Za-z_][A-Za-z0-9_]*
            assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key), (
                f"Invalid environment variable name: {key}"
            )
            assert isinstance(value, str), (
                f"Environment variable {key} must be string, got {type(value)}"
            )

        # Assert output invariant
        assert len(self.variables) >= 0, "variables dict created"

    def to_dict(self) -> dict[str, str]:
        """Convert to dict for backward compatibility."""
        return self.variables.copy()

    @classmethod
    def from_dict(cls, variables: dict[str, str] | None) -> Optional["EnvironmentVariables"]:
        """Create from optional dict (for gradual migration)."""
        if variables is None:
            return None
        return cls(variables=variables)


# ============================================================================
# v2 API Types (functions-over-classes pattern)
# ============================================================================


@dataclass(frozen=True)
class ProcessSpec:
    """Complete specification of a process to run.

    Immutable - represents what to run, not a running process.
    Can be serialized, logged, compared.

    Env vars are passed here, not cached on Session.

    Example:
        spec = ProcessSpec(
            command="python",
            args=("train.py", "--lr", "0.001"),
            cwd="/workspace",
            env={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
    """

    command: str
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env: dict[str, str] | None = None
    cuda_device_ids: tuple[int, ...] | None = None

    def __post_init__(self):
        assert self.command, "command cannot be empty"
        assert isinstance(self.args, tuple), "args must be a tuple"

    def build_command(self) -> str:
        """Build full command string with proper escaping.

        Returns command string suitable for shell execution.
        """
        import shlex

        # Build base command with args
        cmd_parts = [self.command] + list(self.args)
        full_cmd = shlex.join(cmd_parts)

        # Add env vars prefix if needed
        if self.env:
            env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
            full_cmd = f"{env_prefix} {full_cmd}"

        # Add CUDA_VISIBLE_DEVICES if cuda_device_ids specified
        if self.cuda_device_ids is not None:
            devices_str = ",".join(str(d) for d in self.cuda_device_ids)
            full_cmd = f"CUDA_VISIBLE_DEVICES={devices_str} {full_cmd}"

        # Add cd if cwd specified
        if self.cwd:
            full_cmd = f"cd {shlex.quote(self.cwd)} && {full_cmd}"

        return full_cmd


@dataclass(frozen=True)
class JobInfo:
    """Immutable job identifier - just data.

    This is the new v2 JobInfo that follows the functions-over-classes pattern.
    It contains only identifiers, not status. Status comes from job_status().

    Returned by BifrostClient.submit().
    Used with job_status(), job_wait(), job_logs(), job_kill() functions.
    """

    name: str
    tmux_session: str
    log_file: str | None = None
    workspace: str | None = None

    def __post_init__(self):
        assert self.name, "name cannot be empty"
        assert self.tmux_session, "tmux_session cannot be empty"


@dataclass(frozen=True)
class ServerInfo:
    """Immutable server identifier - just data.

    This is the new v2 ServerInfo that follows the functions-over-classes pattern.
    It contains only identifiers, not health status. Status comes from server_is_healthy().

    Returned by BifrostClient.serve().
    Used with server_is_healthy(), server_wait_until_healthy(), server_stop() functions.
    """

    name: str
    tmux_session: str
    log_file: str | None = None
    port: int | None = None
    health_endpoint: str | None = None
    workspace: str | None = None

    def __post_init__(self):
        assert self.name, "name cannot be empty"
        assert self.tmux_session, "tmux_session cannot be empty"

    @property
    def url(self) -> str | None:
        """Get server URL if port is known."""
        if self.port:
            return f"http://localhost:{self.port}"
        return None
