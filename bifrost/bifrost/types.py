"""Bifrost SDK data types and structures."""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    STARTING = "starting"


@dataclass
class SSHConnection:
    """SSH connection information."""
    user: str
    host: str 
    port: int
    
    @classmethod
    def from_string(cls, ssh_string: str) -> 'SSHConnection':
        """Parse SSH string in multiple formats.
        
        Supports:
        - 'user@host:port' format
        - 'ssh -p port user@host' format (standard SSH command)
        """
        # Handle standard SSH command format: "ssh -p port user@host"
        if ssh_string.startswith('ssh '):
            return cls._parse_ssh_command(ssh_string)
        
        # Handle user@host:port format
        if '@' not in ssh_string or ':' not in ssh_string:
            raise ValueError(f"Invalid SSH format: {ssh_string}. Expected: user@host:port or ssh -p port user@host")
        
        user_host, port_str = ssh_string.rsplit(':', 1)
        user, host = user_host.split('@', 1)
        
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port: {port_str}")
        
        return cls(user=user, host=host, port=port)
    
    @classmethod
    def _parse_ssh_command(cls, ssh_cmd: str) -> 'SSHConnection':
        """Parse SSH command format: 'ssh -p port user@host'"""
        import shlex
        
        try:
            parts = shlex.split(ssh_cmd)
        except ValueError as e:
            raise ValueError(f"Failed to parse SSH command: {e}")
        
        if not parts or parts[0] != 'ssh':
            raise ValueError(f"Invalid SSH command: {ssh_cmd}")
        
        port = 22  # default SSH port
        user_host = None
        
        # Parse SSH command arguments
        i = 1
        while i < len(parts):
            if parts[i] == '-p' and i + 1 < len(parts):
                try:
                    port = int(parts[i + 1])
                    i += 2
                except ValueError:
                    raise ValueError(f"Invalid port in SSH command: {parts[i + 1]}")
            elif '@' in parts[i]:
                user_host = parts[i]
                break
            else:
                i += 1
        
        if not user_host or '@' not in user_host:
            raise ValueError(f"No user@host found in SSH command: {ssh_cmd}")
        
        user, host = user_host.split('@', 1)
        
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
        assert isinstance(self.host, str) and len(self.host) > 0, \
            "host must be non-empty string"
        assert isinstance(self.port, int) and 0 < self.port < 65536, \
            f"port must be between 1-65535, got {self.port}"
        assert isinstance(self.user, str) and len(self.user) > 0, \
            "user must be non-empty string"
        assert isinstance(self.key_path, str) and len(self.key_path) > 0, \
            "key_path must be non-empty string"


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
class JobInfo:
    """Information about a detached job."""
    job_id: str
    status: JobStatus
    command: str
    tmux_session: str | None = None  # Main command session
    bootstrap_session: str | None = None  # Bootstrap session (if applicable)
    start_time: datetime | None = None
    end_time: datetime | None = None
    exit_code: int | None = None
    runtime_seconds: float | None = None

    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status in [JobStatus.RUNNING, JobStatus.STARTING]

    @property
    def is_complete(self) -> bool:
        """Check if job has finished (successfully or failed)."""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED]


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
        if not self.path.startswith('/') and not self.path.startswith('~'):
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
            assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key), \
                f"Invalid environment variable name: {key}"
            assert isinstance(value, str), \
                f"Environment variable {key} must be string, got {type(value)}"

        # Assert output invariant
        assert len(self.variables) >= 0, "variables dict created"

    def to_dict(self) -> dict[str, str]:
        """Convert to dict for backward compatibility."""
        return self.variables.copy()

    @classmethod
    def from_dict(cls, variables: dict[str, str] | None) -> Optional['EnvironmentVariables']:
        """Create from optional dict (for gradual migration)."""
        if variables is None:
            return None
        return cls(variables=variables)


@dataclass(frozen=True)
class SessionInfo:
    """Information about tmux sessions for a detached job.

    Provides session names and SSH commands to attach to them.
    Immutable since session info shouldn't change after creation.
    """
    job_id: str
    main_session: str
    attach_main: str
    bootstrap_session: str | None = None
    attach_bootstrap: str | None = None

    def __post_init__(self):
        # Tiger Style: assert inputs and invariants
        assert len(self.job_id) > 0, "job_id cannot be empty"
        assert self.main_session.startswith("bifrost-"), \
            f"Invalid main session name: {self.main_session}"
        assert "ssh" in self.attach_main or "tmux" in self.attach_main, \
            "attach_main must be SSH or tmux command"

        # Validate bootstrap session format if present
        if self.bootstrap_session:
            assert self.bootstrap_session.endswith("-bootstrap"), \
                f"Invalid bootstrap session name: {self.bootstrap_session}"
            assert self.attach_bootstrap is not None, \
                "attach_bootstrap required when bootstrap_session present"

        # Assert output invariant
        assert self.main_session, "session info validated"


@dataclass(frozen=True)
class JobMetadata:
    """Metadata for a detached job execution.

    Stored in ~/.bifrost/jobs/{job_id}/metadata.json on remote.
    Immutable since metadata is write-once, read-many.
    """
    job_id: str
    command: str
    ssh_info: str
    status: str
    start_time: str  # ISO 8601 format
    tmux_session: str
    worktree_path: str
    git_commit: str
    repo_name: str
    end_time: str | None = None  # ISO 8601 format
    exit_code: int | None = None

    def __post_init__(self):
        # Tiger Style: assert all inputs and invariants
        assert len(self.job_id) > 0, "job_id cannot be empty"
        assert len(self.command) > 0, "command cannot be empty"

        # Validate status
        valid_statuses = {"starting", "running", "completed", "failed", "killed"}
        assert self.status in valid_statuses, \
            f"Invalid status: {self.status}, must be one of {valid_statuses}"

        # Validate git commit hash (full SHA-1 or SHA-256)
        assert len(self.git_commit) in (7, 8, 40, 64), \
            f"Invalid git commit hash length: {len(self.git_commit)}"

        # Validate SSH info format
        assert "@" in self.ssh_info and ":" in self.ssh_info, \
            f"Invalid ssh_info format: {self.ssh_info}"

        # Validate exit code range if present
        if self.exit_code is not None:
            assert 0 <= self.exit_code <= 255, \
                f"Invalid exit code: {self.exit_code}"

        # Assert output invariant
        assert self.job_id, "job metadata validated"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "job_id": self.job_id,
            "command": self.command,
            "ssh_info": self.ssh_info,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "exit_code": self.exit_code,
            "tmux_session": self.tmux_session,
            "worktree_path": self.worktree_path,
            "git_commit": self.git_commit,
            "repo_name": self.repo_name
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'JobMetadata':
        """Create from dict (for JSON deserialization)."""
        return cls(
            job_id=data["job_id"],
            command=data["command"],
            ssh_info=data["ssh_info"],
            status=data["status"],
            start_time=data["start_time"],
            tmux_session=data["tmux_session"],
            worktree_path=data["worktree_path"],
            git_commit=data["git_commit"],
            repo_name=data["repo_name"],
            end_time=data.get("end_time"),
            exit_code=data.get("exit_code")
        )