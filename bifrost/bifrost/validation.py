"""
Validation utilities for Bifrost SDK operations.

All validators follow Tiger Style: assert everything, fail fast.
Each validator returns the validated/normalized value on success.
"""

import logging
import secrets
from datetime import datetime

logger = logging.getLogger(__name__)


def validate_job_id(job_id: str, session_name: str | None = None) -> str:
    """Validate job ID format (Tiger Style: assert everything).

    Args:
        job_id: Job identifier to validate
        session_name: Optional session name for context

    Returns:
        Validated job ID

    Raises:
        AssertionError: If validation fails

    Validates:
        - Job ID is non-empty string
        - Length is reasonable (0 < len < 256)
        - Contains expected separators
        - No dangerous shell characters
    """
    # Assert input
    assert isinstance(job_id, str), "job_id must be string"
    assert len(job_id) > 0, "job_id cannot be empty"
    assert len(job_id) < 256, f"job_id too long: {len(job_id)} chars"

    # Assert format (should contain separators from generation)
    assert "-" in job_id, "job_id missing separators (invalid format)"

    # Assert no dangerous characters (shell injection protection)
    dangerous_chars = set(";&|`$(){}[]<>\"'\\")
    assert not any(c in dangerous_chars for c in job_id), (
        f"job_id contains dangerous shell characters: {job_id}"
    )

    # Assert output invariant
    assert len(job_id) > 0, "Validated job_id"
    return job_id


def validate_bootstrap_cmd(bootstrap_cmd: str | list[str]) -> str | list[str]:
    """Validate bootstrap command(s) (Tiger Style: assert everything).

    Args:
        bootstrap_cmd: Single command string or list of commands

    Returns:
        Validated bootstrap command(s)

    Raises:
        AssertionError: If validation fails

    Validates:
        - Command is string or list of strings
        - All commands are non-empty
        - No obvious command injection patterns
    """
    # Assert input type
    assert isinstance(bootstrap_cmd, (str, list)), "bootstrap_cmd must be string or list of strings"

    if isinstance(bootstrap_cmd, str):
        assert len(bootstrap_cmd) > 0, "bootstrap_cmd cannot be empty string"
        _check_command_safety(bootstrap_cmd)
    else:
        assert len(bootstrap_cmd) > 0, "bootstrap_cmd list cannot be empty"
        assert all(isinstance(cmd, str) for cmd in bootstrap_cmd), (
            "All bootstrap commands must be strings"
        )
        assert all(len(cmd) > 0 for cmd in bootstrap_cmd), (
            "All bootstrap commands must be non-empty"
        )
        for cmd in bootstrap_cmd:
            _check_command_safety(cmd)

    # Assert output invariant
    if isinstance(bootstrap_cmd, str):
        assert len(bootstrap_cmd) > 0, "Validated bootstrap command"
    else:
        assert len(bootstrap_cmd) > 0, "Validated bootstrap commands"

    return bootstrap_cmd


def _check_command_safety(command: str) -> None:
    """Basic safety check for shell commands (Tiger Style helper).

    This is NOT a security boundary - it's a sanity check to catch obvious errors.
    For true security, commands should be properly escaped or use subprocess with shell=False.
    """
    # Warn about suspicious patterns (not a hard failure)
    suspicious = ["rm -rf /", "mkfs", "> /dev/", "dd if="]
    for pattern in suspicious:
        if pattern in command.lower():
            logger.warning(f"Suspicious command pattern detected: {pattern}")


def validate_session_name(session_name: str) -> str:
    """Validate session name (Tiger Style: assert everything).

    Args:
        session_name: Human-readable session name

    Returns:
        Validated and sanitized session name

    Raises:
        AssertionError: If validation fails
    """
    # Assert input
    assert isinstance(session_name, str), "session_name must be string"
    assert len(session_name) > 0, "session_name cannot be empty"
    assert len(session_name) < 100, f"session_name too long: {len(session_name)}"

    # Sanitize: keep only alphanumeric and hyphens
    safe_name = "".join(c if c.isalnum() or c == "-" else "_" for c in session_name)
    assert len(safe_name) > 0, "Sanitized session_name is empty"

    # Assert output invariant
    assert len(safe_name) > 0, "Validated session_name"
    return safe_name


def generate_job_id(session_name: str | None = None) -> str:
    """Generate job ID with optional human-readable component.

    Uses timestamp + random suffix to avoid collisions.

    Args:
        session_name: Optional human-readable session name

    Returns:
        Job ID string

    Raises:
        AssertionError: If validation fails
    """
    # Validate input if provided
    if session_name is not None:
        session_name = validate_session_name(session_name)

    # Generate components
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    random_suffix = secrets.token_hex(4)  # 8 hex chars, ~4B combinations

    # Construct job ID
    if session_name:
        job_id = f"{session_name}-{timestamp}-{random_suffix}"
    else:
        job_id = f"job-{timestamp}-{random_suffix}"

    # Validate output
    return validate_job_id(job_id, session_name)


def validate_environment_variables(env: dict[str, str]) -> dict[str, str]:
    """Validate environment variables dictionary (Tiger Style: assert everything).

    Args:
        env: Dict mapping environment variable names to values

    Returns:
        Validated environment variables dict

    Raises:
        AssertionError: If validation fails

    Validates:
        - All keys are valid shell variable names
        - All values are strings
    """
    # Assert input type
    assert isinstance(env, dict), f"env must be dict, got {type(env)}"

    # Validate each environment variable
    for key, value in env.items():
        # Assert key is valid shell variable name
        assert isinstance(key, str), f"env key must be string, got {type(key)}"
        assert len(key) > 0, "env key cannot be empty"
        assert key[0].isalpha() or key[0] == "_", (
            f"env key must start with letter or underscore: {key}"
        )
        assert key.replace("_", "").isalnum(), f"env key must be alphanumeric or underscore: {key}"

        # Assert value is string
        assert isinstance(value, str), f"env value for {key} must be string, got {type(value)}"

    # Assert output invariant
    assert isinstance(env, dict), "Validated environment variables"
    return env


def validate_working_directory(working_dir: str) -> str:
    """Validate working directory path (Tiger Style: assert everything).

    Args:
        working_dir: Working directory path

    Returns:
        Validated working directory

    Raises:
        AssertionError: If validation fails

    Note: Does not check if directory exists (remote path may not exist yet)
    """
    # Assert input type
    assert isinstance(working_dir, str), f"working_dir must be string, got {type(working_dir)}"
    assert len(working_dir) > 0, "working_dir cannot be empty"
    assert len(working_dir) < 4096, f"working_dir too long: {len(working_dir)} chars"

    # Assert no dangerous characters
    dangerous_chars = set(";|&`$(){}[]<>\"'\\")
    assert not any(c in dangerous_chars for c in working_dir), (
        f"working_dir contains dangerous characters: {working_dir}"
    )

    # Warn about relative paths
    if not working_dir.startswith("/") and not working_dir.startswith("~"):
        logger.warning(f"Relative working directory: {working_dir}")

    # Assert output invariant
    assert len(working_dir) > 0, "Validated working_dir"
    return working_dir


def validate_command(command: str) -> str:
    """Validate command string (Tiger Style: assert everything).

    Args:
        command: Command to execute

    Returns:
        Validated command

    Raises:
        AssertionError: If validation fails
    """
    # Assert input type
    assert isinstance(command, str), f"command must be string, got {type(command)}"
    assert len(command) > 0, "command cannot be empty"
    assert len(command) < 100_000, f"command too long: {len(command)} chars"

    # Basic safety check
    _check_command_safety(command)

    # Assert output invariant
    assert len(command) > 0, "Validated command"
    return command


def validate_poll_interval(interval: float) -> float:
    """Validate poll interval (Tiger Style: assert everything).

    Args:
        interval: Polling interval in seconds

    Returns:
        Validated interval

    Raises:
        AssertionError: If validation fails
    """
    # Assert input type
    assert isinstance(interval, (int, float)), f"interval must be numeric, got {type(interval)}"

    # Convert to float
    interval = float(interval)

    # Assert reasonable range
    assert 0.1 <= interval <= 3600, f"interval must be 0.1-3600 seconds, got {interval}"

    # Warn about very frequent polling
    if interval < 1.0:
        logger.warning(f"Very frequent polling: {interval}s")

    # Assert output invariant
    assert interval > 0, "Validated poll_interval"
    return interval
