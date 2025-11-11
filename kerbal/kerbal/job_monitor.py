"""Job monitoring with real-time log streaming.

This module handles monitoring long-running remote jobs with real-time feedback.
Purely about job monitoring - no knowledge of Python envs, GPUs, or deployment specifics.

Key patterns extracted from production deployment code:
- Real-time log streaming with position tracking (clicker: 170a21c, 490f385)
- Early crash detection via tmux session liveness (qwen3_next: d3ebff8)
- Exit code extraction from log markers (qwen3_next: bc35edc)

Tiger Style:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
- Tuple returns for errors

Usage:
    from kerbal import start_tmux_session
    from kerbal.job_monitor import stream_log_until_complete, LogStreamConfig

    # Start job in tmux
    session, err = start_tmux_session(client, "training", "python train.py", log_file="train.log")
    if err:
        print(f"Failed: {err}")
        return

    # Monitor with real-time streaming
    config = LogStreamConfig(
        session_name=session,
        log_file="train.log",
        timeout_sec=7200,  # 2 hours
    )
    success, exit_code, err = stream_log_until_complete(client, config)
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from bifrost import BifrostClient

logger = logging.getLogger(__name__)


@dataclass
class LogStreamConfig:
    """Configuration for log streaming.

    Tiger Style: Explicit configuration, immutable dataclass.

    Attributes:
        session_name: Tmux session name to monitor
        log_file: Absolute path to log file on remote
        timeout_sec: Maximum time to wait for completion (default: 3600 = 1 hour)
        poll_interval_sec: How often to check for new output (default: 2 seconds)
        stop_on_marker: Optional string marker that indicates completion
                       (e.g., "EVAL_EXIT_CODE: 0"). If None, waits for tmux exit.
    """

    session_name: str
    log_file: str
    timeout_sec: int = 3600
    poll_interval_sec: float = 2.0
    stop_on_marker: str | None = None


def stream_log_until_complete(
    client: "BifrostClient",
    config: LogStreamConfig,
) -> tuple[bool, int | None, str | None]:
    """Stream remote log file in real-time until job completes.

    Pattern from qwen3_next/deploy/benchmark_monitor.py and clicker/deploy.py.
    Uses position-tracked tailing for efficient streaming.

    Casey: Granular operation - just stream and monitor, nothing else.
    Tiger Style: < 70 lines, explicit control flow, tuple return.

    Args:
        client: BifrostClient instance
        config: Log streaming configuration

    Returns:
        (success, exit_code, error_message)
        - success=True if job completed successfully
        - exit_code=int if job exited (0 = success)
        - error_message=str if failed or timeout

    Example:
        config = LogStreamConfig(
            session_name="training-job",
            log_file="/workspace/train.log",
            timeout_sec=7200,
        )
        success, exit_code, err = stream_log_until_complete(client, config)
        if not success:
            print(f"Job failed: {err} (exit code: {exit_code})")
    """
    assert client is not None, "BifrostClient required"
    assert config.session_name, "session_name required"
    assert config.log_file, "log_file required"
    assert config.timeout_sec > 0, "timeout_sec must be positive"
    assert config.poll_interval_sec > 0, "poll_interval_sec must be positive"

    logger.info(f"📊 Monitoring job: {config.session_name}")
    logger.info(f"📝 Log: {config.log_file}")
    logger.info(f"⏱️  Timeout: {config.timeout_sec}s")
    logger.info("=" * 60)

    # Stream log and wait for completion
    success, exit_code, err = _stream_and_wait(client, config)

    logger.info("=" * 60)

    if not success:
        logger.error(f"❌ Job failed: {err}")
    else:
        logger.info(f"✅ Job completed (exit code: {exit_code})")

    return success, exit_code, err


def _stream_and_wait(
    client: "BifrostClient",
    config: LogStreamConfig,
) -> tuple[bool, int | None, str | None]:
    """Core streaming loop with position tracking.

    Tiger Style: Split from public function to keep < 70 lines.
    """
    last_position = 0
    start_time = time.time()

    while time.time() - start_time < config.timeout_sec:
        # Check if tmux session still alive (early crash detection)
        alive, err = _is_session_alive(client, config.session_name)
        if err:
            return False, None, f"Session check failed: {err}"

        # Stream new log content
        new_content, new_pos, err = _tail_log_from_position(
            client, config.log_file, last_position
        )
        if err:
            logger.warning(f"⚠️  Log tail failed: {err}")
        elif new_content:
            # Print immediately with flush for real-time feedback
            print(new_content, end='', flush=True)
            last_position = new_pos

            # Check for stop marker if configured
            if config.stop_on_marker and config.stop_on_marker in new_content:
                # Extract exit code from marker line
                exit_code = _extract_exit_code_from_content(new_content, config.stop_on_marker)
                if exit_code == 0:
                    return True, exit_code, None
                else:
                    return False, exit_code, f"Exit code {exit_code}"

        # If session died, extract exit code and return
        if not alive:
            exit_code = _extract_exit_code_from_log(client, config.log_file)
            if exit_code is not None:
                if exit_code == 0:
                    return True, exit_code, None
                else:
                    return False, exit_code, f"Exit code {exit_code}"
            else:
                return False, None, "Session exited but no exit code found"

        time.sleep(config.poll_interval_sec)

    # Timeout - kill session
    _kill_session(client, config.session_name)
    return False, None, f"Timeout after {config.timeout_sec}s"


def _is_session_alive(
    client: "BifrostClient",
    session_name: str,
) -> tuple[bool, str | None]:
    """Check if tmux session is still running.

    Tiger Style: < 70 lines, tuple return for error.

    Returns:
        (alive, error_message)
    """
    result = client.exec(f"tmux has-session -t {session_name} 2>&1")
    if result.exit_code == 0:
        return True, None
    else:
        # Session doesn't exist (may have exited)
        return False, None


def _tail_log_from_position(
    client: "BifrostClient",
    log_file: str,
    last_position: int,
) -> tuple[str, int, str | None]:
    """Tail log file from byte position.

    Position tracking enables efficient streaming - only fetch new content.
    Pattern from clicker:490f385.

    Tiger Style: < 70 lines, tuple return.

    Returns:
        (content, new_position, error_message)
    """
    # tail -c +N reads from byte N (1-indexed)
    tail_cmd = f"tail -c +{last_position + 1} {log_file} 2>/dev/null || true"
    result = client.exec(tail_cmd)

    if result.stdout:
        new_content = result.stdout
        new_position = last_position + len(new_content.encode('utf-8'))
        return new_content, new_position, None
    else:
        # No new content
        return "", last_position, None


def _extract_exit_code_from_log(
    client: "BifrostClient",
    log_file: str,
) -> int | None:
    """Extract exit code from log file.

    Looks for "EXIT_CODE: N" or "EVAL_EXIT_CODE: N" marker.
    Pattern from clicker:170a21c.

    Tiger Style: < 70 lines, explicit None return.

    Returns:
        Exit code as int, or None if not found
    """
    # Look for exit code marker (supports both formats)
    exit_code_cmd = (
        f"grep -E '(EXIT_CODE:|EVAL_EXIT_CODE:)' {log_file} 2>/dev/null | "
        f"tail -1 | awk '{{print $NF}}'"
    )
    result = client.exec(exit_code_cmd)

    if result.stdout:
        exit_code_str = result.stdout.strip()
        if exit_code_str.isdigit():
            return int(exit_code_str)

    return None


def _extract_exit_code_from_content(
    content: str,
    marker: str,
) -> int:
    """Extract exit code from content containing marker.

    Args:
        content: Log content
        marker: Marker string (e.g., "EVAL_EXIT_CODE: 0")

    Returns:
        Exit code (0 if can't parse)
    """
    # Find line with marker
    for line in content.splitlines():
        if marker in line:
            # Extract number after colon
            parts = line.split(':')
            if len(parts) >= 2:
                code_str = parts[-1].strip()
                if code_str.isdigit():
                    return int(code_str)
    return 0


def _kill_session(client: "BifrostClient", session_name: str) -> None:
    """Kill tmux session (timeout cleanup).

    Tiger Style: < 70 lines, explicit cleanup.
    """
    client.exec(f"tmux kill-session -t {session_name} 2>/dev/null || true")
    logger.warning(f"🛑 Killed session: {session_name}")


def stream_log_with_condition(
    client: "BifrostClient",
    log_file: str,
    stop_condition: Callable[[str], bool],
    timeout_sec: int = 300,
    poll_interval_sec: float = 2.0,
) -> tuple[bool, str | None]:
    """Stream log until custom stop condition met.

    More flexible variant - user provides stop condition function.
    Casey: Redundancy principle - provide multiple ways to do same thing.

    Args:
        client: BifrostClient instance
        log_file: Absolute path to log file
        stop_condition: Function that takes log content and returns True when done
        timeout_sec: Maximum time to wait
        poll_interval_sec: How often to poll

    Returns:
        (success, error_message)

    Example:
        # Stop when "Training complete" appears
        success, err = stream_log_with_condition(
            client,
            "train.log",
            lambda content: "Training complete" in content,
            timeout_sec=3600,
        )
    """
    assert client is not None, "BifrostClient required"
    assert log_file, "log_file required"
    assert stop_condition is not None, "stop_condition required"
    assert timeout_sec > 0, "timeout_sec must be positive"

    last_position = 0
    start_time = time.time()

    while time.time() - start_time < timeout_sec:
        content, new_pos, err = _tail_log_from_position(client, log_file, last_position)

        if err:
            logger.warning(f"⚠️  Log tail failed: {err}")
        elif content:
            print(content, end='', flush=True)
            last_position = new_pos

            if stop_condition(content):
                return True, None

        time.sleep(poll_interval_sec)

    return False, f"Timeout after {timeout_sec}s"
