"""External agent drivers.

Drive external agents (Claude Code, Codex, etc.) and emit StreamEvents
that can be consumed by any Frontend.

Usage:
    from rollouts.drivers import ClaudeDriver, run_driver_to_trajectory

    driver = ClaudeDriver(cwd="/path/to/repo")

    # Option 1: Stream events directly
    async for event in driver.run("Fix the bug in auth.py"):
        await frontend.handle_event(event)

    # Option 2: Capture as trajectory (with logging for eval TUI)
    trajectory = await run_driver_to_trajectory(driver, "Fix the bug", sample_id="001")

Session adapters for hot-swap:
    from rollouts.drivers import (
        messages_to_claude_session,
        claude_session_to_messages,
        write_claude_session,
    )
"""

from .claude import ClaudeDriver
from .codex import CodexDriver
from .protocol import ExternalAgentDriver
from .runner import run_driver_to_trajectory, run_external_agent
from .session_adapter import (
    claude_session_to_messages,
    codex_session_to_messages,
    find_claude_session,
    find_codex_session,
    get_claude_session_path,
    get_codex_session_path,
    # Claude Code adapters
    messages_to_claude_session,
    # Codex adapters
    messages_to_codex_session,
    write_claude_session,
    write_codex_session,
)

__all__ = [
    "ExternalAgentDriver",
    "ClaudeDriver",
    "CodexDriver",
    "run_external_agent",
    "run_driver_to_trajectory",
    # Claude Code session adapters
    "messages_to_claude_session",
    "claude_session_to_messages",
    "write_claude_session",
    "find_claude_session",
    "get_claude_session_path",
    # Codex session adapters
    "messages_to_codex_session",
    "codex_session_to_messages",
    "write_codex_session",
    "find_codex_session",
    "get_codex_session_path",
]
