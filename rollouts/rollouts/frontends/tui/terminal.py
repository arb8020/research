"""Terminal abstraction for TUI.

Re-exports from pytui with rollouts-specific additions (session tracking).
"""

from __future__ import annotations

# Re-export everything from pytui
from pytui.terminal import Terminal

# Rollouts-specific: session ID tracking for crash reporting
_active_session_id: str | None = None


def set_active_session_id(session_id: str | None) -> None:
    """Set the active session ID for crash reporting."""
    global _active_session_id
    _active_session_id = session_id


# ProcessTerminal is the old name for pytui.Terminal (with bracketed paste, no alternate screen)
# Keep for backwards compat with interactive_agent.py
class ProcessTerminal(Terminal):
    """Terminal configured for interactive agent use (bracketed paste, session tracking)."""

    def __init__(self) -> None:
        super().__init__(bracketed_paste=True)

    def stop(self) -> None:
        """Stop terminal."""
        super().stop()
        # Note: session info is printed by the caller (runner or interactive_agent)
        # to avoid duplicate prints from multiple cleanup paths
