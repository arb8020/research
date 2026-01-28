"""Terminal abstraction for TUI.

Re-exports from pytui with rollouts-specific additions (session tracking).
"""

from __future__ import annotations

# Re-export everything from pytui
from pytui.terminal import (
    Terminal as _PytTerminal,
)

# Rollouts-specific: session ID tracking for crash reporting
_active_session_id: str | None = None


def set_active_session_id(session_id: str | None) -> None:
    """Set the active session ID for crash reporting."""
    global _active_session_id
    _active_session_id = session_id


# ProcessTerminal is the old name for pytui.Terminal (with bracketed paste, no alternate screen)
# Keep for backwards compat with interactive_agent.py
class ProcessTerminal(_PytTerminal):
    """Terminal configured for interactive agent use (bracketed paste, session tracking)."""

    def __init__(self) -> None:
        super().__init__(bracketed_paste=True)

    def stop(self) -> None:
        """Stop terminal and print session ID for crash recovery."""
        super().stop()
        # Print session ID on exit so user can resume
        if _active_session_id:
            print(f"\nSession: {_active_session_id}")
            print(f"Resume with: --session {_active_session_id}")
