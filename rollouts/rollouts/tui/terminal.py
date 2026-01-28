"""Terminal abstraction for monitor TUI.

Re-exports from pytui.terminal - all code now lives there.
The monitor uses alternate_screen=True by default.
"""

from pytui.terminal import Terminal

__all__ = ["Terminal"]
