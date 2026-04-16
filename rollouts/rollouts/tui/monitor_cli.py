"""Compatibility shim for the moved monitor CLI implementation.

The monitor CLI is not TUI-specific. Keep this import path temporarily so
existing callers do not break while the module settles under `rollouts.monitor`.
"""

from rollouts.monitor.cli import *  # noqa: F401,F403
