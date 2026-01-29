"""Differential terminal renderer.

Extracted from rollouts/frontends/tui/tui.py _do_render() - same algorithm.

Compares new lines against previous lines and only updates changed regions.
Uses synchronized output mode to prevent flicker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .terminal import (
    CLEAR_LINE_FULL,
    CLEAR_SCREEN_HOME,
    SYNC_OUTPUT_OFF,
    SYNC_OUTPUT_ON,
    Terminal,
)
from .text import truncate_to_width, visible_width


@dataclass
class RenderState:
    """Tracks state between renders for differential updates."""

    previous_lines: list[str] = field(default_factory=list)
    previous_width: int = 0
    cursor_row: int = 0  # 0-indexed, relative to first line


def diff_render(
    terminal: Terminal,
    new_lines: list[str],
    state: RenderState,
) -> None:
    """Render lines to terminal with differential updates.

    Only redraws lines that changed since last render.
    Uses synchronized output mode (\x1b[?2026h/l) to prevent flicker.

    Args:
        terminal: Terminal to write to.
        new_lines: Complete list of lines to display.
        state: Mutable render state (updated in place).
    """
    width = terminal.columns
    height = terminal.rows

    # Width changed - need full re-render
    width_changed = state.previous_width != 0 and state.previous_width != width

    # First render - just output everything
    if len(state.previous_lines) == 0:
        buffer = SYNC_OUTPUT_ON
        for i, line in enumerate(new_lines):
            if i > 0:
                buffer += "\r\n"
            buffer += line
        buffer += SYNC_OUTPUT_OFF
        terminal.write(buffer)
        state.cursor_row = len(new_lines) - 1
        state.previous_lines = list(new_lines)
        state.previous_width = width
        return

    # Width changed - full re-render
    if width_changed:
        buffer = SYNC_OUTPUT_ON
        buffer += CLEAR_SCREEN_HOME
        for i, line in enumerate(new_lines):
            if i > 0:
                buffer += "\r\n"
            buffer += line
        buffer += SYNC_OUTPUT_OFF
        terminal.write(buffer)
        state.cursor_row = len(new_lines) - 1
        state.previous_lines = list(new_lines)
        state.previous_width = width
        return

    # Find first changed line
    first_changed = -1
    max_lines = max(len(new_lines), len(state.previous_lines))

    for i in range(max_lines):
        old_line = state.previous_lines[i] if i < len(state.previous_lines) else ""
        new_line = new_lines[i] if i < len(new_lines) else ""
        if old_line != new_line:
            first_changed = i
            break

    # No changes
    if first_changed == -1:
        return

    # Check if first_changed is outside the viewport
    viewport_top = state.cursor_row - height + 1
    if first_changed < viewport_top:
        # First change is above viewport - need full re-render
        buffer = SYNC_OUTPUT_ON
        buffer += CLEAR_SCREEN_HOME
        for i, line in enumerate(new_lines):
            if i > 0:
                buffer += "\r\n"
            buffer += line
        buffer += SYNC_OUTPUT_OFF
        terminal.write(buffer)
        state.cursor_row = len(new_lines) - 1
        state.previous_lines = list(new_lines)
        state.previous_width = width
        return

    # Render from first changed line to end
    buffer = SYNC_OUTPUT_ON

    # Move cursor to first changed line
    line_diff = first_changed - state.cursor_row
    if line_diff > 0:
        buffer += f"\x1b[{line_diff}B"  # Move down
    elif line_diff < 0:
        buffer += f"\x1b[{-line_diff}A"  # Move up

    buffer += "\r"  # Move to column 0

    # Render from first changed line to end, clearing each line before writing
    cursor_after_render = first_changed
    for i in range(first_changed, len(new_lines)):
        if i > first_changed:
            buffer += "\r\n"
            cursor_after_render = i
        buffer += CLEAR_LINE_FULL

        line = new_lines[i]
        if visible_width(line) > width:
            line = truncate_to_width(line, width, ellipsis="\u2026")
        buffer += line

    # If we had more lines before, clear them
    if len(state.previous_lines) > len(new_lines):
        extra_lines = len(state.previous_lines) - len(new_lines)
        for _i in range(extra_lines):
            buffer += "\r\n" + CLEAR_LINE_FULL
        # Move cursor back to correct position
        lines_to_move_up = cursor_after_render + extra_lines - (len(new_lines) - 1)
        if lines_to_move_up > 0:
            buffer += f"\x1b[{lines_to_move_up}A"

    buffer += SYNC_OUTPUT_OFF

    terminal.write(buffer)
    state.cursor_row = len(new_lines) - 1
    state.previous_lines = list(new_lines)
    state.previous_width = width
