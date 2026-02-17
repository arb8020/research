"""
Minimal TUI implementation with differential rendering.

Ported from pi-mono/packages/tui - same architecture, same visual output.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pytui.input import KeyPress
from pytui.text import slice_ansi

from .terminal import Terminal
from .theme import DARK_THEME, Theme
from .utils import truncate_to_width, visible_width

# Overlay anchor positions
OverlayAnchor = Literal[
    "center",
    "top-left",
    "top-center",
    "top-right",
    "left-center",
    "right-center",
    "bottom-left",
    "bottom-center",
    "bottom-right",
]

# Size value: int for absolute, "50%" for percentage
SizeValue = int | str


@dataclass
class OverlayMargin:
    """Margin configuration for overlays."""

    top: int = 0
    right: int = 0
    bottom: int = 0
    left: int = 0


@dataclass
class OverlayOptions:
    """Options for overlay positioning and sizing."""

    # Sizing
    width: SizeValue | None = None  # Default: min(80, available)
    min_width: int | None = None
    max_height: SizeValue | None = None

    # Positioning - anchor-based
    anchor: OverlayAnchor = "center"
    offset_x: int = 0
    offset_y: int = 0

    # Positioning - absolute/percentage (overrides anchor)
    row: SizeValue | None = None
    col: SizeValue | None = None

    # Margin from terminal edges
    margin: OverlayMargin | int = 0

    # Visibility callback: (width, height) -> bool
    visible: Callable[[int, int], bool] | None = None


class OverlayHandle:
    """Handle for controlling an overlay after creation."""

    def __init__(
        self,
        hide_fn: Callable[[], None],
        set_hidden_fn: Callable[[bool], None],
        is_hidden_fn: Callable[[], bool],
    ) -> None:
        self._hide = hide_fn
        self._set_hidden = set_hidden_fn
        self._is_hidden = is_hidden_fn

    def hide(self) -> None:
        """Permanently remove the overlay."""
        self._hide()

    def set_hidden(self, hidden: bool) -> None:
        """Temporarily show/hide the overlay."""
        self._set_hidden(hidden)

    def is_hidden(self) -> bool:
        """Check if overlay is temporarily hidden."""
        return self._is_hidden()


@dataclass
class _OverlayEntry:
    """Internal entry in the overlay stack."""

    component: Component
    options: OverlayOptions
    hidden: bool = False
    pre_focus: Component | None = None


class Component(ABC):
    """Base class for all TUI components."""

    @abstractmethod
    def render(self, width: int) -> list[str]:
        """Render the component to lines for the given viewport width.

        Args:
            width: Current viewport width

        Returns:
            Array of strings, each representing a line
        """
        ...

    def handle_input(self, msg: object) -> None:
        """Optional handler for terminal input when component has focus."""
        pass

    def invalidate(self) -> None:
        """Invalidate any cached rendering state.

        Called when theme changes or when component needs to re-render from scratch.
        """
        pass


class Container(Component):
    """A component that contains other components."""

    def __init__(self, debug_layout: bool = False) -> None:
        self.children: list[Component] = []
        self._debug_layout = debug_layout

    def add_child(self, component: Component) -> None:
        """Add a child component."""
        self.children.append(component)

    def remove_child(self, component: Component) -> None:
        """Remove a child component."""
        if component in self.children:
            self.children.remove(component)

    def clear(self) -> None:
        """Remove all children."""
        self.children.clear()

    def invalidate(self) -> None:
        """Invalidate all children."""
        for child in self.children:
            child.invalidate()

    def render(self, width: int) -> list[str]:
        """Render all children, concatenating their lines."""
        lines: list[str] = []
        for child in self.children:
            lines.extend(child.render(width))
        return lines


class TUI(Container):
    """Main class for managing terminal UI with differential rendering."""

    def __init__(
        self,
        terminal: Terminal,
        theme: Theme | None = None,
        debug: bool = False,
        debug_layout: bool = False,
    ) -> None:
        super().__init__()
        self._terminal = terminal
        self.theme = theme or DARK_THEME
        self._previous_lines: list[str] = []
        self._previous_width: int = 0
        self._focused_component: Component | None = None
        self._render_requested: bool = False
        self._cursor_row: int = 0  # Track where cursor is (0-indexed, relative to our first line)
        self._running: bool = False
        self._debug = debug
        self._debug_layout = debug_layout

        # Loader container - set by InteractiveAgentRunner to render loader in fixed location
        self._loader_container: Component | None = None
        self._loader_text: str | None = None
        self._animation_task_running: bool = False

        # Overlay stack for floating UI elements (dialogs, menus, etc.)
        self._overlay_stack: list[_OverlayEntry] = []

    def invalidate(self) -> None:
        """Invalidate all children and overlays."""
        super().invalidate()
        for entry in self._overlay_stack:
            entry.component.invalidate()

    def set_focus(self, component: Component | None) -> None:
        """Set the focused component for input handling."""
        self._focused_component = component

    # =========================================================================
    # Overlay System
    # =========================================================================

    def add_overlay(
        self,
        component: Component,
        options: OverlayOptions | None = None,
    ) -> OverlayHandle:
        """Show an overlay component with configurable positioning.

        Args:
            component: Component to show as overlay
            options: Positioning and sizing options

        Returns:
            Handle to control the overlay
        """
        entry = _OverlayEntry(
            component=component,
            options=options or OverlayOptions(),
            pre_focus=self._focused_component,
        )
        self._overlay_stack.append(entry)

        # Focus the overlay if visible
        if self._is_overlay_visible(entry):
            self.set_focus(component)

        self.request_render()

        # Create handle callbacks that capture this entry
        def hide() -> None:
            if entry in self._overlay_stack:
                self._overlay_stack.remove(entry)
                # Restore focus to next visible overlay or pre_focus
                top_visible = self._get_topmost_visible_overlay()
                self.set_focus(top_visible.component if top_visible else entry.pre_focus)
                self.request_render()

        def set_hidden(hidden: bool) -> None:
            entry.hidden = hidden
            if hidden and self._focused_component == entry.component:
                # Lost focus, find next target
                top_visible = self._get_topmost_visible_overlay()
                self.set_focus(top_visible.component if top_visible else entry.pre_focus)
            elif not hidden:
                # Restore focus to this overlay
                self.set_focus(entry.component)
            self.request_render()

        def is_hidden() -> bool:
            return entry.hidden

        return OverlayHandle(hide_fn=hide, set_hidden_fn=set_hidden, is_hidden_fn=is_hidden)

    def pop_overlay(self) -> None:
        """Hide the topmost overlay and restore previous focus."""
        if not self._overlay_stack:
            return
        entry = self._overlay_stack.pop()
        top_visible = self._get_topmost_visible_overlay()
        self.set_focus(top_visible.component if top_visible else entry.pre_focus)
        self.request_render()

    def has_visible_overlay(self) -> bool:
        """Check if there are any visible overlays."""
        return any(self._is_overlay_visible(e) for e in self._overlay_stack)

    def _is_overlay_visible(self, entry: _OverlayEntry) -> bool:
        """Check if an overlay entry is currently visible."""
        if entry.hidden:
            return False
        if entry.options.visible:
            return entry.options.visible(self._terminal.columns, self._terminal.rows)
        return True

    def _get_topmost_visible_overlay(self) -> _OverlayEntry | None:
        """Find the topmost visible overlay, if any."""
        for entry in reversed(self._overlay_stack):
            if self._is_overlay_visible(entry):
                return entry
        return None

    def _parse_size_value(self, value: SizeValue | None, reference: int) -> int | None:
        """Parse a size value (int or "50%") into absolute value."""
        if value is None:
            return None
        if isinstance(value, int):
            return value
        # Parse percentage string like "50%"
        match = re.match(r"^(\d+(?:\.\d+)?)%$", value)
        if match:
            return int(reference * float(match.group(1)) / 100)
        return None

    def _resolve_overlay_layout(
        self,
        options: OverlayOptions,
        overlay_height: int,
        term_width: int,
        term_height: int,
    ) -> tuple[int, int, int, int | None]:
        """Resolve overlay layout from options.

        Returns:
            (width, row, col, max_height)
        """
        # Parse margin
        margin = options.margin
        if isinstance(margin, int):
            margin = OverlayMargin(top=margin, right=margin, bottom=margin, left=margin)

        # Available space after margins
        avail_width = max(1, term_width - margin.left - margin.right)
        avail_height = max(1, term_height - margin.top - margin.bottom)

        # Resolve width
        width = self._parse_size_value(options.width, term_width)
        if width is None:
            width = min(80, avail_width)
        if options.min_width:
            width = max(width, options.min_width)
        width = max(1, min(width, avail_width))

        # Resolve max_height
        max_height = self._parse_size_value(options.max_height, term_height)
        if max_height is not None:
            max_height = max(1, min(max_height, avail_height))

        # Effective height for positioning (may be clamped by max_height)
        effective_height = min(overlay_height, max_height) if max_height else overlay_height

        # Resolve row position
        if options.row is not None:
            if isinstance(options.row, str):
                # Percentage: distribute in available space
                match = re.match(r"^(\d+(?:\.\d+)?)%$", options.row)
                if match:
                    max_row = max(0, avail_height - effective_height)
                    percent = float(match.group(1)) / 100
                    row = margin.top + int(max_row * percent)
                else:
                    # Invalid format, fall back to center
                    row = self._resolve_anchor_row(
                        "center", effective_height, avail_height, margin.top
                    )
            else:
                row = options.row
        else:
            row = self._resolve_anchor_row(
                options.anchor, effective_height, avail_height, margin.top
            )

        # Resolve col position
        if options.col is not None:
            if isinstance(options.col, str):
                match = re.match(r"^(\d+(?:\.\d+)?)%$", options.col)
                if match:
                    max_col = max(0, avail_width - width)
                    percent = float(match.group(1)) / 100
                    col = margin.left + int(max_col * percent)
                else:
                    col = self._resolve_anchor_col("center", width, avail_width, margin.left)
            else:
                col = options.col
        else:
            col = self._resolve_anchor_col(options.anchor, width, avail_width, margin.left)

        # Apply offsets
        row += options.offset_y
        col += options.offset_x

        # Clamp to bounds (respecting margins)
        row = max(margin.top, min(row, term_height - margin.bottom - effective_height))
        col = max(margin.left, min(col, term_width - margin.right - width))

        return (width, row, col, max_height)

    def _resolve_anchor_row(
        self, anchor: OverlayAnchor, height: int, avail_height: int, margin_top: int
    ) -> int:
        """Resolve row position from anchor."""
        if anchor in ("top-left", "top-center", "top-right"):
            return margin_top
        elif anchor in ("bottom-left", "bottom-center", "bottom-right"):
            return margin_top + avail_height - height
        else:  # center, left-center, right-center
            return margin_top + (avail_height - height) // 2

    def _resolve_anchor_col(
        self, anchor: OverlayAnchor, width: int, avail_width: int, margin_left: int
    ) -> int:
        """Resolve column position from anchor."""
        if anchor in ("top-left", "left-center", "bottom-left"):
            return margin_left
        elif anchor in ("top-right", "right-center", "bottom-right"):
            return margin_left + avail_width - width
        else:  # center, top-center, bottom-center
            return margin_left + (avail_width - width) // 2

    def _composite_line_at(
        self,
        base_line: str,
        overlay_line: str,
        start_col: int,
        overlay_width: int,
        total_width: int,
    ) -> str:
        """Splice overlay content into a base line at a specific column.

        This is the core compositing operation: given a base line and an overlay line,
        produce a result where the overlay appears at start_col with the specified width.

        Args:
            base_line: The underlying content line
            overlay_line: The overlay content to splice in
            start_col: Column where overlay starts (0-indexed)
            overlay_width: Width of the overlay region
            total_width: Total line width (terminal width)

        Returns:
            Composited line with overlay spliced in
        """
        RESET = "\x1b[0m"

        # Extract "before" segment (columns 0 to start_col)
        before = slice_ansi(base_line, 0, start_col) if start_col > 0 else ""
        before_width = visible_width(before)

        # Extract "after" segment (columns after overlay to end)
        after_start = start_col + overlay_width
        after_len = total_width - after_start
        after = slice_ansi(base_line, after_start, after_start + after_len) if after_len > 0 else ""
        after_width = visible_width(after)

        # Truncate overlay to declared width if needed
        if visible_width(overlay_line) > overlay_width:
            overlay_line = truncate_to_width(overlay_line, overlay_width)
        overlay_actual_width = visible_width(overlay_line)

        # Calculate padding for each segment
        before_pad = max(0, start_col - before_width)
        overlay_pad = max(0, overlay_width - overlay_actual_width)
        after_target = max(0, total_width - start_col - overlay_width)
        after_pad = max(0, after_target - after_width)

        # Compose result with resets between segments to prevent color bleeding
        result = (
            before
            + " " * before_pad
            + RESET
            + overlay_line
            + " " * overlay_pad
            + RESET
            + after
            + " " * after_pad
        )

        # Final safeguard: truncate to terminal width
        if visible_width(result) > total_width:
            result = truncate_to_width(result, total_width)

        return result

    def _composite_overlays(
        self,
        lines: list[str],
        term_width: int,
        term_height: int,
    ) -> list[str]:
        """Composite all overlays into content lines.

        Overlays are composited in stack order (later = on top).

        Args:
            lines: Base content lines
            term_width: Terminal width
            term_height: Terminal height

        Returns:
            Lines with overlays composited in
        """
        if not self._overlay_stack:
            return lines

        result = lines.copy()

        # Pre-render all visible overlays and calculate positions
        rendered: list[tuple[list[str], int, int, int]] = []  # (lines, row, col, width)
        min_lines_needed = len(result)

        for entry in self._overlay_stack:
            if not self._is_overlay_visible(entry):
                continue

            # Get layout (width/maxHeight don't depend on overlay height)
            width, _, _, max_height = self._resolve_overlay_layout(
                entry.options, 0, term_width, term_height
            )

            # Render overlay at calculated width
            overlay_lines = entry.component.render(width)

            # Apply max_height
            if max_height and len(overlay_lines) > max_height:
                overlay_lines = overlay_lines[:max_height]

            # Get final position with actual height
            _, row, col, _ = self._resolve_overlay_layout(
                entry.options, len(overlay_lines), term_width, term_height
            )

            rendered.append((overlay_lines, row, col, width))
            min_lines_needed = max(min_lines_needed, row + len(overlay_lines))

        # Extend result with empty lines if needed for overlay placement
        while len(result) < min_lines_needed:
            result.append("")

        # Calculate viewport start (what portion of content is visible)
        # This matches pi-mono: overlays are positioned relative to the viewport
        viewport_start = max(0, len(result) - term_height)

        # Composite each overlay
        for overlay_lines, row, col, width in rendered:
            for i, overlay_line in enumerate(overlay_lines):
                idx = viewport_start + row + i
                if 0 <= idx < len(result):
                    result[idx] = self._composite_line_at(
                        result[idx], overlay_line, col, width, term_width
                    )

        return result

    def start(self) -> None:
        """Start the TUI, enabling raw mode and input handling."""
        self._terminal.start(
            on_input=self._handle_input,
            on_resize=self.request_render,
        )
        self._terminal.hide_cursor()
        self._running = True
        self.request_render()

    def stop(self) -> None:
        """Stop the TUI and restore terminal state."""
        self._running = False
        self._terminal.show_cursor()
        self._terminal.stop()

    def set_loader_container(self, container: Component) -> None:
        """Set the loader container component.

        Args:
            container: Component that manages loader rendering (e.g., LoaderContainer)
        """
        self._loader_container = container

    def show_loader(
        self,
        text: str,
        spinner_color_fn: Callable[[str], str] = lambda x: x,
        text_color_fn: Callable[[str], str] = lambda x: x,
    ) -> None:
        """Show a loader with spinning animation.

        Args:
            text: Text to display after spinner (e.g. "Calling LLM...")
            spinner_color_fn: Function to colorize spinner (unused, kept for API compatibility)
            text_color_fn: Function to colorize text (unused, kept for API compatibility)
        """
        self._loader_text = text
        if self._loader_container and hasattr(self._loader_container, "set_loader"):
            self._loader_container.set_loader(text)
        self.request_render()

    def hide_loader(self) -> None:
        """Hide the loader."""
        self._loader_text = None
        if self._loader_container and hasattr(self._loader_container, "clear_loader"):
            self._loader_container.clear_loader()
        self.request_render()

    def is_loader_active(self) -> bool:
        """Check if loader is currently showing."""
        if self._loader_container and hasattr(self._loader_container, "is_active"):
            return self._loader_container.is_active()
        return False

    def get_loader_text(self) -> str | None:
        """Return the currently displayed loader text (if any)."""
        return self._loader_text

    async def run_animation_loop(self) -> None:
        """Run the animation timer loop.

        Call this as a background task. The loop runs continuously and triggers
        re-renders every 80ms when the loader is active.
        Why 80ms: matches pi-mono's animation interval, gives smooth 12.5fps animation.

        Usage with Trio:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(tui.run_animation_loop)
                # ... do other work ...
        """
        import trio

        self._animation_task_running = True
        try:
            while self._animation_task_running and self._running:
                await trio.sleep(0.08)  # 80ms
                # Only trigger re-render if loader is active
                if (
                    self._loader_container
                    and hasattr(self._loader_container, "is_active")
                    and self._loader_container.is_active()
                ):
                    self.request_render()
        finally:
            self._animation_task_running = False

    def stop_animation_loop(self) -> None:
        """Signal the animation loop to stop."""
        self._animation_task_running = False

    def render(self, width: int) -> list[str]:
        """Render all children with optional debug gutter."""
        # Debug mode: add single-character gutter showing component types
        if self._debug_layout:
            gutter_width = 2  # Just 1 char + space
            content_width = width - gutter_width

            # Component type to character mapping
            def get_component_char(component: object) -> str:
                name = type(component).__name__
                if name == "Spacer":
                    if hasattr(component, "_debug_label") and component._debug_label:
                        label = component._debug_label
                        if "thinking-to-text" in label:
                            return "·"  # Spacer between thinking and text
                        elif "before-thinking" in label:
                            return "·"  # Spacer before thinking
                    return "·"  # Generic spacer
                elif name == "Container":
                    return "C"
                elif name == "UserMessage":
                    return "U"
                elif name == "AssistantMessage":
                    return "A"
                elif name == "ToolExecution":
                    return "T"
                elif name == "Input":
                    return "I"
                elif name == "LoaderContainer":
                    return "L"
                elif name == "Markdown":
                    return "M"
                elif name == "Text":
                    return "t"
                else:
                    return "?"

            def render_with_gutter(
                component: object, width: int, recurse_containers: bool = True
            ) -> list[str]:
                """Recursively render component and its children with gutter."""
                # Special handling for plain Container: render its children individually
                # But don't recurse into component containers like UserMessage, AssistantMessage
                component_name = type(component).__name__
                is_plain_container = (
                    isinstance(component, Container) and component_name == "Container"
                )

                if is_plain_container and recurse_containers:
                    result = []
                    for child in component.children:
                        result.extend(render_with_gutter(child, width, recurse_containers=True))
                    return result
                else:
                    # Render component as a single unit
                    char = get_component_char(component)
                    component_lines = component.render(width)
                    return [char + " " + line for line in component_lines]

            all_lines: list[str] = []
            for child in self.children:
                all_lines.extend(render_with_gutter(child, content_width, recurse_containers=True))

            lines = all_lines
        else:
            lines = super().render(width)

        return lines

    def request_render(self) -> None:
        """Request a render on the next tick.

        Multiple requests are coalesced into a single render.
        No-op if TUI hasn't started yet (avoids partial renders during setup).
        """
        if not self._running:
            return
        if self._render_requested:
            return
        self._render_requested = True
        # In Python we do immediate render since we don't have process.nextTick
        # For async usage, caller should await trio.sleep(0) or similar
        import time

        start = time.perf_counter()
        self._do_render()
        elapsed = time.perf_counter() - start
        # Always log slow renders (>100ms) to help debug hangs
        if elapsed > 0.1:
            self._log_slow_render(elapsed)
        self._render_requested = False

    def reset_render_state(self) -> None:
        """Reset render state to force complete re-render.

        Call this when returning from external editor or after terminal
        state has been disrupted.
        """
        self._previous_lines = []
        self._previous_width = 0
        self._cursor_row = 0
        # Also invalidate all component caches
        self.invalidate()

    def _log_slow_render(self, elapsed: float) -> None:
        """Log slow render to debug file (always, regardless of --debug flag)."""
        log_path = Path.home() / ".rollouts" / "tui-debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} SLOW_RENDER: {elapsed:.3f}s\n")
            f.write(f"  component_count={len(self.children)}\n")
            f.write(f"  previous_lines={len(self._previous_lines)}\n")

    def _handle_input(self, msg: object) -> None:
        """Handle input, routing to topmost overlay or focused component."""
        if isinstance(msg, str):
            msg = KeyPress(key=msg)

        # Ctrl+C (ASCII 3) should be handled by the application layer.
        if isinstance(msg, KeyPress) and msg.key and ord(msg.key[0]) == 3:
            return

        # Route to topmost visible overlay first, otherwise to focused component
        target: Component | None = None
        if self._overlay_stack:
            top_overlay = self._get_topmost_visible_overlay()
            if top_overlay:
                target = top_overlay.component

        if target is None:
            target = self._focused_component

        if target is not None:
            target.handle_input(msg)
            self.request_render()

    def _debug_log(self, msg: str) -> None:
        """Write debug message to log file."""
        if not self._debug:
            return
        log_path = Path.home() / ".rollouts" / "tui-debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} {msg}\n")

    def _do_render(self) -> None:
        """Perform the actual render with differential updates."""
        width = self._terminal.columns
        height = self._terminal.rows

        # Render all components to get new lines
        # No viewport truncation - render everything and let terminal scrollback handle history
        new_lines = self.render(width)

        # Composite overlays into the rendered lines (before differential compare)
        if self._overlay_stack:
            new_lines = self._composite_overlays(new_lines, width, height)

        # Width changed - need full re-render
        width_changed = self._previous_width != 0 and self._previous_width != width

        # First render - just output everything without clearing
        if len(self._previous_lines) == 0:
            buffer = "\x1b[?2026h"  # Begin synchronized output
            for i, line in enumerate(new_lines):
                if i > 0:
                    buffer += "\r\n"
                buffer += line
            buffer += "\x1b[?2026l"  # End synchronized output
            self._terminal.write(buffer)
            # After rendering N lines, cursor is at end of last line (line N-1)
            self._cursor_row = len(new_lines) - 1
            self._previous_lines = new_lines
            self._previous_width = width
            return

        # Width changed - full re-render
        if width_changed:
            buffer = "\x1b[?2026h"  # Begin synchronized output
            buffer += "\x1b[2J\x1b[H"  # Clear screen and home (preserves scrollback)
            for i, line in enumerate(new_lines):
                if i > 0:
                    buffer += "\r\n"
                buffer += line
            buffer += "\x1b[?2026l"  # End synchronized output
            self._terminal.write(buffer)
            self._cursor_row = len(new_lines) - 1
            self._previous_lines = new_lines
            self._previous_width = width
            return

        # Find first changed line
        first_changed = -1
        max_lines = max(len(new_lines), len(self._previous_lines))

        # Log when line count changes significantly (new content added)
        if abs(len(new_lines) - len(self._previous_lines)) >= 3:
            self._debug_log(
                f"LINE_COUNT_CHANGE prev={len(self._previous_lines)} new={len(new_lines)}"
            )
            # Dump last 10 lines of new content to see padding
            self._debug_log("=== LAST 10 NEW LINES ===")
            for i, line in enumerate(new_lines[-10:]):
                line_idx = len(new_lines) - 10 + i
                self._debug_log(f"  [{line_idx}] {repr(line[-30:])} (len={len(line)})")

        for i in range(max_lines):
            old_line = self._previous_lines[i] if i < len(self._previous_lines) else ""
            new_line = new_lines[i] if i < len(new_lines) else ""

            if old_line != new_line:
                if first_changed == -1:
                    first_changed = i
                    self._debug_log(
                        f"first_changed={i} old={repr(old_line[:80])} new={repr(new_line[:80])}"
                    )

        # No changes
        if first_changed == -1:
            return

        # Check if first_changed is outside the viewport
        # cursor_row is the line where cursor is (0-indexed)
        # Viewport shows lines from (cursor_row - height + 1) to cursor_row
        # If first_changed < viewport_top, we need full re-render
        viewport_top = self._cursor_row - height + 1
        if first_changed < viewport_top:
            # First change is above viewport - need full re-render
            buffer = "\x1b[?2026h"  # Begin synchronized output
            buffer += "\x1b[2J\x1b[H"  # Clear screen and home (preserves scrollback)
            for i, line in enumerate(new_lines):
                if i > 0:
                    buffer += "\r\n"
                buffer += line
            buffer += "\x1b[?2026l"  # End synchronized output
            self._terminal.write(buffer)
            self._cursor_row = len(new_lines) - 1
            self._previous_lines = new_lines
            self._previous_width = width
            return

        # Render from first changed line to end
        buffer = "\x1b[?2026h"  # Begin synchronized output

        # Move cursor to first changed line
        line_diff = first_changed - self._cursor_row
        self._debug_log(
            f"CURSOR_MOVE cursor_row={self._cursor_row} first_changed={first_changed} line_diff={line_diff} total_lines={len(new_lines)} term_height={height}"
        )
        if line_diff > 0:
            buffer += f"\x1b[{line_diff}B"  # Move down
        elif line_diff < 0:
            buffer += f"\x1b[{-line_diff}A"  # Move up

        buffer += "\r"  # Move to column 0

        # Render from first changed line to end, clearing each line before writing
        # Track where cursor ends up after rendering. This is needed because when we
        # clear extra lines below, we need to know the actual cursor position - NOT
        # first_changed, which was a bug that caused ghost artifacts when content shrank
        # (e.g., when loader hides after streaming completes).
        cursor_after_render = first_changed  # Start at first_changed
        for i in range(first_changed, len(new_lines)):
            if i > first_changed:
                buffer += "\r\n"
                cursor_after_render = i
            buffer += "\x1b[2K"  # Clear current line

            line = new_lines[i]
            if visible_width(line) > width:
                # Truncate oversized lines (e.g. progress bars from bash output)
                line = truncate_to_width(line, width, ellipsis="…")
            buffer += line

        # If we had more lines before, clear them
        if len(self._previous_lines) > len(new_lines):
            extra_lines = len(self._previous_lines) - len(new_lines)
            # After render loop, cursor is at cursor_after_render (or first_changed if loop didn't run)
            # We need to move down to clear the extra lines, then back up
            for i in range(extra_lines):
                buffer += "\r\n\x1b[2K"
            # Cursor is now at cursor_after_render + extra_lines
            # We need to end at len(new_lines) - 1
            lines_to_move_up = cursor_after_render + extra_lines - (len(new_lines) - 1)
            if lines_to_move_up > 0:
                buffer += f"\x1b[{lines_to_move_up}A"

        buffer += "\x1b[?2026l"  # End synchronized output

        # Write entire buffer at once
        self._terminal.write(buffer)

        # Cursor is now at end of last line
        self._cursor_row = len(new_lines) - 1

        self._previous_lines = new_lines
        self._previous_width = width
