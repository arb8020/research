# Overlay System Implementation

This document describes the implementation of an overlay/modal system for the rollouts TUI, based on pi-mono's design.

## Overview

The overlay system enables floating UI elements (dialogs, menus, confirmations) that render on top of the main content without disrupting the message flow.

## API Design

```python
from dataclasses import dataclass
from typing import Callable, Literal

OverlayAnchor = Literal[
    "center",
    "top-left", "top-center", "top-right",
    "left-center", "right-center",
    "bottom-left", "bottom-center", "bottom-right",
]

SizeValue = int | str  # int for absolute, "50%" for percentage

@dataclass
class OverlayMargin:
    top: int = 0
    right: int = 0
    bottom: int = 0
    left: int = 0

@dataclass
class OverlayOptions:
    # Sizing
    width: SizeValue | None = None          # Default: min(80, available)
    min_width: int | None = None
    max_height: SizeValue | None = None

    # Positioning - anchor-based
    anchor: OverlayAnchor = "center"
    offset_x: int = 0
    offset_y: int = 0

    # Positioning - absolute/percentage
    row: SizeValue | None = None            # Overrides anchor vertical
    col: SizeValue | None = None            # Overrides anchor horizontal

    # Margin from terminal edges
    margin: OverlayMargin | int = 0

    # Visibility callback
    visible: Callable[[int, int], bool] | None = None  # (width, height) -> bool

class OverlayHandle:
    """Handle for controlling an overlay after creation."""

    def hide(self) -> None:
        """Permanently remove the overlay."""
        ...

    def set_hidden(self, hidden: bool) -> None:
        """Temporarily show/hide the overlay."""
        ...

    def is_hidden(self) -> bool:
        """Check if overlay is temporarily hidden."""
        ...
```

## Usage Examples

### Confirmation Dialog

```python
# Create a confirmation dialog centered with 60% width
dialog = ConfirmDialog("Delete session?", "This cannot be undone.")
handle = tui.add_overlay(dialog, OverlayOptions(
    anchor="center",
    width="60%",
    max_height="50%",
))

# When user confirms/cancels
handle.hide()
```

### Session Selector

```python
# Session browser at top with margin
selector = SessionSelector(sessions)
handle = tui.add_overlay(selector, OverlayOptions(
    anchor="top-center",
    width="80%",
    max_height="70%",
    margin=OverlayMargin(top=2, bottom=2),
))
```

### Autocomplete Dropdown

```python
# Positioned relative to cursor (using absolute positioning)
dropdown = AutocompleteList(completions)
handle = tui.add_overlay(dropdown, OverlayOptions(
    row=cursor_row + 1,  # Below cursor
    col=cursor_col,      # Aligned with cursor
    width=40,
    max_height=10,
))
```

## Implementation

### 1. Add overlay stack to TUI

```python
@dataclass
class OverlayEntry:
    component: Component
    options: OverlayOptions
    hidden: bool = False
    pre_focus: Component | None = None  # Component to restore focus to

class TUI(Container):
    def __init__(self, ...):
        ...
        self._overlay_stack: list[OverlayEntry] = []
```

### 2. Add/remove overlays

```python
def add_overlay(
    self,
    component: Component,
    options: OverlayOptions | None = None,
) -> OverlayHandle:
    """Show an overlay component with configurable positioning."""
    entry = OverlayEntry(
        component=component,
        options=options or OverlayOptions(),
        pre_focus=self._focused_component,
    )
    self._overlay_stack.append(entry)

    # Focus the overlay if visible
    if self._is_overlay_visible(entry):
        self.set_focus(component)

    self.request_render()

    # Return handle for controlling this overlay
    def hide():
        if entry in self._overlay_stack:
            self._overlay_stack.remove(entry)
            # Restore focus
            top_visible = self._get_topmost_visible_overlay()
            self.set_focus(
                top_visible.component if top_visible else entry.pre_focus
            )
            self.request_render()

    def set_hidden(hidden: bool):
        entry.hidden = hidden
        if hidden and self._focused_component == entry.component:
            top_visible = self._get_topmost_visible_overlay()
            self.set_focus(
                top_visible.component if top_visible else entry.pre_focus
            )
        elif not hidden:
            self.set_focus(entry.component)
        self.request_render()

    return OverlayHandle(hide=hide, set_hidden=set_hidden, is_hidden=lambda: entry.hidden)

def pop_overlay(self) -> None:
    """Hide the topmost overlay and restore previous focus."""
    if not self._overlay_stack:
        return
    entry = self._overlay_stack.pop()
    top_visible = self._get_topmost_visible_overlay()
    self.set_focus(top_visible.component if top_visible else entry.pre_focus)
    self.request_render()
```

### 3. Visibility helpers

```python
def _is_overlay_visible(self, entry: OverlayEntry) -> bool:
    """Check if an overlay entry is currently visible."""
    if entry.hidden:
        return False
    if entry.options.visible:
        return entry.options.visible(self._terminal.columns, self._terminal.rows)
    return True

def _get_topmost_visible_overlay(self) -> OverlayEntry | None:
    """Find the topmost visible overlay, if any."""
    for entry in reversed(self._overlay_stack):
        if self._is_overlay_visible(entry):
            return entry
    return None

def has_visible_overlay(self) -> bool:
    """Check if there are any visible overlays."""
    return any(self._is_overlay_visible(e) for e in self._overlay_stack)
```

### 4. Layout resolution

```python
def _parse_size_value(self, value: SizeValue | None, reference: int) -> int | None:
    """Parse a size value (int or "50%") into absolute value."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    # Parse percentage
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
) -> tuple[int, int, int, int | None]:  # (width, row, col, max_height)
    """Resolve overlay layout from options."""
    # Parse margin
    margin = options.margin
    if isinstance(margin, int):
        margin = OverlayMargin(top=margin, right=margin, bottom=margin, left=margin)

    # Available space
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

    # Effective height for positioning
    effective_height = min(overlay_height, max_height) if max_height else overlay_height

    # Resolve row position
    if options.row is not None:
        row = self._parse_size_value(options.row, avail_height) or 0
        if isinstance(options.row, str):
            # Percentage: distribute in available space
            max_row = max(0, avail_height - effective_height)
            percent = float(options.row.rstrip("%")) / 100
            row = margin.top + int(max_row * percent)
        else:
            row = options.row
    else:
        row = self._resolve_anchor_row(
            options.anchor, effective_height, avail_height, margin.top
        )

    # Resolve col position
    if options.col is not None:
        if isinstance(options.col, str):
            max_col = max(0, avail_width - width)
            percent = float(options.col.rstrip("%")) / 100
            col = margin.left + int(max_col * percent)
        else:
            col = options.col
    else:
        col = self._resolve_anchor_col(
            options.anchor, width, avail_width, margin.left
        )

    # Apply offsets
    row += options.offset_y
    col += options.offset_x

    # Clamp to bounds
    row = max(margin.top, min(row, term_height - margin.bottom - effective_height))
    col = max(margin.left, min(col, term_width - margin.right - width))

    return (width, row, col, max_height)

def _resolve_anchor_row(
    self, anchor: OverlayAnchor, height: int, avail_height: int, margin_top: int
) -> int:
    if anchor in ("top-left", "top-center", "top-right"):
        return margin_top
    elif anchor in ("bottom-left", "bottom-center", "bottom-right"):
        return margin_top + avail_height - height
    else:  # center variants
        return margin_top + (avail_height - height) // 2

def _resolve_anchor_col(
    self, anchor: OverlayAnchor, width: int, avail_width: int, margin_left: int
) -> int:
    if anchor in ("top-left", "left-center", "bottom-left"):
        return margin_left
    elif anchor in ("top-right", "right-center", "bottom-right"):
        return margin_left + avail_width - width
    else:  # center variants
        return margin_left + (avail_width - width) // 2
```

### 5. Compositing overlays into rendered content

```python
def _composite_overlays(
    self,
    lines: list[str],
    term_width: int,
    term_height: int,
) -> list[str]:
    """Composite all overlays into content lines."""
    if not self._overlay_stack:
        return lines

    result = lines.copy()

    # Pre-render all visible overlays
    rendered: list[tuple[list[str], int, int, int]] = []  # (lines, row, col, width)
    min_lines_needed = len(result)

    for entry in self._overlay_stack:
        if not self._is_overlay_visible(entry):
            continue

        # Get layout (width, maxHeight don't depend on overlay height)
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

    # Extend result with empty lines if needed
    while len(result) < min_lines_needed:
        result.append("")

    # Calculate viewport start (what portion of content is visible)
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
```

### 6. Line compositing (splice overlay into base line)

```python
def _composite_line_at(
    self,
    base_line: str,
    overlay_line: str,
    start_col: int,
    overlay_width: int,
    total_width: int,
) -> str:
    """Splice overlay content into a base line at a specific column."""
    from pytui.text import slice_ansi, visible_width, truncate_to_width

    RESET = "\x1b[0m"

    # Extract segments: before overlay, after overlay
    before = slice_ansi(base_line, 0, start_col) if start_col > 0 else ""
    before_width = visible_width(before)

    after_start = start_col + overlay_width
    after = slice_ansi(base_line, after_start, total_width - after_start)
    after_width = visible_width(after)

    # Truncate overlay to declared width
    if visible_width(overlay_line) > overlay_width:
        overlay_line = truncate_to_width(overlay_line, overlay_width)
    overlay_actual_width = visible_width(overlay_line)

    # Pad segments
    before_pad = max(0, start_col - before_width)
    overlay_pad = max(0, overlay_width - overlay_actual_width)
    after_target = max(0, total_width - start_col - overlay_width)
    after_pad = max(0, after_target - after_width)

    # Compose result with resets between segments
    result = (
        before + " " * before_pad + RESET +
        overlay_line + " " * overlay_pad + RESET +
        after + " " * after_pad
    )

    # Final safeguard: truncate to terminal width
    if visible_width(result) > total_width:
        result = truncate_to_width(result, total_width)

    return result
```

### 7. Integrate into render pipeline

Modify `_do_render` to composite overlays before differential comparison:

```python
def _do_render(self) -> None:
    width = self._terminal.columns
    height = self._terminal.rows

    # Render all components
    new_lines = self.render(width)

    # Composite overlays (NEW)
    if self._overlay_stack:
        new_lines = self._composite_overlays(new_lines, width, height)

    # ... rest of differential rendering logic unchanged ...
```

### 8. Route input to overlays

Modify `_handle_input` to route to topmost visible overlay:

```python
def _handle_input(self, data: str) -> None:
    # Ctrl+C handling unchanged
    if len(data) > 0 and ord(data[0]) == 3:
        return

    # Route to topmost visible overlay, or focused component
    target = None
    if self._overlay_stack:
        top_overlay = self._get_topmost_visible_overlay()
        if top_overlay:
            target = top_overlay.component

    if target is None:
        target = self._focused_component

    if target is not None:
        target.handle_input(data)
        self.request_render()
```

## Testing Strategy

### Unit Tests

1. **Layout resolution**: Test anchor positions, percentages, margins, offsets
2. **Line compositing**: Test ANSI preservation, CJK characters, edge cases
3. **Visibility**: Test hidden state, visibility callback

### Property-Based Tests (Hypothesis)

```python
from hypothesis import given, strategies as st

@given(
    base=st.text(min_size=0, max_size=100),
    overlay=st.text(min_size=0, max_size=50),
    col=st.integers(min_value=0, max_value=80),
    width=st.integers(min_value=1, max_value=80),
    total=st.integers(min_value=1, max_value=120),
)
def test_composite_never_exceeds_width(base, overlay, col, width, total):
    result = tui._composite_line_at(base, overlay, col, width, total)
    assert visible_width(result) <= total

@given(
    anchor=st.sampled_from(ANCHOR_VALUES),
    height=st.integers(min_value=1, max_value=50),
    term_height=st.integers(min_value=10, max_value=100),
)
def test_overlay_stays_in_bounds(anchor, height, term_height):
    options = OverlayOptions(anchor=anchor)
    _, row, _, _ = tui._resolve_overlay_layout(options, height, 80, term_height)
    assert 0 <= row <= term_height - height
```

### Integration Tests

1. Show overlay, verify it renders at correct position
2. Stack multiple overlays, verify compositing order
3. Hide/show overlay, verify focus restoration
4. Resize terminal, verify overlay repositions correctly

## Migration Path

1. Add overlay infrastructure (no breaking changes)
2. Create `ConfirmDialog` component using overlays
3. Create `SessionSelector` using overlays
4. Migrate existing dialogs to use overlays
5. Add autocomplete dropdown using overlays

## Open Questions

1. **Scroll handling**: When content scrolls, should overlays stay fixed or scroll with content? (pi-mono: fixed to viewport)
2. **Focus trapping**: Should Tab cycle within overlay or escape to parent? (pi-mono: escape with Escape key)
3. **Animation**: Should overlays animate in/out? (pi-mono: no animation)
