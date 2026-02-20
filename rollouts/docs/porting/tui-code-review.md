# TUI Code Review: pi-mono vs rollouts

Reviewing both codebases against the code style principles in `docs/code_style/`.

---

## Executive Summary

**pi-mono's TUI** is cleaner, more compressed, and easier to extend. It follows Casey Muratori's semantic compression principles well.

**rollouts' TUI** is a messier port with accumulated cruft, debug logging mixed into hot paths, and some structural problems that make it harder to reason about.

**Recommendation**: Create `pytui-imperative` as a clean port of pi-mono's TUI, then migrate rollouts to use it.

---

## Analysis by Principle

### 1. Semantic Compression (Casey Muratori)

> "Like a good compressor, I don't reuse anything until I have at least two instances of it occurring."

**pi-mono (good)**:
```typescript
// Clean, minimal overlay entry - just the data needed
private overlayStack: {
    component: Component;
    options?: OverlayOptions;
    preFocus: Component | null;
    hidden: boolean;
}[] = [];
```

**rollouts (worse)**:
```python
# Separate dataclass with default factory - more ceremony
@dataclass
class _OverlayEntry:
    component: Component
    options: OverlayOptions = field(default_factory=OverlayOptions)
    pre_focus: Component | None = None
    hidden: bool = False
```

The extra structure doesn't add value here. pi-mono's inline type is simpler.

### 2. State Ownership (Tiger Style)

> "Minimize owners of state and make it explicit wherever possible."

**pi-mono (good)**:
- Clear ownership: `TUI` owns `overlayStack`, `focusedComponent`, `previousLines`
- State transitions happen in one place (`handleInput`, `doRender`)
- No hidden state mutations

**rollouts (worse)**:
- `_debug_log()` method writes to files inline during render (side effect in render path!)
- Debug state scattered: `_cursor_row`, `_previous_lines`, `_previous_width`, `_render_requested`, etc.
- Some state is duplicated between `TUI` and `TUIFrontend`

```python
# rollouts: debug logging mixed into render hot path
def _do_render(self) -> None:
    ...
    if abs(len(new_lines) - len(self._previous_lines)) >= 3:
        self._debug_log(f"LINE_COUNT_CHANGE prev={len(self._previous_lines)}...")
        self._debug_log("=== LAST 10 NEW LINES ===")
        for i, line in enumerate(new_lines[-10:]):
            ...
```

This is **wrong** - debug logging should be a separate concern, not interleaved with rendering logic.

### 3. Function Shape (Tiger Style)

> "Good function shape is often the inverse of an hourglass: a few parameters, a simple return type, and a lot of meaty logic between the braces."

**pi-mono (good)**:
```typescript
// resolveOverlayLayout: clear inputs, clear output
private resolveOverlayLayout(
    options: OverlayOptions | undefined,
    overlayHeight: number,
    termWidth: number,
    termHeight: number,
): { width: number; row: number; col: number; maxHeight: number | undefined }
```

**rollouts (worse)**:
```python
# _resolve_overlay_layout: incomplete, returns partial data
def _resolve_overlay_layout(
    self,
    options: OverlayOptions,
    overlay_height: int,
    term_width: int,
    term_height: int,
) -> tuple[int, int, int, int | None]:
    """Returns: (width, row, col, max_height)"""
    ...
    # Function is cut off - doesn't complete the return!
```

The rollouts version is incomplete and harder to read.

### 4. Control Flow Centralization (Tiger Style)

> "Push ifs up and fors down."

**pi-mono (good)**:
```typescript
// handleInput: all branching in one place
private handleInput(data: string): void {
    // Input listeners pipeline
    if (this.inputListeners.size > 0) { ... }

    // Cell size response handling
    if (this.cellSizeQueryPending) { ... }

    // Debug key
    if (matchesKey(data, "shift+ctrl+d") && this.onDebug) { ... }

    // Overlay visibility check
    const focusedOverlay = this.overlayStack.find(...);
    if (focusedOverlay && !this.isOverlayVisible(focusedOverlay)) { ... }

    // Pass to focused component
    if (this.focusedComponent?.handleInput) { ... }
}
```

**rollouts (scattered)**:
Input handling is split across `TUIFrontend._run_input_loop()`, `TUI.handle_input()`, and individual components. The flow is harder to trace.

### 5. Granularity (Casey Muratori)

> "Never supply a higher-level function that can't be trivially replaced by a few lower-level functions."

**pi-mono (good)**:
- `Component.render(width)` → returns lines
- `TUI.doRender()` → calls `render()`, then `compositeOverlays()`, then diff-updates terminal
- Each level is independently usable

**rollouts (okay but messier)**:
- Same layering exists, but with more ceremony
- `_composite_overlays` has the same logic but more verbose

### 6. Assertions (Tiger Style)

> "Assert all function arguments and return values, pre/postconditions and invariants."

**pi-mono**: No assertions (TypeScript culture, relies on types)

**rollouts**: No assertions either, but Python would benefit from them more.

Both are weak here.

### 7. Naming

**pi-mono (clearer)**:
- `showOverlay`, `hideOverlay`, `hasOverlay`
- `resolveAnchorRow`, `resolveAnchorCol`
- `compositeOverlays`
- `requestRender`

**rollouts (more verbose, less clear)**:
- `add_overlay`, `pop_overlay`, `has_visible_overlay`
- `_resolve_overlay_layout` (does both row and col)
- `_composite_overlays`
- `request_render`

The rollouts names have underscores for "private" but this adds visual noise without semantic value.

---

## Specific Problems in rollouts/tui.py

### Problem 1: Debug logging in hot path

```python
def _do_render(self) -> None:
    ...
    # This runs on EVERY render
    if abs(len(new_lines) - len(self._previous_lines)) >= 3:
        self._debug_log(...)  # File I/O during render!
```

**Fix**: Remove debug logging from render path entirely. If debugging is needed, use a separate debug mode that instruments externally.

### Problem 2: Incomplete functions

`_resolve_overlay_layout` appears to be incomplete - it starts computing row position but the file read cut off before the return statement.

### Problem 3: Regex parsing on every render

```python
def _parse_size_value(self, value: SizeValue | None, reference: int) -> int | None:
    ...
    match = re.match(r"^(\d+(?:\.\d+)?)%$", value)
```

This regex runs every time an overlay is positioned. Should be parsed once when the option is set.

### Problem 4: Duplicated rendering logic

Both `pytui/renderer.py` and `rollouts/frontends/tui/tui.py` have differential rendering. They should be unified.

### Problem 5: Too many abstraction layers

```
TUIFrontend → TUI → AgentRenderer → Components
```

vs pi-mono:
```
InteractiveMode → TUI → Components
```

The extra `AgentRenderer` layer adds indirection without clear benefit.

---

## What pi-mono Has That's Missing

| Feature | Benefit |
|---------|---------|
| Kitty keyboard protocol | Unambiguous key detection, modifiers work reliably |
| Hardware cursor positioning (CURSOR_MARKER) | IME works correctly for CJK input |
| Input listeners pipeline | Clean way to intercept/modify input |
| Cell size query | Required for image rendering |
| `requestRender(force=true)` | Clean way to force full redraw |
| `wantsKeyRelease` on components | Opt-in for key release events |

---

## Recommendation: pytui-imperative

Create a new `pytui/tui.py` that is a **direct port** of pi-mono's `tui.ts`:

1. **Same structure**: `Component`, `Container`, `TUI` classes with same methods
2. **Same differential rendering**: Port `doRender()` directly
3. **Same overlay system**: Port `showOverlay`, `hideOverlay`, positioning
4. **Same input handling**: Port `handleInput` with listeners
5. **Add Kitty protocol support** (optional, can be incremental)

Then migrate rollouts to import from `pytui.tui` instead of having its own copy.

### File structure

```
pytui/
├── terminal.py      # Keep as-is
├── input.py         # Keep as-is
├── text.py          # Keep as-is
├── renderer.py      # Delete - fold into tui.py
├── theme.py         # Keep as-is
├── app.py           # Keep as-is (Elm runtime)
└── tui.py           # NEW - port of pi-mono/tui.ts
    ├── Component (interface)
    ├── Focusable (interface)
    ├── Container
    └── TUI (main class)
```

### Port priority

1. Component/Container/TUI core
2. Overlay system
3. Differential rendering
4. Input handling with listeners
5. (Optional) Kitty protocol, CURSOR_MARKER, cell size query

---

## Appendix: Line Count Comparison

| Area | pi-mono | rollouts |
|------|---------|----------|
| TUI core | ~800 lines | ~940 lines |
| Rendering | Inline | Inline + separate renderer.py (210 lines) |
| Components | ~30 files | ~15 files |
| Total TUI layer | ~2500 lines | ~3500 lines |

rollouts has more code doing less.
