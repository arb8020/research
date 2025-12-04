# TUI Improvements TODO

## Issue: Loader Spinner Freezes During "Calling LLM..."

**Symptom:** The spinner animation (`⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`) freezes during "Calling LLM..." but animates fine during "Streaming...".

**Root Cause:** The Loader uses time-based frame calculation in `render()`:

```python
elapsed = time.time() - self._start_time
frame_index = int(elapsed * 10) % len(self._spinner_frames)
```

This is correct, BUT `render()` is only called when `request_render()` is triggered. During the API connection phase (between `LLMCallStart` and `StreamStart`), we're blocked waiting for the HTTP connection - no events come in, so nothing triggers a re-render, and the spinner appears frozen.

Once streaming starts, frequent `TextDelta` events trigger `request_render()` and the spinner animates.

**pi-mono's Solution:** Their Loader uses a timer interval that triggers its own re-renders:

```typescript
// From /tmp/pi-mono/packages/tui/src/components/loader.ts
this.intervalId = setInterval(() => {
    this.currentFrame = (this.currentFrame + 1) % this.frames.length;
    this.updateDisplay();
}, 80);
```

The Loader holds a reference to the TUI and calls `ui.requestRender()` every 80ms.

**Possible Fixes:**

1. **Pass TUI to Loader** (quick fix, adds coupling)
   - Loader constructor takes `tui: TUI` parameter
   - Spawns a Trio background task that calls `tui.request_render()` every 80ms
   - Task cancelled in `stop()`

2. **TUI manages animation timer** (cleaner)
   - TUI has a method `start_animation_timer()` / `stop_animation_timer()`
   - When active, TUI spawns a task that calls `request_render()` periodically
   - AgentRenderer calls these when showing/hiding Loader

3. **Loader emits render requests via callback** (most decoupled)
   - Loader takes optional `on_frame: Callable[[], None]` callback
   - Spawns background task that calls callback every 80ms
   - AgentRenderer passes `lambda: self.tui.request_render()`

**Files to modify:**
- `rollouts/frontends/tui/components/loader.py`
- `rollouts/frontends/tui/agent_renderer.py` (to pass TUI or callback)
- Possibly `rollouts/frontends/tui/tui.py` (if using option 2)

---

## Other UI Improvements (from pi-mono comparison)

### Already Done
- [x] Differential rendering with synchronized output
- [x] LLMCallStart event for better status indication
- [x] Basic Loader component

### Not Yet Implemented

**Markdown Rendering**
- Full markdown parsing with `marked`-equivalent library
- Syntax highlighting in code blocks
- Tables, blockquotes, nested lists
- Theming system for markdown elements

**Advanced Editor Keybindings**
- `Ctrl+K` - Delete to end of line
- `Ctrl+U` - Delete to start of line
- `Ctrl+W` / `Alt+Backspace` - Delete word backwards
- `Ctrl+A` / `Ctrl+E` - Jump to line start/end
- `Alt+Left/Right` - Word navigation
- `Shift+Enter` - New line (Enter submits)

**Autocomplete System**
- Slash command autocomplete (`/thinking`, `/model`, etc.)
- File path autocomplete with Tab
- SelectList component for dropdown UI

**Large Paste Handling**
- Pastes >10 lines create `[paste #1 +50 lines]` marker
- Actual content stored and substituted on submit
