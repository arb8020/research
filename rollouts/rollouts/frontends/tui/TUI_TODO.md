# TUI Improvements TODO

## ✅ FIXED: Loader Spinner Freezes During "Calling LLM..."

**Solution Implemented:** Centralized loader state in TUI with background animation loop.

Following the code style principles:
- **"Caller controls flow"** - TUI orchestrates rendering, not Loader
- **"Minimize stateful components"** - Removed Loader class, centralized in TUI
- **"Classes are for legitimate state"** - TUI owns the render loop, so it owns animation timing

**Changes Made:**
1. Replaced `Loader` class with pure `render_loader_line()` function in `tui.py`
2. Added loader state to TUI: `_loader_text`, `_loader_start_time`, color functions
3. Added `show_loader()`, `hide_loader()`, `is_loader_active()` methods to TUI
4. Added `run_animation_loop()` async method - runs every 80ms, triggers re-render when loader active
5. Simplified `AgentRenderer` - now just calls `tui.show_loader()` / `tui.hide_loader()`
6. Removed `status_container` and `self.loader` from AgentRenderer
7. Updated `interactive_agent.py` to start animation loop as background task

**Files Modified:**
- `tui.py` - Added `render_loader_line()`, loader state, animation loop
- `agent_renderer.py` - Removed Loader import, simplified handlers
- `interactive_agent.py` - Added `nursery.start_soon(self.tui.run_animation_loop)`
- `components/__init__.py` - Removed Loader export
- `__init__.py` - Removed Loader export, added `render_loader_line` export

---

## Other UI Improvements (from pi-mono comparison)

### Already Done
- [x] Differential rendering with synchronized output
- [x] LLMCallStart event for better status indication
- [x] Loader animation (fixed - now uses centralized TUI animation loop)

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
