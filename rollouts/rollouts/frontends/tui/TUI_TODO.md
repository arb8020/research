# TUI Improvements TODO

## Recently Completed

### Session Persistence & Unix Mode
- [x] Session persistence (JSONL format)
- [x] `--continue` / `-c` to resume latest session
- [x] `--session` / `-s` interactive picker
- [x] `-p "query"` non-interactive print mode
- [x] Terminal state restoration on exit (atexit handler)

### Loader Animation
- [x] Centralized loader state in TUI with background animation loop
- [x] `LLMCallStart` event for "Calling LLM..." status

### Rendering
- [x] Differential rendering with synchronized output (`\x1b[?2026h/l`)
- [x] Theme system with pi-mono inspired colors
- [x] Crash log on width overflow (`~/.rollouts/crash.log`)

### Coding Environment
- [x] `--env coding` with read/write/edit/bash tools
- [x] Tool result persistence in sessions

---

## Next Priority: Session Resume Display

When resuming a session with `-c` or `-s`, previous messages are loaded into context but NOT rendered to the TUI. User sees empty screen until they send a message.

**Fix needed in `interactive_agent.py`:**
1. After loading session messages, render them to AgentRenderer before showing input
2. Skip system message, render user/assistant/tool messages

---

## Other UI Improvements (from pi-mono comparison)

### Not Yet Implemented

**Markdown Rendering**
- [ ] Full markdown parsing (code blocks, tables, blockquotes, nested lists)
- [ ] Syntax highlighting in code blocks
- [ ] Theming system for markdown elements

**Advanced Editor Keybindings**
- [ ] `Ctrl+K` - Delete to end of line
- [ ] `Ctrl+U` - Delete to start of line
- [ ] `Ctrl+W` / `Alt+Backspace` - Delete word backwards
- [ ] `Ctrl+A` / `Ctrl+E` - Jump to line start/end
- [ ] `Alt+Left/Right` - Word navigation
- [ ] `Shift+Enter` - New line (Enter submits)

**Autocomplete System**
- [ ] Slash command autocomplete (`/thinking`, `/model`, etc.)
- [ ] File path autocomplete with Tab
- [ ] SelectList component for dropdown UI

**Large Paste Handling**
- [ ] Pastes >10 lines create `[paste #1 +50 lines]` marker
- [ ] Actual content stored and substituted on submit

**Session Features**
- [ ] Branching CLI (`--branch`)
- [ ] Compaction CLI (`--compact`)
- [ ] `--output-format json/stream-json` for print mode

**Interactive Features**
- [ ] Interactive tool confirmation (currently auto-confirms)
- [ ] Input history (up/down arrows for previous messages)
- [ ] Scrollback for long conversations
