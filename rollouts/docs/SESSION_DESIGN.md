# Session Persistence Design

## Status: ✅ Implemented

Session persistence for the TUI agent, inspired by pi-mono and Claude Code but simplified.

## Design Principles (from docs/code_style)

1. **Functional over classes** - Session is a frozen dataclass, operations are pure functions
2. **Messages all the way down** - No special entry types, just session header + messages
3. **Write usage code first** - See CLI examples below

## Session Format (JSONL)

```jsonl
{"type": "session", "id": "uuid", "cwd": "/path", "provider": "anthropic", "model": "claude-sonnet-4-5", "timestamp": "..."}
{"type": "message", "timestamp": "...", "message": {"role": "user", "content": "..."}}
{"type": "message", "timestamp": "...", "message": {"role": "assistant", "content": "..."}}
```

No special entry types. Summaries (from compaction) are just regular user messages.

## Storage Location

```
~/.rollouts/sessions/--path-to-project--/
├── 2024-01-15T10-30-00_abc123.jsonl
├── 2024-01-15T11-45-00_def456.jsonl
└── ...
```

Path encoding: `/Users/foo/myproject` → `--Users-foo-myproject--`

## CLI Usage

### Session Management
```bash
# Resume latest session
rollouts --continue
rollouts -c

# Interactive session picker
rollouts --session
rollouts -s

# Resume specific session file
rollouts -s ~/.rollouts/sessions/.../session.jsonl

# Don't persist session
rollouts --no-session
```

### Unix Utility Mode
```bash
# Non-interactive, print result
rollouts -p "query"

# With session continuation
rollouts -c -p "follow up question"

# No session persistence
rollouts -p "quick question" --no-session
```

## API (sessions.py)

```python
@dataclass(frozen=True)
class Session:
    file_path: Path
    session_id: str

# Core functions
create_session(working_dir, provider, model) -> Session
find_latest_session(working_dir) -> Session | None
load_session(file_path) -> Session
list_sessions(working_dir) -> list[Session]
list_sessions_with_info(working_dir) -> list[SessionInfo]

# Message I/O
append_message(session, message) -> None
load_messages(session) -> list[Message]
load_header(session) -> SessionHeader

# Branching/Compaction (implemented but not exposed in CLI yet)
branch_session(source, branch_after_idx, ...) -> Session
compact_session(source, summarize_fn, keep_last_n, ...) -> Session
```

## What's Implemented

- [x] Session creation and persistence
- [x] Message append (user, assistant, tool results)
- [x] Session continuation (`-c`)
- [x] Session picker (`-s`)
- [x] Unix utility mode (`-p`)
- [x] `branch_session()` function
- [x] `compact_session()` function

## Not Yet Implemented

- [ ] CLI commands for branching (`--branch`)
- [ ] CLI commands for compaction (`--compact`)
- [ ] Render previous messages on session resume (currently loads context but doesn't display)
- [ ] `--output-format json/stream-json` for print mode
