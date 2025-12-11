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

# Resume specific session by ID
rollouts -s <session_id>

# Don't persist session
rollouts --no-session
```

### Unix Utility Mode
```bash
# Non-interactive, print result
rollouts -p "query"

# Read query from stdin
rollouts -p -
rollouts -p < prompt.txt
echo "explain this code" | rollouts -p

# With session continuation
rollouts -c -p "follow up question"

# No session persistence
rollouts -p "quick question" --no-session
```

### Handoff (Extract Context for New Session)
```bash
# Extract goal-directed context to stdout
rollouts --handoff "implement the API endpoints" -s <session_id>

# Save to file, edit, then use
rollouts --handoff "fix the tests" -s abc123 > handoff.md
vim handoff.md
rollouts --env coding < handoff.md

# One-liner: pipe directly to new interactive session
rollouts --handoff "continue this work" -s abc123 | rollouts --env coding

# Or non-interactive
rollouts --handoff "continue this work" -s abc123 | rollouts -p
```

Handoff replaces the old `--summarize` and `--compact` commands. Instead of creating
a child session with a baked-in summary, it outputs markdown that you can review,
edit, and pipe into a new session. This is more Unix-like: tools do one thing well.

Stdin input works for both interactive TUI and non-interactive print mode.

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
- [x] Unix utility mode (`-p`) with stdin support
- [x] Handoff (`--handoff GOAL`) - extract goal-directed context to stdout
- [x] `branch_session()` function

## Not Yet Implemented

- [ ] CLI commands for branching (`--branch`)
- [ ] Render previous messages on session resume (currently loads context but doesn't display)
- [ ] `--output-format json/stream-json` for print mode

## Future: Tree Explorer & Content-Addressable Storage

### Tree Explorer (`rollouts tree`)

Git branchless-style visualization of session DAG:

```
○ 20251211_121000 - 3 messages
│
○ 20251211_120456 - model=opus, 8 messages
│
│ ○ 20251211_120123 - env=coding, 12 messages
├─╯
○ 20251211_115811 (root) - 5 messages
```

Current primitives exist:
- `parent_id` and `branch_point` on AgentSession
- `list_children(parent_id)` in SessionStore
- Forking happens automatically when resuming with different config

TODO:
- [ ] `rollouts tree` command - static DAG visualization
- [ ] Interactive tree browser (arrow keys, enter to resume)
- [ ] Show diff between branches (config changes, divergent messages)

### Content-Addressable Storage (Future)

Currently messages are duplicated on fork. For true git semantics:

```
~/.rollouts/
  objects/
    <hash>.json  # Immutable message objects
  sessions/
    <session_id>/
      messages: [hash1, hash2, ...]  # References only
```

Benefits:
- Deduplication (shared history stored once)
- Immutability guarantees
- Easy diffing between sessions
- Enables efficient tree operations

This would make sessions lightweight pointers (like git branches) rather than
containers (like git clones). Low priority - current model works fine.
