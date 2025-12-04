# Session Persistence Design

## Overview

Session persistence for the TUI agent, inspired by pi-mono and Claude Code but simplified.

## Design Principles (from docs/code_style)

1. **Functional over classes** - Session is a frozen dataclass, operations are pure functions
2. **Messages all the way down** - No special entry types, just session header + messages
3. **Write usage code first** - See CLI examples below

## Data Model

Rollouts already has the right primitives:

```
AgentState (frozen)
├── actor: Actor
│   ├── trajectory: Trajectory
│   │   └── messages: list[Message]  # Full conversation history
│   ├── endpoint: Endpoint
│   └── tools: list[Tool]
├── environment: Environment (serializable)
└── turn_idx, pending_tool_calls, stop...
```

## Session Format (JSONL)

```jsonl
{"type": "session", "id": "uuid", "cwd": "/path", "provider": "anthropic", "model": "claude-sonnet-4-5", "timestamp": "..."}
{"type": "message", "message": {"role": "user", "content": "..."}}
{"type": "message", "message": {"role": "assistant", "content": "..."}}
{"type": "message", "message": {"role": "user", "content": "..."}}
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

## API (Functional Style)

```python
@dataclass(frozen=True)
class Session:
    """Just data - the session file path and ID."""
    file_path: Path
    session_id: str

# Session management (pure functions)
def get_sessions_dir(working_dir: Path) -> Path:
    """Get ~/.rollouts/sessions/--encoded-path--/"""

def create_session(working_dir: Path, provider: str, model: str) -> Session:
    """Create new session file with header."""

def find_latest_session(working_dir: Path) -> Session | None:
    """Find most recently modified session."""

def load_session(file_path: Path) -> Session:
    """Load existing session by path."""

def list_sessions(working_dir: Path) -> list[Session]:
    """List all sessions for a working directory."""

# Message I/O (pure functions)
def append_message(session: Session, message: Message) -> None:
    """Append message entry to session file."""

def load_messages(session: Session) -> list[Message]:
    """Load all messages from session."""

# Branching (creates new session)
def branch_session(
    source: Session,
    branch_after_idx: int,
    working_dir: Path,
    provider: str,
    model: str,
) -> Session:
    """Create new session with messages up to branch_after_idx."""

# Compaction (creates new session with summary)
def compact_session(
    source: Session,
    summarize_fn: Callable[[list[Message]], str],
    keep_last_n: int,
    working_dir: Path,
    provider: str,
    model: str,
) -> Session:
    """Create new session with old messages summarized."""
```

## CLI Flags

### Session Management
```bash
# Resume latest session
rollouts --continue
rollouts -c

# Interactive session picker (future)
rollouts --resume
rollouts -r

# Resume specific session
rollouts --session ~/.rollouts/sessions/.../session.jsonl

# Don't persist session
rollouts --no-session
```

### Unix Utility Mode
```bash
# Non-interactive, print result
rollouts -p "query"

# Pipe stdin
cat file.py | rollouts -p "add type hints" > typed.py

# Output formats
rollouts -p "query" --output-format text      # default
rollouts -p "query" --output-format json      # structured
rollouts -p "query" --output-format stream-json  # NDJSON streaming
```

## Comparison

| Feature | Claude Code | Pi-mono | Ours |
|---------|-------------|---------|------|
| Format | JSONL | JSONL | JSONL |
| Storage | `~/.claude/projects/` | `~/.pi/agent/sessions/` | `~/.rollouts/sessions/` |
| Resume latest | `--continue` | `--continue` | `--continue` |
| Resume picker | `--resume` | N/A | `--resume` (future) |
| Branching | N/A | Yes (CompactionEntry) | Yes (new session file) |
| Compaction | N/A | Yes (CompactionEntry) | Yes (summary as message) |
| Entry types | Multiple | Multiple | Just session + message |
| Implementation | Class | Class | Frozen dataclass + functions |

## Why Simpler

Pi-mono has `SessionHeader`, `MessageEntry`, `ThinkingLevelChangeEntry`, `ModelChangeEntry`, `CompactionEntry`. We only need:

1. **Session header** - One-time metadata at file start
2. **Message** - Everything else

- Model changes? Just read the endpoint from AgentState when resuming
- Thinking level? Same
- Compaction? Summary is just a user message
- Branching? New session file with subset of messages

The JSONL file is append-only during a session. Branching/compaction create new files.

## Implementation Plan

1. Add `rollouts/frontends/tui/sessions.py` with frozen dataclass + pure functions
2. Update `cli.py` with `--continue`, `--session`, `--no-session` flags
3. Update `interactive_agent.py` to load/save messages
4. Add `-p` flag for unix utility mode
5. (Future) Add `--resume` interactive picker
6. (Future) Add compaction with LLM summarization
