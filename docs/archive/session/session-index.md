# Session Index

**DRI:** chiraagbalu
**Claude:** [long-horizon RL memory conversation]

## Context
AI CLI sessions (Claude Code, Codex) accumulate rich context about why code changed, but it's trapped in per-session JSONL files with no way to query across them. We want to answer questions like "why did we make this change to auth.py?" by correlating git diffs with the tool calls and conversation that produced them.

## Non-Goals
- Replacing the raw JSONL files as source of truth (index only, never canonical)
- Real-time streaming ingestion (manual sync is fine for now)
- Semantic/embedding search (keyword + structured queries first)
- Full session replay (read the raw file for that)

## Solution
**Input:** Raw session JSONL files from `~/.claude/projects/` (Claude Code) and `~/.codex/sessions/` (Codex)
**Output:** SQLite index at `~/.rollouts/session_index.db` queryable by file touched, user message content, session metadata, and arbitrary tags

## Usage

```python
from rollouts.session_index import SessionIndex

idx = SessionIndex()  # opens ~/.rollouts/session_index.db

# Sync new/modified sessions since last run
idx.sync()

# "Why did we touch auth.py?"
results = idx.query("""
    SELECT t.user_message, tc.tool_name, tc.input_json, tc.output_text
    FROM file_touches ft
    JOIN tool_calls tc ON ft.tool_call_id = tc.id
    JOIN turns t ON tc.turn_id = t.id
    WHERE ft.filepath LIKE '%auth.py%'
    ORDER BY tc.timestamp
""")

# Find sessions by tag
sessions = idx.sessions(tags={"git_branch": "main", "project": "isara-contract"})

# Get all file touches in a session
touches = idx.file_touches(session_id="abc-123")
```

```bash
# CLI
python -m rollouts.session_index sync
python -m rollouts.session_index query "SELECT * FROM file_touches WHERE filepath LIKE '%auth.py%'"
python -m rollouts.session_index tag <session_id> env=prod git_branch=main
```

---

## Details

### Schema

```sql
-- One row per session file. file_path is the source of truth for staleness checks.
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,          -- session UUID (from filename or session_meta)
    provider TEXT NOT NULL,       -- 'claude_code' | 'codex'
    project_path TEXT,            -- decoded cwd at session start
    file_path TEXT NOT NULL,      -- absolute path to source JSONL
    summary TEXT,                 -- first-line summary if present
    model TEXT,
    start_time TEXT,              -- ISO timestamp
    indexed_at TEXT NOT NULL,     -- when we last ingested this file
    file_mtime REAL NOT NULL      -- mtime at index time, used for staleness
);

-- One row per user message. Groups the conversation unit (user ask → tool calls → response).
CREATE TABLE turns (
    id TEXT PRIMARY KEY,          -- uuid or generated
    session_id TEXT NOT NULL REFERENCES sessions(id),
    seq INTEGER NOT NULL,         -- message order within session
    user_message TEXT,            -- plaintext of user message
    timestamp TEXT
);

-- One row per tool invocation.
CREATE TABLE tool_calls (
    id TEXT PRIMARY KEY,
    turn_id TEXT REFERENCES turns(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    tool_name TEXT NOT NULL,      -- 'Read' | 'Edit' | 'Write' | 'Bash' | 'exec_command' | etc.
    input_json TEXT,              -- full tool input as JSON string
    output_text TEXT,             -- tool result/output
    timestamp TEXT
);

-- Denormalized for fast "what touched this file" queries.
-- Populated from tool_calls where we can extract a filepath from input_json.
CREATE TABLE file_touches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    turn_id TEXT REFERENCES turns(id),
    tool_call_id TEXT REFERENCES tool_calls(id),
    filepath TEXT NOT NULL,
    operation TEXT NOT NULL       -- 'read' | 'write' | 'edit' | 'bash'
);

-- Arbitrary KV metadata on sessions. Stripe-style: open key space, string values.
CREATE TABLE session_tags (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (session_id, key)
);

-- Indexes
CREATE INDEX idx_file_touches_filepath ON file_touches(filepath);
CREATE INDEX idx_file_touches_session ON file_touches(session_id);
CREATE INDEX idx_tool_calls_session ON tool_calls(session_id);
CREATE INDEX idx_turns_session ON turns(session_id);
CREATE INDEX idx_session_tags_kv ON session_tags(key, value);
CREATE INDEX idx_sessions_file_path ON sessions(file_path);
```

### Flow

1. `sync()` scans provider dirs for JSONL files
2. For each file, check `file_mtime` against `sessions.file_mtime` — skip if unchanged
3. Parse with provider-specific parser → emit `Session`, `Turn[]`, `ToolCall[]`, `FileTouch[]`
4. Write to SQLite in a single transaction per session (atomic: all or nothing)
5. Mark `indexed_at` and `file_mtime`

### Parsers

**Claude Code** (`_parse_claude_code`):
- Lines with `type: "summary"` → `sessions.summary`
- Lines with `type: "user"` → new `Turn`, extract plaintext from content blocks
- Lines with `type: "assistant"` → scan content blocks for `tool_use` → `ToolCall`
- Lines with `type: "user"` containing `tool_result` blocks → `ToolCall.output_text`
- File touches extracted from tool inputs:
  - `Read(file_path=...)` → operation='read'
  - `Edit(file_path=...)` → operation='edit'
  - `Write(file_path=...)` → operation='write'
  - `Bash(command=...)` → parse command string for file args → operation='bash'

**Codex** (`_parse_codex_rollout`):
- Lines with `type: "session_meta"` → `sessions.*`
- Lines with `type: "user_message"` → new `Turn`
- Lines with `type: "exec_command_begin"` → `ToolCall` (command array, cwd)
- Lines with `type: "exec_command_end"` → matched by `call_id`, fills `output_text`
- File touches: parse `command` array from exec events directly

### Open Questions
- [x] How to handle Codex's `compacted` items: prior turns are already indexed (we saw them as `response_item` lines earlier in the file). When we hit `Compacted`, store `message` as a session tag `compaction_summary`. If `replacement_history` is present, those items are also indexed normally as they appear.
- [x] `Bash` file touch extraction: store raw command in `tool_calls.input_json`, no heuristic extraction. Queries can parse if needed.
- [x] Session deduplication: tag with `provider`, treat as separate sessions.
- [x] Agent sub-sessions (`agent-*.jsonl`): index as separate sessions, tag with `parent_session_id: <parent_uuid>` where parent is the main UUID session in the same project dir.

### Files
**Read:**
- `~/.claude/projects/**/*.jsonl` — Claude Code sessions
- `~/.codex/sessions/**/*.jsonl` — Codex rollout files
- `rollouts/import_cc.py` — existing Claude Code parser (reuse/extend)

**New:**
- `rollouts/session_index.py` — `SessionIndex` class, schema, parsers, CLI entrypoint

## References
- `rollouts/import_cc.py` — existing partial Claude Code importer
- `rollouts/store.py` — `TODO(database)` comment with prior schema sketch
- Codex rollout format: `codex-rs/protocol/src/protocol.rs:2088` (`RolloutItem` enum)
- Codex exec events: `codex-rs/core/src/rollout/recorder.rs`
