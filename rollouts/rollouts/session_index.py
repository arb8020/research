"""Session index: queryable SQLite index over Claude Code and Codex session JSONL files.

The raw JSONL files remain the source of truth. This module builds and maintains
an index for cross-session queries: "what sessions touched auth.py?", "why did
we make this change?", etc.

Usage:
    from rollouts.session_index import SessionIndex

    idx = SessionIndex()
    idx.sync()  # ingest new/modified sessions

    # arbitrary SQL against the index
    rows = idx.query("SELECT * FROM file_touches WHERE filepath LIKE '%auth.py%'")

    # tag a session with arbitrary KV metadata
    idx.tag("session-uuid", env="prod", git_branch="main")

CLI:
    python -m rollouts.session_index sync
    python -m rollouts.session_index query "SELECT ..."
    python -m rollouts.session_index tag <session_id> key=value ...
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CLAUDE_CODE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"
DEFAULT_DB_PATH = Path.home() / ".rollouts" / "session_index.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,      -- 'claude_code' | 'codex'
    project_path TEXT,              -- cwd at session start
    file_path   TEXT NOT NULL,      -- absolute path to source JSONL
    summary     TEXT,               -- first-line summary if present
    model       TEXT,
    start_time  TEXT,               -- ISO timestamp
    indexed_at  TEXT NOT NULL,
    file_mtime  REAL NOT NULL       -- mtime at index time, for staleness check
);

CREATE TABLE IF NOT EXISTS turns (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    seq         INTEGER NOT NULL,   -- message order within session
    user_message TEXT,              -- plaintext of user message
    timestamp   TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id          TEXT PRIMARY KEY,
    turn_id     TEXT REFERENCES turns(id),
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    tool_name   TEXT NOT NULL,
    input_json  TEXT,               -- full tool input as JSON string
    output_text TEXT,               -- tool result/output
    timestamp   TEXT
);

-- Denormalized for fast "what touched this file" queries.
-- Populated only for tool calls where we can extract a filepath from input_json.
CREATE TABLE IF NOT EXISTS file_touches (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    turn_id      TEXT REFERENCES turns(id),
    tool_call_id TEXT REFERENCES tool_calls(id),
    filepath     TEXT NOT NULL,
    operation    TEXT NOT NULL      -- 'read' | 'write' | 'edit' | 'bash'
);

-- Arbitrary KV metadata. Stripe-style: open key space, string values.
CREATE TABLE IF NOT EXISTS session_tags (
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    PRIMARY KEY (session_id, key)
);

CREATE INDEX IF NOT EXISTS idx_file_touches_filepath  ON file_touches(filepath);
CREATE INDEX IF NOT EXISTS idx_file_touches_session   ON file_touches(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session     ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_session          ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_session_tags_kv        ON session_tags(key, value);
CREATE INDEX IF NOT EXISTS idx_sessions_file_path     ON sessions(file_path);
"""

# ---------------------------------------------------------------------------
# Intermediate parse types (never stored directly, just pipeline glue)
# ---------------------------------------------------------------------------


@dataclass
class _Turn:
    id: str
    seq: int
    user_message: str | None
    timestamp: str | None


@dataclass
class _ToolCall:
    id: str
    turn_id: str | None
    tool_name: str
    input_json: str | None
    output_text: str | None
    timestamp: str | None


@dataclass
class _FileTouch:
    tool_call_id: str
    turn_id: str | None
    filepath: str
    operation: str


@dataclass
class _ParsedSession:
    id: str
    provider: str
    project_path: str | None
    file_path: str
    summary: str | None
    model: str | None
    start_time: str | None
    tags: dict[str, str]
    turns: list[_Turn]
    tool_calls: list[_ToolCall]
    file_touches: list[_FileTouch]


# ---------------------------------------------------------------------------
# Claude Code parser
# ---------------------------------------------------------------------------

# Tool names in Claude Code that carry a file path argument
_CC_FILE_TOOLS: dict[str, tuple[str, str]] = {
    # tool_name -> (input_key, operation)
    "Read": ("file_path", "read"),
    "Write": ("file_path", "write"),
    "Edit": ("file_path", "edit"),
    "MultiEdit": ("file_path", "edit"),
    "NotebookEdit": ("notebook_path", "edit"),
}


def _cc_extract_text(content: Any) -> str:
    """Extract plaintext from Claude Code content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "tool_result":
            inner = block.get("content", "")
            parts.append(_cc_extract_text(inner))
    return "\n".join(p for p in parts if p)


def _cc_file_touch_from_tool(tool_name: str, arguments: dict[str, Any], call_id: str, turn_id: str | None) -> _FileTouch | None:
    if tool_name in _CC_FILE_TOOLS:
        key, op = _CC_FILE_TOOLS[tool_name]
        fp = arguments.get(key)
        if fp:
            return _FileTouch(tool_call_id=call_id, turn_id=turn_id, filepath=str(fp), operation=op)
    elif tool_name == "Bash":
        command = arguments.get("command", "")
        # Store the bash command touch as-is against the cwd; filepath=command for queryability
        # We don't heuristically parse the command — callers can do that in SQL if needed
        if command:
            return _FileTouch(tool_call_id=call_id, turn_id=turn_id, filepath=command, operation="bash")
    return None


def _parse_claude_code(file_path: Path, session_id: str, project_path: str | None) -> _ParsedSession:
    summary: str | None = None
    model: str | None = None
    start_time: str | None = None
    tags: dict[str, str] = {"provider": "claude_code"}
    turns: list[_Turn] = []
    tool_calls: list[_ToolCall] = []
    file_touches: list[_FileTouch] = []

    # tool_use blocks emit before tool_result blocks; we need to match them up
    # call_id -> _ToolCall
    pending_tool_calls: dict[str, _ToolCall] = {}
    current_turn_id: str | None = None
    turn_seq = 0

    with open(file_path) as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")

            if entry_type == "summary":
                summary = entry.get("summary")
                continue

            if entry_type == "file-history-snapshot":
                continue

            msg = entry.get("message", {})
            role = msg.get("role") or entry_type
            timestamp = entry.get("timestamp")
            if isinstance(timestamp, (int, float)):
                from datetime import datetime
                timestamp = datetime.fromtimestamp(timestamp / 1000).isoformat()

            if start_time is None and timestamp:
                start_time = timestamp

            if role == "assistant":
                if not model and msg.get("model"):
                    model = msg["model"]

                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_use":
                        continue

                    # Namespace call_id: Claude Code IDs are only locally unique
                    raw_call_id = block.get("id") or str(uuid.uuid4())
                    call_id = f"{session_id}:{raw_call_id}"
                    tool_name = block.get("name", "")
                    arguments = block.get("input", {})

                    tc = _ToolCall(
                        id=call_id,
                        turn_id=current_turn_id,
                        tool_name=tool_name,
                        input_json=json.dumps(arguments),
                        output_text=None,
                        timestamp=timestamp,
                    )
                    pending_tool_calls[call_id] = tc
                    tool_calls.append(tc)

                    touch = _cc_file_touch_from_tool(tool_name, arguments, call_id, current_turn_id)
                    if touch:
                        file_touches.append(touch)

            elif role == "user":
                content = msg.get("content", "")

                # Check if this is a tool result message
                if isinstance(content, list):
                    tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                    if tool_results:
                        for tr in tool_results:
                            raw_id = tr.get("tool_use_id", "")
                            call_id = f"{session_id}:{raw_id}"
                            result_content = tr.get("content", "")
                            output = _cc_extract_text(result_content) if isinstance(result_content, list) else str(result_content)
                            if call_id in pending_tool_calls:
                                pending_tool_calls[call_id].output_text = output
                        continue

                # Regular user message → new turn
                # Namespace with session_id: Claude Code UUIDs are only locally unique
                raw_uuid = entry.get("uuid") or str(uuid.uuid4())
                turn_id = f"{session_id}:{raw_uuid}"
                user_text = _cc_extract_text(content)
                turn = _Turn(id=turn_id, seq=turn_seq, user_message=user_text or None, timestamp=timestamp)
                turns.append(turn)
                current_turn_id = turn_id
                turn_seq += 1

    # Detect agent sub-sessions by filename pattern
    stem = file_path.stem
    if stem.startswith("agent-"):
        tags["session_type"] = "agent"
        # Parent is the non-agent UUID session in the same dir — we tag it here;
        # the link is resolved post-ingestion when the parent is also indexed.
        # Store the project dir so callers can find siblings.
        tags["agent_project_dir"] = str(file_path.parent)
    else:
        tags["session_type"] = "main"

    return _ParsedSession(
        id=session_id,
        provider="claude_code",
        project_path=project_path,
        file_path=str(file_path),
        summary=summary,
        model=model,
        start_time=start_time,
        tags=tags,
        turns=turns,
        tool_calls=tool_calls,
        file_touches=file_touches,
    )


# ---------------------------------------------------------------------------
# Codex parser
# ---------------------------------------------------------------------------


def _parse_codex(file_path: Path, session_id: str) -> _ParsedSession:
    project_path: str | None = None
    model: str | None = None
    start_time: str | None = None
    summary: str | None = None
    tags: dict[str, str] = {"provider": "codex"}
    turns: list[_Turn] = []
    tool_calls: list[_ToolCall] = []
    file_touches: list[_FileTouch] = []

    current_turn_id: str | None = None
    turn_seq = 0
    # call_id -> _ToolCall for exec_command_begin/end matching
    pending_execs: dict[str, _ToolCall] = {}

    with open(file_path) as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")
            timestamp = entry.get("timestamp")
            payload = entry.get("payload", {})

            if start_time is None and timestamp:
                start_time = timestamp

            # Codex uses two nesting styles:
            #   flat:  {"type": "session_meta", "payload": {...}}
            #   event: {"type": "event_msg", "payload": {"type": "user_message", ...}}
            # Resolve the effective type and data once.
            if entry_type == "event_msg":
                effective_type = payload.get("type", "")
                effective_data = payload
            else:
                effective_type = entry_type
                effective_data = payload

            if effective_type == "session_meta":
                project_path = effective_data.get("cwd")
                model = effective_data.get("model_provider")
                meta_id = effective_data.get("id")
                if meta_id:
                    session_id = str(meta_id)
                tags["cli_version"] = effective_data.get("cli_version", "")
                tags["source"] = effective_data.get("source", "")

            elif effective_type == "user_message":
                turn_id = str(uuid.uuid4())
                user_text = effective_data.get("message", "")
                turn = _Turn(id=turn_id, seq=turn_seq, user_message=user_text or None, timestamp=timestamp)
                turns.append(turn)
                current_turn_id = turn_id
                turn_seq += 1

            elif effective_type == "exec_command_begin":
                call_id = effective_data.get("call_id") or str(uuid.uuid4())
                command = effective_data.get("command", [])
                cwd = effective_data.get("cwd", "")
                input_data = {"command": command, "cwd": cwd}

                tc = _ToolCall(
                    id=call_id,
                    turn_id=current_turn_id,
                    tool_name="exec_command",
                    input_json=json.dumps(input_data),
                    output_text=None,
                    timestamp=timestamp,
                )
                pending_execs[call_id] = tc
                tool_calls.append(tc)

                cmd_str = " ".join(str(c) for c in command)
                if cmd_str:
                    file_touches.append(_FileTouch(
                        tool_call_id=call_id,
                        turn_id=current_turn_id,
                        filepath=cmd_str,
                        operation="bash",
                    ))

            elif effective_type == "exec_command_end":
                call_id = effective_data.get("call_id", "")
                output = effective_data.get("aggregated_output") or effective_data.get("stdout", "")
                if call_id in pending_execs:
                    pending_execs[call_id].output_text = output

            elif effective_type == "compacted":
                msg = effective_data.get("message", "")
                if msg:
                    summary = msg
                    tags["compaction_summary"] = msg[:500]

            elif entry_type == "response_item":
                # Model-visible items — extract tool calls from function_call / custom_tool_call
                item_type = payload.get("type", "")
                if item_type == "function_call":
                    call_id = payload.get("call_id") or str(uuid.uuid4())
                    name = payload.get("name", "function_call")
                    args_str = payload.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {"raw": args_str}
                    tc = _ToolCall(
                        id=call_id,
                        turn_id=current_turn_id,
                        tool_name=name,
                        input_json=json.dumps(args),
                        output_text=None,
                        timestamp=timestamp,
                    )
                    pending_execs[call_id] = tc
                    tool_calls.append(tc)

                elif item_type == "function_call_output":
                    call_id = payload.get("call_id", "")
                    output_payload = payload.get("output", {})
                    if isinstance(output_payload, dict):
                        output = output_payload.get("content") or json.dumps(output_payload)
                    else:
                        output = str(output_payload)
                    if call_id in pending_execs:
                        pending_execs[call_id].output_text = output

                elif item_type == "custom_tool_call":
                    call_id = payload.get("call_id") or str(uuid.uuid4())
                    name = payload.get("name", "custom_tool")
                    inp = payload.get("input", "")
                    tc = _ToolCall(
                        id=call_id,
                        turn_id=current_turn_id,
                        tool_name=name,
                        input_json=json.dumps({"input": inp}),
                        output_text=None,
                        timestamp=timestamp,
                    )
                    pending_execs[call_id] = tc
                    tool_calls.append(tc)

                elif item_type == "custom_tool_call_output":
                    call_id = payload.get("call_id", "")
                    output = payload.get("output", "")
                    if call_id in pending_execs:
                        pending_execs[call_id].output_text = str(output)

    return _ParsedSession(
        id=session_id,
        provider="codex",
        project_path=project_path,
        file_path=str(file_path),
        summary=summary,
        model=model,
        start_time=start_time,
        tags=tags,
        turns=turns,
        tool_calls=tool_calls,
        file_touches=file_touches,
    )


# ---------------------------------------------------------------------------
# SessionIndex
# ---------------------------------------------------------------------------


class SessionIndex:
    """SQLite index over Claude Code and Codex session JSONL files.

    The raw JSONL files remain the source of truth. This class maintains an
    index for cross-session structured queries.

    Typical usage:
        idx = SessionIndex()
        idx.sync()
        rows = idx.query("SELECT * FROM file_touches WHERE filepath LIKE '%auth.py%'")
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync(self, verbose: bool = False) -> dict[str, int]:
        """Ingest new or modified session files from all providers.

        Returns counts of sessions indexed/skipped/errored.
        """
        stats = {"indexed": 0, "skipped": 0, "errored": 0}

        for file_path, provider, project_path, session_id in self._discover_files():
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                continue

            if self._is_current(str(file_path), mtime):
                stats["skipped"] += 1
                continue

            try:
                if provider == "claude_code":
                    parsed = _parse_claude_code(file_path, session_id, project_path)
                else:
                    parsed = _parse_codex(file_path, session_id)
                self._ingest(parsed, mtime)
                stats["indexed"] += 1
                if verbose:
                    logger.info("indexed %s (%s)", file_path.name, provider)
            except Exception as e:
                stats["errored"] += 1
                logger.warning("failed to index %s: %s", file_path, e)

        # Second pass: link agent sub-sessions to their parent main session
        self._link_agent_sessions()

        return stats

    def _discover_files(self) -> list[tuple[Path, str, str | None, str]]:
        """Yield (file_path, provider, project_path, session_id) for all known sessions."""
        results = []

        # Claude Code
        if CLAUDE_CODE_PROJECTS_DIR.exists():
            for project_dir in CLAUDE_CODE_PROJECTS_DIR.iterdir():
                if not project_dir.is_dir():
                    continue
                project_path = _cc_decode_path(project_dir.name)
                for session_file in project_dir.glob("*.jsonl"):
                    session_id = session_file.stem
                    results.append((session_file, "claude_code", project_path, session_id))

        # Codex — sessions/YYYY/MM/DD/rollout-*.jsonl
        if CODEX_SESSIONS_DIR.exists():
            for session_file in CODEX_SESSIONS_DIR.rglob("rollout-*.jsonl"):
                # Use the UUID portion of the filename as session_id
                stem = session_file.stem  # e.g. rollout-2025-01-03T12-00-00-<uuid>
                parts = stem.split("-")
                session_id = parts[-1] if parts else stem
                results.append((session_file, "codex", None, session_id))

        return results

    def _is_current(self, file_path: str, mtime: float) -> bool:
        row = self._conn.execute(
            "SELECT file_mtime FROM sessions WHERE file_path = ?", (file_path,)
        ).fetchone()
        if row is None:
            return False
        return abs(row["file_mtime"] - mtime) < 0.01

    def _ingest(self, parsed: _ParsedSession, mtime: float) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        with self._conn:
            # Upsert session
            self._conn.execute("""
                INSERT INTO sessions (id, provider, project_path, file_path, summary, model, start_time, indexed_at, file_mtime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    project_path=excluded.project_path,
                    file_path=excluded.file_path,
                    summary=excluded.summary,
                    model=excluded.model,
                    start_time=excluded.start_time,
                    indexed_at=excluded.indexed_at,
                    file_mtime=excluded.file_mtime
            """, (parsed.id, parsed.provider, parsed.project_path, parsed.file_path,
                  parsed.summary, parsed.model, parsed.start_time, now, mtime))

            # On re-index: delete old child rows first
            self._conn.execute("DELETE FROM file_touches WHERE session_id = ?", (parsed.id,))
            self._conn.execute("DELETE FROM tool_calls WHERE session_id = ?", (parsed.id,))
            self._conn.execute("DELETE FROM turns WHERE session_id = ?", (parsed.id,))
            self._conn.execute("DELETE FROM session_tags WHERE session_id = ?", (parsed.id,))

            for turn in parsed.turns:
                self._conn.execute("""
                    INSERT INTO turns (id, session_id, seq, user_message, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (turn.id, parsed.id, turn.seq, turn.user_message, turn.timestamp))

            for tc in parsed.tool_calls:
                self._conn.execute("""
                    INSERT INTO tool_calls (id, turn_id, session_id, tool_name, input_json, output_text, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (tc.id, tc.turn_id, parsed.id, tc.tool_name, tc.input_json, tc.output_text, tc.timestamp))

            for ft in parsed.file_touches:
                self._conn.execute("""
                    INSERT INTO file_touches (session_id, turn_id, tool_call_id, filepath, operation)
                    VALUES (?, ?, ?, ?, ?)
                """, (parsed.id, ft.turn_id, ft.tool_call_id, ft.filepath, ft.operation))

            for key, value in parsed.tags.items():
                self._conn.execute("""
                    INSERT INTO session_tags (session_id, key, value)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id, key) DO UPDATE SET value=excluded.value
                """, (parsed.id, key, str(value)))

    def _link_agent_sessions(self) -> None:
        """Tag agent sub-sessions with parent_session_id.

        For each agent session in a project dir, find the main session (non-agent
        UUID file) in the same dir and set parent_session_id tag.
        """
        agent_sessions = self._conn.execute("""
            SELECT s.id, st.value AS project_dir
            FROM sessions s
            JOIN session_tags st ON s.id = st.session_id
            WHERE st.key = 'agent_project_dir'
              AND NOT EXISTS (
                SELECT 1 FROM session_tags
                WHERE session_id = s.id AND key = 'parent_session_id'
              )
        """).fetchall()

        for row in agent_sessions:
            agent_id = row["id"]
            project_dir = row["project_dir"]

            # Find main sessions in the same dir
            parent = self._conn.execute("""
                SELECT s.id FROM sessions s
                JOIN session_tags st ON s.id = st.session_id
                WHERE s.file_path LIKE ? || '/%'
                  AND st.key = 'session_type' AND st.value = 'main'
                ORDER BY s.start_time DESC
                LIMIT 1
            """, (project_dir,)).fetchone()

            if parent:
                with self._conn:
                    self._conn.execute("""
                        INSERT INTO session_tags (session_id, key, value)
                        VALUES (?, 'parent_session_id', ?)
                        ON CONFLICT(session_id, key) DO UPDATE SET value=excluded.value
                    """, (agent_id, parent["id"]))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Run arbitrary SQL against the index. Returns list of row dicts."""
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def describe(self) -> str:
        """Return schema + column semantics + example queries for LLM context.

        Call this to understand the index before writing SQL queries.
        """
        example_rows: dict[str, list[dict]] = {}
        for table in ("sessions", "turns", "tool_calls", "file_touches", "session_tags"):
            rows = self._conn.execute(f"SELECT * FROM {table} LIMIT 2").fetchall()
            example_rows[table] = [dict(r) for r in rows]

        counts = self._conn.execute("""
            SELECT
                (SELECT COUNT(*) FROM sessions) AS sessions,
                (SELECT COUNT(*) FROM turns) AS turns,
                (SELECT COUNT(*) FROM tool_calls) AS tool_calls,
                (SELECT COUNT(*) FROM file_touches) AS file_touches,
                (SELECT COUNT(*) FROM session_tags) AS session_tags
        """).fetchone()

        lines = [
            "# Session Index Schema",
            "",
            f"DB: {self.db_path}",
            f"Counts: {dict(counts)}",
            "",
            "## Tables",
            "",
            "### sessions",
            "One row per JSONL session file.",
            "  id          - session UUID (primary key)",
            "  provider    - 'claude_code' | 'codex'",
            "  project_path - cwd at session start (decoded filesystem path)",
            "  file_path   - absolute path to source JSONL file",
            "  summary     - first-line summary if present (Claude Code only)",
            "  model       - model name/provider string",
            "  start_time  - ISO timestamp of first message",
            "  indexed_at  - when this session was last ingested",
            "  file_mtime  - mtime of source file at index time",
            "",
            "### turns",
            "One row per user message. Groups the conversation unit (user ask → tool calls → response).",
            "  id           - namespaced turn ID",
            "  session_id   - FK to sessions",
            "  seq          - message order within session (0-indexed)",
            "  user_message - plaintext of user message",
            "  timestamp    - ISO timestamp",
            "",
            "### tool_calls",
            "One row per tool invocation.",
            "  id           - tool call ID",
            "  turn_id      - FK to turns (the user message that triggered this)",
            "  session_id   - FK to sessions",
            "  tool_name    - e.g. 'Read', 'Edit', 'Write', 'Bash', 'exec_command', 'function_call'",
            "  input_json   - full tool input as JSON string (parse with json() in SQL or in Python)",
            "  output_text  - tool result/output text",
            "  timestamp    - ISO timestamp",
            "",
            "### file_touches",
            "Denormalized for fast filepath queries. Only populated when a filepath is extractable.",
            "  id           - autoincrement PK",
            "  session_id   - FK to sessions",
            "  turn_id      - FK to turns",
            "  tool_call_id - FK to tool_calls",
            "  filepath     - for Read/Edit/Write: the file path; for Bash/exec: the raw command string",
            "  operation    - 'read' | 'write' | 'edit' | 'bash'",
            "",
            "### session_tags",
            "Arbitrary KV metadata on sessions. Open key space, string values.",
            "  session_id   - FK to sessions",
            "  key          - tag key (e.g. 'provider', 'session_type', 'git_branch')",
            "  value        - tag value",
            "Built-in keys: provider, session_type ('main'|'agent'), agent_project_dir,",
            "               parent_session_id, cli_version, source, compaction_summary",
            "",
            "## Example queries",
            "",
            "-- Sessions touching a file:",
            "SELECT DISTINCT s.id, s.summary, s.start_time FROM sessions s",
            "JOIN file_touches ft ON s.id = ft.session_id",
            "WHERE ft.filepath LIKE '%auth.py%' ORDER BY s.start_time DESC;",
            "",
            "-- All turns in a session:",
            "SELECT seq, user_message, timestamp FROM turns",
            "WHERE session_id = '<id>' ORDER BY seq;",
            "",
            "-- Full detail for one turn:",
            "SELECT tc.tool_name, tc.input_json, tc.output_text FROM tool_calls tc",
            "JOIN turns t ON tc.turn_id = t.id",
            "WHERE t.session_id = '<id>' AND t.seq = 3 ORDER BY tc.timestamp;",
            "",
            "-- File touches in a turn range:",
            "SELECT ft.operation, ft.filepath, tc.tool_name FROM file_touches ft",
            "JOIN turns t ON ft.turn_id = t.id",
            "JOIN tool_calls tc ON ft.tool_call_id = tc.id",
            "WHERE ft.session_id = '<id>' AND t.seq BETWEEN 5 AND 12 ORDER BY tc.timestamp;",
            "",
            "## Example rows",
        ]

        for table, rows in example_rows.items():
            lines.append(f"\n### {table} (sample)")
            for row in rows:
                lines.append(f"  {json.dumps(row, default=str)[:200]}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers — earn their place by hiding non-obvious joins
    # ------------------------------------------------------------------

    def turns(self, session_id: str) -> list[dict[str, Any]]:
        """List turns in a session with a tool preview per turn.

        Returns seq, user_message, timestamp, and a comma-separated list of
        tool names used in that turn. Use this to navigate a session before
        calling turn_detail() or file_touches() on a specific range.
        """
        return self.query("""
            SELECT
                t.seq,
                t.user_message,
                t.timestamp,
                GROUP_CONCAT(tc.tool_name, ', ') AS tools_used
            FROM turns t
            LEFT JOIN tool_calls tc ON tc.turn_id = t.id
            WHERE t.session_id = ?
            GROUP BY t.id
            ORDER BY t.seq
        """, (session_id,))

    def turn_detail(self, session_id: str, turn_seq: int) -> dict[str, Any]:
        """Full content of one turn: user message + every tool call with input/output.

        Returns a dict with keys:
          turn   - {seq, user_message, timestamp}
          calls  - list of {tool_name, input_json, output_text, timestamp}
        """
        turn_rows = self.query("""
            SELECT id, seq, user_message, timestamp
            FROM turns WHERE session_id = ? AND seq = ?
        """, (session_id, turn_seq))

        if not turn_rows:
            return {"turn": None, "calls": []}

        turn = turn_rows[0]
        calls = self.query("""
            SELECT tool_name, input_json, output_text, timestamp
            FROM tool_calls
            WHERE turn_id = ?
            ORDER BY timestamp
        """, (turn["id"],))

        return {"turn": turn, "calls": calls}

    def file_touches(
        self,
        session_id: str,
        *,
        from_turn: int | None = None,
        to_turn: int | None = None,
    ) -> list[dict[str, Any]]:
        """File touches in a session, optionally sliced to a turn range (inclusive seq ints).

        The join across file_touches → tool_calls → turns is non-obvious;
        this helper does it correctly and returns a flat list ordered by time.

        filepath for 'bash' operations is the raw command string, not a path —
        filter with LIKE if you want to find specific files within bash calls.
        """
        conditions = ["ft.session_id = ?"]
        params: list[Any] = [session_id]

        if from_turn is not None:
            conditions.append("t.seq >= ?")
            params.append(from_turn)
        if to_turn is not None:
            conditions.append("t.seq <= ?")
            params.append(to_turn)

        where = " AND ".join(conditions)
        return self.query(f"""
            SELECT
                t.seq AS turn_seq,
                t.user_message,
                tc.tool_name,
                ft.operation,
                ft.filepath,
                tc.timestamp
            FROM file_touches ft
            JOIN tool_calls tc ON ft.tool_call_id = tc.id
            JOIN turns t ON ft.turn_id = t.id
            WHERE {where}
            ORDER BY tc.timestamp
        """, tuple(params))

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    def tag(self, session_id: str, **kwargs: str) -> None:
        """Set arbitrary KV tags on a session.

        idx.tag("abc-123", env="prod", git_branch="main")
        """
        with self._conn:
            for key, value in kwargs.items():
                self._conn.execute("""
                    INSERT INTO session_tags (session_id, key, value)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id, key) DO UPDATE SET value=excluded.value
                """, (session_id, key, str(value)))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> SessionIndex:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cc_decode_path(encoded: str) -> str:
    """Decode Claude Code project dir name to filesystem path.

    '-Users-foo-bar' -> '/Users/foo/bar'
    """
    if encoded.startswith("-"):
        return "/" + encoded[1:].replace("-", "/")
    return encoded.replace("-", "/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="session_index", description="Session index CLI")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to SQLite db")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sync_p = sub.add_parser("sync", help="Ingest new/modified sessions")
    sync_p.add_argument("--verbose", "-v", action="store_true")

    query_p = sub.add_parser("query", help="Run SQL against the index")
    query_p.add_argument("sql", help="SQL query string")

    tag_p = sub.add_parser("tag", help="Tag a session with KV metadata")
    tag_p.add_argument("session_id")
    tag_p.add_argument("tags", nargs="+", help="key=value pairs")

    why_p = sub.add_parser("why", help="Why did we touch a file?")
    why_p.add_argument("filepath")
    why_p.add_argument("--session", help="Restrict to session ID")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with SessionIndex(args.db) as idx:
        if args.cmd == "sync":
            stats = idx.sync(verbose=args.verbose)
            print(f"indexed={stats['indexed']} skipped={stats['skipped']} errored={stats['errored']}")

        elif args.cmd == "query":
            rows = idx.query(args.sql)
            for row in rows:
                print(json.dumps(row, default=str))

        elif args.cmd == "tag":
            kwargs: dict[str, str] = {}
            for pair in args.tags:
                if "=" not in pair:
                    print(f"invalid tag (expected key=value): {pair}", file=sys.stderr)
                    sys.exit(1)
                k, v = pair.split("=", 1)
                kwargs[k] = v
            idx.tag(args.session_id, **kwargs)
            print(f"tagged {args.session_id}")

        elif args.cmd == "why":
            rows = idx.why(args.filepath, session_id=args.session)
            for row in rows:
                print(json.dumps(row, default=str))


if __name__ == "__main__":
    _cli()
