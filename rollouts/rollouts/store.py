"""Session storage implementations.

SessionStore protocol and FileSessionStore implementation.
"""

# TODO(database): Consider migrating to database backend for session storage
#
# Motivation:
# - Full-text search on message content (find sessions by conversation)
# - Complex queries for training data (filter by model, reward, status, etc.)
# - Atomic transactions (fixes read-modify-write races in update())
# - Better multi-process safety (WAL mode vs current filesystem races)
#
# Implementation approach:
# - Create new DatabaseSessionStore implementing SessionStore protocol
# - Keep FileSessionStore as fallback (human-readable, easy debugging)
# - Select via config/env var at runtime
#
# Open questions:
# - SQLite vs DuckDB: SQLite better for write-heavy (append_message),
#   DuckDB better for analytics (aggregate rewards across sessions)
# - ORM (SQLAlchemy) vs raw SQL: ORM convenient if schema evolves often
# - Migration tooling: Alembic (heavy, needs ORM) vs yoyo-migrations (lightweight)
#   vs manual PRAGMA user_version (simple but manual)
#
# Schema sketch:
# - sessions: id, parent_id, branch_point, endpoint_json, environment_json,
#             status, reward_json, tags_json, created_at, updated_at
# - messages: id, session_id (FK), role, content_json, provider, model,
#             timestamp, position (for ordering)
# - FTS virtual table on message content for search

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import trio

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .dtypes import RequestSpan

from .core import (
    Endpoint,
    EnvironmentConfig,
    Message,
    SessionSummary,
    StopReason,
    Trajectory,
    TrajectoryEnvironment,
    TrajectorySession,
)

_UNSET = object()


def generate_session_id() -> str:
    """Generate a unique session ID.

    Format: timestamp_random (e.g., "20241205_143052_a1b2c3")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = os.urandom(3).hex()
    return f"{timestamp}_{random_suffix}"


def generate_message_id() -> str:
    """Generate a unique message ID within a session.

    Short hex (8 chars). Collisions within a session are astronomically
    unlikely at our message counts (~10^4 messages max would give < 10^-10
    collision probability in a 2^32 space).
    """
    return os.urandom(4).hex()


class SessionStore(Protocol):
    """Storage backend for persisted trajectories and live session handles.

    Implementations should be frozen dataclasses (just config, no mutable state).
    This allows SessionStore to be serializable and passed around freely.
    """

    # Core CRUD
    async def create(
        self,
        endpoint: Endpoint,
        environment: EnvironmentConfig,
        parent_id: str | None = None,
        branch_point: int | None = None,
        tags: dict[str, str] | None = None,
        vcs: dict[str, str] | None = None,
    ) -> Trajectory:
        """Create new session, return a canonical persisted trajectory with generated session_id.

        Args:
            vcs: Optional VCS info for agent-trace attribution.
                 Format: {"type": "git", "revision": "abc...", "root": "/path"}
                 Use agent_trace.get_git_info() to capture this.
        """
        ...

    async def get(self, session_id: str) -> tuple[Trajectory | None, str | None]:
        """Load a full persisted trajectory by ID."""
        ...

    async def get_trajectory(self, session_id: str) -> tuple[Trajectory | None, str | None]:
        """Load a canonical trajectory by session ID."""
        ...

    async def save_trajectory(self, trajectory: Trajectory) -> tuple[Trajectory | None, str | None]:
        """Persist a canonical trajectory and return the persisted trajectory."""
        ...

    async def update(
        self,
        session_id: str,
        stop_reason: StopReason | None | object = _UNSET,
        environment_state: dict | None = None,
        tags: dict[str, str] | None = None,
        endpoint: Endpoint | None = None,
    ) -> tuple[None, str | None]:
        """Update session metadata. Returns (None, None) or (None, error)."""
        ...

    # Streaming append
    async def append_message(self, session_id: str, message: Message) -> Message:
        """Append message to trajectory (streaming, append-only).

        Returns the message as stored, with `id` and `parent_id` resolved
        (sub-step 1b of the session refactor). Callers that thread a leaf
        cursor should use the returned message's id as the next parent.
        """
        ...

    # Queries
    async def list(
        self,
        filter_tags: dict[str, str] | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[SessionSummary]:
        """List session summaries, optionally filtered by tags and status."""
        ...

    async def list_children(self, parent_id: str) -> list[SessionSummary]:
        """List child session summaries (branches/resumes)."""
        ...

    async def get_latest(
        self,
        status: str | None = None,
    ) -> tuple[SessionSummary | None, str | None]:
        """Get the most recent session summary, optionally filtered by status."""
        ...

    # Cleanup
    async def delete(self, session_id: str) -> tuple[None, str | None]:
        """Delete session and associated data."""
        ...


@dataclass(frozen=True)
class FileSessionStore:
    """File-based implementation of SessionStore.

    Frozen dataclass - just holds the base_dir path. Serializable.
    Methods are essentially pure functions that take base_dir as implicit arg.

    Layout:
        ~/.rollouts/sessions/
            <session_id>/
                session.json     # metadata: endpoint, environment, tags, status, parent_id, etc.
                messages.jsonl   # trajectory (append-only)
                environment.json # serialized env state (written at checkpoints)
    """

    base_dir: Path = Path.home() / ".rollouts" / "sessions"
    atif_filename: str | None = None
    atif_output_path: Path | None = None

    def _ensure_base_dir(self) -> None:
        """Lazy directory creation - called by methods that need it.

        Avoids side effects in __post_init__ which would break frozen dataclass semantics.
        Idempotent due to exist_ok=True.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        """Get the directory for a session."""
        return self.base_dir / session_id

    def _atif_path(self, session_id: str) -> Path | None:
        """Get the optional ATIF artifact path for a session."""
        if self.atif_filename is None:
            return None
        return self._session_dir(session_id) / self.atif_filename

    async def _write_json(self, path: Path, data: dict) -> None:
        """Write JSON to file atomically."""
        content = json.dumps(data, indent=2)
        # Write to temp file then rename for atomicity
        temp_path = path.with_suffix(".tmp")
        async with await trio.open_file(temp_path, "w") as f:
            await f.write(content)
        temp_path.rename(path)

    async def _read_json(self, path: Path) -> dict:
        """Read JSON from file."""
        async with await trio.open_file(path, "r") as f:
            content = await f.read()
        return json.loads(content)

    async def _sync_atif_artifact(self, session_id: str) -> None:
        """Mirror the canonical trajectory into optional Harbor ATIF artifacts."""
        trajectory, err = await self.get_trajectory(session_id)
        if err is not None or trajectory is None or not trajectory.messages:
            return

        from .adapters import trajectory_to_atif_dict

        atif_payload = trajectory_to_atif_dict(trajectory)

        session_artifact_path = self._atif_path(session_id)
        if session_artifact_path is not None:
            await self._write_json(session_artifact_path, atif_payload)

        if self.atif_output_path is not None:
            self.atif_output_path.parent.mkdir(parents=True, exist_ok=True)
            await self._write_json(self.atif_output_path, atif_payload)

    async def create(
        self,
        endpoint: Endpoint,
        environment: EnvironmentConfig,
        parent_id: str | None = None,
        branch_point: int | None = None,
        tags: dict[str, str] | None = None,
        vcs: dict[str, str] | None = None,
    ) -> Trajectory:
        """Create new session, return a persisted trajectory with generated session_id.

        Args:
            vcs: Optional VCS info for agent-trace attribution.
                 Format: {"type": "git", "revision": "abc...", "root": "/path"}
                 Use agent_trace.get_git_info() to capture this.
        """
        self._ensure_base_dir()
        session_id = generate_session_id()
        trajectory = Trajectory(
            session=TrajectorySession(
                session_id=session_id,
                parent_id=parent_id,
                branch_point=branch_point,
                endpoint=endpoint,
                tags=tags or {},
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                vcs=vcs,
            ),
            environment=TrajectoryEnvironment.from_session_parts(environment),
        )
        await self.save_trajectory(trajectory)
        return trajectory

    async def save(self, trajectory: Trajectory) -> None:
        """Save a complete trajectory snapshot.

        Used for saving transformed sessions (compact, summarize).

        Session refactor (1b): assign tree ids to messages that don't have
        them, linking in a linear chain. Also records the final leaf_id in
        session.json so subsequent appends (via append_message) thread from
        the correct point.
        """
        self._ensure_base_dir()
        session_id = trajectory.session.session_id or generate_session_id()
        if not trajectory.session.session_id:
            trajectory = replace(
                trajectory,
                session=replace(trajectory.session, session_id=session_id),
            )

        session_dir = self._session_dir(session_id)
        session_dir.mkdir(exist_ok=True)

        # Assign tree ids for any messages missing them, linking linearly.
        # Messages that already carry ids are kept as-is (caller-provided
        # tree structure wins).
        assigned_messages: list[Message] = []
        prev_id: str | None = None
        for msg in trajectory.messages:
            if msg.id is None:
                msg_id = generate_message_id()
                parent = msg.parent_id if msg.parent_id is not None else prev_id
                msg = replace(msg, id=msg_id, parent_id=parent)
            assigned_messages.append(msg)
            prev_id = msg.id

        # Write session.json with leaf_id pointing at the last assigned id.
        session_record = trajectory.to_session_record()
        if prev_id is not None:
            session_record["leaf_id"] = prev_id
        await self._write_json(session_dir / "session.json", session_record)

        # Write messages.jsonl
        messages_file = session_dir / "messages.jsonl"
        async with await trio.open_file(messages_file, "w") as f:
            for msg in assigned_messages:
                await f.write(msg.to_json() + "\n")

        await self._sync_atif_artifact(session_id)

    async def save_trajectory(self, trajectory: Trajectory) -> tuple[Trajectory | None, str | None]:
        """Persist a canonical trajectory and return the persisted trajectory."""
        session_id = trajectory.session.session_id or generate_session_id()
        if not trajectory.session.session_id:
            trajectory = replace(
                trajectory,
                session=replace(trajectory.session, session_id=session_id),
            )

        await self.save(trajectory)
        return trajectory, None

    def get_config_sync(self, session_id: str) -> tuple[dict | None, str | None]:
        """Sync load of session config (just session.json, not messages).

        Used during CLI arg parsing before async context is available.
        Returns (config_dict, None) or (None, error).
        """
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return None, f"Session not found: {session_id}"

        session_file = session_dir / "session.json"
        if not session_file.exists():
            return None, f"Session config not found: {session_id}"

        return json.loads(session_file.read_text()), None

    def get_latest_id_sync(self) -> str | None:
        """Sync get the most recent session ID.

        Used during CLI arg parsing before async context is available.
        """
        self._ensure_base_dir()
        if not self.base_dir.exists():
            return None

        session_dirs = sorted(self.base_dir.iterdir(), reverse=True)
        for session_dir in session_dirs:
            if session_dir.is_dir() and (session_dir / "session.json").exists():
                return session_dir.name
        return None

    async def get(self, session_id: str) -> tuple[Trajectory | None, str | None]:
        """Load a full persisted trajectory by ID."""
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return None, f"Session not found: {session_id}"

        # Load session.json
        session_data = await self._read_json(session_dir / "session.json")

        # Load messages.jsonl
        messages: list[Message] = []
        messages_file = session_dir / "messages.jsonl"
        if messages_file.exists():
            async with await trio.open_file(messages_file, "r") as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        messages.append(Message.from_json(line))

        # Session refactor (sub-step 1a) — migration-on-read.
        #
        # Pre-refactor sessions have no `id` / `parent_id` fields on their
        # messages. Synthesize them so loaded messages satisfy the tree
        # invariants (unique id per message, parent_id points at the prior
        # message for a linear history). Messages that already have `id`
        # are left alone.
        #
        # This only affects the in-memory view. On-disk JSONL stays as it
        # was written. Future appends to the session will read leaf_id
        # from session.json (may be absent for pre-refactor sessions,
        # which is fine — the first append starts a fresh linear thread).
        #
        # TODO(session-refactor sub-step 1b): seed messages constructed in
        # Python (initial system/user messages built before run_agent) are
        # not appended to the store until after the first LLM turn. This
        # means on disk the seed messages have id=None/parent_id=None and
        # the first truly-appended message has parent_id=None too, so the
        # on-disk tree has multiple roots. Migration-on-read patches seeds
        # into a coherent chain but the first stored message still appears
        # as a separate root on disk. Fix: either append seed messages to
        # the store in `ensure_persisted_session`, or carry a leaf cursor
        # in AgentState that's initialized from the seed and threaded into
        # the first append's `parent_id`. Lands with the explicit-cursor
        # work in sub-step 1b.
        prev_id: str | None = None
        for i, msg in enumerate(messages):
            if msg.id is not None:
                prev_id = msg.id
                continue
            synthesized_id = generate_message_id()
            messages[i] = replace(msg, id=synthesized_id, parent_id=prev_id)
            prev_id = synthesized_id

        return Trajectory.from_session_record(session_data, messages), None

    async def get_trajectory(self, session_id: str) -> tuple[Trajectory | None, str | None]:
        """Load a canonical trajectory by session ID."""
        session, err = await self.get(session_id)
        if err or session is None:
            return None, err
        return session, None

    async def update(
        self,
        session_id: str,
        stop_reason: StopReason | None | object = _UNSET,
        environment_state: dict | None = None,
        tags: dict[str, str] | None = None,
        endpoint: Endpoint | None = None,
    ) -> tuple[None, str | None]:
        """Update session metadata. Returns (None, None) or (None, error)."""
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return None, f"Session not found: {session_id}"

        # Load current session
        session_data = await self._read_json(session_dir / "session.json")

        # Update fields
        if stop_reason is not _UNSET:
            session_data["stop_reason"] = stop_reason.value if stop_reason is not None else None
        if environment_state is not None:
            session_data["environment_state"] = environment_state
        if tags is not None:
            session_data["tags"] = tags
        if endpoint is not None:
            session_data["endpoint"] = endpoint.to_dict(exclude_secrets=True)

        session_data["updated_at"] = datetime.now().isoformat()

        # Write back
        await self._write_json(session_dir / "session.json", session_data)
        await self._sync_atif_artifact(session_id)

        return None, None

    async def append_message(self, session_id: str, message: Message) -> Message:
        """Append message to trajectory (streaming, append-only).

        Session refactor (sub-step 1a): if `message.id` is unset we assign a
        fresh short hex id. If `message.parent_id` is unset we set it to the
        session's current leaf (read from session.json's `leaf_id` field).
        After the append, session.json's `leaf_id` is updated to the new
        message's id so the next append continues the linear thread.

        Callers can override either field to explicitly construct tree
        structure (branching). Sub-step 1b threads explicit cursors through
        the native loop and external adapters; 1a keeps everything
        backwards-compatible by defaulting to linear-history behavior.

        Returns the message as stored (with id and parent_id resolved) so
        callers tracking a leaf cursor can read the assigned id directly.

        See rollouts/rollouts/agents/runtime_refactor.md.
        """
        session_dir = self._session_dir(session_id)
        messages_file = session_dir / "messages.jsonl"
        session_file = session_dir / "session.json"

        # Add timestamp if not present
        if message.timestamp is None:
            message = replace(message, timestamp=datetime.now().isoformat())

        # Read current leaf from session.json to set parent_id if caller
        # didn't provide one. Tolerant of missing file / missing field —
        # older sessions pre-refactor have no leaf_id; first append starts
        # a new linear thread.
        current_leaf: str | None = None
        session_data: dict | None = None
        if session_file.exists():
            try:
                session_data = await self._read_json(session_file)
                current_leaf = session_data.get("leaf_id")
            except Exception:
                session_data = None

        # Assign id if unset.
        assigned_id = message.id or generate_message_id()

        # Parent_id contract (sub-step 1b):
        # - If caller provided parent_id, use it. This is the explicit-cursor
        #   path — native loop and migrated adapters pass their tracked leaf.
        # - If caller did not provide parent_id, fall back to session.json's
        #   leaf_id and emit a warning. This keeps existing callers working
        #   while surfacing the sites that still need cursor threading.
        #   Exception: the very first message in a session legitimately has
        #   no parent (current_leaf is None), so don't warn there.
        if message.parent_id is not None:
            assigned_parent = message.parent_id
        else:
            assigned_parent = current_leaf
            if current_leaf is not None:
                logger.warning(
                    "session_store.append_message called without explicit "
                    "parent_id for session=%s; auto-inferred parent=%s from "
                    "session.json leaf_id. Caller should thread AgentState."
                    "leaf_id (or equivalent cursor) through. See "
                    "rollouts/rollouts/agents/runtime_refactor.md.",
                    session_id,
                    current_leaf,
                )

        if message.id != assigned_id or message.parent_id != assigned_parent:
            message = replace(message, id=assigned_id, parent_id=assigned_parent)

        # Append the message (append-only, streaming safe).
        async with await trio.open_file(messages_file, "a") as f:
            await f.write(message.to_json() + "\n")

        # Update the session's leaf cursor to point at this message. This
        # is a small mutable pointer on an otherwise append-only record —
        # callers branching off an earlier message will override parent_id
        # and subsequent appends under the same session_id will extend from
        # this new leaf (which may not be what a branch author wants; that
        # case is handled by passing leaf explicitly in sub-step 1b).
        if session_data is not None:
            session_data["leaf_id"] = assigned_id
            await self._write_json(session_file, session_data)

        await self._sync_atif_artifact(session_id)
        return message

    async def append_span(self, session_id: str, span: RequestSpan) -> None:
        """Append request span to spans.jsonl (streaming, append-only).

        Captures per-request metrics: timing, tokens, cost, provider info.
        """

        session_dir = self._session_dir(session_id)
        spans_file = session_dir / "spans.jsonl"

        # Append-only (streaming safe)
        async with await trio.open_file(spans_file, "a") as f:
            await f.write(span.to_json() + "\n")

    async def load_spans(self, session_id: str) -> list[RequestSpan]:
        """Load all spans for a session."""
        from .dtypes import RequestSpan

        session_dir = self._session_dir(session_id)
        spans_file = session_dir / "spans.jsonl"

        if not spans_file.exists():
            return []

        spans = []
        async with await trio.open_file(spans_file, "r") as f:
            async for line in f:
                if line.strip():
                    spans.append(RequestSpan.from_json(line))
        return spans

    async def list(
        self,
        filter_tags: dict[str, str] | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[SessionSummary]:
        """List session summaries, optionally filtered by tags and status."""
        self._ensure_base_dir()

        sessions: list[SessionSummary] = []

        # Iterate through session directories
        if not self.base_dir.exists():
            return sessions

        session_dirs = sorted(self.base_dir.iterdir(), reverse=True)  # newest first

        for session_dir in session_dirs:
            if not session_dir.is_dir():
                continue

            session_file = session_dir / "session.json"
            if not session_file.exists():
                continue

            # Load session metadata (not messages for efficiency)
            async with await trio.open_file(session_file, "r") as f:
                content = await f.read()
            session_data = json.loads(content)

            # Filter by tags
            if filter_tags:
                session_tags = session_data.get("tags", {})
                if not all(session_tags.get(k) == v for k, v in filter_tags.items()):
                    continue

            # Count messages without loading them
            messages_file = session_dir / "messages.jsonl"
            message_count = 0
            if messages_file.exists():
                async with await trio.open_file(messages_file, "r") as f:
                    async for line in f:
                        if line.strip():
                            message_count += 1

            # Create session without loading messages
            trajectory = Trajectory.from_session_record(session_data, messages=[])
            summary = trajectory.to_session_summary(message_count=message_count)
            if status is not None and summary.status != status:
                continue
            sessions.append(summary)

            if len(sessions) >= limit:
                break

        return sessions

    async def list_children(self, parent_id: str) -> list[SessionSummary]:
        """List child session summaries (branches/resumes)."""
        self._ensure_base_dir()

        children: list[SessionSummary] = []

        if not self.base_dir.exists():
            return children

        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir():
                continue

            session_file = session_dir / "session.json"
            if not session_file.exists():
                continue

            async with await trio.open_file(session_file, "r") as f:
                content = await f.read()
            session_data = json.loads(content)

            if session_data.get("parent_id") == parent_id:
                trajectory = Trajectory.from_session_record(session_data, messages=[])
                children.append(trajectory.to_session_summary())

        return children

    async def get_latest(
        self,
        status: str | None = None,
    ) -> tuple[SessionSummary | None, str | None]:
        """Get the most recent session summary, optionally filtered by status."""
        sessions = await self.list(status=status, limit=1)
        if not sessions:
            return None, "No sessions found"
        return sessions[0], None

    async def delete(self, session_id: str) -> tuple[None, str | None]:
        """Delete session and associated data."""
        import shutil

        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return None, f"Session not found: {session_id}"

        # Remove entire directory
        shutil.rmtree(session_dir)

        return None, None


def log_crash(
    error: Exception,
    provider: str,
    model: str,
    *,
    session_id: str | None = None,
    messages: list | None = None,
) -> Path:
    """Log crash info to ~/.rollouts/crashes/ with optional message dump.

    Returns path to the crash file for reference in error messages.
    """
    import traceback

    crashes_dir = Path.home() / ".rollouts" / "crashes"
    crashes_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = os.urandom(3).hex()
    crash_file = crashes_dir / f"{timestamp}_{random_suffix}.txt"

    crash_info: dict = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "session_id": session_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }

    if messages is not None:
        crash_info["messages"] = messages

    crash_file.write_text(json.dumps(crash_info, indent=2, default=str))
    return crash_file
