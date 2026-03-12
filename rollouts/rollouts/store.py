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
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import trio

if TYPE_CHECKING:
    pass

from .core import (
    Endpoint,
    EnvironmentConfig,
    Message,
    PendingInput,
    SessionHandle,
    SessionStatus,
    SessionSummary,
    Trajectory,
    TrajectoryEnvironment,
    TrajectorySession,
)


def generate_session_id() -> str:
    """Generate a unique session ID.

    Format: timestamp_random (e.g., "20241205_143052_a1b2c3")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = os.urandom(3).hex()
    return f"{timestamp}_{random_suffix}"


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
    ) -> SessionHandle:
        """Create new session, return a live session handle with generated session_id.

        Args:
            vcs: Optional VCS info for agent-trace attribution.
                 Format: {"type": "git", "revision": "abc...", "root": "/path"}
                 Use agent_trace.get_git_info() to capture this.
        """
        ...

    async def get(self, session_id: str) -> tuple[SessionHandle | None, str | None]:
        """Load a full live session handle by ID."""
        ...

    async def get_trajectory(self, session_id: str) -> tuple[Trajectory | None, str | None]:
        """Load a canonical trajectory by session ID."""
        ...

    async def save_trajectory(
        self, trajectory: Trajectory
    ) -> tuple[SessionHandle | None, str | None]:
        """Persist a canonical trajectory and return a live session handle."""
        ...

    async def update(
        self,
        session_id: str,
        status: SessionStatus | None = None,
        environment_state: dict | None = None,
        reward: float | dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        endpoint: Endpoint | None = None,
    ) -> tuple[None, str | None]:
        """Update session metadata. Returns (None, None) or (None, error)."""
        ...

    # Streaming append
    async def append_message(self, session_id: str, message: Message) -> None:
        """Append message to trajectory (streaming, append-only)."""
        ...

    # Queries
    async def list(
        self,
        filter_tags: dict[str, str] | None = None,
        status: SessionStatus | None = None,
        limit: int = 100,
    ) -> list[SessionSummary]:
        """List session summaries, optionally filtered by tags and status."""
        ...

    async def list_children(self, parent_id: str) -> list[SessionSummary]:
        """List child session summaries (branches/resumes)."""
        ...

    async def get_latest(
        self,
        status: SessionStatus | None = None,
    ) -> tuple[SessionSummary | None, str | None]:
        """Get the most recent session summary, optionally filtered by status."""
        ...

    # Cleanup
    async def delete(self, session_id: str) -> tuple[None, str | None]:
        """Delete session and associated data."""
        ...

    async def write_pending_input(self, session_id: str, data: dict) -> tuple[None, str | None]:
        """Persist pending input metadata for detached/waiting sessions."""
        ...

    async def read_pending_input(self, session_id: str) -> dict | None:
        """Load pending input metadata for a waiting session."""
        ...

    async def clear_pending_input(self, session_id: str) -> tuple[None, str | None]:
        """Delete pending input metadata after resuming a waiting session."""
        ...

    async def enqueue_message(self, session_id: str, message: Message) -> tuple[None, str | None]:
        """Persist queued user input to be consumed on the next resume."""
        ...

    async def consume_queued_message(self, session_id: str) -> tuple[Message | None, str | None]:
        """Consume and return the oldest queued message for a session."""
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

    def _pending_input_path(self, session_id: str) -> Path:
        """Get the pending input metadata path for a session."""
        return self._session_dir(session_id) / "pending_input.json"

    def _atif_path(self, session_id: str) -> Path | None:
        """Get the optional ATIF artifact path for a session."""
        if self.atif_filename is None:
            return None
        return self._session_dir(session_id) / self.atif_filename

    def _normalize_control_state(
        self, data: dict | None
    ) -> tuple[PendingInput | None, tuple[Message, ...]]:
        if data is None:
            return None, ()

        if "pending_input" in data:
            pending_payload = data.get("pending_input")
        elif "type" in data or "prompt" in data:
            pending_payload = data
        else:
            pending_payload = None
        pending_input = None
        if isinstance(pending_payload, dict) and pending_payload:
            pending_input = PendingInput.from_dict(pending_payload)

        queued_raw = data.get("queued_messages", [])
        queued_messages: list[Message] = []
        if isinstance(queued_raw, list):
            for raw in queued_raw:
                if isinstance(raw, dict):
                    queued_messages.append(Message.from_json(json.dumps(raw)))

        return pending_input, tuple(queued_messages)

    async def _read_control_state(
        self, session_id: str
    ) -> tuple[PendingInput | None, tuple[Message, ...]]:
        pending_path = self._pending_input_path(session_id)
        if not pending_path.exists():
            return None, ()
        raw_data = await self._read_json(pending_path)
        return self._normalize_control_state(raw_data)

    async def _write_control_state(
        self,
        session_id: str,
        *,
        pending_input: PendingInput | None,
        queued_messages: tuple[Message, ...],
    ) -> tuple[None, str | None]:
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return None, f"Session not found: {session_id}"

        if pending_input is None and not queued_messages:
            pending_path = self._pending_input_path(session_id)
            if pending_path.exists():
                pending_path.unlink()
            return None, None

        payload: dict[str, object] = {
            "queued_messages": [json.loads(msg.to_json()) for msg in queued_messages]
        }
        if pending_input is not None:
            payload["pending_input"] = pending_input.to_dict()
        await self._write_json(self._pending_input_path(session_id), payload)
        return None, None

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
    ) -> SessionHandle:
        """Create new session, return a live session handle with generated session_id.

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
                status=SessionStatus.PENDING.value,
                tags=tags or {},
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                vcs=vcs,
            ),
            environment=TrajectoryEnvironment.from_session_parts(environment),
        )
        await self.save_trajectory(trajectory)
        return SessionHandle.from_trajectory(trajectory)

    async def save(self, trajectory: Trajectory) -> None:
        """Save a complete trajectory snapshot.

        Used for saving transformed sessions (compact, summarize).
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

        # Write session.json
        await self._write_json(session_dir / "session.json", trajectory.to_session_record())

        # Write messages.jsonl
        messages_file = session_dir / "messages.jsonl"
        async with await trio.open_file(messages_file, "w") as f:
            for msg in trajectory.messages:
                await f.write(msg.to_json() + "\n")

        await self._sync_atif_artifact(session_id)

    async def save_trajectory(
        self, trajectory: Trajectory
    ) -> tuple[SessionHandle | None, str | None]:
        """Persist a canonical trajectory and return a live session handle."""
        session_id = trajectory.session.session_id or generate_session_id()
        if not trajectory.session.session_id:
            trajectory = replace(
                trajectory,
                session=replace(trajectory.session, session_id=session_id),
            )

        await self.save(trajectory)
        return SessionHandle.from_trajectory(trajectory), None

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

    async def get(self, session_id: str) -> tuple[SessionHandle | None, str | None]:
        """Load a full session handle by ID."""
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

        pending_input, queued_messages = await self._read_control_state(session_id)
        return SessionHandle.from_dict(
            session_data,
            messages,
            pending_input=pending_input,
            queued_messages=queued_messages,
        ), None

    async def get_trajectory(self, session_id: str) -> tuple[Trajectory | None, str | None]:
        """Load a canonical trajectory by session ID."""
        session, err = await self.get(session_id)
        if err or session is None:
            return None, err
        return session.to_trajectory(), None

    async def update(
        self,
        session_id: str,
        status: SessionStatus | None = None,
        environment_state: dict | None = None,
        reward: float | dict[str, float] | None = None,
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
        if status is not None:
            session_data["status"] = status.value
        if environment_state is not None:
            session_data["environment_state"] = environment_state
        if reward is not None:
            session_data["reward"] = reward
        if tags is not None:
            session_data["tags"] = tags
        if endpoint is not None:
            session_data["endpoint"] = endpoint.to_dict(exclude_secrets=True)

        session_data["updated_at"] = datetime.now().isoformat()

        # Write back
        await self._write_json(session_dir / "session.json", session_data)
        await self._sync_atif_artifact(session_id)

        return None, None

    async def append_message(self, session_id: str, message: Message) -> None:
        """Append message to trajectory (streaming, append-only)."""
        session_dir = self._session_dir(session_id)
        messages_file = session_dir / "messages.jsonl"

        # Add timestamp if not present
        if message.timestamp is None:
            message = replace(message, timestamp=datetime.now().isoformat())

        # Append-only (streaming safe)
        async with await trio.open_file(messages_file, "a") as f:
            await f.write(message.to_json() + "\n")

        await self._sync_atif_artifact(session_id)

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
        status: SessionStatus | None = None,
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

            # Filter by status
            if status is not None and session_data.get("status") != status.value:
                continue

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
            sessions.append(trajectory.to_session_summary(message_count=message_count))

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
        status: SessionStatus | None = None,
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

    async def write_pending_input(self, session_id: str, data: dict) -> tuple[None, str | None]:
        """Persist pending input metadata and mark the session waiting."""
        pending_input = PendingInput.from_dict(data)
        _existing_pending, queued_messages = await self._read_control_state(session_id)
        _, err = await self._write_control_state(
            session_id,
            pending_input=pending_input,
            queued_messages=queued_messages,
        )
        if err is not None:
            return None, err
        return await self.update(session_id, status=SessionStatus.WAITING)

    async def read_pending_input(self, session_id: str) -> dict | None:
        """Load pending input metadata if present."""
        pending_input, _queued_messages = await self._read_control_state(session_id)
        if pending_input is None:
            return None
        return pending_input.to_dict()

    async def clear_pending_input(self, session_id: str) -> tuple[None, str | None]:
        """Delete pending input metadata and mark the session pending again."""
        _pending_input, queued_messages = await self._read_control_state(session_id)
        _, err = await self._write_control_state(
            session_id,
            pending_input=None,
            queued_messages=queued_messages,
        )
        if err is not None:
            return None, err
        return await self.update(session_id, status=SessionStatus.PENDING)

    async def enqueue_message(self, session_id: str, message: Message) -> tuple[None, str | None]:
        """Queue a user message to be consumed on the next resumed turn."""
        pending_input, queued_messages = await self._read_control_state(session_id)
        next_queue = queued_messages + (message,)
        return await self._write_control_state(
            session_id,
            pending_input=pending_input,
            queued_messages=next_queue,
        )

    async def consume_queued_message(self, session_id: str) -> tuple[Message | None, str | None]:
        """Consume the oldest queued user message for a session."""
        pending_input, queued_messages = await self._read_control_state(session_id)
        if not queued_messages:
            return None, None

        next_message = queued_messages[0]
        _, err = await self._write_control_state(
            session_id,
            pending_input=pending_input,
            queued_messages=queued_messages[1:],
        )
        if err is not None:
            return None, err
        return next_message, None


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
