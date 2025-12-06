"""Session storage implementations.

SessionStore protocol and FileSessionStore implementation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import trio

if TYPE_CHECKING:
    pass

from .dtypes import (
    AgentSession,
    EndpointConfig,
    EnvironmentConfig,
    SessionMessage,
    SessionStatus,
)


def generate_session_id() -> str:
    """Generate a unique session ID.

    Format: timestamp_random (e.g., "20241205_143052_a1b2c3")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = os.urandom(3).hex()
    return f"{timestamp}_{random_suffix}"


class SessionStore(Protocol):
    """Storage backend for AgentSessions.

    Implementations should be frozen dataclasses (just config, no mutable state).
    This allows SessionStore to be serializable and passed around freely.
    """

    # Core CRUD
    async def create(
        self,
        endpoint: EndpointConfig,
        environment: EnvironmentConfig,
        parent_id: str | None = None,
        branch_point: int | None = None,
        tags: dict[str, str] | None = None,
    ) -> AgentSession:
        """Create new session, return AgentSession with generated session_id."""
        ...

    async def get(self, session_id: str) -> tuple[AgentSession | None, str | None]:
        """Load session by ID. Returns (session, None) or (None, error)."""
        ...

    async def update(
        self,
        session_id: str,
        status: SessionStatus | None = None,
        environment_state: dict | None = None,
        reward: float | dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
    ) -> tuple[None, str | None]:
        """Update session metadata. Returns (None, None) or (None, error)."""
        ...

    # Streaming append
    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        """Append message to trajectory (streaming, append-only)."""
        ...

    # Queries
    async def list(
        self,
        filter_tags: dict[str, str] | None = None,
        status: SessionStatus | None = None,
        limit: int = 100,
    ) -> list[AgentSession]:
        """List sessions, optionally filtered by tags and status."""
        ...

    async def list_children(self, parent_id: str) -> list[AgentSession]:
        """List child sessions (branches/resumes)."""
        ...

    async def get_latest(
        self,
        status: SessionStatus | None = None,
    ) -> tuple[AgentSession | None, str | None]:
        """Get the most recent session, optionally filtered by status."""
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

    def _ensure_base_dir(self) -> None:
        """Lazy directory creation - called by methods that need it.

        Avoids side effects in __post_init__ which would break frozen dataclass semantics.
        Idempotent due to exist_ok=True.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        """Get the directory for a session."""
        return self.base_dir / session_id

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

    async def create(
        self,
        endpoint: EndpointConfig,
        environment: EnvironmentConfig,
        parent_id: str | None = None,
        branch_point: int | None = None,
        tags: dict[str, str] | None = None,
    ) -> AgentSession:
        """Create new session, return AgentSession with generated session_id."""
        self._ensure_base_dir()
        session_id = generate_session_id()
        session_dir = self._session_dir(session_id)
        session_dir.mkdir()

        now = datetime.now().isoformat()
        session = AgentSession(
            session_id=session_id,
            parent_id=parent_id,
            branch_point=branch_point,
            endpoint=endpoint,
            environment=environment,
            messages=[],
            status=SessionStatus.PENDING,
            tags=tags or {},
            created_at=now,
            updated_at=now,
        )

        # Write session.json
        await self._write_json(session_dir / "session.json", session.to_dict())

        # Create empty messages.jsonl
        (session_dir / "messages.jsonl").touch()

        return session

    async def get(self, session_id: str) -> tuple[AgentSession | None, str | None]:
        """Load session by ID. Returns (session, None) or (None, error)."""
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return None, f"Session not found: {session_id}"

        # Load session.json
        session_data = await self._read_json(session_dir / "session.json")

        # Load messages.jsonl
        messages: list[SessionMessage] = []
        messages_file = session_dir / "messages.jsonl"
        if messages_file.exists():
            async with await trio.open_file(messages_file, "r") as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        messages.append(SessionMessage.from_dict(json.loads(line)))

        return AgentSession.from_dict(session_data, messages), None

    async def update(
        self,
        session_id: str,
        status: SessionStatus | None = None,
        environment_state: dict | None = None,
        reward: float | dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
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

        session_data["updated_at"] = datetime.now().isoformat()

        # Write back
        await self._write_json(session_dir / "session.json", session_data)

        return None, None

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        """Append message to trajectory (streaming, append-only)."""
        session_dir = self._session_dir(session_id)
        messages_file = session_dir / "messages.jsonl"

        # Add timestamp if not present
        if message.timestamp is None:
            message = SessionMessage(
                role=message.role,
                content=message.content,
                tool_call_id=message.tool_call_id,
                timestamp=datetime.now().isoformat(),
            )

        # Append-only (streaming safe)
        async with await trio.open_file(messages_file, "a") as f:
            await f.write(json.dumps(message.to_dict()) + "\n")

    async def list(
        self,
        filter_tags: dict[str, str] | None = None,
        status: SessionStatus | None = None,
        limit: int = 100,
    ) -> list[AgentSession]:
        """List sessions, optionally filtered by tags and status."""
        self._ensure_base_dir()

        sessions: list[AgentSession] = []

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

            # Create session without loading messages
            session = AgentSession.from_dict(session_data, messages=[])
            sessions.append(session)

            if len(sessions) >= limit:
                break

        return sessions

    async def list_children(self, parent_id: str) -> list[AgentSession]:
        """List child sessions (branches/resumes)."""
        self._ensure_base_dir()

        children: list[AgentSession] = []

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
                session = AgentSession.from_dict(session_data, messages=[])
                children.append(session)

        return children

    async def get_latest(
        self,
        status: SessionStatus | None = None,
    ) -> tuple[AgentSession | None, str | None]:
        """Get the most recent session, optionally filtered by status."""
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
