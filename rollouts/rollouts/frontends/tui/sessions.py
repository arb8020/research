"""
Session persistence for TUI agent.

Functional approach: frozen Session dataclass + pure functions.
Format: JSONL with session header + message entries.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable
import json
import uuid

from rollouts.dtypes import Message


# ── Session dataclass ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Session:
    """Immutable session reference - just the file path and ID."""
    file_path: Path
    session_id: str


@dataclass(frozen=True)
class SessionHeader:
    """Session metadata written as first line of JSONL."""
    type: str  # Always "session"
    id: str
    timestamp: str
    cwd: str
    provider: str
    model: str


@dataclass(frozen=True)
class MessageEntry:
    """Message entry in session JSONL."""
    type: str  # Always "message"
    timestamp: str
    message: dict  # Serialized Message


# ── Path utilities ────────────────────────────────────────────────────────────


def get_sessions_dir(working_dir: Path) -> Path:
    """Get session directory for a working directory.

    Returns: ~/.rollouts/sessions/--encoded-path--/

    Path encoding: /Users/foo/myproject → --Users-foo-myproject--
    """
    # Encode path: replace leading slash, then all slashes with dashes
    path_str = str(working_dir.resolve())
    encoded = "--" + path_str.lstrip("/").replace("/", "-") + "--"

    sessions_dir = Path.home() / ".rollouts" / "sessions" / encoded
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def _generate_session_filename(session_id: str) -> str:
    """Generate timestamped session filename."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return f"{timestamp}_{session_id}.jsonl"


# ── Session creation ──────────────────────────────────────────────────────────


def create_session(
    working_dir: Path,
    provider: str,
    model: str,
) -> Session:
    """Create new session file with header.

    Returns: Session with new file path and ID.
    """
    session_id = str(uuid.uuid4())
    sessions_dir = get_sessions_dir(working_dir)
    filename = _generate_session_filename(session_id)
    file_path = sessions_dir / filename

    # Write header
    header = SessionHeader(
        type="session",
        id=session_id,
        timestamp=datetime.now().isoformat(),
        cwd=str(working_dir.resolve()),
        provider=provider,
        model=model,
    )

    with open(file_path, "w") as f:
        f.write(json.dumps(asdict(header)) + "\n")

    return Session(file_path=file_path, session_id=session_id)


# ── Session discovery ─────────────────────────────────────────────────────────


def find_latest_session(working_dir: Path) -> Session | None:
    """Find most recently modified session for working directory.

    Returns: Session if found, None otherwise.
    """
    sessions_dir = get_sessions_dir(working_dir)

    if not sessions_dir.exists():
        return None

    # Find all .jsonl files, sort by modification time
    session_files = list(sessions_dir.glob("*.jsonl"))
    if not session_files:
        return None

    latest = max(session_files, key=lambda p: p.stat().st_mtime)
    return load_session(latest)


def load_session(file_path: Path) -> Session:
    """Load session from file path.

    Reads header to extract session ID.
    """
    assert file_path.exists(), f"Session file not found: {file_path}"

    with open(file_path) as f:
        first_line = f.readline()

    header = json.loads(first_line)
    assert header.get("type") == "session", f"Invalid session header: {header}"

    return Session(
        file_path=file_path,
        session_id=header["id"],
    )


def list_sessions(working_dir: Path) -> list[Session]:
    """List all sessions for a working directory, newest first."""
    sessions_dir = get_sessions_dir(working_dir)

    if not sessions_dir.exists():
        return []

    session_files = list(sessions_dir.glob("*.jsonl"))
    # Sort by modification time, newest first
    session_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return [load_session(f) for f in session_files]


@dataclass(frozen=True)
class SessionInfo:
    """Session metadata for display in picker."""
    session: Session
    header: SessionHeader
    message_count: int
    first_user_message: str | None
    modified: datetime


def get_session_info(session: Session) -> SessionInfo:
    """Get detailed info about a session for display."""
    header = load_header(session)
    messages = load_messages(session)

    # Find first user message for preview
    first_user = None
    for msg in messages:
        if msg.role == "user":
            content = msg.content
            if isinstance(content, str):
                first_user = content[:60] + "..." if len(content) > 60 else content
            elif isinstance(content, list):
                # Extract text from content blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        first_user = text[:60] + "..." if len(text) > 60 else text
                        break
            break

    modified = datetime.fromtimestamp(session.file_path.stat().st_mtime)

    return SessionInfo(
        session=session,
        header=header,
        message_count=len(messages),
        first_user_message=first_user,
        modified=modified,
    )


def list_sessions_with_info(working_dir: Path) -> list[SessionInfo]:
    """List all sessions with detailed info, newest first."""
    sessions = list_sessions(working_dir)
    return [get_session_info(s) for s in sessions]


# ── Message I/O ───────────────────────────────────────────────────────────────


def append_message(session: Session, message: Message) -> None:
    """Append message entry to session file."""
    entry = MessageEntry(
        type="message",
        timestamp=datetime.now().isoformat(),
        message=json.loads(message.to_json()),
    )

    with open(session.file_path, "a") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def load_messages(session: Session) -> list[Message]:
    """Load all messages from session file."""
    messages = []

    with open(session.file_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") == "message":
                messages.append(Message.from_json(json.dumps(entry["message"])))

    return messages


def load_header(session: Session) -> SessionHeader:
    """Load session header."""
    with open(session.file_path) as f:
        first_line = f.readline()

    data = json.loads(first_line)
    return SessionHeader(
        type=data["type"],
        id=data["id"],
        timestamp=data["timestamp"],
        cwd=data["cwd"],
        provider=data["provider"],
        model=data["model"],
    )


# ── Branching ─────────────────────────────────────────────────────────────────


def branch_session(
    source: Session,
    branch_after_idx: int,
    working_dir: Path,
    provider: str,
    model: str,
) -> Session:
    """Create new session with messages up to branch_after_idx (inclusive).

    Args:
        source: Session to branch from
        branch_after_idx: Include messages 0..branch_after_idx
        working_dir: Working directory for new session
        provider: Provider for new session
        model: Model for new session

    Returns: New Session with branched messages.
    """
    # Load messages up to branch point
    all_messages = load_messages(source)
    branched_messages = all_messages[:branch_after_idx + 1]

    # Create new session
    new_session = create_session(working_dir, provider, model)

    # Write messages
    for msg in branched_messages:
        append_message(new_session, msg)

    return new_session


# ── Compaction ────────────────────────────────────────────────────────────────


def compact_session(
    source: Session,
    summarize_fn: Callable[[list[Message]], str],
    keep_last_n: int,
    working_dir: Path,
    provider: str,
    model: str,
) -> Session:
    """Create new session with old messages summarized.

    Args:
        source: Session to compact
        summarize_fn: Function to summarize messages (typically LLM call)
        keep_last_n: Number of recent messages to keep verbatim
        working_dir: Working directory for new session
        provider: Provider for new session
        model: Model for new session

    Returns: New Session with compacted messages.
    """
    all_messages = load_messages(source)

    # Nothing to compact
    if len(all_messages) <= keep_last_n:
        return source

    # Split messages
    old_messages = all_messages[:-keep_last_n]
    recent_messages = all_messages[-keep_last_n:]

    # Generate summary
    summary = summarize_fn(old_messages)
    summary_message = Message(
        role="user",
        content=f"[Previous conversation summary]\n{summary}",
    )

    # Create new session with summary + recent messages
    new_session = create_session(working_dir, provider, model)
    append_message(new_session, summary_message)
    for msg in recent_messages:
        append_message(new_session, msg)

    return new_session


# ── Delete session ────────────────────────────────────────────────────────────


def delete_session(session: Session) -> None:
    """Delete a session file.

    Args:
        session: Session to delete
    """
    if session.file_path.exists():
        session.file_path.unlink()


def clear_all_sessions(working_dir: Path) -> int:
    """Delete all sessions in the working directory.

    Args:
        working_dir: Working directory containing .rollouts folder

    Returns:
        Number of sessions deleted
    """
    sessions = list_sessions(working_dir)
    for session in sessions:
        delete_session(session)
    return len(sessions)
