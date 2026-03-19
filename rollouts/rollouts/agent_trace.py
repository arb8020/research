"""Agent Trace support for code attribution.

Implements the agent-trace.dev spec (v0.1.0) for tracking AI-generated code.
https://agent-trace.dev/

This module provides:
- Data types for trace records, file attributions, and line ranges
- VCS (git) metadata capture
- Conversion from rollouts trajectories to agent-trace format

Usage:

    # 1. Capture VCS info when creating a session
    from rollouts.agent_trace import capture_vcs_for_session

    vcs = capture_vcs_for_session(Path.cwd())
    session = await store.create(endpoint, env, vcs=vcs)

    # 2. After session completes, export the trace
    from rollouts.agent_trace import export_session_trace

    session, _ = await store.get(session_id)
    trace_path = export_session_trace(session)
    # Writes to .agent-trace.jsonl in repo root

For tools to be tracked, they must populate ToolResult.details with:
    {
        "file_path": "/abs/path/to/file.py",
        "start_line": 10,
        "end_line": 25,
        "operation": "edit",  # or "create"
    }

The coding and git_worktree environments do this automatically for
write and edit operations.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .core import JsonSerializable, Message, Trajectory

# -----------------------------------------------------------------------------
# agent-trace.dev spec types (v0.1.0)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Contributor(JsonSerializable):
    """Attribution for who wrote the code.

    type: "human" | "ai" | "mixed" | "unknown"
    model_id: models.dev format, e.g. "anthropic/claude-sonnet-4-5-20250929"
    """

    type: str  # "human", "ai", "mixed", "unknown"
    model_id: str | None = None


@dataclass(frozen=True)
class LineRange(JsonSerializable):
    """A range of lines in a file (1-indexed, inclusive).

    content_hash: Optional position-independent tracking, format "algorithm:hash"
    contributor: Optional override for this specific range
    """

    start_line: int
    end_line: int
    content_hash: str | None = None
    contributor: Contributor | None = None

    def __post_init__(self) -> None:
        assert self.start_line >= 1, f"start_line must be >= 1, got {self.start_line}"
        assert self.end_line >= self.start_line, (
            f"end_line ({self.end_line}) must be >= start_line ({self.start_line})"
        )


@dataclass(frozen=True)
class Conversation(JsonSerializable):
    """A conversation that contributed code to a file.

    Groups line ranges by their source conversation/session.
    """

    contributor: Contributor
    ranges: tuple[LineRange, ...]
    url: str | None = None  # Link to conversation/session viewer
    related: tuple[str, ...] | None = None  # Links to sub-sessions, prompts, etc.


@dataclass(frozen=True)
class FileAttribution(JsonSerializable):
    """Attribution data for a single file."""

    path: str  # Repo-relative path
    conversations: tuple[Conversation, ...]


@dataclass(frozen=True)
class VCSInfo(JsonSerializable):
    """Version control metadata.

    type: "git" | "jj" | "hg" | "svn"
    revision: Commit SHA (git), change ID (jj), changeset (hg), etc.
    """

    type: str
    revision: str


@dataclass(frozen=True)
class ToolInfo(JsonSerializable):
    """Tool that generated the trace."""

    name: str
    version: str


@dataclass(frozen=True)
class TraceRecord(JsonSerializable):
    """A complete agent-trace record.

    This is the top-level data structure written to .agent-trace.jsonl
    """

    version: str  # Spec version, e.g. "0.1.0"
    id: str  # UUID
    timestamp: str  # RFC 3339
    files: tuple[FileAttribution, ...]
    vcs: VCSInfo | None = None
    tool: ToolInfo | None = None
    metadata: dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# VCS helpers
# -----------------------------------------------------------------------------


def get_git_info(repo_path: Path | None = None) -> VCSInfo | None:
    """Capture git metadata from repo.

    Returns None if not in a git repo or git not available.
    """
    try:
        cwd = str(repo_path) if repo_path else None
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        revision = result.stdout.strip()
        assert len(revision) == 40, f"Expected 40-char SHA, got {len(revision)}"
        return VCSInfo(type="git", revision=revision)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def get_git_root(path: Path | None = None) -> Path | None:
    """Get the root directory of the git repo containing path.

    Returns None if not in a git repo.
    """
    try:
        cwd = str(path) if path else None
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        return Path(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def capture_vcs_for_session(working_dir: Path | None = None) -> dict[str, str] | None:
    """Capture VCS info for storing in session metadata.

    Returns a dict suitable for persisted session VCS metadata, or None if not in a repo.

    Usage:
        vcs = capture_vcs_for_session(Path.cwd())
        session = await store.create(endpoint, env, vcs=vcs)
    """
    git_info = get_git_info(working_dir)
    if git_info is None:
        return None

    git_root = get_git_root(working_dir)
    result = {
        "type": git_info.type,
        "revision": git_info.revision,
    }
    if git_root is not None:
        result["root"] = str(git_root)
    return result


# -----------------------------------------------------------------------------
# Model ID formatting (models.dev convention)
# -----------------------------------------------------------------------------


def format_model_id(provider: str, model: str) -> str:
    """Format provider/model into models.dev format.

    Examples:
        format_model_id("anthropic", "claude-sonnet-4-5-20250929")
        -> "anthropic/claude-sonnet-4-5-20250929"
    """
    assert provider, "provider must not be empty"
    assert model, "model must not be empty"
    return f"{provider}/{model}"


# -----------------------------------------------------------------------------
# File edit details schema
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FileEditDetails(JsonSerializable):
    """Standardized details for file edit tool results.

    Tools that modify files should populate ToolResultReceived.details
    with this schema to enable agent-trace attribution.
    """

    file_path: str  # Absolute or repo-relative path
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed, inclusive
    operation: str  # "create", "edit", "delete"
    content_hash: str | None = None  # "algorithm:hash", e.g. "murmur3:abc123"

    def __post_init__(self) -> None:
        assert self.file_path, "file_path must not be empty"
        assert self.start_line >= 1, f"start_line must be >= 1, got {self.start_line}"
        assert self.end_line >= self.start_line, (
            f"end_line ({self.end_line}) must be >= start_line ({self.start_line})"
        )
        assert self.operation in ("create", "edit", "delete"), (
            f"operation must be create/edit/delete, got {self.operation}"
        )


# -----------------------------------------------------------------------------
# Trajectory to agent-trace conversion
# -----------------------------------------------------------------------------


def extract_file_edits(messages: list[Message]) -> list[FileEditDetails]:
    """Extract file edit details from a message trajectory.

    Looks for tool results with 'file_edit' or similar details.
    Returns empty list if no structured edit details found.
    """
    edits: list[FileEditDetails] = []
    for msg in messages:
        if msg.role != "tool":
            continue
        if msg.details is None:
            continue
        # Check for file_edit schema in details
        if "file_path" in msg.details and "start_line" in msg.details:
            try:
                edit = FileEditDetails(
                    file_path=msg.details["file_path"],
                    start_line=msg.details["start_line"],
                    end_line=msg.details.get("end_line", msg.details["start_line"]),
                    operation=msg.details.get("operation", "edit"),
                    content_hash=msg.details.get("content_hash"),
                )
                edits.append(edit)
            except (KeyError, AssertionError):
                # Malformed details, skip
                continue
    return edits


def make_path_relative(file_path: str, repo_root: Path | None) -> str:
    """Make a file path relative to the repo root.

    If repo_root is None or path is not under repo, returns original path.
    """
    if repo_root is None:
        return file_path
    try:
        path = Path(file_path)
        if path.is_absolute():
            return str(path.relative_to(repo_root))
        return file_path
    except ValueError:
        # Path not under repo_root
        return file_path


def session_to_trace_record(
    session: Trajectory,
    *,
    repo_root: Path | None = None,
    tool_name: str = "rollouts",
    tool_version: str = "0.1.0",
) -> TraceRecord | None:
    """Convert a rollouts session to an agent-trace record.

    Returns None if no file edits were found in the session.

    Args:
        session: The agent session with messages
        repo_root: Git repo root for making paths relative
        tool_name: Name of the tool generating the trace
        tool_version: Version of the tool
    """
    # Extract file edits from messages
    edits = extract_file_edits(session.messages)
    if not edits:
        return None

    # Build model ID
    model_id = None
    if session.endpoint.provider and session.endpoint.model:
        model_id = format_model_id(session.endpoint.provider, session.endpoint.model)

    contributor = Contributor(type="ai", model_id=model_id)

    # Group edits by file
    edits_by_file: dict[str, list[FileEditDetails]] = {}
    for edit in edits:
        rel_path = make_path_relative(edit.file_path, repo_root)
        if rel_path not in edits_by_file:
            edits_by_file[rel_path] = []
        edits_by_file[rel_path].append(edit)

    # Build file attributions
    file_attributions: list[FileAttribution] = []
    for path, file_edits in edits_by_file.items():
        ranges = tuple(
            LineRange(
                start_line=e.start_line,
                end_line=e.end_line,
                content_hash=e.content_hash,
            )
            for e in file_edits
        )
        conversation = Conversation(
            contributor=contributor,
            ranges=ranges,
            # Could generate URL if we have a session viewer
            url=None,
        )
        file_attributions.append(FileAttribution(path=path, conversations=(conversation,)))

    # Get VCS info - prefer session's captured VCS, fallback to current state
    vcs_info = None
    if session.vcs is not None:
        vcs_info = VCSInfo(
            type=session.vcs["type"],
            revision=session.vcs["revision"],
        )
    elif repo_root is not None:
        vcs_info = get_git_info(repo_root)

    return TraceRecord(
        version="0.1.0",
        id=str(uuid.uuid4()),
        timestamp=session.created_at or datetime.now().isoformat(),
        files=tuple(file_attributions),
        vcs=vcs_info,
        tool=ToolInfo(name=tool_name, version=tool_version),
        metadata={"session_id": session.session_id},
    )


def write_trace_record(record: TraceRecord, output_path: Path) -> None:
    """Append a trace record to a .agent-trace.jsonl file."""
    import json

    with open(output_path, "a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def export_session_trace(
    session: Trajectory,
    output_path: Path | None = None,
    *,
    tool_name: str = "rollouts",
    tool_version: str = "0.1.0",
) -> Path | None:
    """Export agent-trace attribution for a completed session.

    Writes a trace record to .agent-trace.jsonl in the repo root,
    or to the specified output_path.

    Returns the path written to, or None if no file edits were found.

    Usage:
        session, _ = await store.get(session_id)
        trace_path = export_session_trace(session)
        if trace_path:
            print(f"Wrote trace to {trace_path}")
    """
    # Determine repo root from session VCS or working directory
    repo_root = None
    if session.vcs is not None and "root" in session.vcs:
        repo_root = Path(session.vcs["root"])

    # Convert session to trace record
    record = session_to_trace_record(
        session,
        repo_root=repo_root,
        tool_name=tool_name,
        tool_version=tool_version,
    )

    if record is None:
        return None

    # Determine output path
    if output_path is None:
        if repo_root is not None:
            output_path = repo_root / ".agent-trace.jsonl"
        else:
            output_path = Path.cwd() / ".agent-trace.jsonl"

    write_trace_record(record, output_path)
    return output_path
