"""Import sessions from Claude Code into rollouts.

Claude Code stores sessions at:
    ~/.claude/projects/<encoded-path>/<session-uuid>.jsonl

Each line is a JSON object with:
    - type: "user" | "assistant" | "summary" | "file-history-snapshot"
    - sessionId: UUID
    - uuid: message UUID
    - parentUuid: parent message UUID (for threading)
    - timestamp: ISO string or unix ms
    - message: {role, content, model, ...}

This module converts Claude Code sessions to rollouts format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .dtypes import (
    Endpoint,
    EnvironmentConfig,
    Message,
    TextContent,
    ToolCallContent,
)
from .store import FileSessionStore

# Claude Code stores projects with path encoded as dashes
CLAUDE_CODE_DIR = Path.home() / ".claude"
CLAUDE_CODE_PROJECTS_DIR = CLAUDE_CODE_DIR / "projects"


def encode_project_path(path: Path) -> str:
    """Encode a path to Claude Code's project directory format.

    /Users/foo/bar -> -Users-foo-bar
    """
    return str(path.absolute()).replace("/", "-")


def decode_project_path(encoded: str) -> Path:
    """Decode Claude Code's project directory format to a path.

    -Users-foo-bar -> /Users/foo/bar
    """
    # First char is always "-" for root "/"
    if encoded.startswith("-"):
        return Path("/" + encoded[1:].replace("-", "/"))
    return Path(encoded.replace("-", "/"))


@dataclass
class ClaudeCodeSession:
    """Metadata about a Claude Code session."""

    session_id: str
    project_path: Path
    file_path: Path
    message_count: int
    last_modified: datetime
    model: str | None = None

    @property
    def display_name(self) -> str:
        """Short display name for session picker."""
        return f"{self.project_path.name}/{self.session_id[:8]}"


def list_claude_code_projects() -> list[tuple[str, Path]]:
    """List all Claude Code projects.

    Returns list of (encoded_name, decoded_path) tuples.
    """
    if not CLAUDE_CODE_PROJECTS_DIR.exists():
        return []

    projects = []
    for project_dir in CLAUDE_CODE_PROJECTS_DIR.iterdir():
        if project_dir.is_dir():
            decoded = decode_project_path(project_dir.name)
            projects.append((project_dir.name, decoded))

    return sorted(projects, key=lambda x: x[1])


def list_claude_code_sessions(
    project_filter: str | None = None,
    limit: int = 20,
) -> list[ClaudeCodeSession]:
    """List Claude Code sessions, optionally filtered by project.

    Args:
        project_filter: Project name or path substring to filter by
        limit: Maximum number of sessions to return

    Returns:
        List of ClaudeCodeSession objects, sorted by last modified (newest first)
    """
    if not CLAUDE_CODE_PROJECTS_DIR.exists():
        return []

    sessions: list[ClaudeCodeSession] = []

    for project_dir in CLAUDE_CODE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue

        project_path = decode_project_path(project_dir.name)

        # Apply project filter
        if project_filter:
            if project_filter not in str(project_path) and project_filter not in project_dir.name:
                continue

        # Find session files
        for session_file in project_dir.glob("*.jsonl"):
            # Skip non-UUID files
            if len(session_file.stem) != 36:  # UUID length
                continue

            try:
                stat = session_file.stat()
                # Count lines (messages) - quick approximation
                with open(session_file) as f:
                    line_count = sum(1 for _ in f)

                # Try to get model from first assistant message
                model = None
                with open(session_file) as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("type") == "assistant":
                                model = entry.get("message", {}).get("model")
                                if model:
                                    break
                        except json.JSONDecodeError:
                            continue

                sessions.append(
                    ClaudeCodeSession(
                        session_id=session_file.stem,
                        project_path=project_path,
                        file_path=session_file,
                        message_count=line_count,
                        last_modified=datetime.fromtimestamp(stat.st_mtime),
                        model=model,
                    )
                )
            except (OSError, ValueError):
                continue

    # Sort by last modified, newest first
    sessions.sort(key=lambda s: s.last_modified, reverse=True)

    return sessions[:limit]


def parse_claude_code_session(session_file: Path) -> tuple[list[Message], dict[str, Any]]:
    """Parse a Claude Code session file into rollouts Messages.

    Args:
        session_file: Path to the .jsonl session file

    Returns:
        Tuple of (messages, metadata) where metadata contains:
            - session_id: str
            - model: str | None
            - cwd: str | None
            - git_branch: str | None
    """
    messages: list[Message] = []
    metadata: dict[str, Any] = {
        "session_id": session_file.stem,
        "model": None,
        "provider": "anthropic",
        "cwd": None,
        "git_branch": None,
    }

    with open(session_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")

            # Skip metadata entries
            if entry_type in ("summary", "file-history-snapshot"):
                continue

            # Extract metadata from first entry
            if metadata["cwd"] is None:
                metadata["cwd"] = entry.get("cwd")
                metadata["git_branch"] = entry.get("gitBranch")

            # Parse message
            msg_data = entry.get("message", {})
            role = msg_data.get("role") or entry_type

            if not role:
                continue

            # Extract model info
            if role == "assistant" and msg_data.get("model"):
                metadata["model"] = msg_data["model"]

            # Parse timestamp
            timestamp = entry.get("timestamp")
            if isinstance(timestamp, (int, float)):
                # Unix ms -> ISO string
                timestamp = datetime.fromtimestamp(timestamp / 1000).isoformat()

            # Convert content
            content = msg_data.get("content", "")
            converted_content = _convert_content(content, role)

            # Determine final role
            final_role = role
            tool_call_id = None

            # Handle tool results (Claude Code stores as role="user" with tool_result blocks)
            if role == "user" and isinstance(content, list):
                tool_results = [
                    b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if tool_results:
                    # Create separate tool messages for each result
                    for tr in tool_results:
                        tool_msg = Message(
                            role="tool",
                            content=tr.get("content", ""),
                            tool_call_id=tr.get("tool_use_id"),
                            provider="anthropic",
                            model=metadata["model"],
                            timestamp=timestamp,
                        )
                        messages.append(tool_msg)
                    continue  # Skip adding as user message

            messages.append(
                Message(
                    role=final_role,
                    content=converted_content,
                    tool_call_id=tool_call_id,
                    provider="anthropic",
                    model=metadata["model"],
                    timestamp=timestamp,
                )
            )

    return messages, metadata


def _convert_content(content: Any, role: str) -> str | list[TextContent | ToolCallContent]:
    """Convert Claude Code content format to rollouts format.

    Claude Code format:
        - str: plain text
        - list of {type: "text", text: "..."} or {type: "tool_use", id, name, input}

    Rollouts format:
        - str: plain text
        - list of TextContent or ToolCallContent
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content) if content else ""

    blocks: list[TextContent | ToolCallContent] = []

    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type == "text":
            blocks.append(
                TextContent(
                    text=block.get("text", ""),
                )
            )
        elif block_type == "tool_use":
            blocks.append(
                ToolCallContent(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                )
            )
        # Skip tool_result blocks - handled separately

    # If only text blocks, consider returning as string
    if len(blocks) == 1 and isinstance(blocks[0], TextContent):
        return blocks[0].text

    return blocks if blocks else ""


async def import_claude_code_session(
    session: ClaudeCodeSession,
    session_store: FileSessionStore,
) -> tuple[str | None, str | None]:
    """Import a Claude Code session into rollouts.

    Args:
        session: ClaudeCodeSession to import
        session_store: Rollouts session store

    Returns:
        Tuple of (new_session_id, error) - one will be None
    """
    # Parse the session
    messages, metadata = parse_claude_code_session(session.file_path)

    if not messages:
        return None, "No messages found in session"

    # Create endpoint config (secrets excluded)
    model = metadata.get("model") or "claude-sonnet-4-20250514"
    endpoint = Endpoint(
        provider="anthropic",
        model=model,
        api_base="https://api.anthropic.com",
    )

    # Create environment config
    # Default to coding environment since Claude Code is a coding assistant
    env_config = EnvironmentConfig(
        type="LocalFilesystemEnvironment",
        config={
            "working_dir": metadata.get("cwd") or str(session.project_path),
        },
    )

    # Create new session
    new_session = await session_store.create(
        endpoint=endpoint,
        environment=env_config,
        tags={
            "imported_from": "claude_code",
            "original_session_id": session.session_id,
            "original_project": str(session.project_path),
        },
    )

    # Append messages
    for msg in messages:
        await session_store.append_message(new_session.session_id, msg)

    return new_session.session_id, None
