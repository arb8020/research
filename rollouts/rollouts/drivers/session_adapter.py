"""Session adapters for hot-swapping between drivers.

Converts between rollouts' canonical session format (SessionHandle.messages)
and external driver formats (Claude Code JSONL, Codex threads).

The canonical format is a list of Message objects with role/content. Each
driver adapter handles bidirectional conversion:

    rollouts (direct API) <-> Our Session <-> Claude Code
                                   ^
                                   v
                                 Codex

Usage:
    from rollouts.drivers.session_adapter import (
        messages_to_claude_session,
        claude_session_to_messages,
    )

    # Export to Claude Code
    claude_session_to_claude_jsonl = messages_to_claude_session(messages, session_id)

    # Import from Claude Code
    messages = claude_session_to_messages(claude_session_path)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..dtypes import (
    Message,
    TextContent,
    ThinkingContent,
    ToolCallContent,
)


def _generate_uuid() -> str:
    """Generate a UUID for Claude Code message linking."""
    return str(uuid.uuid4())


def _now_iso() -> str:
    """Get current time in ISO format with timezone."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# -----------------------------------------------------------------------------
# Rollouts -> Claude Code
# -----------------------------------------------------------------------------


def message_to_claude_format(
    msg: Message,
    parent_uuid: str | None,
    session_id: str,
    cwd: str = "/",
) -> tuple[dict[str, Any], str]:
    """Convert a rollouts Message to Claude Code JSONL format.

    Args:
        msg: The rollouts Message to convert
        parent_uuid: UUID of the parent message (None for first message)
        session_id: Claude Code session ID
        cwd: Working directory for the session

    Returns:
        Tuple of (claude_message_dict, this_message_uuid)
    """
    msg_uuid = _generate_uuid()
    timestamp = msg.timestamp or _now_iso()

    base = {
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "userType": "external",
        "cwd": cwd,
        "sessionId": session_id,
        "version": "2.0.55",  # Claude Code version
        "gitBranch": "",
        "uuid": msg_uuid,
        "timestamp": timestamp,
    }

    if msg.role == "user":
        # User message - content is always string
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return {
            **base,
            "type": "user",
            "message": {"role": "user", "content": content},
            "thinkingMetadata": {"level": "none", "disabled": True, "triggers": []},
            "todos": [],
        }, msg_uuid

    elif msg.role == "assistant":
        # Assistant message - convert ContentBlocks to Claude's format
        content_blocks = _content_to_claude_blocks(msg.content)
        return {
            **base,
            "type": "assistant",
            "message": {
                "model": msg.model or "unknown",
                "id": f"msg_{_generate_uuid()[:24]}",
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }, msg_uuid

    elif msg.role == "tool":
        # Tool result message
        return {
            **base,
            "type": "tool_result",
            "tool_use_id": msg.tool_call_id or "",
            "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            "is_error": False,
        }, msg_uuid

    else:
        # Unknown role - treat as user
        return {
            **base,
            "type": "user",
            "message": {"role": "user", "content": str(msg.content)},
        }, msg_uuid


def _content_to_claude_blocks(content: str | list | None) -> list[dict[str, Any]]:
    """Convert rollouts content to Claude Code content blocks."""
    if content is None:
        return []

    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    # List of ContentBlocks
    blocks = []
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                blocks.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "thinking":
                # Skip thinking blocks - Claude API requires a valid signature field
                # which we can't generate. The thinking content isn't essential for resume.
                pass
            elif block_type == "toolCall":
                blocks.append({
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("arguments", {}),
                })
        elif isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingContent):
            # Skip thinking blocks - Claude API requires a valid signature field
            pass
        elif isinstance(block, ToolCallContent):
            blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.arguments,
            })

    return blocks


def messages_to_claude_session(
    messages: list[Message],
    session_id: str,
    cwd: str = "/",
) -> list[dict[str, Any]]:
    """Convert a list of rollouts Messages to Claude Code JSONL format.

    Args:
        messages: List of rollouts Messages
        session_id: Claude Code session ID
        cwd: Working directory for the session

    Returns:
        List of dicts ready to be written as JSONL
    """
    result = []
    parent_uuid: str | None = None

    # Add summary as first line (Claude Code convention)
    result.append({
        "type": "summary",
        "summary": "Session imported from rollouts",
        "leafUuid": _generate_uuid(),
    })

    for msg in messages:
        claude_msg, msg_uuid = message_to_claude_format(msg, parent_uuid, session_id, cwd)
        result.append(claude_msg)
        parent_uuid = msg_uuid

    return result


def write_claude_session(
    messages: list[Message],
    session_id: str,
    output_path: Path,
    cwd: str = "/",
) -> None:
    """Write a rollouts session as Claude Code JSONL file.

    Args:
        messages: List of rollouts Messages
        session_id: Claude Code session ID
        output_path: Path to write the JSONL file
        cwd: Working directory for the session
    """
    claude_messages = messages_to_claude_session(messages, session_id, cwd)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for msg in claude_messages:
            f.write(json.dumps(msg) + "\n")


# -----------------------------------------------------------------------------
# Claude Code -> Rollouts
# -----------------------------------------------------------------------------


def claude_message_to_rollouts(msg: dict[str, Any]) -> Message | None:
    """Convert a Claude Code message to rollouts Message.

    Args:
        msg: Claude Code message dict from JSONL

    Returns:
        Message or None if not a content message (e.g., summary, system)
    """
    msg_type = msg.get("type")
    timestamp = msg.get("timestamp")

    if msg_type == "user":
        inner = msg.get("message", {})
        content = inner.get("content", "")
        return Message(
            role="user",
            content=content,
            timestamp=timestamp,
        )

    elif msg_type == "assistant":
        inner = msg.get("message", {})
        content_blocks = inner.get("content", [])
        model = inner.get("model")

        # Convert Claude blocks to rollouts ContentBlocks
        rollouts_blocks = _claude_blocks_to_content(content_blocks)

        return Message(
            role="assistant",
            content=rollouts_blocks if rollouts_blocks else "",
            model=model,
            timestamp=timestamp,
        )

    elif msg_type == "tool_result":
        return Message(
            role="tool",
            content=msg.get("content", ""),
            tool_call_id=msg.get("tool_use_id"),
            timestamp=timestamp,
        )

    # Skip non-content messages (summary, system, file-history-snapshot, etc.)
    return None


def _claude_blocks_to_content(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Claude Code content blocks to rollouts ContentBlock dicts."""
    result = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            result.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "thinking":
            result.append({"type": "thinking", "thinking": block.get("thinking", "")})
        elif block_type == "tool_use":
            result.append({
                "type": "toolCall",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": block.get("input", {}),
            })
    return result


def claude_session_to_messages(session_path: Path) -> list[Message]:
    """Read a Claude Code session JSONL and convert to rollouts Messages.

    Args:
        session_path: Path to Claude Code session JSONL file

    Returns:
        List of rollouts Messages
    """
    messages = []
    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                rollouts_msg = claude_message_to_rollouts(msg)
                if rollouts_msg is not None:
                    messages.append(rollouts_msg)
            except json.JSONDecodeError:
                continue
    return messages


def read_claude_session_id(session_path: Path) -> str | None:
    """Extract session ID from a Claude Code session file.

    Args:
        session_path: Path to Claude Code session JSONL file

    Returns:
        Session ID or None if not found
    """
    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if "sessionId" in msg:
                    return msg["sessionId"]
            except json.JSONDecodeError:
                continue
    return None


# -----------------------------------------------------------------------------
# Claude Code session path helpers
# -----------------------------------------------------------------------------


def get_claude_session_path(session_id: str, cwd: str = "/") -> Path:
    """Get the path to a Claude Code session file.

    Claude Code stores sessions at:
    ~/.claude/projects/<escaped-cwd>/<session-id>.jsonl

    Args:
        session_id: Claude Code session ID
        cwd: Working directory (used to find project folder)

    Returns:
        Path to the session file
    """
    # Resolve symlinks (e.g., /tmp -> /private/tmp on macOS)
    resolved_cwd = str(Path(cwd).resolve())

    # Claude Code escapes paths by replacing / with -
    # The leading dash is kept (e.g., /private/tmp -> -private-tmp)
    escaped_cwd = resolved_cwd.replace("/", "-")

    return Path.home() / ".claude" / "projects" / escaped_cwd / f"{session_id}.jsonl"


def find_claude_session(session_id: str) -> Path | None:
    """Find a Claude Code session file by ID (searches all projects).

    Args:
        session_id: Claude Code session ID

    Returns:
        Path to the session file or None if not found
    """
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return None

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        session_file = project_dir / f"{session_id}.jsonl"
        if session_file.exists():
            return session_file

    return None


# -----------------------------------------------------------------------------
# Codex -> Rollouts
# -----------------------------------------------------------------------------


def codex_message_to_rollouts(entry: dict[str, Any]) -> Message | None:
    """Convert a Codex session entry to rollouts Message.

    Codex JSONL format:
        {"timestamp": "...", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [...]}}
        {"timestamp": "...", "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [...]}}

    Args:
        entry: Codex session entry dict from JSONL

    Returns:
        Message or None if not a content message
    """
    entry_type = entry.get("type")
    timestamp = entry.get("timestamp")

    # Only process response_item entries with message payloads
    if entry_type != "response_item":
        return None

    payload = entry.get("payload", {})
    if payload.get("type") != "message":
        return None

    role = payload.get("role")
    content_blocks = payload.get("content", [])

    if role == "user":
        # Extract text from user content blocks
        texts = []
        for block in content_blocks:
            if block.get("type") == "input_text":
                texts.append(block.get("text", ""))
        return Message(
            role="user",
            content="\n".join(texts) if texts else "",
            timestamp=timestamp,
        )

    elif role == "assistant":
        # Convert Codex content blocks to rollouts format
        rollouts_blocks = _codex_blocks_to_content(content_blocks)
        return Message(
            role="assistant",
            content=rollouts_blocks if rollouts_blocks else "",
            timestamp=timestamp,
        )

    return None


def _codex_blocks_to_content(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Codex content blocks to rollouts ContentBlock dicts."""
    result = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "output_text":
            result.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "reasoning":
            # Codex reasoning -> thinking
            result.append({"type": "thinking", "thinking": block.get("text", "")})
        elif block_type == "function_call":
            result.append({
                "type": "toolCall",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": block.get("arguments", {}),
            })
    return result


def codex_session_to_messages(session_path: Path) -> list[Message]:
    """Read a Codex session JSONL and convert to rollouts Messages.

    Args:
        session_path: Path to Codex session JSONL file

    Returns:
        List of rollouts Messages
    """
    messages = []
    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                rollouts_msg = codex_message_to_rollouts(entry)
                if rollouts_msg is not None:
                    messages.append(rollouts_msg)
            except json.JSONDecodeError:
                continue
    return messages


def read_codex_session_id(session_path: Path) -> str | None:
    """Extract session ID from a Codex session file.

    Args:
        session_path: Path to Codex session JSONL file

    Returns:
        Session ID or None if not found
    """
    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") == "session_meta":
                    return entry.get("payload", {}).get("id")
            except json.JSONDecodeError:
                continue
    return None


# -----------------------------------------------------------------------------
# Rollouts -> Codex
# -----------------------------------------------------------------------------


def message_to_codex_format(
    msg: Message,
    session_id: str,
    cwd: str = "/",
) -> list[dict[str, Any]]:
    """Convert a rollouts Message to Codex JSONL entries.

    Args:
        msg: The rollouts Message to convert
        session_id: Codex session ID
        cwd: Working directory for the session

    Returns:
        List of Codex JSONL entry dicts (may be multiple per message)
    """
    timestamp = msg.timestamp or _now_iso()
    entries = []

    if msg.role == "user":
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        entries.append({
            "timestamp": timestamp,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            },
        })

    elif msg.role == "assistant":
        content_blocks = _content_to_codex_blocks(msg.content)
        entries.append({
            "timestamp": timestamp,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
            },
        })

    return entries


def _content_to_codex_blocks(content: str | list | None) -> list[dict[str, Any]]:
    """Convert rollouts content to Codex content blocks."""
    if content is None:
        return []

    if isinstance(content, str):
        return [{"type": "output_text", "text": content}]

    blocks = []
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                blocks.append({"type": "output_text", "text": block.get("text", "")})
            elif block_type == "thinking":
                blocks.append({"type": "reasoning", "text": block.get("thinking", "")})
            elif block_type == "toolCall":
                blocks.append({
                    "type": "function_call",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": block.get("arguments", {}),
                })
        elif isinstance(block, TextContent):
            blocks.append({"type": "output_text", "text": block.text})
        elif isinstance(block, ThinkingContent):
            blocks.append({"type": "reasoning", "text": block.thinking})
        elif isinstance(block, ToolCallContent):
            blocks.append({
                "type": "function_call",
                "id": block.id,
                "name": block.name,
                "arguments": block.arguments,
            })

    return blocks


def messages_to_codex_session(
    messages: list[Message],
    session_id: str,
    cwd: str = "/",
) -> list[dict[str, Any]]:
    """Convert a list of rollouts Messages to Codex JSONL format.

    Args:
        messages: List of rollouts Messages
        session_id: Codex session ID
        cwd: Working directory for the session

    Returns:
        List of dicts ready to be written as JSONL
    """
    result = []

    # Add session metadata as first entry
    result.append({
        "timestamp": _now_iso(),
        "type": "session_meta",
        "payload": {
            "id": session_id,
            "timestamp": _now_iso(),
            "cwd": cwd,
            "originator": "rollouts",
            "cli_version": "0.1.0",
            "instructions": None,
            "git": None,
        },
    })

    for msg in messages:
        entries = message_to_codex_format(msg, session_id, cwd)
        result.extend(entries)

    return result


def write_codex_session(
    messages: list[Message],
    session_id: str,
    output_path: Path,
    cwd: str = "/",
) -> None:
    """Write a rollouts session as Codex JSONL file.

    Args:
        messages: List of rollouts Messages
        session_id: Codex session ID
        output_path: Path to write the JSONL file
        cwd: Working directory for the session
    """
    codex_entries = messages_to_codex_session(messages, session_id, cwd)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in codex_entries:
            f.write(json.dumps(entry) + "\n")


# -----------------------------------------------------------------------------
# Codex session path helpers
# -----------------------------------------------------------------------------


def get_codex_session_path(session_id: str) -> Path:
    """Get the path where Codex would store a session file.

    Codex stores sessions at:
    ~/.codex/sessions/<year>/<month>/<day>/rollout-<timestamp>-<session-id>.jsonl

    Since we don't know the timestamp, we generate one for new sessions.

    Args:
        session_id: Codex session ID (UUID)

    Returns:
        Path for a new session file
    """
    now = datetime.now(timezone.utc)
    date_path = now.strftime("%Y/%m/%d")
    timestamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"rollout-{timestamp}-{session_id}.jsonl"

    return Path.home() / ".codex" / "sessions" / date_path / filename


def find_codex_session(session_id: str) -> Path | None:
    """Find a Codex session file by ID (searches all session directories).

    Args:
        session_id: Codex session ID (UUID portion)

    Returns:
        Path to the session file or None if not found
    """
    sessions_dir = Path.home() / ".codex" / "sessions"
    if not sessions_dir.exists():
        return None

    # Search recursively for files containing the session ID
    for session_file in sessions_dir.rglob("*.jsonl"):
        if session_id in session_file.name:
            return session_file

    return None
