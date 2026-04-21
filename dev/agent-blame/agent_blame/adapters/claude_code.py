"""Claude Code transcript adapter.

Claude Code stores sessions at:

    ~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl

where `<encoded-cwd>` is the absolute cwd with `/` replaced by `-`. Each line
is one JSON record. Types we care about:

    {"type": "assistant", "sessionId": ..., "uuid": ..., "parentUuid": ...,
     "timestamp": ..., "message": {"role": "assistant",
                                   "content": [ {"type": "tool_use",
                                                 "id": "toolu_...",
                                                 "name": "Edit" | "Write" | "MultiEdit",
                                                 "input": {...}}, ... ]}}

We skip everything else ("user", "summary", "file-history-snapshot") — they
are not sources of filesystem mutation.

Tool input shapes (Claude Code specific):
    Write:      {file_path, content}
    Edit:       {file_path, old_string, new_string, replace_all?}
    MultiEdit:  {file_path, edits: [{old_string, new_string, replace_all?}, ...]}

We expand MultiEdit into multiple FileEdits at adapter time — downstream
never needs to know it was a MultiEdit.

This adapter reads JSONL directly rather than importing rollouts'
`import_cc.py`, to keep `agent-blame` decoupled. The JSONL schema is stable
enough that duplication is the honest choice.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..effects import FileEdit

logger = logging.getLogger(__name__)

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def decode_project_cwd(encoded: str) -> Path:
    """`-Users-foo-bar` -> `/Users/foo/bar`.

    Note: Claude Code's encoding is lossy — real dashes in path components
    become indistinguishable from separators. We accept that ambiguity; the
    cwd only serves to filter candidate sessions before parsing.
    """
    if encoded.startswith("-"):
        return Path("/" + encoded[1:].replace("-", "/"))
    return Path(encoded.replace("-", "/"))


def list_sessions_for_cwd(repo_path: Path) -> list[Path]:
    """Return JSONL paths whose encoded cwd is a prefix-match for `repo_path`.

    We match against both the exact cwd and any cwd that sits inside
    `repo_path`, so sessions run from a subdirectory of the repo are included.
    """
    if not CLAUDE_PROJECTS_DIR.exists():
        return []

    target = repo_path.resolve()
    results: list[Path] = []
    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        cwd = decode_project_cwd(project_dir.name)
        try:
            cwd.resolve().relative_to(target)
        except ValueError:
            # cwd is not inside target
            continue
        for session_file in project_dir.glob("*.jsonl"):
            if len(session_file.stem) == 36:  # UUID
                results.append(session_file)
    return results


def _parse_timestamp(raw: Any) -> datetime | None:
    """Claude Code uses either ISO strings or unix ms. Always return UTC-aware."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw / 1000, tz=timezone.utc)
    if isinstance(raw, str):
        # `fromisoformat` handles trailing Z in 3.11+; be explicit just in case
        s = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _edits_from_tool_use(
    *,
    session_id: str,
    message_uuid: str,
    timestamp: datetime,
    tool_use: dict[str, Any],
) -> Iterator[FileEdit]:
    """Expand one tool_use block into zero or more FileEdits."""
    name = tool_use.get("name")
    tool_id = tool_use.get("id")
    raw_input = tool_use.get("input")
    if not isinstance(raw_input, dict) or not isinstance(tool_id, str):
        return

    common = {
        "source": "claude_code",
        "session_id": session_id,
        "message_uuid": message_uuid,
        "tool_call_id": tool_id,
        "timestamp": timestamp,
    }

    if name == "Write":
        path = raw_input.get("file_path")
        content = raw_input.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            return
        yield FileEdit(
            **common,
            path=path,
            op="write",
            new_content=content,
            old_content=None,
        )
        return

    if name == "Edit":
        path = raw_input.get("file_path")
        old = raw_input.get("old_string")
        new = raw_input.get("new_string")
        if not isinstance(path, str) or not isinstance(old, str) or not isinstance(new, str):
            return
        yield FileEdit(
            **common,
            path=path,
            op="edit",
            new_content=new,
            old_content=old,
        )
        return

    if name == "MultiEdit":
        path = raw_input.get("file_path")
        edits = raw_input.get("edits")
        if not isinstance(path, str) or not isinstance(edits, list):
            return
        # Expand into a sequence of edits, tagged so we can still trace them
        # back to the same tool_call_id. Downstream fold sees ordered edits;
        # it does not need to know they were grouped.
        for i, edit in enumerate(edits):
            if not isinstance(edit, dict):
                continue
            old = edit.get("old_string")
            new = edit.get("new_string")
            if not isinstance(old, str) or not isinstance(new, str):
                continue
            yield FileEdit(
                source="claude_code",
                session_id=session_id,
                message_uuid=message_uuid,
                tool_call_id=f"{tool_id}#{i}",
                timestamp=timestamp,
                path=path,
                op="edit",
                new_content=new,
                old_content=old,
            )
        return

    # Silently skip other tools (Read, Bash, Glob, Grep, ...) — they are not
    # filesystem mutations we can attribute.


def iter_file_edits(session_path: Path) -> Iterable[FileEdit]:
    """Yield FileEdits from one Claude Code JSONL session, in log order.

    Order matters: fold replays edits in adapter-emit order within a session
    (timestamps alone do not disambiguate edits issued in the same message).
    """
    session_id = session_path.stem
    with session_path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("skipping non-JSON line %d in %s", lineno, session_path)
                continue
            if entry.get("type") != "assistant":
                continue
            msg = entry.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            message_uuid = entry.get("uuid")
            if not isinstance(message_uuid, str):
                continue
            timestamp = _parse_timestamp(entry.get("timestamp"))
            if timestamp is None:
                # We cannot order edits without a timestamp. Skip rather
                # than fabricate — loud is better than silent.
                logger.warning(
                    "assistant message %s in %s has no timestamp; skipping",
                    message_uuid, session_path,
                )
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                yield from _edits_from_tool_use(
                    session_id=session_id,
                    message_uuid=message_uuid,
                    timestamp=timestamp,
                    tool_use=block,
                )
