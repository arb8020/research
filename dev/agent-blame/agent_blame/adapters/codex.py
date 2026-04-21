"""Codex transcript adapter.

Codex stores sessions at:

    ~/.codex/sessions/<YYYY>/<MM>/<DD>/rollout-<ISO>-<uuid>.jsonl

Each line is `{timestamp, type, payload}`. Our event vocabulary (cross-ref
Euphony's parser at /tmp/euphony/src/utils/codex-session.ts):

    type="session_meta"   -> has payload.id (session uuid), payload.cwd
    type="response_item"  -> has payload.type in {message, function_call,
                             custom_tool_call, ..., reasoning}

Codex's main edit channel is `apply_patch`, emitted as either:

    custom_tool_call with name="apply_patch", input="*** Begin Patch\\n..."
    function_call    with name="apply_patch", arguments='{"input": "..."}'

We support both. The patch format (OpenAI's "apply_patch" style, not unified
diff) is content-anchored:

    *** Begin Patch
    *** Update File: /abs/path
    @@
     context_line
    -old_line
    +new_line
     more_context
    @@
     ...
    *** Update File: /abs/path2
    ...
    *** End Patch

Hunks are separated by `@@`. Within a hunk, lines are prefixed by:
    ' '  context
    '-'  deleted
    '+'  added

No `@@ -N,M +N,M @@` line-number header — matching is by content. That is
exactly the shape our fold wants: `old_content` = the context + deleted
lines joined; `new_content` = context + added lines joined.

## What we deliberately skip

- `exec_command` / `shell` calls. These are bash commands; we do not parse
  heredocs or `sed` one-liners here. A separate `shell_edits` adapter can
  mine them later with per-pattern confidence.
- `*** Delete File:` headers. The file is gone on disk anyway; there's no
  line to attribute. We log and move on.
- Patches that reference files outside the session's cwd. We still emit
  them; reconcile will just find no match and they become "no-op" edits.
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

CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"


def list_sessions_for_cwd(repo_path: Path) -> list[Path]:
    """Return Codex JSONL paths whose session_meta.cwd is inside repo_path.

    Codex does not encode cwd in the filename (unlike Claude Code), so we
    must open each session's first line. Sessions are laid out by date under
    YYYY/MM/DD/, so we walk the whole tree. For very large session counts
    this is O(n) file opens — acceptable for v0 (sessions are tiny JSONL
    first lines).
    """
    if not CODEX_SESSIONS_DIR.exists():
        return []
    target = repo_path.resolve()
    results: list[Path] = []
    for jsonl in CODEX_SESSIONS_DIR.rglob("*.jsonl"):
        cwd = _read_session_cwd(jsonl)
        if cwd is None:
            continue
        try:
            Path(cwd).resolve().relative_to(target)
        except (ValueError, OSError):
            continue
        results.append(jsonl)
    return results


def _read_session_cwd(session_path: Path) -> str | None:
    """Read the session_meta event (should be first line) and return cwd."""
    try:
        with session_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    return None
                if evt.get("type") == "session_meta":
                    payload = evt.get("payload") or {}
                    cwd = payload.get("cwd")
                    return cwd if isinstance(cwd, str) else None
                # Only the first non-empty line should be session_meta.
                # If not, this isn't a codex session we understand.
                return None
    except OSError:
        return None
    return None


def _read_session_id(session_path: Path) -> str:
    """Return the session uuid from session_meta, falling back to filename."""
    try:
        with session_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    break
                if evt.get("type") == "session_meta":
                    payload = evt.get("payload") or {}
                    sid = payload.get("id")
                    if isinstance(sid, str):
                        return sid
                break
    except OSError:
        pass
    # Fallback: filename usually ends with `-<uuid>.jsonl`
    return session_path.stem


def _parse_timestamp(raw: Any) -> datetime | None:
    """Codex timestamps are ISO 8601 strings with trailing Z."""
    if not isinstance(raw, str):
        return None
    s = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# --------------------------------------------------------------------------
# apply_patch parsing
# --------------------------------------------------------------------------


class ApplyPatchParseError(ValueError):
    """Raised when an apply_patch input is malformed in a way we cannot recover from.

    Callers catch and log; we do not try to salvage partial hunks because a
    silently-partial patch would attribute lines to the wrong edit.
    """


def _parse_apply_patch(patch_text: str) -> list[tuple[str, str, str | None, str]]:
    """Parse an apply_patch input into a list of `(path, op, old, new)` tuples.

    op is "write" for Add File, "edit" for each hunk of Update File.
    For Add File, old is None and new is the full content.
    For Update File hunks, old and new are the concatenated context+delete
    and context+add views respectively; they are suitable for fold._apply_edit.

    We intentionally produce one output tuple per hunk (not per file), so the
    fold can track per-hunk attribution independently.
    """
    lines = patch_text.replace("\r\n", "\n").split("\n")
    # Normalize boundaries
    # Find Begin Patch
    start = next(
        (i for i, line in enumerate(lines) if line == "*** Begin Patch"), None,
    )
    if start is None:
        raise ApplyPatchParseError("missing '*** Begin Patch'")
    end = next(
        (i for i, line in enumerate(lines) if line == "*** End Patch"), len(lines),
    )
    body = lines[start + 1 : end]

    out: list[tuple[str, str, str | None, str]] = []

    i = 0
    current_path: str | None = None
    current_op: str | None = None  # "update" | "add" | "delete"
    pending_add_body: list[str] = []
    pending_update_hunks: list[tuple[str, str]] = []  # (old, new) per hunk
    hunk_lines: list[str] = []

    def flush_hunk() -> None:
        """Turn `hunk_lines` into an (old, new) pair and append to pending_update_hunks."""
        old_lines: list[str] = []
        new_lines: list[str] = []
        for hl in hunk_lines:
            if hl.startswith("+"):
                new_lines.append(hl[1:])
            elif hl.startswith("-"):
                old_lines.append(hl[1:])
            elif hl.startswith(" "):
                # context appears in both
                old_lines.append(hl[1:])
                new_lines.append(hl[1:])
            elif hl == "":
                # empty line in hunk body is context with an empty string
                old_lines.append("")
                new_lines.append("")
            else:
                # unexpected — treat as context to avoid losing data,
                # but flag. apply_patch format does not define naked lines.
                logger.debug(
                    "unexpected hunk body line (treating as context): %r", hl,
                )
                old_lines.append(hl)
                new_lines.append(hl)
        pending_update_hunks.append(
            ("\n".join(old_lines), "\n".join(new_lines))
        )

    def flush_current_file() -> None:
        nonlocal current_path, current_op
        if current_path is None:
            return
        if current_op == "add":
            out.append((current_path, "write", None, "\n".join(pending_add_body)))
        elif current_op == "update":
            if hunk_lines:
                flush_hunk()
            for old, new in pending_update_hunks:
                out.append((current_path, "edit", old, new))
        elif current_op == "delete":
            # No FileEdit emitted — nothing to attribute. Logged at caller.
            pass
        current_path = None
        current_op = None
        pending_add_body.clear()
        pending_update_hunks.clear()
        hunk_lines.clear()

    while i < len(body):
        line = body[i]

        if line.startswith("*** Update File: "):
            flush_current_file()
            current_path = line[len("*** Update File: "):]
            current_op = "update"
            i += 1
            continue

        if line.startswith("*** Add File: "):
            flush_current_file()
            current_path = line[len("*** Add File: "):]
            current_op = "add"
            i += 1
            continue

        if line.startswith("*** Delete File: "):
            flush_current_file()
            current_path = line[len("*** Delete File: "):]
            current_op = "delete"
            i += 1
            continue

        if current_op == "update":
            if line == "@@" or line.startswith("@@ "):
                if hunk_lines:
                    flush_hunk()
                    hunk_lines.clear()
                i += 1
                continue
            hunk_lines.append(line)
            i += 1
            continue

        if current_op == "add":
            # Add File body: every subsequent line is the new file content,
            # prefixed with '+' in the apply_patch format.
            if line.startswith("+"):
                pending_add_body.append(line[1:])
            elif line == "":
                pending_add_body.append("")
            else:
                logger.debug(
                    "unexpected line in Add File body (ignoring): %r", line,
                )
            i += 1
            continue

        if current_op == "delete":
            # Delete body lines are informational; we don't use them.
            i += 1
            continue

        # Outside any file section (e.g. blank line before the first header)
        i += 1

    flush_current_file()
    return out


# --------------------------------------------------------------------------
# JSONL -> FileEdit stream
# --------------------------------------------------------------------------


def _extract_apply_patch_input(payload: dict[str, Any]) -> str | None:
    """Codex emits apply_patch as either custom_tool_call (input string) or
    function_call (arguments JSON with input field). Normalize both."""
    payload_type = payload.get("type")
    if payload_type == "custom_tool_call":
        inp = payload.get("input")
        return inp if isinstance(inp, str) else None
    if payload_type == "function_call":
        args = payload.get("arguments")
        if not isinstance(args, str):
            return None
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            inp = parsed.get("input") or parsed.get("patch")
            return inp if isinstance(inp, str) else None
        return None
    return None


def iter_file_edits(session_path: Path) -> Iterable[FileEdit]:
    """Yield FileEdits from one Codex JSONL session, in log order.

    Only `apply_patch` tool calls produce edits in v0. Shell-based file
    mutations are out of scope.
    """
    session_id = _read_session_id(session_path)
    with session_path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("skipping non-JSON line %d in %s", lineno, session_path)
                continue
            if evt.get("type") != "response_item":
                continue
            payload = evt.get("payload")
            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            if name != "apply_patch":
                continue
            call_id = payload.get("call_id")
            if not isinstance(call_id, str):
                continue
            timestamp = _parse_timestamp(evt.get("timestamp"))
            if timestamp is None:
                logger.warning(
                    "apply_patch call %s in %s has no timestamp; skipping",
                    call_id, session_path,
                )
                continue
            patch_text = _extract_apply_patch_input(payload)
            if patch_text is None:
                logger.debug(
                    "apply_patch call %s in %s had no extractable input",
                    call_id, session_path,
                )
                continue
            try:
                parsed = _parse_apply_patch(patch_text)
            except ApplyPatchParseError as e:
                logger.warning(
                    "malformed apply_patch %s in %s: %s", call_id, session_path, e,
                )
                continue
            # One response_item can touch multiple files and multiple hunks.
            # We tag each hunk with a synthetic id so downstream still sees
            # a unique (source, session, tool_call_id).
            for idx, (path, op, old, new) in enumerate(parsed):
                yield FileEdit(
                    source="codex",
                    session_id=session_id,
                    message_uuid=call_id,  # Codex has no separate message uuid
                    tool_call_id=f"{call_id}#{idx}",
                    timestamp=timestamp,
                    path=path,
                    op=op,  # type: ignore[arg-type]  # "write" | "edit"
                    new_content=new,
                    old_content=old,
                )
