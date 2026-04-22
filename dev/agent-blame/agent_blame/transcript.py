"""Load a session's transcript from disk, normalized across agent sources.

Downstream of the attribution pipeline. When the UI wants to show the
conversation that produced a specific blamed line, it asks for
`(source, session_id)` and gets back a flat list of messages with tool
calls inlined.

Shape returned by `load_transcript`:

    [
      {
        "role": "user" | "assistant" | "system" | "tool_result",
        "timestamp": ISO8601 | None,
        "text": str,                     # plaintext rendering
        "tool_calls": [                  # may be empty
          {
            "tool_call_id": str,
            "name": str,
            "input": {...},              # tool-specific
          }, ...
        ],
        "tool_result_for": str | None,   # if role=tool_result, which tool_call_id
        "source_type": str,              # the source's own type tag, for debugging
      },
      ...
    ]

We deliberately keep this shape narrow and UI-oriented. It is NOT an
attempt to perfectly round-trip every Claude Code / Codex field —
anything not useful to a transcript viewer is dropped. Callers that
need the raw bytes should hit the original JSONL themselves.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .adapters import claude_code, codex
from .tool_diff import render_tool_call

logger = logging.getLogger(__name__)

Source = Literal["claude_code", "codex"]


class TranscriptNotFound(ValueError):
    """Session JSONL could not be located for the given source + id."""


def _find_claude_code_path(session_id: str) -> Path | None:
    """Walk the Claude Code projects dir looking for <session_id>.jsonl.

    CC sessions are namespaced by project cwd; we don't know the cwd here,
    so brute-force glob. Cheap at our scale (tens of sessions).
    """
    if not claude_code.CLAUDE_PROJECTS_DIR.exists():
        return None
    for project_dir in claude_code.CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        candidate = project_dir / f"{session_id}.jsonl"
        if candidate.exists():
            return candidate
    return None


def _find_codex_path(session_id: str) -> Path | None:
    """Codex session uuids appear in the filename; glob by suffix."""
    if not codex.CODEX_SESSIONS_DIR.exists():
        return None
    for p in codex.CODEX_SESSIONS_DIR.rglob(f"*{session_id}*.jsonl"):
        return p  # first hit; names are unique
    return None


def _parse_ts(raw: Any) -> str | None:
    """Return ISO8601 UTC string, or None if unparseable."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw / 1000, tz=timezone.utc).isoformat()
    if isinstance(raw, str):
        s = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    return None


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------


def _render_cc_content(content: Any) -> tuple[str, list[dict]]:
    """Return (text, tool_calls) for a Claude Code message.content.

    CC content is either a plain string or a list of blocks
    ({type: "text"|"tool_use"|"tool_result", ...}). We join text blocks
    and collect tool_use blocks separately. tool_result blocks are
    handled at the outer level (they arrive as role=user messages
    which we re-label as role=tool_result for clarity).
    """
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content) if content else "", []
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "text":
            text_parts.append(block.get("text", ""))
        elif t == "tool_use":
            name = block.get("name", "")
            inp = block.get("input", {})
            tool_calls.append({
                "tool_call_id": block.get("id", ""),
                "name": name,
                "input": inp,
                "render": render_tool_call(name, inp),
            })
    return "\n".join(p for p in text_parts if p), tool_calls


def _load_claude_code(session_id: str) -> list[dict]:
    path = _find_claude_code_path(session_id)
    if path is None:
        raise TranscriptNotFound(f"claude_code session {session_id} not found under {claude_code.CLAUDE_PROJECTS_DIR}")
    messages: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_type = entry.get("type")
            if entry_type in ("summary", "file-history-snapshot"):
                continue
            msg_data = entry.get("message", {}) or {}
            role = msg_data.get("role") or entry_type
            if not role:
                continue
            timestamp = _parse_ts(entry.get("timestamp"))
            content = msg_data.get("content", "")

            # tool_result blocks live inside role=user content. Split them out
            # so the UI can render them close to the producing tool call.
            if role == "user" and isinstance(content, list):
                tool_results = [
                    b for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if tool_results:
                    for tr in tool_results:
                        tr_content = tr.get("content", "")
                        if isinstance(tr_content, list):
                            # may itself be blocks with {type: "text", ...}
                            pieces = []
                            for b in tr_content:
                                if isinstance(b, dict) and b.get("type") == "text":
                                    pieces.append(b.get("text", ""))
                            tr_text = "\n".join(pieces)
                        else:
                            tr_text = str(tr_content) if tr_content else ""
                        messages.append({
                            "role": "tool_result",
                            "timestamp": timestamp,
                            "text": tr_text,
                            "tool_calls": [],
                            "tool_result_for": tr.get("tool_use_id"),
                            "source_type": "user(tool_result)",
                        })
                    continue  # don't also render as user message

            text, tool_calls = _render_cc_content(content)
            messages.append({
                "role": role,
                "timestamp": timestamp,
                "text": text,
                "tool_calls": tool_calls,
                "tool_result_for": None,
                "source_type": entry_type or role,
            })
    return messages


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------


def _load_codex(session_id: str) -> list[dict]:
    path = _find_codex_path(session_id)
    if path is None:
        raise TranscriptNotFound(f"codex session {session_id} not found under {codex.CODEX_SESSIONS_DIR}")
    messages: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = evt.get("type")
            payload = evt.get("payload") or {}
            timestamp = _parse_ts(evt.get("timestamp"))

            if etype == "response_item":
                ptype = payload.get("type")
                if ptype == "message":
                    role = payload.get("role") or "assistant"
                    content = payload.get("content") or []
                    text_parts = []
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                text_parts.append(part["text"])
                    messages.append({
                        "role": role,
                        "timestamp": timestamp,
                        "text": "\n".join(text_parts),
                        "tool_calls": [],
                        "tool_result_for": None,
                        "source_type": "response_item.message",
                    })
                elif ptype == "reasoning":
                    summary = payload.get("summary") or []
                    text_parts = []
                    if isinstance(summary, list):
                        for part in summary:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                text_parts.append(part["text"])
                    if text_parts:
                        messages.append({
                            "role": "assistant",
                            "timestamp": timestamp,
                            "text": "\n".join(text_parts),
                            "tool_calls": [],
                            "tool_result_for": None,
                            "source_type": "response_item.reasoning",
                        })
                elif ptype in ("function_call", "custom_tool_call"):
                    call_id = payload.get("call_id", "")
                    name = payload.get("name", "")
                    if ptype == "custom_tool_call":
                        inp = payload.get("input")
                    else:
                        raw_args = payload.get("arguments")
                        try:
                            inp = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except json.JSONDecodeError:
                            inp = raw_args
                    messages.append({
                        "role": "assistant",
                        "timestamp": timestamp,
                        "text": "",
                        "tool_calls": [{
                            "tool_call_id": call_id,
                            "name": name,
                            "input": inp,
                            "render": render_tool_call(name, inp),
                        }],
                        "tool_result_for": None,
                        "source_type": f"response_item.{ptype}",
                    })
                elif ptype in ("function_call_output", "custom_tool_call_output"):
                    call_id = payload.get("call_id", "")
                    output = payload.get("output")
                    if isinstance(output, (dict, list)):
                        text = json.dumps(output, indent=2)
                    else:
                        text = "" if output is None else str(output)
                    messages.append({
                        "role": "tool_result",
                        "timestamp": timestamp,
                        "text": text,
                        "tool_calls": [],
                        "tool_result_for": call_id,
                        "source_type": f"response_item.{ptype}",
                    })
            elif etype == "event_msg":
                ptype = payload.get("type")
                if ptype == "user_message":
                    messages.append({
                        "role": "user",
                        "timestamp": timestamp,
                        "text": payload.get("message", "") or "",
                        "tool_calls": [],
                        "tool_result_for": None,
                        "source_type": "event_msg.user_message",
                    })
                elif ptype == "agent_message":
                    messages.append({
                        "role": "assistant",
                        "timestamp": timestamp,
                        "text": payload.get("message", "") or "",
                        "tool_calls": [],
                        "tool_result_for": None,
                        "source_type": "event_msg.agent_message",
                    })
                # other event_msg types (token_count, task_complete, etc.)
                # are skipped — not useful for a reader-facing transcript.
    return messages


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_transcript(source: str, session_id: str) -> list[dict]:
    """Dispatch to the right adapter by source."""
    if source == "claude_code":
        return _load_claude_code(session_id)
    if source == "codex":
        # Codex's session_id field and filename-uuid match; the adapter
        # globs by substring so trimmed ids usually still resolve.
        return _load_codex(session_id)
    raise ValueError(f"unknown source {source!r}")
