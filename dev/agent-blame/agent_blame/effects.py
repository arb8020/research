"""Source-agnostic effect type for filesystem edits by coding agents.

The central idea of this project: every coding agent — Claude Code, Codex,
OpenCode, rollouts-native — ultimately produces a stream of filesystem edits.
Once we normalize into a common `FileEdit` record, blame becomes a fold over
that stream.

Keep this file boring. It is the hot-path type that every adapter targets and
every downstream stage (fold, reconcile) consumes. Do not add fields unless
a real consumer needs them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

EditOp = Literal["write", "edit", "multi_edit"]
"""What kind of filesystem mutation this effect represents.

- write:      full-file write (Claude Code `Write`, Codex file write)
- edit:       in-place substring replacement (Claude Code `Edit`)
- multi_edit: ordered series of edits applied atomically (Claude Code `MultiEdit`)

We deliberately do not model `Read`, `Bash`, or `Glob` here — they are not
mutations. A later pass may want `delete` / `rename`, but today's agents
perform those via Bash shell-outs which we cannot reliably parse.
"""


@dataclass(frozen=True)
class FileEdit:
    """One filesystem mutation produced by an agent, source-agnostic.

    Every field below is required. If an adapter cannot produce a field,
    that is a signal the source does not carry enough information and the
    adapter should skip the record rather than fabricate one.
    """

    # Identity of the producing agent turn
    source: Literal["claude_code", "codex", "opencode", "rollouts"]
    session_id: str
    """Stable identifier for the session within its source.

    For Claude Code this is the JSONL filename UUID; for Codex it is the
    Codex session id; for rollouts it is the session id used by FileSessionStore.
    """

    message_uuid: str
    """Identifier of the assistant message that issued this edit.

    Used to index back into the transcript for UI ("show the conversation
    that produced this line").
    """

    tool_call_id: str
    """Identifier of the specific tool_use block within the message."""

    timestamp: datetime
    """When the edit was issued. Critical for fold ordering across sessions.

    Must be timezone-aware. Adapters must convert source timestamps to UTC.
    """

    # The effect itself
    path: str
    """Absolute path as recorded by the agent.

    Reconciliation with a repo happens later; we record what the agent saw.
    """

    op: EditOp

    new_content: str
    """For `write`: the full new file contents.
    For `edit`: the `new_string` that replaced `old_string`.
    For `multi_edit`: joined `new_string`s (but prefer expanding multi_edit
    into multiple `edit` FileEdits at adapter time)."""

    old_content: str | None
    """For `edit`: the `old_string` being replaced. None for `write`.

    Needed at fold time to locate the insertion point within the virtual
    file state.
    """

    def __post_init__(self) -> None:
        assert self.timestamp.tzinfo is not None, (
            f"FileEdit.timestamp must be timezone-aware, got naive {self.timestamp!r} "
            f"for {self.source}:{self.session_id}:{self.tool_call_id}"
        )
        if self.op == "write":
            assert self.old_content is None, (
                f"write edits must have old_content=None, got {self.old_content!r} "
                f"for {self.source}:{self.session_id}:{self.tool_call_id}"
            )
        if self.op == "edit":
            assert self.old_content is not None, (
                f"edit edits must have old_content, got None "
                f"for {self.source}:{self.session_id}:{self.tool_call_id}"
            )
