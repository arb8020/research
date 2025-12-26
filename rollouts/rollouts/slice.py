"""
Session slicing: select, summarize, compact, and inject messages.

Usage:
    rollouts --slice "0:4, summarize:4:18, 18:20, inject:'focus on tests'" -s abc123

Slice spec grammar:
    slice_spec := segment ("," segment)*
    segment    := range | summarize | compact | inject
    range      := start ":" end?                      # Python slice notation
    summarize  := "summarize:" start ":" end (":" quoted_string)?
    compact    := "compact:" start ":" end
    inject     := "inject:" quoted_string

Examples:
    "0:4"                              → messages 0,1,2,3
    "0:4, 10:"                         → messages 0-3, then 10 to end
    "summarize:4:18"                   → LLM summary of messages 4-17
    "summarize:4:18:'security review'" → summary focused on goal
    "compact:5:15"                     → keep structure, shrink tool results
    "inject:'check tests'"             → insert user message
    "0:4, compact:5:10, summarize:10:18:'fix bugs', 18:, inject:'now test'"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rollouts.store import SessionStore

from .dtypes import AgentSession, Endpoint, Message


@dataclass
class SliceSegment:
    """One segment of a slice operation."""

    type: Literal["range", "summarize", "compact", "inject"]
    start: int | str | None = None  # int, "N%", or None
    end: int | str | None = None  # int, "N%", or None
    content: str | None = None  # For inject
    goal: str | None = None  # For summarize with goal

    def __repr__(self) -> str:
        if self.type == "range":
            end = self.end if self.end is not None else ""
            return f"range({self.start}:{end})"
        elif self.type == "summarize":
            goal_str = f":{self.goal!r}" if self.goal else ""
            return f"summarize({self.start}:{self.end}{goal_str})"
        elif self.type == "compact":
            return f"compact({self.start}:{self.end})"
        else:
            return f"inject({self.content!r})"

    def resolve_index(self, value: int | str | None, total: int) -> int:
        """Resolve an index value (int, percentage string, or None) to an int."""
        if value is None:
            return total
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.endswith("%"):
            pct = int(value[:-1])
            return (total * pct) // 100
        return int(value)


def parse_slice_spec(spec: str) -> list[SliceSegment]:
    """Parse slice spec string into segments.

    Args:
        spec: Slice specification like "0:4, summarize:4:18:'goal', compact:18:25"

    Returns:
        List of SliceSegment objects

    Raises:
        ValueError: If spec is malformed
    """
    segments: list[SliceSegment] = []

    # Split by comma, but respect quoted strings
    parts = _split_respecting_quotes(spec)

    for raw_part in parts:
        part = raw_part.strip()
        if not part:
            continue

        segment = _parse_segment(part)
        segments.append(segment)

    return segments


def _split_respecting_quotes(s: str) -> list[str]:
    """Split by comma but respect quoted strings."""
    parts = []
    current = []
    in_quotes = False
    quote_char = None

    for char in s:
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = char
            current.append(char)
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
            current.append(char)
        elif char == "," and not in_quotes:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)

    if current:
        parts.append("".join(current))

    return parts


def _parse_index(s: str | None) -> int | str | None:
    """Parse an index string to int, percentage string, or None."""
    if not s:
        return None
    if s.endswith("%"):
        return s  # Keep as string like "80%"
    return int(s)


def _parse_segment(part: str) -> SliceSegment:
    """Parse a single segment like '0:4' or '80%:' or 'summarize:4:80%' or 'compact:0:50%'."""

    # Check for inject: prefix
    inject_match = re.match(r"inject:\s*['\"](.+)['\"]", part)
    if inject_match:
        return SliceSegment(type="inject", content=inject_match.group(1))

    # Check for summarize: prefix (with optional goal, end is optional)
    # summarize:4:18 or summarize:4: or summarize:0%:80% or summarize:4:18:'goal text'
    summarize_match = re.match(
        r"summarize:\s*(\d+%?):(\d*%?)(?::\s*['\"](.+)['\"])?", part
    )
    if summarize_match:
        return SliceSegment(
            type="summarize",
            start=_parse_index(summarize_match.group(1)),
            end=_parse_index(summarize_match.group(2)),
            goal=summarize_match.group(3),  # None if not provided
        )

    # Check for compact: prefix (end is optional, defaults to all remaining)
    # compact:0:50% or compact:0: or compact:50%:
    compact_match = re.match(r"compact:\s*(\d+%?):(\d*%?)", part)
    if compact_match:
        return SliceSegment(
            type="compact",
            start=_parse_index(compact_match.group(1)),
            end=_parse_index(compact_match.group(2)),
        )

    # Must be a range like "0:4" or "10:" or ":5" or "80%:" or ":50%"
    range_match = re.match(r"(-?\d*%?):(-?\d*%?)", part)
    if range_match:
        start_str, end_str = range_match.groups()
        return SliceSegment(
            type="range",
            start=_parse_index(start_str),
            end=_parse_index(end_str),
        )

    raise ValueError(f"Invalid slice segment: {part!r}")


# ── Compact Logic ───────────────────────────────────────────────────────────


def _compact_tool_result(
    content: str, tool_call_id: str | None, messages: list[Message], idx: int
) -> str:
    """Compact a tool result message.

    Looks back at the assistant message to find the tool name, then compacts appropriately.
    """
    # Find the tool name by looking for the matching tool call in prior assistant message
    tool_name = _find_tool_name(tool_call_id, messages, idx)

    # Compact based on tool type
    if tool_name == "read":
        return _compact_read(content)
    elif tool_name == "write":
        return _compact_write(content)
    elif tool_name == "edit":
        return _compact_edit(content)
    elif tool_name == "bash":
        return _compact_bash(content)
    elif tool_name == "list_files":
        return _compact_list_files(content)
    else:
        # Generic compaction for unknown tools
        return _compact_generic(content, tool_name)


def _find_tool_name(tool_call_id: str | None, messages: list[Message], idx: int) -> str | None:
    """Find the tool name for a tool_call_id by looking at prior assistant messages."""
    if not tool_call_id:
        return None

    # Search backwards for an assistant message with matching tool call
    for i in range(idx - 1, -1, -1):
        msg = messages[i]
        if msg.role != "assistant":
            continue

        # Check content blocks for tool calls
        content = msg.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_use" and block.get("id") == tool_call_id:
                        return block.get("name")
                    if block.get("type") == "toolCall" and block.get("id") == tool_call_id:
                        return block.get("name")
                elif hasattr(block, "id") and block.id == tool_call_id:
                    return getattr(block, "name", None)

        # Also check tool_calls attribute if present
        tool_calls = msg.get_tool_calls() if hasattr(msg, "get_tool_calls") else []
        for tc in tool_calls:
            if tc.id == tool_call_id:
                return tc.name

    return None


def _compact_read(content: str) -> str:
    """Compact a read tool result."""
    lines = content.count("\n") + 1
    chars = len(content)
    # Try to extract filename from first line or content pattern
    first_line = content.split("\n")[0][:100] if content else ""
    return f"📄 [read: {lines} lines, {chars:,} chars] {first_line}..."


def _compact_write(content: str) -> str:
    """Compact a write tool result."""
    # Write results are usually confirmations
    if "wrote" in content.lower() or "created" in content.lower():
        return f"✏️ {content[:200]}"
    lines = content.count("\n") + 1
    return f"✏️ [wrote {lines} lines]"


def _compact_edit(content: str) -> str:
    """Compact an edit tool result."""
    # Count added/removed lines from diff-like output
    added = content.count("\n+")
    removed = content.count("\n-")
    if added or removed:
        return f"🔧 [edit: +{added}/-{removed} lines]"
    return f"🔧 [edit applied] {content[:100]}..."


def _compact_bash(content: str) -> str:
    """Compact a bash tool result."""
    lines = content.split("\n")
    total_lines = len(lines)

    # Keep first 3 lines, indicate more
    if total_lines <= 5:
        return content

    preview = "\n".join(lines[:3])
    return f"{preview}\n... ({total_lines - 3} more lines)"


def _compact_list_files(content: str) -> str:
    """Compact a list_files tool result."""
    lines = content.strip().split("\n")
    count = len(lines)
    return f"📁 [listed {count} files]"


def _compact_generic(content: str, tool_name: str | None) -> str:
    """Generic compaction for unknown tools."""
    chars = len(content)
    lines = content.count("\n") + 1
    name = tool_name or "tool"

    if chars <= 500:
        return content

    preview = content[:200].replace("\n", " ")
    return f"🔧 [{name}: {lines} lines, {chars:,} chars] {preview}..."


def compact_messages(messages: list[Message]) -> list[Message]:
    """Compact tool results in a list of messages.

    Preserves structure but shrinks tool result content.
    User and assistant messages are kept as-is.
    """
    result = []

    for i, msg in enumerate(messages):
        if msg.role == "tool":
            # Compact the tool result
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            compacted = _compact_tool_result(content, msg.tool_call_id, messages, i)
            result.append(
                Message(
                    role=msg.role,
                    content=compacted,
                    tool_call_id=msg.tool_call_id,
                    timestamp=msg.timestamp,
                )
            )
        else:
            # Keep user/assistant messages as-is
            result.append(msg)

    return result


# ── Summarize Logic ─────────────────────────────────────────────────────────


async def summarize_messages(
    messages: list[Message],
    endpoint: Endpoint,
    goal: str | None = None,
) -> str:
    """Summarize a list of messages using LLM.

    Args:
        messages: Messages to summarize
        endpoint: LLM endpoint for summarization
        goal: Optional goal to focus the summary

    Returns:
        Summary text
    """
    from .dtypes import Actor, StreamEvent, TextDelta, Trajectory
    from .dtypes import Message as Msg
    from .providers import get_provider_function

    # Format messages for summarization
    formatted_parts = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # Truncate very long messages
        if len(content) > 2000:
            content = content[:2000] + "... (truncated)"
        formatted_parts.append(f"[{msg.role}] {content}")

    formatted_messages = "\n\n".join(formatted_parts)

    goal_instruction = ""
    if goal:
        goal_instruction = f"\n\nFocus the summary on context relevant to: {goal}"

    prompt = f"""Summarize the following conversation segment concisely. 
Preserve key decisions, findings, file paths mentioned, and important context.
Keep it under 500 words.{goal_instruction}

---
{formatted_messages}
---

Summary:"""

    actor = Actor(
        trajectory=Trajectory(messages=[Msg(role="user", content=prompt)]),
        endpoint=endpoint,
        tools=[],
    )

    result_parts: list[str] = []

    async def collect(event: StreamEvent) -> None:
        if isinstance(event, TextDelta):
            result_parts.append(event.delta)

    provider_fn = get_provider_function(endpoint.provider, endpoint.model)
    await provider_fn(actor, collect)

    return "".join(result_parts).strip()


# ── Apply Slice ─────────────────────────────────────────────────────────────


async def apply_slice(
    session: AgentSession,
    segments: list[SliceSegment],
    endpoint: Endpoint | None = None,
    summarize_goal: str | None = None,
) -> list[Message]:
    """Apply slice operations to build new message list.

    Segments are processed in order. Each segment appends to the result:
    - range: appends messages[start:end]
    - summarize: appends a summary of messages[start:end] as a user message
    - compact: appends compacted versions of messages[start:end]
    - inject: appends a user message with the given content

    Args:
        session: Source session
        segments: Parsed slice segments
        endpoint: LLM endpoint (required if any segment is summarize)
        summarize_goal: Default goal for summaries (segment goal overrides)

    Returns:
        New message list
    """
    result: list[Message] = []
    messages = session.messages
    total = len(messages)

    for seg in segments:
        if seg.type == "range":
            start = seg.resolve_index(seg.start, total) if seg.start is not None else 0
            end = seg.resolve_index(seg.end, total) if seg.end is not None else None
            sliced = messages[start:end]
            result.extend(sliced)

        elif seg.type == "summarize":
            if endpoint is None:
                raise ValueError("Endpoint required for summarize segments")

            start = seg.resolve_index(seg.start, total) if seg.start is not None else 0
            end = seg.resolve_index(seg.end, total) if seg.end is not None else total

            to_summarize = messages[start:end]
            if not to_summarize:
                continue

            # Use segment goal if provided, otherwise fall back to global goal
            goal = seg.goal or summarize_goal
            summary = await summarize_messages(to_summarize, endpoint, goal)

            # Create a user message with the summary
            summary_msg = Message(
                role="user",
                content=f"[Summary of messages {start}-{end - 1}]\n\n{summary}",
            )
            result.append(summary_msg)

        elif seg.type == "compact":
            start = seg.resolve_index(seg.start, total) if seg.start is not None else 0
            end = seg.resolve_index(seg.end, total) if seg.end is not None else total

            to_compact = messages[start:end]
            if not to_compact:
                continue

            compacted = compact_messages(to_compact)
            result.extend(compacted)

        elif seg.type == "inject":
            if seg.content is None:
                continue
            inject_msg = Message(role="user", content=seg.content)
            result.append(inject_msg)

    return result


async def slice_session(
    session: AgentSession,
    spec: str,
    endpoint: Endpoint,
    session_store: SessionStore,
    summarize_goal: str | None = None,
) -> AgentSession:
    """Create new session from sliced/summarized messages.

    Args:
        session: Source session
        spec: Slice specification string
        endpoint: LLM endpoint (for summaries and new session)
        session_store: Storage backend
        summarize_goal: Optional goal to focus summaries

    Returns:
        New child session with sliced messages
    """

    # Parse spec
    segments = parse_slice_spec(spec)

    # Apply operations
    new_messages = await apply_slice(session, segments, endpoint, summarize_goal)

    if not new_messages:
        raise ValueError("Slice resulted in empty message list")

    # Create child session
    child = await session_store.create(
        endpoint=session.endpoint,
        environment=session.environment,
        parent_id=session.session_id,
        branch_point=len(new_messages),
        tags={"sliced": "true", "slice_spec": spec},
    )

    # Save messages
    for msg in new_messages:
        await session_store.append_message(child.session_id, msg)

    return child


# ── CLI Integration ─────────────────────────────────────────────────────────


async def run_slice_command(
    session: AgentSession,
    spec: str,
    endpoint: Endpoint,
    session_store: SessionStore,
    summarize_goal: str | None = None,
) -> tuple[AgentSession | None, str | None]:
    """Run --slice command.

    Args:
        session: Source session
        spec: Slice specification
        endpoint: LLM endpoint
        session_store: Storage backend
        summarize_goal: Optional goal for summaries

    Returns:
        (new_session, None) on success, (None, error) on failure
    """
    try:
        segments = parse_slice_spec(spec)
    except ValueError as e:
        return None, f"Invalid slice spec: {e}"

    # Check if summarize is needed
    has_summarize = any(s.type == "summarize" for s in segments)
    if has_summarize and endpoint is None:
        return None, "Summarize requires --model to be specified"

    try:
        child = await slice_session(
            session=session,
            spec=spec,
            endpoint=endpoint,
            session_store=session_store,
            summarize_goal=summarize_goal,
        )
    except Exception as e:
        return None, f"Slice failed: {e}"
    else:
        return child, None
