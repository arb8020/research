"""
Session export to Markdown and HTML, plus session transformations.

Usage:
    from rollouts import session_to_markdown, session_to_html
    # or: from rollouts.export import session_to_markdown, session_to_html

    md = session_to_markdown(session)
    html = session_to_html(session)

    # Create compacted child session
    child = compact_session(parent)
"""

from __future__ import annotations

import html
import json
from dataclasses import replace
from datetime import datetime
from typing import Any, TYPE_CHECKING
import uuid

from rollouts.dtypes import AgentSession, SessionMessage, SessionStatus


def format_content_block(block: dict[str, Any]) -> str:
    """Format a single content block to markdown."""
    block_type = block.get("type", "")

    if block_type == "text":
        return block.get("text", "")

    elif block_type == "thinking":
        thinking = block.get("thinking", "")
        return f"*<thinking>*\n{thinking}\n*</thinking>*"

    elif block_type == "toolCall":
        name = block.get("name", "unknown")
        args = block.get("arguments", {})
        args_str = json.dumps(args, indent=2)
        return f"**Tool Call: {name}**\n```json\n{args_str}\n```"

    elif block_type == "image":
        url = block.get("image_url", "")
        if url.startswith("data:"):
            return "[Embedded Image]"
        return f"![Image]({url})"

    else:
        # Unknown block type - dump as JSON
        return f"```json\n{json.dumps(block, indent=2)}\n```"


def format_message_content(content: str | list[dict[str, Any]]) -> str:
    """Format message content (string or content blocks) to markdown."""
    if isinstance(content, str):
        return content

    # Content blocks
    parts = [format_content_block(block) for block in content]
    return "\n\n".join(parts)


def session_to_markdown(session: AgentSession, include_metadata: bool = True) -> str:
    """Convert session to markdown.

    Args:
        session: The session to convert
        include_metadata: Whether to include header with session metadata

    Returns:
        Markdown string
    """
    lines: list[str] = []

    if include_metadata:
        lines.append(f"# Session {session.session_id}")
        lines.append("")
        lines.append(f"- **Created**: {session.created_at}")
        lines.append(f"- **Model**: {session.endpoint.provider}/{session.endpoint.model}")
        lines.append(f"- **Status**: {session.status.value}")
        if session.parent_id:
            lines.append(f"- **Branched from**: {session.parent_id} (at message {session.branch_point})")
        lines.append("")
        lines.append("---")
        lines.append("")

    for msg in session.messages:
        role = msg.role.upper()
        content = format_message_content(msg.content)

        # Role header
        if msg.role == "system":
            lines.append("## System")
        elif msg.role == "user":
            lines.append("## User")
        elif msg.role == "assistant":
            lines.append("## Assistant")
        elif msg.role == "tool":
            tool_id = msg.tool_call_id or "unknown"
            lines.append(f"## Tool Result ({tool_id})")
        else:
            lines.append(f"## {role}")

        lines.append("")
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


def session_to_html(session: AgentSession) -> str:
    """Convert session to standalone HTML.

    Args:
        session: The session to convert

    Returns:
        HTML string (complete document)
    """
    # Get markdown first, then wrap in HTML with styling
    # For now, just escape and wrap - could use a proper md->html converter later

    html_parts: list[str] = []

    # HTML header with dark theme styling
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Session {session_id}</title>
    <style>
        :root {{
            --bg: #1a1a1a;
            --fg: #e0e0e0;
            --muted: #888;
            --accent: #8abeb7;
            --user-bg: #2a2a3a;
            --assistant-bg: #1a1a1a;
            --tool-bg: #1a2a1a;
            --system-bg: #2a2a2a;
            --border: #404040;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: var(--bg);
            color: var(--fg);
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
        }}
        h1 {{
            color: var(--accent);
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
        }}
        h2 {{
            color: var(--muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            margin-top: 2rem;
            margin-bottom: 0.5rem;
        }}
        .metadata {{
            color: var(--muted);
            font-size: 0.85rem;
            margin-bottom: 2rem;
        }}
        .message {{
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            white-space: pre-wrap;
        }}
        .message.user {{ background: var(--user-bg); }}
        .message.assistant {{ background: var(--assistant-bg); border-left: 2px solid var(--accent); }}
        .message.tool {{ background: var(--tool-bg); font-family: monospace; font-size: 0.9rem; }}
        .message.system {{ background: var(--system-bg); color: var(--muted); }}
        .thinking {{
            color: var(--muted);
            font-style: italic;
            border-left: 2px solid var(--muted);
            padding-left: 1rem;
            margin: 0.5rem 0;
        }}
        pre {{
            background: #0a0a0a;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
        }}
        code {{
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <h1>Session {session_id}</h1>
    <div class="metadata">
        <div>Created: {created_at}</div>
        <div>Model: {provider}/{model}</div>
        <div>Status: {status}</div>
        {parent_info}
    </div>
    <hr>
""".format(
        session_id=html.escape(session.session_id),
        created_at=html.escape(session.created_at),
        provider=html.escape(session.endpoint.provider),
        model=html.escape(session.endpoint.model),
        status=html.escape(session.status.value),
        parent_info=f"<div>Branched from: {html.escape(session.parent_id or '')} (at message {session.branch_point})</div>" if session.parent_id else "",
    ))

    # Messages
    for msg in session.messages:
        role_class = msg.role
        role_label = msg.role.upper()

        if msg.role == "tool":
            role_label = f"TOOL RESULT ({html.escape(msg.tool_call_id or 'unknown')})"

        content_html = format_content_html(msg.content)

        html_parts.append(f"""
    <h2>{role_label}</h2>
    <div class="message {role_class}">
{content_html}
    </div>
""")

    # HTML footer
    html_parts.append("""
</body>
</html>
""")

    return "".join(html_parts)


def format_content_html(content: str | list[dict[str, Any]]) -> str:
    """Format message content to HTML."""
    if isinstance(content, str):
        return html.escape(content)

    # Content blocks
    parts: list[str] = []
    for block in content:
        block_type = block.get("type", "")

        if block_type == "text":
            parts.append(html.escape(block.get("text", "")))

        elif block_type == "thinking":
            thinking = html.escape(block.get("thinking", ""))
            parts.append(f'<div class="thinking">{thinking}</div>')

        elif block_type == "toolCall":
            name = html.escape(block.get("name", "unknown"))
            args = json.dumps(block.get("arguments", {}), indent=2)
            parts.append(f"<strong>Tool Call: {name}</strong><pre><code>{html.escape(args)}</code></pre>")

        elif block_type == "image":
            url = block.get("image_url", "")
            if url.startswith("data:"):
                parts.append(f'<img src="{url}" style="max-width: 100%;">')
            else:
                parts.append(f'<img src="{html.escape(url)}" style="max-width: 100%;">')

        else:
            # Unknown - dump as JSON
            parts.append(f"<pre><code>{html.escape(json.dumps(block, indent=2))}</code></pre>")

    return "\n".join(parts)


# --- Session Transformations ---


def compact_content_block(block: dict[str, Any], max_length: int = 500) -> dict[str, Any]:
    """Compact a single content block by truncating verbose content."""
    block_type = block.get("type", "")

    if block_type == "text":
        text = block.get("text", "")
        if len(text) > max_length:
            return {**block, "text": text[:max_length] + f"\n... [truncated {len(text) - max_length} chars]"}
        return block

    elif block_type == "toolResult":
        # Tool results are often verbose - truncate aggressively
        content = block.get("content", "")
        if isinstance(content, str) and len(content) > max_length:
            return {**block, "content": content[:max_length] + f"\n... [truncated {len(content) - max_length} chars]"}
        elif isinstance(content, list):
            # Content blocks in tool result
            compacted = [compact_content_block(b, max_length) for b in content]
            return {**block, "content": compacted}
        return block

    # Other types pass through unchanged
    return block


def compact_message_content(
    content: str | list[dict[str, Any]],
    max_length: int = 500,
) -> str | list[dict[str, Any]]:
    """Compact message content by truncating verbose parts."""
    if isinstance(content, str):
        if len(content) > max_length:
            return content[:max_length] + f"\n... [truncated {len(content) - max_length} chars]"
        return content

    # Content blocks
    return [compact_content_block(block, max_length) for block in content]


def compact_session(
    session: AgentSession,
    max_content_length: int = 500,
) -> AgentSession:
    """Create a compacted child session with truncated tool results.

    Args:
        session: Parent session to compact
        max_content_length: Max chars before truncation (default 500)

    Returns:
        New child session with compacted messages
    """
    # Generate new session ID
    now = datetime.now()
    session_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}"

    # Compact each message
    compacted_messages: list[SessionMessage] = []
    for msg in session.messages:
        compacted_content = compact_message_content(msg.content, max_content_length)
        compacted_msg = SessionMessage(
            role=msg.role,
            content=compacted_content,
            tool_call_id=msg.tool_call_id,
        )
        compacted_messages.append(compacted_msg)

    # Create child session
    return AgentSession(
        session_id=session_id,
        created_at=now.isoformat(),
        endpoint=session.endpoint,
        environment=session.environment,
        messages=compacted_messages,
        status=SessionStatus.PENDING,
        parent_id=session.session_id,
        branch_point=len(session.messages),
    )


def summarize_session(
    session: AgentSession,
    summary: str,
) -> AgentSession:
    """Create a summarized child session with LLM-generated summary as first message.

    Args:
        session: Parent session to summarize
        summary: LLM-generated summary text

    Returns:
        New child session with summary as system/user message
    """
    # Generate new session ID
    now = datetime.now()
    session_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}"

    # Create summary message as user message (context for next assistant turn)
    summary_msg = SessionMessage(
        role="user",
        content=f"[Session Summary from {session.session_id}]\n\n{summary}\n\nPlease continue from here.",
    )

    # Create child session with just the summary
    return AgentSession(
        session_id=session_id,
        created_at=now.isoformat(),
        endpoint=session.endpoint,
        environment=session.environment,
        messages=[summary_msg],
        status=SessionStatus.PENDING,
        parent_id=session.session_id,
        branch_point=len(session.messages),
    )


# --- CLI Command Runners ---


async def run_compact_command(
    session_store: "SessionStore",
    session: AgentSession,
) -> tuple[AgentSession, None] | tuple[None, str]:
    """Run the compact command on a session.

    Args:
        session_store: Store to save the compacted session
        session: Session to compact

    Returns:
        (child_session, None) on success, (None, error) on failure
    """
    child_session = compact_session(session)
    await session_store.save(child_session)
    return child_session, None


async def run_summarize_command(
    session_store: "SessionStore",
    session: AgentSession,
    endpoint: "Endpoint",
) -> tuple[AgentSession, None] | tuple[None, str]:
    """Run the summarize command on a session.

    Args:
        session_store: Store to save the summarized session
        session: Session to summarize
        endpoint: LLM endpoint for generating summary

    Returns:
        (child_session, None) on success, (None, error) on failure
    """
    from rollouts.dtypes import Actor, Endpoint, Message, Trajectory, TextDelta, StreamEvent
    from rollouts.providers import get_provider_function

    print(f"Summarizing session {session.session_id} ({len(session.messages)} messages)...")

    # Convert session to markdown for LLM
    session_md = session_to_markdown(session, include_metadata=False)
    summary_prompt = f"""Summarize this conversation session concisely. Focus on:
1. What was the main task/goal
2. Key decisions made
3. What was accomplished
4. Any open items or next steps

Session content:
{session_md}

Provide a clear, actionable summary that would help someone continue this work."""

    # Create actor with summary prompt
    actor = Actor(
        trajectory=Trajectory(messages=[Message(role="user", content=summary_prompt)]),
        endpoint=endpoint,
        tools=[],
    )

    # Stream response
    summary_parts: list[str] = []

    async def collect_text(event: StreamEvent) -> None:
        if isinstance(event, TextDelta):
            summary_parts.append(event.delta)
            print(event.delta, end="", flush=True)

    provider_fn = get_provider_function(endpoint.provider, endpoint.model)
    await provider_fn(actor, collect_text)
    print()  # newline after streaming

    summary = "".join(summary_parts)

    # Create and save summarized child
    child_session = summarize_session(session, summary)
    await session_store.save(child_session)
    return child_session, None


# Type hint for SessionStore (avoid circular import)
if TYPE_CHECKING:
    from rollouts.store import SessionStore
    from rollouts.dtypes import Endpoint
