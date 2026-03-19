from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import trio

from ..core import (
    Trajectory,
    TrajectoryEnvironment,
    TrajectorySession,
)
from ..store import FileSessionStore


def format_time_ago(dt_str: str) -> str:
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt

        if diff.days > 0:
            return f"{diff.days}d ago"
        hours = diff.seconds // 3600
        if hours > 0:
            return f"{hours}h ago"
        minutes = diff.seconds // 60
        if minutes > 0:
            return f"{minutes}m ago"
        return "just now"
    except Exception:
        return dt_str[:10]


async def pick_session_async(session_store: FileSessionStore) -> Trajectory | None:
    sessions = await session_store.list(limit=20)
    if not sessions:
        print("No sessions found.", file=sys.stderr)
        return None

    print("Recent sessions:")
    for i, session in enumerate(sessions, 1):
        updated = format_time_ago(session.updated_at) if session.updated_at else "?"
        message_count = session.message_count if hasattr(session, "message_count") else "?"
        print(f"{i:2}. {session.session_id} ({message_count} msgs, {updated})")

    while True:
        try:
            choice = input("\nSelect session (number, or Enter to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not choice:
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                session_id = sessions[idx].session_id
                session, err = await session_store.get(session_id)
                if err or session is None:
                    print(f"Error loading session: {err}", file=sys.stderr)
                    return None
                return session
        except ValueError:
            pass
        print("Invalid choice.", file=sys.stderr)


def cmd_export(config: Any, session_store: FileSessionStore) -> int:
    from ..export import session_to_html, session_to_markdown

    async def export_action() -> int:
        if config.session is not None and config.session != "":
            session, err = await session_store.get(config.session)
            if err or session is None:
                print(f"Error loading session: {err}", file=sys.stderr)
                return 1
        elif config.session == "" or config.continue_session:
            if config.continue_session:
                summary, err = await session_store.get_latest()
                if err or summary is None:
                    print("No sessions found", file=sys.stderr)
                    return 1
                session, err = await session_store.get(summary.session_id)
                if err or session is None:
                    print(f"Error loading session: {err}", file=sys.stderr)
                    return 1
            else:
                session = await pick_session_async(session_store)
                if session is None:
                    return 0
        else:
            summary, err = await session_store.get_latest()
            if err or summary is None:
                print("No sessions found. Use -s to select a session.", file=sys.stderr)
                return 1
            session, err = await session_store.get(summary.session_id)
            if err or session is None:
                print(f"Error loading session: {err}", file=sys.stderr)
                return 1

        if config.export_md is not None:
            output = session_to_markdown(session)
            export_path = config.export_md
        else:
            output = session_to_html(session)
            export_path = config.export_html

        assert export_path is not None
        if export_path == "":
            print(output)
        else:
            Path(export_path).write_text(output)
            print(f"Exported to {export_path}")
        return 0

    return trio.run(export_action)


def _diagnose_session_issues(
    session: Trajectory,
) -> list[tuple[str, str, list[int]]]:
    issues: list[tuple[str, str, list[int]]] = []

    tool_result_ids: dict[str, list[int]] = {}
    for i, msg in enumerate(session.messages):
        if msg.role == "tool" and msg.tool_call_id:
            tool_result_ids.setdefault(msg.tool_call_id, []).append(i)

    duplicate_results = {k: v for k, v in tool_result_ids.items() if len(v) > 1}
    for tool_id, indices in duplicate_results.items():
        issues.append((
            "duplicate_tool_result",
            f"Tool result '{tool_id[:20]}...' appears {len(indices)} times at messages {indices}",
            indices[1:],
        ))

    tool_call_ids: set[str] = set()
    for msg in session.messages:
        if msg.role == "assistant" and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    tool_call_ids.add(block.get("id", ""))

    for i, msg in enumerate(session.messages):
        if msg.role == "tool" and msg.tool_call_id and msg.tool_call_id not in tool_call_ids:
            issues.append((
                "orphaned_tool_result",
                f"Tool result '{msg.tool_call_id[:20]}...' at message {i} has no matching tool_use",
                [i],
            ))

    for i, msg in enumerate(session.messages):
        content_len = len(str(msg.content))
        if content_len > 100_000:
            issues.append((
                "oversized_message",
                f"Message {i} ({msg.role}) is {content_len:,} chars ({content_len // 4:,} est. tokens)",
                [i],
            ))

    return issues


async def _fix_session_issues(
    session: Trajectory,
    issues: list[tuple[str, str, list[int]]],
    session_store: FileSessionStore,
) -> int:
    indices_to_remove: set[int] = set()
    for issue_type, _, affected in issues:
        if issue_type in ("duplicate_tool_result", "orphaned_tool_result"):
            indices_to_remove.update(affected)

    if not indices_to_remove:
        print("\nNo auto-fixable issues found. Use --trim N for oversized messages.")
        return 0

    fixed_messages = [msg for i, msg in enumerate(session.messages) if i not in indices_to_remove]
    fixed_trajectory = Trajectory(
        messages=fixed_messages,
        session=TrajectorySession(
            parent_id=session.session_id,
            branch_point=len(fixed_messages),
            endpoint=session.endpoint,
            tags={"doctor": "fixed", "removed_indices": str(sorted(indices_to_remove))},
            vcs=session.vcs,
        ),
        environment=TrajectoryEnvironment.from_session_parts(
            session.environment_config(),
            session.environment_state(),
        ),
    )

    new_session, err = await session_store.save_trajectory(fixed_trajectory)
    if err is not None or new_session is None:
        print(f"\nFailed to create fixed session: {err or 'unknown error'}", file=sys.stderr)
        return 1

    print(f"\nCreated fixed session: {new_session.session_id}")
    print(f"  Removed {len(indices_to_remove)} message(s) at indices: {sorted(indices_to_remove)}")
    print(f"  Parent: {session.session_id}")
    print(f"\nResume with: rollouts --session {new_session.session_id}")
    return 0


async def _trim_session(
    session: Trajectory,
    trim_count: int,
    session_store: FileSessionStore,
) -> int:
    if trim_count <= 0:
        print("\n--trim must be a positive integer", file=sys.stderr)
        return 1
    if trim_count >= len(session.messages):
        print(
            f"\nCannot trim {trim_count} messages from session with {len(session.messages)} messages",
            file=sys.stderr,
        )
        return 1

    trimmed_messages = session.messages[:-trim_count]
    trimmed_trajectory = Trajectory(
        messages=trimmed_messages,
        session=TrajectorySession(
            parent_id=session.session_id,
            branch_point=len(trimmed_messages),
            endpoint=session.endpoint,
            tags={"doctor": "trimmed", "trimmed_count": str(trim_count)},
            vcs=session.vcs,
        ),
        environment=TrajectoryEnvironment.from_session_parts(
            session.environment_config(),
            session.environment_state(),
        ),
    )

    new_session, err = await session_store.save_trajectory(trimmed_trajectory)
    if err is not None or new_session is None:
        print(f"\nFailed to create fixed session: {err or 'unknown error'}", file=sys.stderr)
        return 1

    print(f"\nCreated fixed session: {new_session.session_id}")
    print(f"  Trimmed {trim_count} messages (kept {len(trimmed_messages)})")
    print(f"  Parent: {session.session_id}")
    print(f"\nResume with: rollouts --session {new_session.session_id}")
    return 0


def cmd_doctor(config: Any, session_store: FileSessionStore) -> int:
    async def doctor_action() -> int:
        if config.session and config.session != "":
            target_session_id = config.session
        elif config.continue_session:
            target_session_id = session_store.get_latest_id_sync()
        else:
            target_session_id = session_store.get_latest_id_sync()

        if not target_session_id:
            print("No session found. Use --session <id> to specify.", file=sys.stderr)
            return 1

        session, err = await session_store.get(target_session_id)
        if err or not session:
            print(f"Error loading session: {err}", file=sys.stderr)
            return 1

        total_chars = sum(len(str(msg.content)) for msg in session.messages)
        estimated_tokens = total_chars // 4
        print(f"Session: {session.session_id}")
        print(f"  Messages: {len(session.messages)}")
        print(f"  Total chars: {total_chars:,}")
        print(f"  Est. tokens: {estimated_tokens:,}")
        print(f"  Status: {session.status}")
        if session.parent_id:
            print(f"  Parent: {session.parent_id}")

        issues = _diagnose_session_issues(session)
        if issues:
            print(f"\n⚠️  Found {len(issues)} issue(s):")
            for issue_type, desc, _ in issues:
                print(f"  [{issue_type}] {desc}")

        if config.fix and issues:
            return await _fix_session_issues(session, issues, session_store)

        if session.messages:
            print("\nLast 5 messages:")
            for msg in session.messages[-5:]:
                content_preview = str(msg.content)[:80].replace("\n", " ")
                content_len = len(str(msg.content))
                print(f"  [{msg.role}] {content_preview}... ({content_len:,} chars)")

        if config.trim is not None:
            return await _trim_session(session, config.trim, session_store)

        return 0

    return trio.run(doctor_action)


def cmd_handoff(config: Any, session_store: FileSessionStore) -> int:
    from ..export import run_handoff_command

    async def handoff_action() -> int:
        if config.session is None:
            print("Error: --handoff requires -s <session_id>", file=sys.stderr)
            return 1

        if config.session == "":
            session = await pick_session_async(session_store)
            if session is None:
                return 0
        else:
            session, err = await session_store.get(config.session)
            if err or session is None:
                print(f"Error loading session: {err}", file=sys.stderr)
                return 1

        assert config.endpoint is not None
        assert config.handoff is not None

        if config.fast_handoff:
            print(f"Extracting context (fast) for: {config.handoff}", file=sys.stderr)
            print(
                f"From session: {session.session_id} ({len(session.messages)} messages)",
                file=sys.stderr,
            )
            print(file=sys.stderr)
            handoff_md, err = await run_handoff_command(session, config.endpoint, config.handoff)
        else:
            from ..environments.handoff import generate_handoff_context_agent

            print(f"Extracting context for: {config.handoff}", file=sys.stderr)
            print(
                f"From session: {session.session_id} ({len(session.messages)} messages)",
                file=sys.stderr,
            )
            print(file=sys.stderr)
            handoff_md, err = await generate_handoff_context_agent(
                session_id=session.session_id,
                goal=config.handoff,
                endpoint=config.endpoint,
                sessions_dir=session_store.base_dir,
                working_dir=Path.cwd(),
            )

        if err:
            print(f"Error: {err}", file=sys.stderr)
            return 1
        print(handoff_md)
        return 0

    return trio.run(handoff_action)


def cmd_slice(config: Any, session_store: FileSessionStore) -> int:
    from ..slice import run_slice_command

    def estimate_tokens(messages: list[Any]) -> int:
        total_chars = sum(
            len(m.content) if isinstance(m.content, str) else len(str(m.content)) for m in messages
        )
        return total_chars // 4

    async def slice_action() -> int:
        if config.session is None:
            print("Error: --slice requires -s <session_id>", file=sys.stderr)
            return 1

        if config.session == "":
            session = await pick_session_async(session_store)
            if session is None:
                return 0
        else:
            session, err = await session_store.get(config.session)
            if err or session is None:
                print(f"Error loading session: {err}", file=sys.stderr)
                return 1

        assert config.endpoint is not None
        assert config.slice is not None

        if config.slice.strip().lower() == "count":
            print(len(session.messages))
            return 0

        source_tokens = estimate_tokens(session.messages)
        print(
            f"Slicing: {session.session_id} ({len(session.messages)} messages, ~{source_tokens:,} tokens)",
            file=sys.stderr,
        )
        print(f"Spec: {config.slice}", file=sys.stderr)

        child, err = await run_slice_command(
            session=session,
            spec=config.slice,
            endpoint=config.endpoint,
            session_store=session_store,
            summarize_goal=config.slice_goal,
        )
        if err:
            print(f"Error: {err}", file=sys.stderr)
            return 1
        assert child is not None

        child_full, _ = await session_store.get(child.session_id)
        if child_full:
            child_tokens = estimate_tokens(child_full.messages)
            reduction = (1 - child_tokens / source_tokens) * 100 if source_tokens > 0 else 0
            print(
                f"Created: {child.session_id} ({len(child_full.messages)} messages, ~{child_tokens:,} tokens)",
                file=sys.stderr,
            )
            if reduction > 0:
                print(f"Reduction: {reduction:.0f}% fewer tokens", file=sys.stderr)
        else:
            print(f"Created: {child.session_id}", file=sys.stderr)

        print(child.session_id)
        return 0

    return trio.run(slice_action)


def cmd_ls(session_store: FileSessionStore, include_all: bool = False) -> int:
    async def ls_action() -> int:
        sessions = await session_store.list(limit=100)
        if not include_all:
            sessions = [s for s in sessions if s.status == "pending"]

        if not sessions:
            print(
                "No sessions found."
                if include_all
                else "No active sessions. Use --ls-all to see all sessions."
            )
            return 0

        print(f"{'SESSION ID':<28} {'STATUS':<12} {'MODEL':<25} {'UPDATED':<12}")
        print("-" * 77)
        for session in sessions:
            model = f"{session.endpoint.provider}/{session.endpoint.model}"
            if len(model) > 25:
                model = model[:22] + "..."
            updated = format_time_ago(session.updated_at) if session.updated_at else "?"
            print(f"{session.session_id:<28} {session.status:<12} {model:<25} {updated:<12}")
        return 0

    return trio.run(ls_action)


def cmd_status(session_store: FileSessionStore, session_id: str) -> int:
    async def status_action() -> int:
        session, err = await session_store.get(session_id)
        if err or not session:
            print(f"Session not found: {session_id}", file=sys.stderr)
            return 1

        print(f"Session: {session.session_id}")
        print(f"Status:  {session.status}")
        print(f"Model:   {session.endpoint.provider}/{session.endpoint.model}")
        print(f"Messages: {len(session.messages)}")
        if session.updated_at:
            print(f"Updated: {session.updated_at}")
        return 0

    return trio.run(status_action)


def cmd_attach(config: Any, session_id: str) -> int:
    config.session = session_id
    config.frontend = "tui"
    config.detached = False
    return -1
