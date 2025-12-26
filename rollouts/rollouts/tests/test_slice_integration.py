"""Integration tests for session slicing.

Grugbrain philosophy:
- Focus on integration tests (sweet spot)
- Test actual slice → new session flow
- Minimal unit tests, mostly for getting started
- One curated end-to-end test for the main use case
"""

import pytest
import trio

from rollouts.dtypes import (
    AgentSession,
    Endpoint,
    EnvironmentConfig,
    Message,
    SessionStatus,
)
from rollouts.slice import apply_slice, parse_slice_spec, slice_session
from rollouts.store import FileSessionStore


# ── Fixtures ────────────────────────────────────────────────────────────────


def make_realistic_session(n_turns: int = 10) -> AgentSession:
    """Create a session that looks like a real coding session."""
    messages = [
        Message(role="system", content="You are a coding assistant."),
        Message(role="user", content="Help me refactor the auth module."),
    ]

    # Simulate turns with tool calls
    for i in range(n_turns):
        # Assistant reads a file
        messages.append(
            Message(
                role="assistant",
                content=[
                    {"type": "text", "text": f"Let me check file {i}."},
                    {
                        "type": "tool_use",
                        "id": f"tool_{i}",
                        "name": "read",
                        "input": {"path": f"src/file{i}.py"},
                    },
                ],
            )
        )
        # Tool result with realistic file content
        messages.append(
            Message(
                role="tool",
                content=f"def function_{i}():\n    # Implementation\n    pass\n" * 20,
                tool_call_id=f"tool_{i}",
            )
        )
        # Assistant comments
        messages.append(
            Message(role="assistant", content=f"I see file {i} has function_{i}.")
        )
        # User follow-up
        if i < n_turns - 1:
            messages.append(Message(role="user", content=f"Good, now check file {i+1}."))

    return AgentSession(
        session_id="test-realistic",
        endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4-5-20250929"),
        environment=EnvironmentConfig(type="coding", config={}),
        messages=messages,
        status=SessionStatus.PENDING,
        tags={},
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
    )


# ── Integration Tests (the sweet spot) ──────────────────────────────────────


class TestSliceIntegration:
    """Integration tests for the slice flow."""

    @pytest.mark.trio
    async def test_compact_reduces_token_count(self) -> None:
        """Compact should significantly reduce message sizes."""
        session = make_realistic_session(10)

        # Measure original size
        original_chars = sum(
            len(m.content) if isinstance(m.content, str) else len(str(m.content))
            for m in session.messages
        )

        # Compact everything
        segments = parse_slice_spec("compact:0:")
        result = await apply_slice(session, segments)

        # Measure compacted size
        compacted_chars = sum(
            len(m.content) if isinstance(m.content, str) else len(str(m.content))
            for m in result
        )

        # Should be significantly smaller (tool results shrunk)
        assert compacted_chars < original_chars * 0.5, (
            f"Expected >50% reduction, got {compacted_chars}/{original_chars} "
            f"= {compacted_chars/original_chars:.0%}"
        )

    @pytest.mark.trio
    async def test_slice_preserves_structure(self) -> None:
        """Slicing should preserve message structure and order."""
        session = make_realistic_session(10)

        # Keep first 5 messages, compact next 10, keep rest
        segments = parse_slice_spec("0:5, compact:5:15, 15:")
        result = await apply_slice(session, segments)

        # Should have same count (compact doesn't remove messages)
        assert len(result) == len(session.messages)

        # First 5 should be unchanged
        for i in range(5):
            assert result[i].role == session.messages[i].role
            assert result[i].content == session.messages[i].content

        # Last messages should be unchanged
        assert result[-1].content == session.messages[-1].content

    @pytest.mark.trio
    async def test_percentage_slicing(self) -> None:
        """Percentage-based slicing should work correctly."""
        session = make_realistic_session(10)
        total = len(session.messages)

        # Keep last 20%
        segments = parse_slice_spec("80%:")
        result = await apply_slice(session, segments)

        expected_count = total - (total * 80 // 100)
        assert len(result) == expected_count

    @pytest.mark.trio
    async def test_inject_adds_user_message(self) -> None:
        """Inject should add a user message at the right spot."""
        session = make_realistic_session(5)

        segments = parse_slice_spec("0:5, inject:'Now focus on security'")
        result = await apply_slice(session, segments)

        # Should have one more message
        assert len(result) == 6

        # Last message should be our injection
        assert result[-1].role == "user"
        assert result[-1].content == "Now focus on security"

    @pytest.mark.trio
    async def test_full_slice_workflow(self) -> None:
        """Test the complete slice workflow: parse → apply → verify."""
        session = make_realistic_session(20)

        # Realistic use case: keep system+first user, compact middle, keep recent
        spec = "0:2, compact:2:80%, 80%:"
        segments = parse_slice_spec(spec)
        result = await apply_slice(session, segments)

        # Verify structure
        assert result[0].role == "system"
        assert result[1].role == "user"

        # Recent messages should be unchanged
        total = len(session.messages)
        keep_from = total * 80 // 100
        recent_original = session.messages[keep_from:]
        recent_result = result[-(total - keep_from) :]
        assert len(recent_result) == len(recent_original)


class TestSliceSessionPersistence:
    """Test that slice creates proper child sessions."""

    @pytest.mark.trio
    async def test_slice_creates_child_session(self, tmp_path) -> None:
        """Slicing should create a new session with correct parent reference."""
        store = FileSessionStore(base_dir=tmp_path)

        # Create parent session
        parent = await store.create(
            endpoint=Endpoint(provider="test", model="test"),
            environment=EnvironmentConfig(type="none", config={}),
        )

        # Add some messages
        for i in range(10):
            await store.append_message(
                parent.session_id,
                Message(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}"),
            )

        # Load it back
        parent_loaded, _ = await store.get(parent.session_id)
        assert parent_loaded is not None

        # Slice it (no summarize, so no endpoint needed for LLM call)
        child = await slice_session(
            session=parent_loaded,
            spec="0:3, 7:",
            endpoint=Endpoint(provider="test", model="test"),
            session_store=store,
        )

        # Verify child exists and has correct parent
        assert child.session_id != parent.session_id
        child_loaded, _ = await store.get(child.session_id)
        assert child_loaded is not None
        assert len(child_loaded.messages) == 6  # 3 + 3

        # Parent should be unchanged
        parent_check, _ = await store.get(parent.session_id)
        assert len(parent_check.messages) == 10


# ── Minimal Unit Tests (just for getting started) ───────────────────────────


class TestParseBasics:
    """Minimal parsing tests - just enough to know it works."""

    def test_parses_common_patterns(self) -> None:
        """Test the patterns users will actually use."""
        # These are the main use cases
        cases = [
            ("0:10", 1),  # Simple range
            ("compact:0:", 1),  # Compact all
            ("0:2, summarize:2:, inject:'continue'", 3),  # Full workflow
            ("80%:", 1),  # Percentage
        ]

        for spec, expected_count in cases:
            segments = parse_slice_spec(spec)
            assert len(segments) == expected_count, f"Failed for: {spec}"

    def test_invalid_spec_raises(self) -> None:
        """Invalid specs should raise ValueError."""
        invalid = ["garbage", "compact", "summarize:abc"]

        for spec in invalid:
            with pytest.raises(ValueError):
                parse_slice_spec(spec)


# ── End-to-End Test (curated, must always pass) ─────────────────────────────


class TestEndToEnd:
    """One curated end-to-end test for the main use case."""

    @pytest.mark.trio
    async def test_self_compaction_workflow(self, tmp_path) -> None:
        """
        Test the main use case: agent self-compacts when context fills up.

        This is the critical path that must always work:
        1. Session exists with many messages
        2. Agent runs --slice to compact/summarize
        3. New child session is created
        4. Agent can continue in child session
        """
        store = FileSessionStore(base_dir=tmp_path)

        # 1. Create a "full" session (simulating context approaching limits)
        parent = await store.create(
            endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4-5-20250929"),
            environment=EnvironmentConfig(type="coding", config={}),
        )

        # Add realistic content
        await store.append_message(
            parent.session_id, Message(role="system", content="You are a coding assistant.")
        )
        await store.append_message(
            parent.session_id, Message(role="user", content="Help me with auth.")
        )

        # Simulate 50 turns of work (using string content to avoid serialization issues)
        for i in range(50):
            await store.append_message(
                parent.session_id,
                Message(role="assistant", content=f"Reading file {i}..."),
            )
            await store.append_message(
                parent.session_id,
                Message(role="tool", content="x" * 1000, tool_call_id=f"t{i}"),
            )

        parent_loaded, _ = await store.get(parent.session_id)
        assert parent_loaded is not None
        original_msg_count = len(parent_loaded.messages)
        assert original_msg_count == 102  # 2 + 50*2

        # 2. Self-compact: keep system+first user, compact middle 80%, keep recent
        child = await slice_session(
            session=parent_loaded,
            spec="0:2, compact:2:80%, 80%:",
            endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4-5-20250929"),
            session_store=store,
        )

        # 3. Verify child session
        child_loaded, _ = await store.get(child.session_id)
        assert child_loaded is not None

        # Should have same message count (compact doesn't remove)
        assert len(child_loaded.messages) == original_msg_count

        # But should be much smaller in bytes (tool results compacted)
        parent_size = sum(
            len(str(m.content)) for m in parent_loaded.messages
        )
        child_size = sum(
            len(str(m.content)) for m in child_loaded.messages
        )
        # Compact only shrinks tool results, so expect ~50% reduction
        # (tool results are ~50% of content in our test)
        assert child_size < parent_size * 0.6, (
            f"Expected >40% size reduction, got {child_size}/{parent_size} = {child_size/parent_size:.0%}"
        )

        # 4. Verify lineage is preserved
        assert child_loaded.session_id != parent.session_id
        # Parent unchanged
        parent_check, _ = await store.get(parent.session_id)
        assert len(parent_check.messages) == original_msg_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
