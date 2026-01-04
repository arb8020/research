"""Tests for session slicing.

Grugbrain philosophy:
- Integration tests are the sweet spot
- Minimal unit tests (just enough to get started, don't get attached)
- One curated end-to-end test for the main use case
- No mocking
"""

from pathlib import Path

import pytest

from ..dtypes import AgentSession, Endpoint, EnvironmentConfig, Message, SessionStatus
from ..slice import apply_slice, parse_slice_spec, slice_session
from ..store import FileSessionStore

# ── Fixtures ────────────────────────────────────────────────────────────────


def make_session(n_messages: int = 20) -> AgentSession:
    """Create a test session with n messages."""
    messages = []
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(Message(role=role, content=f"Message {i}"))

    return AgentSession(
        session_id="test-session",
        endpoint=Endpoint(provider="test", model="test"),
        environment=EnvironmentConfig(type="none", config={}),
        messages=messages,
        status=SessionStatus.PENDING,
        tags={},
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
    )


def make_session_with_tools() -> AgentSession:
    """Create a session that looks like a real coding session with tool calls."""
    messages = [
        Message(role="system", content="You are a coding assistant."),
        Message(role="user", content="Read the auth file"),
        Message(
            role="assistant",
            content=[
                {"type": "text", "text": "I'll read that file."},
                {"type": "tool_use", "id": "tool_1", "name": "read", "input": {"path": "auth.py"}},
            ],
        ),
        Message(
            role="tool",
            content="def authenticate(user, password):\n    # Check credentials\n    return True\n"
            * 50,
            tool_call_id="tool_1",
        ),
        Message(
            role="assistant", content="I see the auth file has a simple authenticate function."
        ),
        Message(role="user", content="Now run the tests"),
        Message(
            role="assistant",
            content=[
                {"type": "text", "text": "Running tests."},
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "bash",
                    "input": {"command": "pytest"},
                },
            ],
        ),
        Message(
            role="tool",
            content="PASSED test_auth.py::test_login\nPASSED test_auth.py::test_logout\n" * 20,
            tool_call_id="tool_2",
        ),
        Message(role="assistant", content="All tests pass!"),
    ]

    return AgentSession(
        session_id="test-session-tools",
        endpoint=Endpoint(provider="test", model="test"),
        environment=EnvironmentConfig(type="none", config={}),
        messages=messages,
        status=SessionStatus.PENDING,
        tags={},
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
    )


# ── Integration Tests (the sweet spot) ──────────────────────────────────────


class TestSliceIntegration:
    """Integration tests: parse spec → apply → verify result."""

    def test_parse_common_patterns(self) -> None:
        """All the patterns users will actually use should parse correctly."""
        cases = [
            "0:10",  # simple range
            "0.8:",  # decimal (last 20%)
            "0:0.5",  # first half
            "compact:0:",  # compact all
            "summarize:0:0.8:'goal'",  # summarize with goal
            "0:2, compact:2:0.8, 0.8:",  # mixed workflow
            "inject:'focus on tests'",  # inject
        ]
        for spec in cases:
            segments = parse_slice_spec(spec)
            assert len(segments) >= 1, f"Failed to parse: {spec}"

    def test_invalid_specs_raise(self) -> None:
        """Garbage input should raise ValueError."""
        for spec in ["garbage", "compact", "summarize:abc"]:
            with pytest.raises(ValueError):
                parse_slice_spec(spec)

    @pytest.mark.trio
    async def test_range_slicing(self) -> None:
        """Range slicing extracts correct messages."""
        session = make_session(20)

        # Parse and apply: first 4, then last 2
        segments = parse_slice_spec("0:4, 18:")
        result = await apply_slice(session, segments)

        assert len(result) == 6
        assert result[0].content == "Message 0"
        assert result[3].content == "Message 3"
        assert result[4].content == "Message 18"
        assert result[5].content == "Message 19"

    @pytest.mark.trio
    async def test_decimal_slicing(self) -> None:
        """Decimal fractions resolve to correct indices."""
        session = make_session(100)

        # 0.8: means start at 80% = index 80
        segments = parse_slice_spec("0.8:")
        result = await apply_slice(session, segments)

        assert len(result) == 20
        assert result[0].content == "Message 80"
        assert result[-1].content == "Message 99"

    @pytest.mark.trio
    async def test_compact_reduces_size(self) -> None:
        """Compact should significantly shrink tool results."""
        session = make_session_with_tools()

        original_size = sum(
            len(m.content) if isinstance(m.content, str) else len(str(m.content))
            for m in session.messages
        )

        segments = parse_slice_spec("compact:0:")
        result = await apply_slice(session, segments)

        compacted_size = sum(
            len(m.content) if isinstance(m.content, str) else len(str(m.content)) for m in result
        )

        # Should have same count but smaller size
        assert len(result) == len(session.messages)
        assert compacted_size < original_size * 0.5

    @pytest.mark.trio
    async def test_inject_adds_message(self) -> None:
        """Inject inserts a user message."""
        session = make_session(5)

        segments = parse_slice_spec("0:5, inject:'now focus on security'")
        result = await apply_slice(session, segments)

        assert len(result) == 6
        assert result[-1].role == "user"
        assert result[-1].content == "now focus on security"


# ── End-to-End Test (curated, must always pass) ─────────────────────────────


class TestEndToEnd:
    """The main use case: agent self-compacts when context fills up."""

    @pytest.mark.trio
    async def test_self_compaction_workflow(self, tmp_path: Path) -> None:
        """
        Critical path that must always work:
        1. Session exists with many messages
        2. Agent runs --slice to compact
        3. New child session is created with reduced size
        4. Parent unchanged
        """
        store = FileSessionStore(base_dir=tmp_path)

        # 1. Create a "full" session
        parent = await store.create(
            endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4-5-20250929"),
            environment=EnvironmentConfig(type="coding", config={}),
        )

        await store.append_message(
            parent.session_id, Message(role="system", content="You are a coding assistant.")
        )
        await store.append_message(
            parent.session_id, Message(role="user", content="Help me with auth.")
        )

        # Simulate 50 turns of work
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
        assert len(parent_loaded.messages) == 102  # 2 + 50*2

        # 2. Slice: keep system+first user, compact middle, keep recent 20%
        child = await slice_session(
            session=parent_loaded,
            spec="0:2, compact:2:0.8, 0.8:",
            endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4-5-20250929"),
            session_store=store,
        )

        # 3. Verify child
        child_loaded, _ = await store.get(child.session_id)
        assert child_loaded is not None
        assert len(child_loaded.messages) == 102  # same count (compact doesn't remove)

        parent_size = sum(len(str(m.content)) for m in parent_loaded.messages)
        child_size = sum(len(str(m.content)) for m in child_loaded.messages)
        assert child_size < parent_size * 0.6  # >40% reduction

        # 4. Parent unchanged
        parent_check, _ = await store.get(parent.session_id)
        assert len(parent_check.messages) == 102


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
