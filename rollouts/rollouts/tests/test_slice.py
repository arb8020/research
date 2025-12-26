"""Tests for session slicing."""

import pytest
from rollouts.dtypes import AgentSession, Endpoint, EnvironmentConfig, Message, SessionStatus
from rollouts.slice import (
    SliceSegment,
    apply_slice,
    compact_messages,
    parse_slice_spec,
)


def make_test_session(n_messages: int = 20) -> AgentSession:
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
    """Create a test session with tool calls and results."""
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
            content="def authenticate(user, password):\n    # Check credentials\n    return True\n" * 50,
            tool_call_id="tool_1",
        ),
        Message(role="assistant", content="I see the auth file has a simple authenticate function."),
        Message(role="user", content="Now run the tests"),
        Message(
            role="assistant",
            content=[
                {"type": "text", "text": "Running tests."},
                {"type": "tool_use", "id": "tool_2", "name": "bash", "input": {"command": "pytest"}},
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


class TestParseSliceSpec:
    """Tests for parse_slice_spec."""

    def test_simple_range(self) -> None:
        segments = parse_slice_spec("0:4")
        assert len(segments) == 1
        assert segments[0].type == "range"
        assert segments[0].start == 0
        assert segments[0].end == 4

    def test_range_open_end(self) -> None:
        segments = parse_slice_spec("10:")
        assert len(segments) == 1
        assert segments[0].type == "range"
        assert segments[0].start == 10
        assert segments[0].end is None

    def test_range_open_start(self) -> None:
        segments = parse_slice_spec(":5")
        assert len(segments) == 1
        assert segments[0].type == "range"
        assert segments[0].start is None
        assert segments[0].end == 5

    def test_multiple_ranges(self) -> None:
        segments = parse_slice_spec("0:4, 10:15")
        assert len(segments) == 2
        assert segments[0].type == "range"
        assert segments[0].start == 0
        assert segments[0].end == 4
        assert segments[1].type == "range"
        assert segments[1].start == 10
        assert segments[1].end == 15

    def test_summarize(self) -> None:
        segments = parse_slice_spec("summarize:4:18")
        assert len(segments) == 1
        assert segments[0].type == "summarize"
        assert segments[0].start == 4
        assert segments[0].end == 18
        assert segments[0].goal is None

    def test_summarize_with_goal(self) -> None:
        segments = parse_slice_spec("summarize:4:18:'security review'")
        assert len(segments) == 1
        assert segments[0].type == "summarize"
        assert segments[0].start == 4
        assert segments[0].end == 18
        assert segments[0].goal == "security review"

    def test_summarize_with_double_quotes(self) -> None:
        segments = parse_slice_spec('summarize:4:18:"fix the bug"')
        assert len(segments) == 1
        assert segments[0].type == "summarize"
        assert segments[0].goal == "fix the bug"

    def test_compact(self) -> None:
        segments = parse_slice_spec("compact:5:15")
        assert len(segments) == 1
        assert segments[0].type == "compact"
        assert segments[0].start == 5
        assert segments[0].end == 15

    def test_inject_single_quotes(self) -> None:
        segments = parse_slice_spec("inject:'focus on tests'")
        assert len(segments) == 1
        assert segments[0].type == "inject"
        assert segments[0].content == "focus on tests"

    def test_inject_double_quotes(self) -> None:
        segments = parse_slice_spec('inject:"focus on tests"')
        assert len(segments) == 1
        assert segments[0].type == "inject"
        assert segments[0].content == "focus on tests"

    def test_full_example(self) -> None:
        spec = "0:4, summarize:4:18:'review', 18:20, inject:'focus on tests'"
        segments = parse_slice_spec(spec)
        
        assert len(segments) == 4
        
        assert segments[0].type == "range"
        assert segments[0].start == 0
        assert segments[0].end == 4
        
        assert segments[1].type == "summarize"
        assert segments[1].start == 4
        assert segments[1].end == 18
        assert segments[1].goal == "review"
        
        assert segments[2].type == "range"
        assert segments[2].start == 18
        assert segments[2].end == 20
        
        assert segments[3].type == "inject"
        assert segments[3].content == "focus on tests"

    def test_compact_in_spec(self) -> None:
        spec = "0:4, compact:4:10, 10:"
        segments = parse_slice_spec(spec)
        
        assert len(segments) == 3
        assert segments[0].type == "range"
        assert segments[1].type == "compact"
        assert segments[1].start == 4
        assert segments[1].end == 10
        assert segments[2].type == "range"

    def test_whitespace_handling(self) -> None:
        segments = parse_slice_spec("  0:4  ,  10:  ")
        assert len(segments) == 2
        assert segments[0].start == 0
        assert segments[0].end == 4
        assert segments[1].start == 10
        assert segments[1].end is None


class TestCompactMessages:
    """Tests for compact_messages."""

    def test_compact_preserves_user_messages(self) -> None:
        messages = [
            Message(role="user", content="Hello, this is a long message " * 100),
            Message(role="assistant", content="Response " * 100),
        ]
        
        result = compact_messages(messages)
        
        assert len(result) == 2
        assert result[0].content == messages[0].content  # Unchanged
        assert result[1].content == messages[1].content  # Unchanged

    def test_compact_shrinks_tool_results(self) -> None:
        messages = [
            Message(
                role="assistant",
                content=[{"type": "tool_use", "id": "t1", "name": "read", "input": {}}],
            ),
            Message(
                role="tool",
                content="line 1\nline 2\nline 3\n" * 100,  # Long content
                tool_call_id="t1",
            ),
        ]
        
        result = compact_messages(messages)
        
        assert len(result) == 2
        assert result[0].content == messages[0].content  # Assistant unchanged
        assert len(result[1].content) < len(messages[1].content)  # Tool compacted
        assert "📄" in result[1].content or "read" in result[1].content


class TestApplySlice:
    """Tests for apply_slice (without summarize)."""

    @pytest.mark.trio
    async def test_simple_range(self) -> None:
        session = make_test_session(20)
        segments = [SliceSegment(type="range", start=0, end=4)]
        
        result = await apply_slice(session, segments)
        
        assert len(result) == 4
        assert result[0].content == "Message 0"
        assert result[3].content == "Message 3"

    @pytest.mark.trio
    async def test_multiple_ranges(self) -> None:
        session = make_test_session(20)
        segments = [
            SliceSegment(type="range", start=0, end=2),
            SliceSegment(type="range", start=18, end=20),
        ]
        
        result = await apply_slice(session, segments)
        
        assert len(result) == 4
        assert result[0].content == "Message 0"
        assert result[1].content == "Message 1"
        assert result[2].content == "Message 18"
        assert result[3].content == "Message 19"

    @pytest.mark.trio
    async def test_inject(self) -> None:
        session = make_test_session(20)
        segments = [
            SliceSegment(type="range", start=0, end=2),
            SliceSegment(type="inject", content="focus on tests"),
        ]
        
        result = await apply_slice(session, segments)
        
        assert len(result) == 3
        assert result[0].content == "Message 0"
        assert result[1].content == "Message 1"
        assert result[2].role == "user"
        assert result[2].content == "focus on tests"

    @pytest.mark.trio
    async def test_open_range(self) -> None:
        session = make_test_session(10)
        segments = [SliceSegment(type="range", start=8, end=None)]
        
        result = await apply_slice(session, segments)
        
        assert len(result) == 2
        assert result[0].content == "Message 8"
        assert result[1].content == "Message 9"

    @pytest.mark.trio
    async def test_compact_segment(self) -> None:
        session = make_session_with_tools()
        # Compact messages 2-4 (the read tool call and result)
        segments = [
            SliceSegment(type="range", start=0, end=2),
            SliceSegment(type="compact", start=2, end=5),
            SliceSegment(type="range", start=5, end=None),
        ]
        
        result = await apply_slice(session, segments)
        
        # Should have all messages, but tool result at index 3 should be compacted
        assert len(result) == len(session.messages)
        
        # Find the tool result message and check it's shorter
        tool_result_original = session.messages[3]
        tool_result_compacted = result[3]
        assert len(tool_result_compacted.content) < len(tool_result_original.content)


class TestSliceSegmentRepr:
    """Tests for SliceSegment repr."""

    def test_range_repr(self) -> None:
        seg = SliceSegment(type="range", start=0, end=4)
        assert repr(seg) == "range(0:4)"

    def test_range_open_end_repr(self) -> None:
        seg = SliceSegment(type="range", start=10, end=None)
        assert repr(seg) == "range(10:)"

    def test_summarize_repr(self) -> None:
        seg = SliceSegment(type="summarize", start=4, end=18)
        assert repr(seg) == "summarize(4:18)"

    def test_summarize_with_goal_repr(self) -> None:
        seg = SliceSegment(type="summarize", start=4, end=18, goal="security")
        assert repr(seg) == "summarize(4:18:'security')"

    def test_compact_repr(self) -> None:
        seg = SliceSegment(type="compact", start=5, end=15)
        assert repr(seg) == "compact(5:15)"

    def test_inject_repr(self) -> None:
        seg = SliceSegment(type="inject", content="test msg")
        assert repr(seg) == "inject('test msg')"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestOpenEndedRanges:
    """Tests for open-ended compact and summarize."""

    def test_compact_open_end(self) -> None:
        segments = parse_slice_spec("compact:0:")
        assert len(segments) == 1
        assert segments[0].type == "compact"
        assert segments[0].start == 0
        assert segments[0].end is None

    def test_summarize_open_end(self) -> None:
        segments = parse_slice_spec("summarize:5:")
        assert len(segments) == 1
        assert segments[0].type == "summarize"
        assert segments[0].start == 5
        assert segments[0].end is None

    def test_summarize_open_end_with_goal(self) -> None:
        segments = parse_slice_spec("summarize:5::'my goal'")
        assert len(segments) == 1
        assert segments[0].type == "summarize"
        assert segments[0].start == 5
        assert segments[0].end is None
        assert segments[0].goal == "my goal"

    @pytest.mark.trio
    async def test_compact_open_end_applies_to_all(self) -> None:
        session = make_session_with_tools()
        segments = [SliceSegment(type="compact", start=0, end=None)]
        
        result = await apply_slice(session, segments)
        
        # Should have same number of messages
        assert len(result) == len(session.messages)
