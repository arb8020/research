"""Tests for session slicing."""

import pytest
from rollouts.dtypes import AgentSession, Endpoint, EnvironmentConfig, Message, SessionStatus
from rollouts.slice import (
    SliceSegment,
    apply_slice,
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
        spec = "0:4, summarize:4:18, 18:20, inject:'focus on tests'"
        segments = parse_slice_spec(spec)
        
        assert len(segments) == 4
        
        assert segments[0].type == "range"
        assert segments[0].start == 0
        assert segments[0].end == 4
        
        assert segments[1].type == "summarize"
        assert segments[1].start == 4
        assert segments[1].end == 18
        
        assert segments[2].type == "range"
        assert segments[2].start == 18
        assert segments[2].end == 20
        
        assert segments[3].type == "inject"
        assert segments[3].content == "focus on tests"

    def test_whitespace_handling(self) -> None:
        segments = parse_slice_spec("  0:4  ,  10:  ")
        assert len(segments) == 2
        assert segments[0].start == 0
        assert segments[0].end == 4
        assert segments[1].start == 10
        assert segments[1].end is None


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
