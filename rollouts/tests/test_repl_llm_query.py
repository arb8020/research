"""Test that llm_query works inside REPL exec'd code."""

from typing import Any

import pytest

from rollouts.dtypes import (
    Actor,
    AgentState,
    Endpoint,
    Message,
    RunConfig,
    ToolCall,
    Trajectory,
)
from rollouts.environments.repl import MessageParsingREPLEnvironment, REPLEnvironment

# Mock endpoint for testing
MOCK_ENDPOINT = Endpoint(
    provider="anthropic",
    model="claude-3-5-haiku-20241022",
    api_base="https://api.anthropic.com",
    api_key="test-key",
)


async def _noop_on_chunk(event: Any) -> None:
    """No-op chunk handler for testing."""
    pass


def _make_run_config() -> RunConfig:
    """Create a RunConfig for testing."""
    return RunConfig(on_chunk=_noop_on_chunk)


class MockREPLEnvironment(REPLEnvironment):
    """REPL environment with mocked llm_query for testing."""

    async def _async_llm_query(self, prompt: str) -> str:
        """Mock LLM query that echoes the prompt."""
        return f"MOCK_RESPONSE: {prompt[:50]}"


class MockMessageParsingREPLEnvironment(MessageParsingREPLEnvironment):
    """Message parsing REPL with mocked llm_query for testing."""

    async def _async_llm_query(self, prompt: str) -> str:
        """Mock LLM query that echoes the prompt."""
        return f"MOCK_RESPONSE: {prompt[:50]}"


@pytest.mark.trio
async def test_llm_query_inside_repl_tool() -> None:
    """Test that llm_query can be called from inside exec'd code via tool."""
    env = MockREPLEnvironment(
        context="The magic number is 42.",
        sub_endpoint=MOCK_ENDPOINT,
    )

    # Create a tool call that uses llm_query inside the code
    tool_call = ToolCall(
        id="test-1",
        name="repl",
        args={
            "code": """
result = llm_query("What is the magic number in: " + context)
print(result)
"""
        },
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    assert not result.is_error, f"Tool execution failed: {result.error}"
    assert "MOCK_RESPONSE" in result.content
    assert "magic number" in result.content.lower()


@pytest.mark.trio
async def test_llm_query_loop_inside_repl() -> None:
    """Test that llm_query works in a loop (like the RLM chunking pattern)."""
    env = MockREPLEnvironment(
        context="chunk1\n---\nchunk2\n---\nchunk3",
        sub_endpoint=MOCK_ENDPOINT,
    )

    tool_call = ToolCall(
        id="test-2",
        name="repl",
        args={
            "code": """
chunks = context.split('---')
results = []
for i, chunk in enumerate(chunks):
    answer = llm_query(f"Process chunk {i}: {chunk.strip()}")
    results.append(answer)
print(f"Processed {len(results)} chunks")
for r in results:
    print(r)
"""
        },
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    assert not result.is_error, f"Tool execution failed: {result.error}"
    assert "Processed 3 chunks" in result.content
    assert result.content.count("MOCK_RESPONSE") == 3


@pytest.mark.trio
async def test_llm_query_in_message_parsing_env() -> None:
    """Test that llm_query works in MessageParsingREPLEnvironment."""
    env = MockMessageParsingREPLEnvironment(
        context="The secret is 'hello world'.",
        sub_endpoint=MOCK_ENDPOINT,
    )

    # Create a mock agent state
    actor = Actor(
        trajectory=Trajectory(messages=[]),
        endpoint=MOCK_ENDPOINT,
        tools=[],
    )
    state = AgentState(actor=actor, environment=env)

    # Create assistant message with code block that uses llm_query
    message = Message(
        role="assistant",
        content="""Let me analyze the context:

```repl
answer = llm_query("What is the secret in: " + context)
print(answer)
```
""",
    )

    new_state = await env.on_assistant_message(message, state)

    # Check that output was injected
    assert len(new_state.actor.trajectory.messages) == 1
    output_msg = new_state.actor.trajectory.messages[0]
    assert output_msg.role == "user"
    assert "MOCK_RESPONSE" in output_msg.content
    assert "secret" in output_msg.content.lower()


@pytest.mark.trio
async def test_llm_query_error_handling() -> None:
    """Test that errors in llm_query are handled gracefully."""

    class ErrorREPLEnvironment(REPLEnvironment):
        async def _async_llm_query(self, prompt: str) -> str:
            raise ValueError("Simulated API error")

    env = ErrorREPLEnvironment(
        context="test",
        sub_endpoint=MOCK_ENDPOINT,
    )

    tool_call = ToolCall(
        id="test-3",
        name="repl",
        args={"code": "result = llm_query('test')\nprint(result)"},
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    # Should not crash, should return error message in output
    assert "llm_query error" in result.content or "error" in result.content.lower()


@pytest.mark.trio
async def test_agent_in_namespace() -> None:
    """Test that agent() function is available in REPL namespace."""
    env = MockREPLEnvironment(
        context="Parent context with data.",
        sub_endpoint=MOCK_ENDPOINT,
        max_depth=2,
    )

    # Test that agent is available and respects max_depth
    tool_call = ToolCall(
        id="test-4",
        name="repl",
        args={
            "code": """
# Check agent is available
print(f"agent available: {callable(agent)}")

# Check context is available
print(f"context length: {len(context)}")
"""
        },
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    assert not result.is_error, f"Tool execution failed: {result.error}"
    assert "agent available: True" in result.content
    assert "context length:" in result.content


@pytest.mark.trio
async def test_agent_max_depth_reached() -> None:
    """Test that agent() returns error when max depth reached."""
    env = MockREPLEnvironment(
        context="test",
        sub_endpoint=MOCK_ENDPOINT,
        max_depth=1,
        _current_depth=1,  # Already at max
    )

    tool_call = ToolCall(
        id="test-5",
        name="repl",
        args={"code": "result = agent('test task', 'test context')\nprint(result)"},
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    assert not result.is_error
    assert "max depth" in result.content.lower()


@pytest.mark.trio
async def test_agent_tool_direct_call() -> None:
    """Test the agent tool can be called directly (not via repl)."""
    env = MockREPLEnvironment(
        context="Parent context.",
        sub_endpoint=MOCK_ENDPOINT,
        max_depth=2,
        _current_depth=2,  # At max depth
    )

    tool_call = ToolCall(
        id="test-6",
        name="agent",
        args={"task": "Analyze the context", "context": "Some data to analyze"},
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    # Should fail due to max depth
    assert result.is_error
    assert "max" in result.error.lower() or "depth" in result.error.lower()


@pytest.mark.trio
async def test_agent_tool_inherits_parent_context() -> None:
    """Test that agent tool uses parent context if not specified."""
    env = MockREPLEnvironment(
        context="This is the parent context with important data.",
        sub_endpoint=MOCK_ENDPOINT,
        max_depth=2,
    )

    # Check that agent defaults to parent context
    tool_call = ToolCall(
        id="test-7",
        name="repl",
        args={
            "code": """
# agent() should use self.context if context not provided
# We can't fully test spawning without mocking run_agent,
# but we can verify the function signature works
import inspect
sig = str(inspect.signature(agent))
print(f"agent signature: {sig}")
"""
        },
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    assert not result.is_error, f"Tool execution failed: {result.error}"
    assert "context" in result.content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
