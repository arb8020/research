"""Test that agent() works inside REPL exec'd code for recursive exploration."""

from typing import Any

import pytest

from rollouts.dtypes import (
    Endpoint,
    RunConfig,
    ToolCall,
)
from rollouts.environments.repl import REPLEnvironment

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
    """REPL environment with mocked llm_query and agent for testing."""

    async def _async_llm_query(self, prompt: str) -> str:
        """Mock LLM query that echoes the prompt."""
        return f"MOCK_LLM: {prompt[:50]}"

    async def _async_agent(self, task: str, context: str, run_config: RunConfig) -> str:
        """Mock agent that echoes task and context length."""
        return f"MOCK_AGENT: task='{task[:30]}', context_len={len(context)}"


@pytest.mark.trio
async def test_agent_inside_repl_tool() -> None:
    """Test that agent() can be called from inside exec'd code via tool."""
    env = MockREPLEnvironment(
        context="This is the main context with lots of information.",
        sub_endpoint=MOCK_ENDPOINT,
        max_depth=3,
    )

    # Create a tool call that uses agent inside the code
    tool_call = ToolCall(
        id="test-1",
        name="repl",
        args={
            "code": """
subset = context[:20]
result = agent("Find the key info", subset)
print(result)
"""
        },
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    assert not result.is_error, f"Tool execution failed: {result.error}"
    assert "MOCK_AGENT" in result.content
    assert "context_len=20" in result.content


@pytest.mark.trio
async def test_agent_tool_direct() -> None:
    """Test calling agent as a direct tool (not through repl)."""
    env = MockREPLEnvironment(
        context="Main context",
        sub_endpoint=MOCK_ENDPOINT,
        max_depth=3,
    )

    tool_call = ToolCall(
        id="test-2",
        name="agent",
        args={
            "task": "Analyze this code for bugs",
            "context": "def foo(): return bar",
        },
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    assert not result.is_error, f"Tool execution failed: {result.error}"
    assert "MOCK_AGENT" in result.content


@pytest.mark.trio
async def test_agent_max_depth() -> None:
    """Test that agent respects max_depth limit."""
    env = MockREPLEnvironment(
        context="test",
        sub_endpoint=MOCK_ENDPOINT,
        max_depth=2,
        _current_depth=2,  # Already at max depth
    )

    tool_call = ToolCall(
        id="test-3",
        name="agent",
        args={
            "task": "Do something",
            "context": "some context",
        },
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    assert result.is_error
    assert "max depth" in result.error.lower()


@pytest.mark.trio
async def test_agent_inside_repl_max_depth() -> None:
    """Test that agent() in repl respects max_depth."""
    env = MockREPLEnvironment(
        context="test",
        sub_endpoint=MOCK_ENDPOINT,
        max_depth=1,
        _current_depth=1,  # Already at max depth
    )

    tool_call = ToolCall(
        id="test-4",
        name="repl",
        args={
            "code": """
result = agent("task", "context")
print(result)
"""
        },
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    assert not result.is_error
    assert "max depth" in result.content.lower()


@pytest.mark.trio
async def test_agent_requires_context() -> None:
    """Test that agent tool requires context parameter."""
    env = MockREPLEnvironment(
        context="main",
        sub_endpoint=MOCK_ENDPOINT,
    )

    tool_call = ToolCall(
        id="test-5",
        name="agent",
        args={
            "task": "Do something",
            # No context provided
        },
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    assert result.is_error
    assert "context" in result.error.lower()


@pytest.mark.trio
async def test_both_llm_query_and_agent() -> None:
    """Test using both llm_query and agent in the same repl session."""
    env = MockREPLEnvironment(
        context="chunk1---chunk2---chunk3",
        sub_endpoint=MOCK_ENDPOINT,
        max_depth=3,
    )

    tool_call = ToolCall(
        id="test-6",
        name="repl",
        args={
            "code": """
# Use llm_query for quick classification
label = llm_query("Is this code or text: " + context[:10])
print(f"Label: {label}")

# Use agent for deeper exploration
chunks = context.split('---')
for i, chunk in enumerate(chunks):
    result = agent(f"Analyze chunk {i}", chunk)
    print(f"Chunk {i}: {result}")
"""
        },
    )

    run_config = _make_run_config()
    result = await env.exec_tool(tool_call, None, run_config)

    assert not result.is_error, f"Tool execution failed: {result.error}"
    assert "MOCK_LLM" in result.content
    assert result.content.count("MOCK_AGENT") == 3


@pytest.mark.trio
async def test_tools_list_includes_agent() -> None:
    """Test that get_tools() returns the agent tool."""
    env = REPLEnvironment(
        context="test",
        sub_endpoint=MOCK_ENDPOINT,
    )

    tools = env.get_tools()
    tool_names = [t.function.name for t in tools]

    assert "repl" in tool_names
    assert "llm_query" in tool_names
    assert "agent" in tool_names
    assert "final_answer" in tool_names


@pytest.mark.trio
async def test_system_prompt_documents_agent() -> None:
    """Test that system prompt explains agent() usage."""
    env = REPLEnvironment(
        context="test",
        sub_endpoint=MOCK_ENDPOINT,
    )

    prompt = env.get_system_prompt()

    assert "agent(" in prompt
    assert "sub-agent" in prompt.lower()
    assert "llm_query" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
