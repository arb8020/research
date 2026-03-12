"""Test that large tool outputs are saved to files instead of truncated."""

from pathlib import Path

import pytest
import trio

from rollouts.agents import Actor, AgentState, RunConfig
from rollouts.core import Endpoint, ToolCall
from rollouts.environments.coding import (
    MAX_OUTPUT_SIZE,
    WEB_FETCH_MAX_CONTENT,
    LocalFilesystemEnvironment,
)

pytestmark = pytest.mark.live


def test_large_bash_output_saved_to_file() -> None:
    """Large outputs (>30KB) should be saved to a file with full content preserved."""

    async def run() -> None:
        env = LocalFilesystemEnvironment()

        async def noop(_: object) -> None:
            pass

        run_config = RunConfig(on_chunk=noop)
        state = AgentState(
            actor=Actor(
                endpoint=Endpoint(
                    model="test/test", base_url="http://test", api_format="openai-completions"
                ),
                trajectory=[],
                tools=[],
            ),
            environment=env,
            session_id="test_large_output",
        )

        # Generate ~50KB output
        result = await env.exec_tool(
            ToolCall(id="t1", name="bash", args={"command": "seq 1 10000"}),
            state,
            run_config,
        )

        # File should be created with full content
        assert result.details and "output_file" in result.details
        output_file = Path(result.details["output_file"])
        assert output_file.exists()
        assert len(output_file.read_text()) > MAX_OUTPUT_SIZE

        # Result should contain instructions for reading the file
        assert "read path=" in result.content

        # Cleanup
        output_file.unlink()
        output_file.parent.rmdir()

    trio.run(run)


def test_large_web_fetch_saved_to_file() -> None:
    """Large web content (>100KB) should be saved to a file with full content preserved."""

    async def run() -> None:
        env = LocalFilesystemEnvironment(summarize_web_fetch=False)

        async def noop(_: object) -> None:
            pass

        run_config = RunConfig(on_chunk=noop)
        state = AgentState(
            actor=Actor(
                endpoint=Endpoint(
                    model="test/test", base_url="http://test", api_format="openai-completions"
                ),
                trajectory=[],
                tools=[],
            ),
            environment=env,
            session_id="test_large_web_fetch",
        )

        # Fetch a large page (Python docs)
        result = await env.exec_tool(
            ToolCall(
                id="wf1",
                name="web_fetch",
                args={
                    "url": "https://docs.python.org/3/library/functions.html",
                    "prompt": "list functions",
                },
            ),
            state,
            run_config,
        )

        # File should be created with full content
        assert result.details and "output_file" in result.details
        output_file = Path(result.details["output_file"])
        assert output_file.exists()
        assert len(output_file.read_text()) > WEB_FETCH_MAX_CONTENT

        # Result should contain instructions for reading the file
        assert "read path=" in result.content

        # Cleanup
        output_file.unlink()
        output_file.parent.rmdir()

    trio.run(run)
