"""Test that large bash outputs are saved to files instead of truncated."""

from pathlib import Path

import trio

from rollouts.dtypes import Actor, AgentState, Endpoint, RunConfig, ToolCall
from rollouts.environments.coding import MAX_OUTPUT_SIZE, LocalFilesystemEnvironment


def test_large_bash_output_saved_to_file() -> None:
    """Large outputs (>30KB) should be saved to a file with full content preserved."""

    async def run() -> None:
        env = LocalFilesystemEnvironment()

        async def noop(_: object) -> None:
            pass

        run_config = RunConfig(on_chunk=noop)
        state = AgentState(
            actor=Actor(endpoint=Endpoint(provider="test", model="test"), trajectory=[], tools=[]),
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


if __name__ == "__main__":
    test_large_bash_output_saved_to_file()
    print("✓ test_large_bash_output_saved_to_file")
