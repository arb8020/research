from __future__ import annotations

from pathlib import Path

import pytest

from rollouts.core import ToolCall
from rollouts.environments.coding import LocalFilesystemEnvironment


@pytest.mark.trio
async def test_local_filesystem_environment_uses_injected_paths(tmp_path: Path) -> None:
    resolved_file = tmp_path / "resolved.txt"
    output_dir = tmp_path / "tool-outputs"
    env = LocalFilesystemEnvironment(
        path_resolver=lambda _: resolved_file,
        output_dir=output_dir,
    )

    write_result = await env.exec_tool(
        ToolCall(id="tc-1", name="write", args={"path": "ignored.txt", "content": "hello"}),
        current_state=None,
        run_config=None,
    )
    output_path = Path(env._write_large_output("big output", "tc-2", "session-1"))

    assert write_result.is_error is False
    assert resolved_file.read_text() == "hello"
    assert output_path.exists()
    assert output_path.is_relative_to(output_dir)
