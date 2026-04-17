"""Construct ClaudeCodeEnvironment / CodexEnvironment and print their launch
flags + a sample translated event. Does not launch the real binaries.

Run: uv run --no-sync -q python dev/session_refactor_tests/external_env_sanity.py
"""

from __future__ import annotations

from pathlib import Path

from rollouts.environments.external_agent_environments import (
    ClaudeCodeEnvironment,
    CodexEnvironment,
)
from rollouts.environments.local_workspace_resource import LocalWorkspaceResource


def main() -> None:
    workspace = LocalWorkspaceResource.from_existing(Path("/tmp"))

    claude_env = ClaudeCodeEnvironment(
        workspace=workspace,
        allowed_builtin_tools=["bash", "read", "write", "edit"],
        mcp_tools=[
            {"name": "calc", "spec": {"command": "python", "args": ["-m", "calc_mcp"]}},
        ],
        mcp_config_path=Path("/tmp/rollouts-mcp-config.json"),
    )
    print("── ClaudeCodeEnvironment ──")
    print("launch flags:", claude_env.get_launch_flags())
    print("mcp config json:\n" + claude_env.mcp_config_json())

    claude_sample = {
        "type": "assistant",
        "message": {
            "id": "msg_abc",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello world"}],
        },
    }
    print("translated claude event:", claude_env.translate_harness_event(claude_sample))

    codex_env = CodexEnvironment(
        workspace=workspace,
        sandbox_mode="workspace-write",
        mcp_tools=[
            {"name": "calc", "spec": {"command": "python", "args": ["-m", "calc_mcp"]}},
        ],
        codex_config_path=Path("/tmp/rollouts-codex-home/config.toml"),
    )
    print()
    print("── CodexEnvironment ──")
    print("launch flags:", codex_env.get_launch_flags())
    print("codex config.toml:\n" + codex_env.codex_mcp_config_toml())

    codex_sample = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi from codex"}],
        },
    }
    print("translated codex event:", codex_env.translate_harness_event(codex_sample))


if __name__ == "__main__":
    main()
