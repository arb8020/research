from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rollouts.eval.configs import AgentRunSpec
from rollouts.eval.launch import (
    _interactive_claude_launch_kwargs,
    _interactive_codex_launch_kwargs,
)


def _config_module_with_external_args(**external_agent_args: object) -> SimpleNamespace:
    return SimpleNamespace(
        run_spec=AgentRunSpec(
            prepare_messages=lambda sample: [sample],
            external_agent_args=dict(external_agent_args),
        )
    )


def test_interactive_claude_launch_kwargs_merge_external_agent_args() -> None:
    config_module = _config_module_with_external_args(
        model="wrong-model",
        system_prompt="custom system",
        allowed_tools=["bash", "edit"],
    )

    kwargs = _interactive_claude_launch_kwargs(
        config_module,
        cwd=Path("/tmp/workdir"),
        model="sonnet",
    )

    assert kwargs["cwd"] == Path("/tmp/workdir")
    assert kwargs["model"] == "sonnet"
    assert kwargs["system_prompt"] == "custom system"
    assert kwargs["allowed_tools"] == ["bash", "edit"]


def test_interactive_codex_launch_kwargs_default_workspace_write() -> None:
    config_module = _config_module_with_external_args(ask_for_approval="never")

    kwargs = _interactive_codex_launch_kwargs(
        config_module,
        cwd=Path("/tmp/workdir"),
        model="gpt-5.1-codex-mini",
        default_sandbox="workspace-write",
    )

    assert kwargs["cwd"] == Path("/tmp/workdir")
    assert kwargs["model"] == "gpt-5.1-codex-mini"
    assert kwargs["sandbox"] == "workspace-write"
    assert kwargs["ask_for_approval"] == "never"


def test_interactive_codex_launch_kwargs_preserve_external_override() -> None:
    config_module = _config_module_with_external_args(
        model="wrong-model",
        sandbox="danger-full-access",
        ask_for_approval="on-failure",
    )

    kwargs = _interactive_codex_launch_kwargs(
        config_module,
        cwd=Path("/tmp/workdir"),
        model="gpt-5.1-codex-mini",
        default_sandbox="workspace-write",
    )

    assert kwargs["cwd"] == Path("/tmp/workdir")
    assert kwargs["model"] == "gpt-5.1-codex-mini"
    assert kwargs["sandbox"] == "danger-full-access"
    assert kwargs["ask_for_approval"] == "on-failure"


def test_interactive_claude_launch_kwargs_preserve_allowed_tools() -> None:
    config_module = _config_module_with_external_args(
        model="wrong-model",
        allowed_tools=["bash"],
        dangerously_skip_permissions=False,
    )

    kwargs = _interactive_claude_launch_kwargs(
        config_module,
        cwd=Path("/tmp/workdir"),
        model="sonnet",
    )

    assert kwargs["cwd"] == Path("/tmp/workdir")
    assert kwargs["model"] == "sonnet"
    assert kwargs["allowed_tools"] == ["bash"]
    assert kwargs["dangerously_skip_permissions"] is False
