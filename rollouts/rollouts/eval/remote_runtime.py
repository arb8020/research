"""Remote sandbox execution for external agent CLIs.

Handles launching CLIs inside SandboxWorkspaceResource containers,
including bootstrap, polling, ACP preparation, and trajectory parsing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import trio

from ..core import Message, Trajectory
from ..drivers.claude import _ClaudeEventParser
from ..drivers.codex import _CodexEventParser
from ..drivers.runner import (
    _EventAccumulator,
    _FlushAssistantMessage,
    _make_raw_driver_line_handler,
)
from ..environments.local_workspace_resource import LocalWorkspaceResource
from ..environments.resources import ExecSpec, SandboxWorkspaceResource
from ..eval.types import ExternalAttemptArtifact

_event_logger = logging.getLogger("rollouts.eval.events")

REMOTE_AGENT_USER = "rollouts-agent"


@dataclass(frozen=True)
class RemoteRuntimePreparation:
    mode: Literal["preinstalled", "uv"] = "uv"
    uv_config_toml: str | None = None
    npmrc: str | None = "min-release-age=8\nignore-scripts=true\n"


async def _workspace_exec(
    workspace: SandboxWorkspaceResource,
    command: str,
    *,
    cwd: str,
    timeout: float,
) -> Any:
    return await workspace.exec(
        ExecSpec(
            command=command,
            cwd=cwd,
            timeout=timeout,
        )
    )


def _make_eval_on_event(
    sample_id: str,
    caller_on_event: Callable[[Any], Awaitable[None]] | None,
) -> Callable[[Any], Awaitable[None]]:
    """Emit coarse external-driver progress while forwarding real events."""
    from ..dtypes import LLMCallStart, ToolCallEnd, ToolCallStart

    turn: list[int] = [0]

    async def on_event(event: Any) -> None:
        if isinstance(event, LLMCallStart):
            _event_logger.info(
                "turn",
                extra={"sample_id": sample_id, "turn": turn[0], "status": "streaming..."},
            )
            turn[0] += 1
        elif isinstance(event, ToolCallStart):
            _event_logger.info(
                "turn",
                extra={"sample_id": sample_id, "turn": turn[0], "status": "calling tool..."},
            )
        elif isinstance(event, ToolCallEnd):
            _event_logger.info(
                "turn",
                extra={"sample_id": sample_id, "turn": turn[0], "status": "tool done"},
            )
        if caller_on_event is not None:
            await caller_on_event(event)

    return on_event


def _sample_id_slug(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", sample_id).strip("-") or "sample"


def _remote_runtime_state_dir(runtime: str, sample_id: str) -> str:
    return f"/tmp/rollouts-external-runtime/{runtime}/{_sample_id_slug(sample_id)}"


def _remote_runtime_cli(runtime: str) -> tuple[str, str]:
    if runtime == "claude_code":
        return "claude", "@anthropic-ai/claude-code"
    if runtime == "claude_acp":
        return "claude-agent-acp", "@agentclientprotocol/claude-agent-acp"
    if runtime == "codex":
        return "codex", "@openai/codex"
    if runtime == "codex_acp":
        return "codex-acp", "@zed-industries/codex-acp"
    raise ValueError(f"Unsupported remote runtime: {runtime}")


def _remote_runtime_env(runtime: str) -> dict[str, str]:
    if runtime == "claude_code":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for remote Claude Code runs")
        return {
            "ANTHROPIC_API_KEY": key,
            "CLAUDE_CODE_ENTRYPOINT": "rollouts-remote-eval",
        }
    if runtime == "claude_acp":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for remote Claude ACP runs")
        return {
            "ANTHROPIC_API_KEY": key,
        }
    if runtime == "codex":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for remote Codex runs")
        return {
            "CODEX_API_KEY": key,
            "OPENAI_API_KEY": key,
            "CODEX_ENTRYPOINT": "rollouts-remote-eval",
        }
    if runtime == "codex_acp":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for remote Codex ACP runs")
        return {
            "CODEX_API_KEY": key,
            "OPENAI_API_KEY": key,
        }
    raise ValueError(f"Unsupported remote runtime: {runtime}")


def _remote_acp_python(runtime: str) -> str:
    del runtime
    return "/opt/venvs/rollouts/bin/python"


def _remote_codex_acp_auth_payload(api_key: str) -> dict[str, Any]:
    return {
        "auth_mode": "apikey",
        "OPENAI_API_KEY": api_key,
        "tokens": None,
        "last_refresh": None,
    }


def _remote_acp_validate_command(runtime: str) -> str:
    cli_name, _npm_package = _remote_runtime_cli(runtime)
    python_bin = _remote_acp_python(runtime)
    return (
        "set -euo pipefail\n"
        'NODE_MAJOR="$(node -p \'process.versions.node.split(".")[0]\' 2>/dev/null || echo 0)"\n'
        'if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1 || [ "${NODE_MAJOR}" -lt 18 ]; then\n'
        '  echo "missing required Node/npm runtime (need node>=18 and npm)" >&2\n'
        "  exit 1\n"
        "fi\n"
        f"if ! command -v {cli_name} >/dev/null 2>&1; then\n"
        f'  echo "missing preinstalled ACP CLI: {cli_name}" >&2\n'
        "  exit 1\n"
        "fi\n"
        f'{python_bin} -c "import acp, trio_asyncio" >/dev/null 2>&1\n'
    )


def _emit_remote_runtime_stage(
    *,
    runtime: str,
    sample_id: str | None,
    stage: str,
    status: str,
) -> None:
    _event_logger.info(
        "remote_runtime_stage",
        extra={
            "runtime": runtime,
            "sample_id": sample_id,
            "stage": stage,
            "status": status,
        },
    )


def _remote_acp_ensure_node_command() -> str:
    return (
        "set -euo pipefail\n"
        'NODE_MAJOR="$(node -p \'process.versions.node.split(".")[0]\' 2>/dev/null || echo 0)"\n'
        'if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1 || [ "${NODE_MAJOR}" -lt 18 ]; then\n'
        "  apt-get update\n"
        "  DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates\n"
        "  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -\n"
        "  DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs\n"
        "fi\n"
    )


def _remote_acp_ensure_uv_command() -> str:
    return (
        "set -euo pipefail\n"
        "if ! command -v uv >/dev/null 2>&1; then\n"
        '  if [ -x "$HOME/.local/bin/uv" ]; then\n'
        '    export PATH="$HOME/.local/bin:$PATH"\n'
        "  else\n"
        "    if ! command -v curl >/dev/null 2>&1; then\n"
        "      apt-get update\n"
        "      DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates\n"
        "    fi\n"
        "    curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        '    export PATH="$HOME/.local/bin:$PATH"\n'
        "  fi\n"
        "fi\n"
        "command -v uv >/dev/null 2>&1\n"
    )


def _remote_acp_uv_prepare_command(
    runtime: str,
    *,
    uv_config_path: str | None = None,
    npmrc_path: str | None = None,
) -> str:
    python_bin = _remote_acp_python(runtime)
    uv_args = f"--config-file {shlex.quote(uv_config_path)} " if uv_config_path else ""
    npm_env = f"export NPM_CONFIG_USERCONFIG={shlex.quote(npmrc_path)}\n" if npmrc_path else ""
    return (
        "set -euo pipefail\n"
        f"{npm_env}"
        'if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then\n'
        '  export PATH="$HOME/.local/bin:$PATH"\n'
        "fi\n"
        f'{python_bin} -c "import acp, trio_asyncio" >/dev/null 2>&1 || '
        f"uv pip install {uv_args}--python {shlex.quote(python_bin)} "
        "'agent-client-protocol>=0.8.1' 'trio-asyncio>=0.15.0'\n"
    )


def _remote_npm_registry_probe_command(
    npm_package: str,
    *,
    npmrc_path: str | None = None,
) -> str:
    npm_env = f"export NPM_CONFIG_USERCONFIG={shlex.quote(npmrc_path)}\n" if npmrc_path else ""
    return f"set -euo pipefail\n{npm_env}npm view {shlex.quote(npm_package)} version >/dev/null\n"


def _remote_acp_run_env_exports(*, npmrc_path: str | None = None) -> str:
    exports: list[str] = []
    if npmrc_path is not None:
        exports.append(f"export NPM_CONFIG_USERCONFIG={shlex.quote(npmrc_path)}")
    # TODO(remote-pnpm-preparation): if the ACP adapter path ever switches from
    # npm/npx to pnpm dlx, materialize a remote pnpm rc file here and export the
    # corresponding config env so remote release-age policy stays explicit.
    # TODO(remote-bun-preparation): if the ACP adapter path ever switches from
    # npm/npx to bunx, materialize a remote bunfig.toml here and export BUN_CONFIG
    # (or pass --config) explicitly rather than relying on ambient host config.
    if not exports:
        return ""
    return "".join(f"{line}\n" for line in exports)


def _remote_acp_command(
    runtime: Literal["claude_acp", "codex_acp"],
    *,
    preparation: RemoteRuntimePreparation,
    extra_args: list[str] | None = None,
) -> list[str]:
    cli_name, npm_package = _remote_runtime_cli(runtime)
    base = [cli_name] if preparation.mode == "preinstalled" else ["npx", "-y", npm_package]
    return [*base, *(extra_args or [])]


async def _ensure_remote_acp_uv_preparation(
    workspace: SandboxWorkspaceResource,
    *,
    runtime: Literal["claude_acp", "codex_acp"],
    cwd: str,
    preparation: RemoteRuntimePreparation,
    sample_id: str | None = None,
) -> None:
    _cli_name, npm_package = _remote_runtime_cli(runtime)
    marker_dir = f"/tmp/rollouts-external-runtime/{runtime}"
    marker_path = f"{marker_dir}/bootstrap-ready"
    await _workspace_exec(workspace, f"mkdir -p {marker_dir}", cwd=cwd, timeout=30.0)

    marker_result = await _workspace_exec(
        workspace,
        f"test -f {marker_path}",
        cwd=cwd,
        timeout=10.0,
    )
    if marker_result.returncode == 0:
        _emit_remote_runtime_stage(
            runtime=runtime,
            sample_id=sample_id,
            stage="prepare_cached",
            status="success",
        )
        return

    uv_config_path: str | None = None
    npmrc_path: str | None = None
    if preparation.uv_config_toml is not None:
        uv_config_path = f"{marker_dir}/uv.toml"
        await workspace.write_file(uv_config_path, preparation.uv_config_toml.encode("utf-8"))
    if preparation.npmrc is not None:
        npmrc_path = f"{marker_dir}/.npmrc"
        await workspace.write_file(npmrc_path, preparation.npmrc.encode("utf-8"))

    stages = (
        ("node_runtime", _remote_acp_ensure_node_command(), 5 * 60),
        ("uv_runtime", _remote_acp_ensure_uv_command(), 5 * 60),
        (
            "python_deps",
            _remote_acp_uv_prepare_command(
                runtime,
                uv_config_path=uv_config_path,
                npmrc_path=npmrc_path,
            ),
            10 * 60,
        ),
        (
            "npm_registry",
            _remote_npm_registry_probe_command(npm_package, npmrc_path=npmrc_path),
            60.0,
        ),
        ("mark_ready", f"touch {marker_path}", 10.0),
    )

    for stage_name, command, timeout in stages:
        _emit_remote_runtime_stage(
            runtime=runtime,
            sample_id=sample_id,
            stage=stage_name,
            status="start",
        )
        result = await _workspace_exec(
            workspace,
            command,
            cwd=cwd,
            timeout=timeout,
        )
        if result.returncode != 0:
            _emit_remote_runtime_stage(
                runtime=runtime,
                sample_id=sample_id,
                stage=stage_name,
                status="error",
            )
            raise RuntimeError(
                f"Failed to prepare remote {runtime} during {stage_name}: "
                f"{result.stderr or result.stdout}"
            )
        _emit_remote_runtime_stage(
            runtime=runtime,
            sample_id=sample_id,
            stage=stage_name,
            status="success",
        )


async def _ensure_remote_acp_preinstalled(
    workspace: SandboxWorkspaceResource,
    *,
    runtime: Literal["claude_acp", "codex_acp"],
    cwd: str,
) -> None:
    result = await _workspace_exec(
        workspace,
        _remote_acp_validate_command(runtime),
        cwd=cwd,
        timeout=60.0,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Remote {runtime} preinstalled runtime is incomplete: {result.stderr or result.stdout}"
        )


async def _prepare_remote_acp_runtime(
    workspace: SandboxWorkspaceResource,
    *,
    runtime: Literal["claude_acp", "codex_acp"],
    cwd: str,
    preparation: RemoteRuntimePreparation,
    sample_id: str | None = None,
) -> None:
    if preparation.mode == "preinstalled":
        await _ensure_remote_acp_preinstalled(workspace, runtime=runtime, cwd=cwd)
        return
    if preparation.mode == "uv":
        await _ensure_remote_acp_uv_preparation(
            workspace,
            runtime=runtime,
            cwd=cwd,
            preparation=preparation,
            sample_id=sample_id,
        )
        return
    raise AssertionError(f"Unhandled remote ACP preparation mode: {preparation.mode!r}")


def _assistant_block_from_text(idx: int, text: str) -> tuple[int, dict[str, Any]]:
    return idx, {"type": "text", "text": text}


def _assistant_block_from_thinking(idx: int, text: str) -> tuple[int, dict[str, Any]]:
    return idx, {"type": "thinking", "thinking": text}


def _assistant_block_from_tool_call(
    idx: int,
    *,
    tool_call_id: str,
    name: str,
    args: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    return idx, {"type": "toolCall", "id": tool_call_id, "name": name, "arguments": args}


def _trajectory_from_remote_acp_payload(payload: dict[str, Any]) -> Trajectory:
    messages: list[Message] = []
    completed_blocks: list[tuple[int, dict[str, Any]]] = []

    def flush_assistant() -> None:
        nonlocal completed_blocks
        if not completed_blocks:
            return
        completed_blocks.sort(key=lambda item: item[0])
        messages.append(Message(role="assistant", content=[block for _, block in completed_blocks]))
        completed_blocks = []

    for entry in payload.get("events", []):
        event_type = entry.get("event")
        if event_type == "text":
            completed_blocks.append(
                _assistant_block_from_text(int(entry["content_index"]), str(entry["content"]))
            )
            continue
        if event_type == "thinking":
            completed_blocks.append(
                _assistant_block_from_thinking(int(entry["content_index"]), str(entry["content"]))
            )
            continue
        if event_type == "tool_call":
            completed_blocks.append(
                _assistant_block_from_tool_call(
                    int(entry["content_index"]),
                    tool_call_id=str(entry["tool_call_id"]),
                    name=str(entry["tool_name"]),
                    args=dict(entry.get("args", {})),
                )
            )
            continue
        if event_type == "tool_result":
            flush_assistant()
            messages.append(
                Message(
                    role="tool",
                    content=str(entry.get("content", "")),
                    tool_call_id=str(entry["tool_call_id"]),
                )
            )
            continue

    flush_assistant()
    return Trajectory(messages=messages)


async def _run_remote_acp_runtime(
    *,
    runtime: Literal["claude_acp", "codex_acp"],
    prompt: str,
    sample_id: str,
    workspace: SandboxWorkspaceResource,
    cwd: str,
    run_config: Any | None,
    command: list[str],
    timeout_seconds: float,
    preparation: RemoteRuntimePreparation,
) -> ExternalAttemptArtifact:
    del run_config
    await _prepare_remote_acp_runtime(
        workspace,
        runtime=runtime,
        cwd=cwd,
        preparation=preparation,
        sample_id=sample_id,
    )

    state_dir = _remote_runtime_state_dir(runtime, sample_id)
    prompt_path = f"{state_dir}/prompt.txt"
    env_path = f"{state_dir}/env.json"
    output_path = f"{state_dir}/result.json"
    python_bin = _remote_acp_python(runtime)
    npmrc_path = f"/tmp/rollouts-external-runtime/{runtime}/.npmrc"
    codex_home_path: str | None = None
    await _workspace_exec(workspace, f"mkdir -p {state_dir}", cwd=cwd, timeout=30.0)
    await workspace.write_file(prompt_path, prompt.encode("utf-8"))
    env_payload = _remote_runtime_env(runtime)
    if runtime == "codex_acp":
        codex_home_path = f"{state_dir}/codex-home"
        auth_path = f"{codex_home_path}/auth.json"
        await _workspace_exec(workspace, f"mkdir -p {codex_home_path}", cwd=cwd, timeout=30.0)
        await workspace.write_file(
            auth_path,
            json.dumps(_remote_codex_acp_auth_payload(str(env_payload["OPENAI_API_KEY"]))).encode(
                "utf-8"
            ),
        )
        env_payload["CODEX_HOME"] = codex_home_path
    await workspace.write_file(
        env_path,
        json.dumps(env_payload).encode("utf-8"),
    )

    runner = (
        "cat <<'PY' > "
        f"{state_dir}/run_remote_acp.py\n"
        "from __future__ import annotations\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import trio\n"
        "import trio_asyncio\n"
        "from acp import InitializeResponse, PromptResponse, spawn_agent_process\n"
        "from acp.schema import ClientCapabilities, Implementation, TextContentBlock\n"
        "class Bridge:\n"
        "    def __init__(self):\n"
        "        self.events = []\n"
        "        self._content_index = 0\n"
        "        self._pending = {}\n"
        "    def _next_idx(self):\n"
        "        idx = self._content_index\n"
        "        self._content_index += 1\n"
        "        return idx\n"
        "    def on_connect(self, conn):\n"
        "        return None\n"
        "    async def request_permission(self, options, session_id, tool_call, **kwargs):\n"
        "        del session_id, tool_call, kwargs\n"
        "        from acp import RequestPermissionResponse\n"
        "        from acp.schema import AllowedOutcome, DeniedOutcome\n"
        "        allowed = next((opt for opt in options if getattr(opt, 'kind', None) in {'allow_once', 'allow_always'}), None)\n"
        "        if allowed is None:\n"
        "            return RequestPermissionResponse(outcome=DeniedOutcome(outcome='cancelled'))\n"
        "        return RequestPermissionResponse(outcome=AllowedOutcome(optionId=allowed.optionId, outcome='selected'))\n"
        "    async def session_update(self, session_id, update, **kwargs):\n"
        "        del session_id, kwargs\n"
        "        update_type = getattr(update, 'sessionUpdate', None)\n"
        "        if update_type == 'agent_message_chunk':\n"
        "            text = self._content_to_text(getattr(update, 'content', None))\n"
        "            if text:\n"
        "                self.events.append({'event': 'text', 'content_index': self._next_idx(), 'content': text})\n"
        "            return\n"
        "        if update_type == 'agent_thought_chunk':\n"
        "            text = self._content_to_text(getattr(update, 'content', None))\n"
        "            if text:\n"
        "                self.events.append({'event': 'thinking', 'content_index': self._next_idx(), 'content': text})\n"
        "            return\n"
        "        if update_type not in {'tool_call', 'tool_call_update'}:\n"
        "            return\n"
        "        tool_call_id = getattr(update, 'toolCallId', None)\n"
        "        if not tool_call_id:\n"
        "            return\n"
        "        pending = self._pending.get(tool_call_id)\n"
        "        name = self._tool_name(update)\n"
        "        args = self._json_safe(getattr(update, 'rawInput', None))\n"
        "        if not isinstance(args, dict):\n"
        "            args = {'raw_input': args}\n"
        "        if pending is None:\n"
        "            pending = {'content_index': self._next_idx(), 'tool_name': name, 'args': args}\n"
        "            self._pending[tool_call_id] = pending\n"
        "        else:\n"
        "            pending['tool_name'] = name or pending['tool_name']\n"
        "            pending['args'] = args or pending['args']\n"
        "        status = getattr(update, 'status', None)\n"
        "        if status not in {'completed', 'failed'}:\n"
        "            return\n"
        "        self.events.append({'event': 'tool_call', 'content_index': pending['content_index'], 'tool_call_id': tool_call_id, 'tool_name': pending['tool_name'], 'args': pending['args']})\n"
        "        raw_output = self._json_safe(getattr(update, 'rawOutput', None))\n"
        "        content = self._json_safe(getattr(update, 'content', None))\n"
        "        result = self._stringify(raw_output) if raw_output is not None else self._stringify(content)\n"
        "        self.events.append({'event': 'tool_result', 'tool_call_id': tool_call_id, 'content': result, 'is_error': status == 'failed'})\n"
        "        self._pending.pop(tool_call_id, None)\n"
        "    def _tool_name(self, update):\n"
        "        title = getattr(update, 'title', None)\n"
        "        if title:\n"
        "            return str(title)\n"
        "        kind = getattr(update, 'kind', None)\n"
        "        if kind:\n"
        "            return str(kind)\n"
        "        return 'tool'\n"
        "    def _json_safe(self, value):\n"
        "        if isinstance(value, list):\n"
        "            return [self._json_safe(item) for item in value]\n"
        "        if isinstance(value, tuple):\n"
        "            return [self._json_safe(item) for item in value]\n"
        "        if isinstance(value, dict):\n"
        "            return {str(k): self._json_safe(v) for k, v in value.items() if not (k == 'field_meta' and v is None)}\n"
        "        if hasattr(value, 'model_dump'):\n"
        "            return self._json_safe(value.model_dump())\n"
        "        if hasattr(value, '__dict__') and not isinstance(value, (str, bytes, bytearray)):\n"
        "            try:\n"
        "                return self._json_safe(vars(value))\n"
        "            except TypeError:\n"
        "                pass\n"
        "        try:\n"
        "            json.dumps(value)\n"
        "            return value\n"
        "        except TypeError:\n"
        "            return repr(value)\n"
        "    def _content_to_text(self, content):\n"
        "        if content is None:\n"
        "            return ''\n"
        "        if isinstance(content, list):\n"
        "            return '\\n'.join(filter(None, (self._content_to_text(item) for item in content)))\n"
        "        block_type = getattr(content, 'type', None)\n"
        "        if block_type == 'text':\n"
        "            return str(getattr(content, 'text', ''))\n"
        "        return json.dumps(self._json_safe(content), indent=2)\n"
        "    def _stringify(self, value):\n"
        "        if isinstance(value, str):\n"
        "            return value\n"
        "        try:\n"
        "            return json.dumps(value, indent=2)\n"
        "        except TypeError:\n"
        "            return repr(value)\n"
        "async def main():\n"
        f"    prompt = Path({prompt_path!r}).read_text()\n"
        f"    env = dict(os.environ)\n"
        f"    env.update(json.loads(Path({env_path!r}).read_text()))\n"
        f"    if Path({npmrc_path!r}).exists():\n"
        f"        env['NPM_CONFIG_USERCONFIG'] = {npmrc_path!r}\n"
        f"    command = {command!r}\n"
        "    async def run_once_asyncio():\n"
        "        bridge = Bridge()\n"
        "        async with spawn_agent_process(bridge, command[0], *command[1:], env=env, cwd=os.getcwd()) as (conn, process):\n"
        "            init = await conn.initialize(protocol_version=1, client_capabilities=ClientCapabilities(terminal=False), client_info=Implementation(name='rollouts-remote', version='0.1.0'))\n"
        "            if not isinstance(init, InitializeResponse):\n"
        "                raise RuntimeError(f'unexpected initialize response: {init!r}')\n"
        "            session = await conn.new_session(cwd=os.getcwd(), mcp_servers=[])\n"
        "            response = await conn.prompt([TextContentBlock(type='text', text=prompt)], session_id=session.sessionId)\n"
        "            if not isinstance(response, PromptResponse):\n"
        "                raise RuntimeError(f'unexpected prompt response: {response!r}')\n"
        "            if process.returncode is None:\n"
        "                process.terminate()\n"
        "                await process.wait()\n"
        "            return bridge, response, process.returncode\n"
        "    async with trio_asyncio.open_loop():\n"
        "        bridge, response, returncode = await trio_asyncio.aio_as_trio(run_once_asyncio)()\n"
        "    payload = {'events': bridge.events, 'stop_reason': response.stopReason, 'returncode': returncode}\n"
        f"    Path({output_path!r}).write_text(json.dumps(payload), encoding='utf-8')\n"
        "trio.run(main)\n"
        "PY\n"
        f"{_remote_acp_run_env_exports(npmrc_path=npmrc_path)}"
        f"{python_bin} {state_dir}/run_remote_acp.py\n"
    )
    result = await _workspace_exec(
        workspace,
        runner,
        cwd=cwd,
        timeout=timeout_seconds,
    )
    output_result = await _workspace_exec(
        workspace,
        f"cat {output_path}",
        cwd=cwd,
        timeout=30.0,
    )
    await _workspace_exec(workspace, f"rm -rf {state_dir}", cwd=cwd, timeout=30.0)
    if result.returncode != 0:
        raise RuntimeError(
            f"Remote {runtime} failed before producing trajectory output: "
            f"{result.stderr or result.stdout}"
        )
    payload = json.loads(output_result.stdout)
    trajectory = _trajectory_from_remote_acp_payload(payload)
    metadata: dict[str, Any] = {
        "runtime": runtime,
        "driver": runtime,
        "cwd": cwd,
        "remote_execution": True,
        "source": "remote_acp_buffered",
        "stop_reason": payload.get("stop_reason"),
    }
    if payload.get("returncode") not in (None, 0):
        metadata["remote_returncode"] = payload["returncode"]
    return ExternalAttemptArtifact(trajectory=trajectory, metadata=metadata)


def _remote_runtime_bootstrap_command(runtime: str) -> str:
    cli_name, npm_package = _remote_runtime_cli(runtime)
    return (
        "set -euo pipefail\n"
        'NODE_MAJOR="$(node -p \'process.versions.node.split(".")[0]\' 2>/dev/null || echo 0)"\n'
        'if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1 || [ "${NODE_MAJOR}" -lt 18 ]; then\n'
        "  apt-get update\n"
        "  DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates\n"
        "  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -\n"
        "  DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs\n"
        "fi\n"
        f"if ! id -u {REMOTE_AGENT_USER} >/dev/null 2>&1; then\n"
        f"  useradd -m -s /bin/bash {REMOTE_AGENT_USER}\n"
        "fi\n"
        f"if ! command -v {cli_name} >/dev/null 2>&1; then\n"
        f"  npm install -g {npm_package}\n"
        "fi\n"
    )


async def _ensure_remote_runtime_bootstrap(
    workspace: SandboxWorkspaceResource,
    *,
    runtime: str,
    cwd: str,
) -> None:
    marker_dir = f"/tmp/rollouts-external-runtime/{runtime}"
    marker_path = f"{marker_dir}/bootstrap-ready"
    result = await _workspace_exec(
        workspace,
        (
            "set -euo pipefail\n"
            f"mkdir -p {marker_dir}\n"
            f"if [ ! -f {marker_path} ]; then\n"
            f"{_remote_runtime_bootstrap_command(runtime)}"
            f"  touch {marker_path}\n"
            "fi\n"
        ),
        cwd=cwd,
        timeout=10 * 60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to bootstrap remote {runtime}: {result.stderr or result.stdout}"
        )


# TODO(remote-external-runtime-streaming): This function is the fallback for
# the buffered remote execution path. Once session-file polling lands, this
# should only be called when polling fails to find the session file (e.g. the
# CLI crashed before writing anything). The primary path should be live polling.
async def _build_trajectory_from_remote_jsonl(
    *,
    runtime: Literal["claude_code", "codex"],
    raw_output: str,
    sample_id: str,
    run_config: Any | None,
    driver_name: str,
) -> tuple[Trajectory, str | None]:
    parser = _ClaudeEventParser() if runtime == "claude_code" else _CodexEventParser()
    accumulator = _EventAccumulator()
    on_event = _make_eval_on_event(sample_id, getattr(run_config, "on_chunk", None))
    on_raw_line = _make_raw_driver_line_handler(run_config, driver=driver_name)

    for raw_line in raw_output.splitlines():
        line = raw_line.rstrip("\r")
        if on_raw_line is not None and line:
            await on_raw_line(line)
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        for event in parser.parse(msg):
            if isinstance(event, _FlushAssistantMessage):
                accumulator.handle(event)
                continue
            await on_event(event)
            accumulator.handle(event)

    return Trajectory(messages=accumulator.finalize()), getattr(parser, "_session_id", None)


def _remote_session_file_discovery_script(runtime: str, agent_home: str, cwd: str) -> str:
    """Return a shell snippet that prints the session file path to stdout.

    For claude_code: looks in ~/.claude/projects/<escaped-cwd>/ for the newest .jsonl.
    For codex: looks in ~/.codex/sessions/YYYY/MM/DD/ for the newest rollout-*.jsonl.
    Prints nothing (exits 0) if not found yet.
    """
    if runtime == "claude_code":
        # Claude Code escapes cwd by replacing / with -
        # e.g. /workspace -> -workspace
        # We use Python inline to avoid bash quoting hell with arbitrary cwd paths.
        return (
            f'python3 -c "\n'
            f"import os, pathlib\n"
            f"cwd = {cwd!r}\n"
            f"escaped = cwd.replace('/', '-')\n"
            f"projects = pathlib.Path({agent_home!r}) / '.claude' / 'projects' / escaped\n"
            f"if not projects.exists():\n"
            f"    exit(0)\n"
            f"files = sorted(projects.glob('*.jsonl'), key=lambda p: p.stat().st_mtime)\n"
            f"if files:\n"
            f"    print(files[-1])\n"
            f'"'
        )
    if runtime == "codex":
        return (
            f'python3 -c "\n'
            f"import pathlib\n"
            f"sessions = pathlib.Path({agent_home!r}) / '.codex' / 'sessions'\n"
            f"if not sessions.exists():\n"
            f"    exit(0)\n"
            f"files = sorted(sessions.rglob('rollout-*.jsonl'), key=lambda p: p.stat().st_mtime)\n"
            f"if files:\n"
            f"    print(files[-1])\n"
            f'"'
        )
    raise ValueError(f"Unsupported runtime: {runtime}")


async def _run_agent_in_workspace(
    *,
    runtime: Literal["claude_code", "codex"],
    prompt: str,
    sample_id: str,
    workspace: SandboxWorkspaceResource | LocalWorkspaceResource,
    cwd: str,
    run_config: Any | None,
    command: list[str],
    timeout_seconds: float,
) -> ExternalAttemptArtifact:
    """Launch the CLI in the background and build trajectory from native session file.

    Uses `exec_background` + periodic `download_bytes(offset=...)` reads for both
    local and sandbox workspaces. Remote-only bootstrap/chown/getent logic is still
    required for sandbox-backed execution, while local workspaces skip that path.
    """
    from ..drivers.session_adapter import claude_message_to_rollouts, codex_message_to_rollouts

    is_remote_workspace = isinstance(workspace, SandboxWorkspaceResource)
    if is_remote_workspace:
        await _ensure_remote_runtime_bootstrap(workspace, runtime=runtime, cwd=cwd)

    state_dir = _remote_runtime_state_dir(runtime, sample_id)
    prompt_path = f"{state_dir}/prompt.txt"
    env_path = f"{state_dir}/env.json"
    stdout_path = f"{state_dir}/agent.stdout"
    stderr_path = f"{state_dir}/agent.stderr"

    state_cmd = f"mkdir -p {state_dir}"
    if is_remote_workspace:
        await _workspace_exec(workspace, state_cmd, cwd=cwd, timeout=30.0)
    else:
        state_result = await workspace.run(state_cmd, cwd=cwd, timeout=30.0)
        if state_result.returncode != 0:
            raise RuntimeError(
                f"Failed to prepare local run state directory for {runtime}: "
                f"{state_result.stderr or state_result.stdout}"
            )

    await workspace.write_file(prompt_path, prompt.encode("utf-8"))
    await workspace.write_file(
        env_path,
        json.dumps(_remote_runtime_env(runtime)).encode("utf-8"),
    )

    agent_home = os.path.expanduser("~")
    if is_remote_workspace:
        await _workspace_exec(
            workspace,
            f"chown -R {REMOTE_AGENT_USER}:{REMOTE_AGENT_USER} {cwd} {state_dir}",
            cwd=cwd,
            timeout=30.0,
        )
        home_result = await _workspace_exec(
            workspace,
            f"getent passwd {REMOTE_AGENT_USER} | cut -d: -f6",
            cwd=cwd,
            timeout=10.0,
        )
        agent_home = home_result.stdout.strip() or f"/home/{REMOTE_AGENT_USER}"

    cli_command = shlex.join(command + [prompt])

    if is_remote_workspace:
        env_export = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in _remote_runtime_env(runtime).items()
        )
        launch_command = (
            f"{env_export} HOME={shlex.quote(agent_home)} {cli_command}"
            if env_export
            else f"HOME={shlex.quote(agent_home)} {cli_command}"
        )
        pid = await workspace.exec_background(
            f"su -s /bin/bash {REMOTE_AGENT_USER} -c {shlex.quote(launch_command)}",
            cwd=cwd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    else:
        pid = await workspace.exec_background(
            cli_command,
            cwd=cwd,
            env=_remote_runtime_env(runtime),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    async def _read_new_session_bytes() -> bytes:
        nonlocal file_offset
        if session_file is None:
            return b""
        raw_session_bytes = await workspace.download_bytes(session_file, offset=file_offset)
        if not raw_session_bytes:
            return b""
        return raw_session_bytes

    _store = getattr(run_config, "session_store", None)
    _harness_session_id = getattr(run_config, "session_id", None)

    async def _append_session_entries(chunk: bytes) -> int:
        nonlocal session_id, assistant_turn
        if not chunk:
            return 0
        text = chunk.decode("utf-8", errors="replace")
        added = 0
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if runtime == "claude_code":
                if session_id is None:
                    cli_session_id = entry.get("sessionId")
                    if cli_session_id is not None:
                        session_id = cli_session_id
                        if _store is not None and _harness_session_id is not None:
                            await _store.update(
                                _harness_session_id,
                                tags={"cli_session_id": cli_session_id},
                            )
                new_msgs = claude_message_to_rollouts(entry)
            else:
                if session_id is None and entry.get("type") == "session_meta":
                    cli_session_id = entry.get("payload", {}).get("id")
                    if cli_session_id is not None:
                        session_id = cli_session_id
                        if _store is not None and _harness_session_id is not None:
                            await _store.update(
                                _harness_session_id,
                                tags={"cli_session_id": cli_session_id},
                            )
                codex_msg = codex_message_to_rollouts(entry)
                new_msgs = [codex_msg] if codex_msg is not None else []
            for msg in new_msgs:
                messages.append(msg)
                if _store is not None and _harness_session_id is not None:
                    await _store.append_message(_harness_session_id, msg)
                added += 1
                if msg.role == "assistant":
                    _event_logger.info(
                        "turn",
                        extra={
                            "sample_id": sample_id,
                            "turn": assistant_turn,
                            "status": "streaming...",
                        },
                    )
                    assistant_turn += 1
        return added

    # Poll for the run process and parse its session log in-flight.
    messages: list[Any] = []
    session_file: str | None = None
    session_id: str | None = None
    poll_interval = 2.0
    elapsed = 0.0
    file_offset = 0
    assistant_turn = 0
    discovery_script = _remote_session_file_discovery_script(runtime, agent_home, cwd)

    while elapsed < timeout_seconds:
        await trio.sleep(poll_interval)
        elapsed += poll_interval

        if session_file is None:
            if is_remote_workspace:
                disc = await _workspace_exec(
                    workspace,
                    discovery_script,
                    cwd=cwd,
                    timeout=10.0,
                )
            else:
                disc = await workspace.run(discovery_script, cwd=cwd, timeout=10.0)
            path = disc.stdout.strip()
            if path:
                session_file = path
                _event_logger.info(
                    "session_file_found",
                    extra={"runtime": runtime, "sample_id": sample_id, "path": path},
                )

        if session_file is not None:
            session_bytes = await _read_new_session_bytes()
            if session_bytes:
                file_offset += len(session_bytes)
                await _append_session_entries(session_bytes)

        alive_command = f"kill -0 {pid} 2>/dev/null && echo alive || echo dead"
        if is_remote_workspace:
            alive_result = await _workspace_exec(workspace, alive_command, cwd=cwd, timeout=10.0)
        else:
            alive_result = await workspace.run(alive_command, cwd=cwd, timeout=10.0)
        if alive_result.stdout.strip() == "dead":
            break

    else:
        raise RuntimeError(
            f"Remote {runtime} timed out after {timeout_seconds:.0f}s "
            f"(session_file={session_file!r})"
        )

    # Final read for trailing lines not yet picked up by polling boundary.
    if session_file is not None:
        final_session_bytes = await _read_new_session_bytes()
        file_offset += len(final_session_bytes)
        await _append_session_entries(final_session_bytes)

    if is_remote_workspace:
        stdout_bytes = await workspace.download_bytes(stdout_path)
        stderr_bytes = await workspace.download_bytes(stderr_path)
        cleanup_result = await _workspace_exec(
            workspace, f"rm -rf {state_dir}", cwd=cwd, timeout=30.0
        )
    else:
        stdout_bytes = await workspace.download_bytes(stdout_path)
        stderr_bytes = await workspace.download_bytes(stderr_path)
        cleanup_result = await workspace.run(f"rm -rf {state_dir}", cwd=cwd, timeout=30.0)
    if cleanup_result.returncode != 0:
        _event_logger.warning(
            "remote_runtime_cleanup_failed",
            extra={"sample_id": sample_id, "runtime": runtime, "stderr": cleanup_result.stderr},
        )

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    has_agent_output = any(
        isinstance(msg, Message) and msg.role in {"assistant", "tool"} for msg in messages
    )
    if not has_agent_output:
        trajectory, fallback_session_id = await _build_trajectory_from_remote_jsonl(
            runtime=runtime,
            raw_output=stdout_text,
            sample_id=sample_id,
            run_config=run_config,
            driver_name="claude" if runtime == "claude_code" else "codex",
        )
        fallback_has_agent_output = any(
            isinstance(msg, Message) and msg.role in {"assistant", "tool"}
            for msg in trajectory.messages
        )
        if fallback_has_agent_output:
            return ExternalAttemptArtifact(
                trajectory=trajectory,
                metadata={
                    "runtime": runtime,
                    "driver": "claude" if runtime == "claude_code" else "codex",
                    "cwd": cwd,
                    "session_id": fallback_session_id,
                    "remote_execution": True,
                    "source": "stream_json_fallback",
                },
            )
        raise RuntimeError(
            f"Remote {runtime} exited without producing agent output. stderr: {stderr_text[-2000:]}"
        )

    trajectory = Trajectory(messages=list(messages))
    metadata: dict[str, Any] = {
        "runtime": runtime,
        "driver": "claude" if runtime == "claude_code" else "codex",
        "cwd": cwd,
        "session_id": session_id,
        "remote_execution": True,
        "source": "session_file",
    }
    return ExternalAttemptArtifact(trajectory=trajectory, metadata=metadata)


async def trajectory_from_remote_claude_code(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    workspace: SandboxWorkspaceResource | LocalWorkspaceResource,
    cwd: str,
    run_config: Any | None = None,
    *,
    model: str = "sonnet",
    include_partial: bool = True,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    timeout_seconds: float = 600.0,
) -> ExternalAttemptArtifact:
    del sample_data
    command = [
        "claude",
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
        "--model",
        model,
    ]
    if include_partial:
        command.append("--include-partial-messages")
    if system_prompt:
        command.extend(["--system-prompt", system_prompt])
    if allowed_tools:
        command.extend(["--allowedTools", ",".join(allowed_tools)])
    artifact = await _run_agent_in_workspace(
        runtime="claude_code",
        prompt=prompt,
        sample_id=sample_id,
        workspace=workspace,
        cwd=cwd,
        run_config=run_config,
        command=command,
        timeout_seconds=timeout_seconds,
    )
    artifact.metadata["model"] = model
    return artifact


async def trajectory_from_remote_codex(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    workspace: SandboxWorkspaceResource | LocalWorkspaceResource,
    cwd: str,
    run_config: Any | None = None,
    *,
    model: str = "gpt-5.1-codex-mini",
    sandbox: str = "read-only",
    timeout_seconds: float = 600.0,
) -> ExternalAttemptArtifact:
    del sample_data
    command = [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--model",
        model,
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    artifact = await _run_agent_in_workspace(
        runtime="codex",
        prompt=prompt,
        sample_id=sample_id,
        workspace=workspace,
        cwd=cwd,
        run_config=run_config,
        command=command,
        timeout_seconds=timeout_seconds,
    )
    artifact.metadata["model"] = model
    artifact.metadata["sandbox"] = sandbox
    return artifact


async def trajectory_from_remote_claude_acp(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    workspace: SandboxWorkspaceResource,
    cwd: str,
    run_config: Any | None = None,
    *,
    model: str = "claude-agent-acp",
    timeout_seconds: float = 600.0,
    preparation: RemoteRuntimePreparation = RemoteRuntimePreparation(mode="uv"),
) -> ExternalAttemptArtifact:
    del sample_data
    artifact = await _run_remote_acp_runtime(
        runtime="claude_acp",
        prompt=prompt,
        sample_id=sample_id,
        workspace=workspace,
        cwd=cwd,
        run_config=run_config,
        command=_remote_acp_command("claude_acp", preparation=preparation),
        timeout_seconds=timeout_seconds,
        preparation=preparation,
    )
    artifact.metadata["model"] = model
    return artifact


async def trajectory_from_remote_codex_acp(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    workspace: SandboxWorkspaceResource,
    cwd: str,
    run_config: Any | None = None,
    *,
    model: str = "codex-acp",
    timeout_seconds: float = 600.0,
    preparation: RemoteRuntimePreparation = RemoteRuntimePreparation(mode="uv"),
) -> ExternalAttemptArtifact:
    del sample_data
    artifact = await _run_remote_acp_runtime(
        runtime="codex_acp",
        prompt=prompt,
        sample_id=sample_id,
        workspace=workspace,
        cwd=cwd,
        run_config=run_config,
        command=_remote_acp_command(
            "codex_acp",
            preparation=preparation,
            extra_args=[
                "-c",
                'forced_login_method="api"',
                "-c",
                'preferred_auth_method="apikey"',
                "-c",
                'approval_policy="never"',
                "-c",
                'sandbox_mode="workspace-write"',
            ],
        ),
        timeout_seconds=timeout_seconds,
        preparation=preparation,
    )
    artifact.metadata["model"] = model
    return artifact
