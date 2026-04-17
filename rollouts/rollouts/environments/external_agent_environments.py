"""External-agent environments.

Each external-agent runtime (claude-code, codex, opencode, ...) gets its own
Environment type. These are first-class homes for:

  1. The agent's tool vocabulary — which of the harness's built-in tools are
     allowed (`allowed_builtin_tools`) and what we inject via MCP (`mcp_tools`).
  2. The workspace — the `SandboxWorkspaceResource` the agent operates in.
  3. The translation boundary — `translate_harness_event(raw)` maps one line
     of harness-native JSONL into rollouts `Message`s.

These coexist with `rollouts/eval/remote_runtime.py` for now; the bespoke
`trajectory_from_remote_claude_code` / `_codex` functions stay working and
are not consumers of these types yet. Migration is tracked in
`rollouts/rollouts/agents/runtime_refactor.md`.

See `/docs/design/session_ownership.md` for the ownership model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ..core import Message
from ..drivers.session_adapter import claude_message_to_rollouts, codex_message_to_rollouts

if TYPE_CHECKING:
    from .resources import SandboxWorkspaceResource


# ── Protocol ────────────────────────────────────────────────────────────────


class ExternalAgentEnvironment(Protocol):
    """Protocol for external-agent-runtime-specific environments.

    Each concrete type (ClaudeCodeEnvironment, CodexEnvironment, ...) owns
    the translation between its harness's native session format and rollouts
    `Message`s, plus the MCP/allowed-tools flag story for that harness.

    The Protocol intentionally types `translate_harness_event` as returning
    `list[Message]` rather than a single `Message | None` / `SessionEntry`:
    claude-code emits assistant+tool_use+tool_result as separate top-level
    JSONL records that can expand into multiple `Message`s per line; codex
    emits at most one. A uniform `list[Message]` covers both without losing
    information. When `SessionEntry` lands (see runtime_refactor.md) the
    signature will shift to `list[SessionEntry]`.
    """

    workspace: SandboxWorkspaceResource
    allowed_builtin_tools: list[str] | None
    mcp_tools: list[Any]  # list[MCPTool] once that type exists

    def get_tools(self) -> list[Any]: ...

    def get_launch_flags(self) -> list[str]: ...

    def translate_harness_event(self, raw: dict[str, Any]) -> list[Message]: ...

    async def apply_effect(self, effect: Any) -> None: ...


# ── MCP config helpers ──────────────────────────────────────────────────────


def _mcp_tools_to_claude_config(mcp_tools: list[Any]) -> dict[str, Any]:
    """Build the JSON body claude-code expects for `--mcp-config`.

    Claude-code reads a JSON file of shape `{"mcpServers": {<name>: <spec>}}`
    where each spec has `command`, `args`, optional `env`. We accept either a
    pre-shaped dict (with keys `name` + `spec`) or anything with a `to_claude_mcp_spec()`
    method so callers can pass their own tool abstractions without us owning
    the MCP tool type yet.
    """
    servers: dict[str, Any] = {}
    for tool in mcp_tools:
        if isinstance(tool, dict) and "name" in tool and "spec" in tool:
            servers[tool["name"]] = tool["spec"]
            continue
        to_spec = getattr(tool, "to_claude_mcp_spec", None)
        if callable(to_spec):
            name, spec = to_spec()
            servers[name] = spec
            continue
        raise TypeError(
            f"mcp_tools entry {tool!r} is not a dict with name/spec and has no "
            "to_claude_mcp_spec() method. Pass a concrete MCP tool shape."
        )
    return {"mcpServers": servers}


# ── Concrete environments ──────────────────────────────────────────────────


@dataclass
class ClaudeCodeEnvironment:
    """Environment for claude-code CLI runs.

    Claude-code ships built-in tools (bash, read, write, edit, glob, grep,
    web_fetch, ...). This type describes which of those are enabled and what
    extra tools we inject via MCP.

    Session files live at `~/.claude/projects/<escaped-cwd>/*.jsonl`. Each
    line is a dict with a top-level `type` ("user", "assistant", "tool_use",
    "tool_result", etc.); `translate_harness_event` normalizes those to
    rollouts `Message`s by delegating to `claude_message_to_rollouts` in
    `drivers/session_adapter.py`.
    """

    workspace: SandboxWorkspaceResource

    # `None` = claude-code's full default tool set.
    # `[]` = no built-in tools (MCP-only agent).
    # list = subset enforced via --allowed-tools.
    allowed_builtin_tools: list[str] | None = None

    # Additional tools exposed via MCP.
    mcp_tools: list[Any] = field(default_factory=list)

    # Where to write the generated MCP config JSON. Caller-supplied because
    # the path has to be reachable from wherever claude-code launches (local
    # temp dir for LocalWorkspaceResource, in-sandbox path otherwise). If
    # None and mcp_tools is non-empty, get_launch_flags will raise.
    mcp_config_path: Path | None = None

    def get_tools(self) -> list[Any]:
        # TODO(external-env / claude-code): return concrete Tool objects once
        # the Tool type from the runtime refactor lands. Today we don't have
        # a canonical representation of claude-code's built-in tool schemas,
        # so composing them with mcp_tools would be premature.
        raise NotImplementedError(
            "ClaudeCodeEnvironment.get_tools: deferred until the Tool type "
            "from the session refactor lands."
        )

    def get_launch_flags(self) -> list[str]:
        """Translate allowed_builtin_tools + mcp_tools into CLI flags.

        Returns flags suitable for appending to a `claude` invocation, e.g.
        `--allowed-tools bash,read --mcp-config /path/to/mcp.json`.

        Does NOT write the MCP config file. The caller is responsible for
        writing `mcp_config_json()` to `self.mcp_config_path` before launch
        (paths may need to live inside a remote workspace, not locally).
        """
        flags: list[str] = []
        if self.allowed_builtin_tools is not None:
            # claude-code accepts empty list as "no builtins" via an empty arg.
            flags.extend(["--allowed-tools", ",".join(self.allowed_builtin_tools)])
        if self.mcp_tools:
            if self.mcp_config_path is None:
                raise ValueError(
                    "ClaudeCodeEnvironment has mcp_tools but no mcp_config_path; "
                    "caller must provide a writable path for the MCP config JSON."
                )
            flags.extend(["--mcp-config", str(self.mcp_config_path)])
        return flags

    def mcp_config_json(self) -> str:
        """Serialize mcp_tools into the JSON body claude-code expects.

        Caller writes this to `self.mcp_config_path` before launching.
        """
        return json.dumps(_mcp_tools_to_claude_config(self.mcp_tools), indent=2)

    def translate_harness_event(self, raw: dict[str, Any]) -> list[Message]:
        """Translate one claude-code session JSONL line to rollouts Messages.

        Delegates to `claude_message_to_rollouts`, which is the existing
        session-file adapter and already handles user/assistant/tool_use/
        tool_result entries. Returns `[]` for uninteresting lines (status
        pings, meta records) — matches the adapter's contract.

        Note: this handles the native session-file shape (`~/.claude/projects/.../*.jsonl`).
        The stream-json stdout shape that `_ClaudeEventParser` handles is a
        different format; if we need to translate that too, a second method
        (or a `format=` arg) will be added. Keeping them separate for now so
        neither consumer has to guess which format they're on.
        """
        return claude_message_to_rollouts(raw)

    async def apply_effect(self, effect: Any) -> None:
        # TODO(external-env / claude-code): fold-contract replay. Stub until
        # the SessionEntry sum type exists — we don't have a typed `effect`
        # to dispatch on yet. For filesystem-only tool calls this will be
        # "run the recorded bash/write/edit call against self.workspace";
        # for MCP-tool effects it routes to the tool's apply method.
        raise NotImplementedError(
            "ClaudeCodeEnvironment.apply_effect: deferred until SessionEntry lands."
        )


@dataclass
class CodexEnvironment:
    """Environment for OpenAI codex CLI runs.

    Codex session files live at `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`.
    Each line is `{"type": "session_meta" | "response_item" | ..., "payload": {...}}`;
    `translate_harness_event` delegates to `codex_message_to_rollouts`.

    Codex's MCP story is different from claude-code's: MCP servers are
    declared in `$CODEX_HOME/config.toml` under `[mcp_servers.<name>]`, not
    via a CLI flag. For a run-local config the idiomatic path is to point
    `CODEX_HOME` at a fresh directory (done by `remote_runtime.py` already
    for `codex_acp`). We therefore don't emit an MCP flag; instead we
    expose `codex_mcp_config_toml()` so the caller can write it into their
    chosen CODEX_HOME.
    """

    workspace: SandboxWorkspaceResource

    allowed_builtin_tools: list[str] | None = None
    mcp_tools: list[Any] = field(default_factory=list)

    # codex's own --sandbox flag. Passed through verbatim when set.
    # None means "don't emit --sandbox" — use when the base command already
    # handles sandbox policy (e.g. --dangerously-bypass-approvals-and-sandbox).
    sandbox_mode: str | None = None

    # Where to write the generated codex config.toml. Unlike claude-code,
    # codex does not take a --mcp-config flag: the caller must write this
    # into a CODEX_HOME and set CODEX_HOME in the launch env. If None and
    # mcp_tools is non-empty, get_launch_flags will raise.
    codex_config_path: Path | None = None

    def get_tools(self) -> list[Any]:
        # TODO(external-env / codex): same deferral as ClaudeCodeEnvironment.
        raise NotImplementedError(
            "CodexEnvironment.get_tools: deferred until the Tool type "
            "from the session refactor lands."
        )

    def get_launch_flags(self) -> list[str]:
        """Translate sandbox/allowed/MCP settings into codex CLI flags.

        Emits:
          - `--sandbox <mode>` when sandbox_mode is set (not None).
          - Nothing for mcp_tools (see class docstring); raises if mcp_tools
            is set without codex_config_path so the caller doesn't silently
            lose tool injection.

        Note: `allowed_builtin_tools` on codex is not a CLI flag — codex's
        tool restriction is policy-config-based. We keep the field for
        parity with ClaudeCodeEnvironment and surface it via
        `codex_mcp_config_toml`, which can embed a `[tools]` filter.
        """
        if self.mcp_tools and self.codex_config_path is None:
            raise ValueError(
                "CodexEnvironment has mcp_tools but no codex_config_path; "
                "caller must provide a writable path for codex's config.toml "
                "and set CODEX_HOME in the launch env."
            )
        flags: list[str] = []
        if self.sandbox_mode is not None:
            flags.extend(["--sandbox", self.sandbox_mode])
        return flags

    def codex_mcp_config_toml(self) -> str:
        """Serialize mcp_tools into a codex config.toml fragment.

        Caller writes this to `self.codex_config_path` and points
        CODEX_HOME at its parent directory before launching.

        TODO(external-env / codex): this is the minimal shape — each tool
        contributes an `[mcp_servers.<name>]` block. We don't own the MCP
        tool type yet, so we accept dicts with `name` + `spec` (with
        `command`, `args`, `env`) or objects with `to_codex_mcp_spec()`.
        Once the MCPTool type lands, this collapses into a typed loop.
        """
        lines: list[str] = []
        for tool in self.mcp_tools:
            if isinstance(tool, dict) and "name" in tool and "spec" in tool:
                name, spec = tool["name"], tool["spec"]
            else:
                to_spec = getattr(tool, "to_codex_mcp_spec", None)
                if not callable(to_spec):
                    raise TypeError(
                        f"mcp_tools entry {tool!r} is not a dict with name/spec "
                        "and has no to_codex_mcp_spec() method."
                    )
                name, spec = to_spec()
            lines.append(f"[mcp_servers.{name}]")
            for key, value in spec.items():
                lines.append(f"{key} = {json.dumps(value)}")
            lines.append("")
        return "\n".join(lines)

    def translate_harness_event(self, raw: dict[str, Any]) -> list[Message]:
        """Translate one codex session JSONL line to rollouts Messages.

        Delegates to `codex_message_to_rollouts`, which returns a single
        `Message` or `None`. We normalize to `list[Message]` for Protocol
        uniformity with ClaudeCodeEnvironment.
        """
        msg = codex_message_to_rollouts(raw)
        return [msg] if msg is not None else []

    async def apply_effect(self, effect: Any) -> None:
        # TODO(external-env / codex): fold-contract replay. Same deferral
        # rationale as ClaudeCodeEnvironment.apply_effect.
        raise NotImplementedError(
            "CodexEnvironment.apply_effect: deferred until SessionEntry lands."
        )


# TODO(external-env / opencode): OpencodeEnvironment. Same shape. Defer
# until we actually integrate opencode; stubs without a consumer just bitrot.


# TODO(external-env / harbor): if Harbor's BaseEnvironment becomes a peer of
# these (rather than a provider *of* one of these), it slots in here too.
# More likely: Harbor is the workspace provider (docker/modal/daytona), and
# these external-agent environments wrap a Harbor-provided workspace plus
# claude-code-or-whatever tool-set config. See runtime_refactor.md.
