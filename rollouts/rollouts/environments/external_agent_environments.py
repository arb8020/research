"""External-agent environments — SKELETAL STUBS.

Each external-agent runtime (claude-code, codex, opencode, ...) gets its own
Environment type. This is a sketch, not an implementation. Inline TODOs mark
the work; pass 1 of the session refactor (see
`rollouts/rollouts/agents/runtime_refactor.md`) fleshes them out.

Why these exist:

An `Environment` in the native case is "the agent's tools + the workspace
state those tools manipulate." For external agents (claude-code, codex, ...)
that fuses two distinct things:

  1. The agent's tool vocabulary — what tools the harness exposes to the
     model. For external agents we don't write these tools; the harness ships
     them. We can restrict the set (via `--allowed-tools` or equivalent) and
     we can *augment* the set with our own tools via MCP injection.
  2. The workspace — the directory, container, or sandbox the agent's tools
     manipulate. Same concept as native.

Today these are fused in the native case (one `CodingEnvironment`) and barely
modeled in the external case (adapters take a workspace and launch a binary;
no first-class "what tools did this run have access to" object).

These types make the distinction explicit and give MCP injection a home.

What "Environment" for external agents entails:

  - `workspace`: the resource the agent operates in. Directory on disk,
    container, sandbox. The same `SandboxWorkspaceResource` abstractions we
    already have work here.
  - `allowed_builtin_tools`: which of the harness's built-in tools the agent
    may use. `None` = all defaults; `[]` = none (MCP-only); `["bash", "read"]`
    = restricted subset. Translates to CLI flags at launch.
  - `mcp_tools`: additional tools we inject via MCP so the external harness
    can call tools it wouldn't normally have (e.g., `terminal_bench.tmux`,
    `calculator`, custom eval-specific tools). At launch we write an MCP
    config file the harness reads.
  - Translation boundary: `translate_harness_entry(raw) -> SessionEntry` maps
    harness-native session events into our session entries, so the session
    record is shape-equivalent across runtimes. Today this logic is buried
    inside `_run_agent_in_workspace` polling; it belongs on the environment.
  - Fold contract: given a sequence of our effect entries, reconstruct the
    workspace state. For filesystem-backed agents this is trivial (replay
    bash/write/edit). For environments with out-of-band state (a TerminalBench
    container with running processes), the fold semantics are the same but
    the implementation is subtler.

See `/docs/design/session_ownership.md` for the ownership model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .resources import SandboxWorkspaceResource


# TODO(session-refactor / external-env): define the concrete Protocol that
# external-agent environment types satisfy. Expected members:
#
#   workspace: SandboxWorkspaceResource
#   allowed_builtin_tools: list[str] | None
#   mcp_tools: list[MCPTool]
#
#   def get_tools(self) -> list[Tool]:
#       """Return the effective tool set for this run (restricted builtins
#       + injected MCP tools). Used for UI, logging, analysis."""
#
#   def get_launch_flags(self) -> list[str]:
#       """Return CLI flags the harness launcher needs (e.g., --allowed-tools,
#       --mcp-config). Harness-specific; abstracted here for uniform launch."""
#
#   def translate_harness_event(self, raw: dict) -> SessionEntry | None:
#       """Translate one line from the harness's native session format into
#       our session entry shape. Return None if the event is uninteresting
#       (status pings, etc.)."""
#
#   async def apply_effect(self, effect) -> None:
#       """Apply a recorded effect entry to this workspace. Used by the fold
#       contract for cold-restore / replay."""
#
# For now we sketch it as a Protocol without fleshing out the Tool / MCPTool
# / SessionEntry types — those are part of the same refactor and will land
# in lockstep.


class ExternalAgentEnvironment(Protocol):
    """Protocol for external-agent-runtime-specific environments.

    Each concrete type (ClaudeCodeEnvironment, CodexEnvironment, ...) owns
    the translation between its harness's native session format and our
    session entries, plus the MCP injection story for that harness.

    TODO(session-refactor): flesh this out once SessionEntry / MCPTool /
    Tool land from the runtime refactor. Keeping it minimal so the stubs
    can live alongside the real code as it evolves.
    """

    workspace: SandboxWorkspaceResource
    allowed_builtin_tools: list[str] | None
    mcp_tools: list[Any]  # list[MCPTool] once that type exists


# ── Concrete stubs ──────────────────────────────────────────────────────────


@dataclass
class ClaudeCodeEnvironment:
    """Environment for claude-code CLI runs.

    Claude-code ships built-in tools (bash, read, write, edit, glob, grep,
    web_fetch, ...). This type describes *which* of those the agent may use
    in a given run, and *what additional tools* we inject via MCP.

    Session files live at `~/.claude/projects/<escaped-cwd>/*.jsonl`. The
    translation boundary maps claude-code's JSONL events to our session
    entries.

    TODO(external-env / claude-code): implementation. Currently this is a
    container for configuration fields; nothing consumes it yet.

    Usage sketch (post-refactor):

        env = ClaudeCodeEnvironment(
            workspace=LocalWorkspaceResource.from_existing(dir),
            allowed_builtin_tools=["bash", "read", "write", "edit"],
            mcp_tools=[terminal_bench_tmux_tool()],
        )
        # ... passed into the external-runtime launcher, which consults
        # env.get_launch_flags() to build the CLI invocation, env.mcp_tools
        # to write the MCP config, and env.translate_harness_event() to
        # convert each polled JSONL line into a session entry.
    """

    workspace: SandboxWorkspaceResource

    # `None` = claude-code's full default tool set.
    # `[]` = no built-in tools (MCP-only agent).
    # list = subset enforced via --allowed-tools.
    allowed_builtin_tools: list[str] | None = None

    # Additional tools exposed via MCP. Empty by default.
    mcp_tools: list[Any] = field(default_factory=list)

    # TODO(external-env / claude-code): method stubs below. Each links back
    # to runtime_refactor.md so the intent is discoverable.

    def get_tools(self) -> list[Any]:
        # TODO(external-env / claude-code): return Tool objects for the
        # effective tool set. Compose allowed builtins (with claude-code's
        # schemas) and MCP tools. Useful for UI, for scorers asking "what
        # was available," for test assertions.
        raise NotImplementedError("ClaudeCodeEnvironment.get_tools stub")

    def get_launch_flags(self) -> list[str]:
        # TODO(external-env / claude-code): translate allowed_builtin_tools +
        # mcp_tools into the CLI flags claude-code expects. Roughly:
        #   if allowed_builtin_tools is not None:
        #       flags += ["--allowed-tools", ",".join(allowed_builtin_tools)]
        #   if mcp_tools:
        #       flags += ["--mcp-config", path_to_mcp_config_json]
        raise NotImplementedError("ClaudeCodeEnvironment.get_launch_flags stub")

    def translate_harness_event(self, raw: dict[str, Any]) -> Any | None:
        # TODO(external-env / claude-code): map one line of claude-code's
        # JSONL to one of our session entries (AssistantTurn, ToolCall,
        # ToolResult, ...). Today this translation logic is buried in
        # _ClaudeEventParser in drivers/claude.py; we want it to live here
        # so the Environment owns its harness's format.
        raise NotImplementedError("ClaudeCodeEnvironment.translate_harness_event stub")

    async def apply_effect(self, effect: Any) -> None:
        # TODO(external-env / claude-code): replay a recorded effect against
        # this workspace. For filesystem-backed agents this is "run the
        # recorded bash/write/edit call." For environments with side channels
        # (a custom MCP tool), this needs to invoke the tool's apply method.
        raise NotImplementedError("ClaudeCodeEnvironment.apply_effect stub")


@dataclass
class CodexEnvironment:
    """Environment for OpenAI codex CLI runs.

    Codex ships its own tool vocabulary, sandbox modes, and session format.
    Session files live at `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`.

    TODO(external-env / codex): same shape as ClaudeCodeEnvironment but with
    codex-specific flags and translation. Currently a stub.
    """

    workspace: SandboxWorkspaceResource

    allowed_builtin_tools: list[str] | None = None
    mcp_tools: list[Any] = field(default_factory=list)

    # Codex-specific knobs.
    sandbox_mode: str = "read-only"  # codex's own --sandbox flag

    def get_tools(self) -> list[Any]:
        raise NotImplementedError("CodexEnvironment.get_tools stub")

    def get_launch_flags(self) -> list[str]:
        raise NotImplementedError("CodexEnvironment.get_launch_flags stub")

    def translate_harness_event(self, raw: dict[str, Any]) -> Any | None:
        raise NotImplementedError("CodexEnvironment.translate_harness_event stub")

    async def apply_effect(self, effect: Any) -> None:
        raise NotImplementedError("CodexEnvironment.apply_effect stub")


# TODO(external-env / opencode): OpencodeEnvironment. Same shape. Defer
# until we actually integrate opencode; stubs without a consumer just bitrot.


# TODO(external-env / harbor): if Harbor's BaseEnvironment becomes a peer of
# these (rather than a provider *of* one of these), it slots in here too.
# More likely: Harbor is the workspace provider (docker/modal/daytona), and
# these external-agent environments wrap a Harbor-provided workspace plus
# claude-code-or-whatever tool-set config. See runtime_refactor.md.
