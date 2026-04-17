# Agent / Environment Decoupling

## The model

Three things, each with a distinct lifecycle and responsibility:

**Agent** — stateless process. Input is message history + tool definitions. Output is
text + tool calls. Doesn't care where tool results come from or where it's running.
Between turns it has no state except the message history.

**Environment** — owns a workspace resource. Durable state lives in the workspace
(files, containers, whatever), not in memory. Can reconstruct what happened to it from
the workspace and session log on revival. Exposes tools to the agent.

**Harness** — coordinates the two. Reads agent output, fires events at environment,
appends returned messages to the session store, feeds messages back to agent, loops.
On crash + revival: rehydrates from session store + reprovisioned environment.

**Session store** — append-only log. Source of truth for what the agent said and what
came back. Independent of both agent and environment.

---

## The event model

The environment is an event handler:

```
on_event(event) -> Message | None
```

Events: `tool_call`, `assistant_message`, `session_start`, `run_end`.

The harness fires events, environment returns messages, harness appends them to the
session store and feeds them back to the agent. The environment never touches agent
state directly.

`exec_tool` already fits this — returns a `ToolResult` which becomes a message. The
only thing wrong today is the result goes directly into the agent loop instead of
through the session store first.

`on_assistant_message` doesn't fit — it currently returns a mutated `AgentState` instead
of a `Message | None`. That's the tight coupling. Fix: return `Message | None`, harness
handles appending and feeding back, same as tool results.

Default behavior:
- trigger = `tool_call`, append = tool result  (this is `exec_tool` today)
- `on_assistant_message` is a special case of the same pattern

---

## Environment protocol (target)

```python
class Environment(Protocol):
    # Lifecycle
    async def initialize(session_id: str | None = None) -> None
    async def close() -> None

    # Tool contract
    def get_tools() -> list[Tool]
    async def on_event(event: EnvironmentEvent) -> Message | None

    # Durability
    async def serialize() -> dict
    @staticmethod
    async def deserialize(data: dict) -> Environment

    # Optional: for external agents
    def get_mcp_servers() -> list[MCPServerConfig] | None
```

`on_assistant_message` goes away as a separate method. It becomes an event type that
`on_event` handles.

---

## Workspace resource

The environment holds a reference to a `WorkspaceResource`. The resource lifecycle is
managed externally (injected into the environment). The environment can request
start/revive but doesn't own it.

`WorkspaceResource` is the thing that knows how to:
- exec a command (local: subprocess, remote: SSH/Docker/Modal/bifrost)
- read a file at an offset (local: open(), remote: download_bytes)
- write a file (local: write(), remote: upload)

The existing `SandboxWorkspaceResource` / `RemoteSession` in `resources.py` is close
to this already. It needs `exec_background` (launch and detach) added. The blocking
`exec` stays for setup commands.

Local gets a `LocalWorkspaceResource` implementing the same protocol — subprocess for
exec, filesystem for read/write. Then the harness has one code path for both topologies.

---

## The harness loop (target)

Same loop for native and external agents:

```python
async def run(agent, environment, session_store, session_id):
    await environment.initialize(session_id)

    while True:
        # 1. Get next agent output (the only difference between native and external)
        output = await agent.next_turn(history)

        # 2. Append agent message to session store
        await session_store.append_message(session_id, output.message)

        # 3. Fire events at environment, collect messages
        for tool_call in output.tool_calls:
            result = await environment.on_event(ToolCallEvent(tool_call))
            if result:
                await session_store.append_message(session_id, result)
                history.append(result)

        # 4. Check stop condition
        if output.stop:
            break

    await environment.close()
```

The only difference between native and external is step 1:
- Native: calls SDK/provider directly, gets StreamEvents
- External: reads from pipe or polls session file the CLI writes

Everything else — session persistence, environment events, stop conditions — is identical.

---

## Native agent observation (future direction)

The native rollouts agent already has `HeadlessJsonFrontend` — bidirectional NDJSON
over stdin/stdout, same wire format as Claude Code CLI. If native agent runs go through
this frontend, then native and external become the same from the harness perspective:
both are processes that speak NDJSON. The harness observation loop is identical.

This is a bigger move — not required for the near-term work but worth designing toward.
The implication: "native agent" is just an external agent where the CLI happens to be
the rollouts agent itself.

---

## Current problems (concrete, in priority order)

### Bug 1: environment discarded before CLI launch

`execute_external_attempt` does `del environment`. MCP servers, workspace state, all
of it gone before the CLI starts. Fix: keep environment alive, call
`get_mcp_servers()`, write `--mcp-config`, thread environment through to launch.

### Bug 2: no per-turn persistence on external path

Polling loop in `_run_remote_external_runtime_session_file` reads turns incrementally
but throws them away. Fix: call `session_store.append_message` inside the loop.

### Structural problem: parallel function hierarchies

`trajectory_from_claude_code` (local) and `trajectory_from_remote_claude_code` (remote)
are the same function split across topology. Every new capability added twice.
Root cause: topology encoded in the function signature (`cwd: Path` vs
`SandboxWorkspaceResource`). Fix: environment owns the workspace resource, trajectory
function takes `environment`, topology is inside the environment.

### Structural problem: `on_assistant_message` mutates agent state

Environment returns mutated `AgentState` instead of `Message | None`. Breaks the
"environment produces messages, harness appends them" model. Fix: change signature to
return `Message | None`.

### Dead code to delete

- `CodexEnvironment`, `ClaudeCodeEnvironment` — external agent harnesses pretending to
  be environments. Only referenced in live tests, not in any production path.
  `execute_external_attempt` is the real path.
- `EnvironmentResumeMode` / `resume_mode` / `_env_ref` — declared, never consumed,
  `_env_ref` never set. Delete and replace with: environment's `deserialize()` is the
  single construction path; environments that can't cold-resume raise `NotImplementedError`
  (as `TerminalBenchEnvironment` already does honestly).
- `ProjectedTrajectoryAdapter` / `RemoteTrajectoryAdapter` type aliases — gone once
  topology collapses into environment.
- `_trajectory_adapter_accepts_run_config` — runtime signature inspection, symptom of
  inconsistent adapter API. Gone once adapter signature is unified.

---

## Sequencing

These are in dependency order, not time order. Each step leaves the codebase in a
working state.

**Step 1: Delete dead code**
Remove `CodexEnvironment`, `ClaudeCodeEnvironment`, `EnvironmentResumeMode`,
`resume_mode`, `_env_ref`. Tests that reference them: only live tests, move to
`execute_external_attempt` pattern. Mechanical, no design decisions.

**Step 2: Fix Bug 1 (MCP threading)**
Add `get_mcp_servers()` to Environment protocol (optional method). Thread environment
through `execute_external_attempt` instead of deleting it. Write MCP config file,
pass `--mcp-config` to CLI. No topology changes needed — works with current two-adapter
structure.

**Step 3: Fix Bug 2 (per-turn persistence)**
Thread `session_store` + `session_id` into polling loop. Call `append_message` per
turn. Works with current structure, no topology changes needed.

**Step 4: Add `LocalWorkspaceResource`**
Implement `WorkspaceResource` protocol for local filesystem — subprocess exec,
filesystem read/write. Same interface as `SandboxWorkspaceResource`. Add
`exec_background` to both.

**Step 5: Topology collapse**
Environment holds `WorkspaceResource`. Trajectory functions take `environment` instead
of `cwd: Path` or `workspace: SandboxWorkspaceResource`. Local and remote collapse to
one function per runtime. `ProjectedTrajectoryAdapter` / `RemoteTrajectoryAdapter` gone.

**Step 6: Fix `on_assistant_message`**
Change signature from returning `AgentState` to returning `Message | None`. Harness
handles appending. Unifies native and external event model.

**Step 7: Unify harness loop (optional, bigger)**
Native path runs through same loop as external. Possibly via `HeadlessJsonFrontend`
for native agent. Native and external become the same from harness perspective.

---

## What to read

```
rollouts/rollouts/eval/external_attempts.py     current external path entry point
rollouts/rollouts/eval/remote_runtime.py        remote execution + polling loop
rollouts/rollouts/agents/runtime.py             native harness loop
rollouts/rollouts/dtypes.py                     Environment protocol, TrajectoryEnvironment
rollouts/rollouts/environments/resources.py     WorkspaceResource / RemoteSession protocol
rollouts/rollouts/frontends/headless_json.py    NDJSON frontend (step 7 reference)
~/research/docs/references/ant-managed-agents.md
~/research/docs/code_style/
```
