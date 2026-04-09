# Handoff: Session Decoupling + External Agent MCP Threading

## What this is

Two related tracks of work that share a root cause: the external agent execution path
(`execute_external_attempt`) discards the environment before launching the CLI, so external
agents have no access to environment capabilities and no session persistence during runs.

The goal is not a specific implementation — read the reference material below, understand
the current shape, and develop your own approach. The design space is genuinely open.

---

## Branch / worktree

Work off `main` in `~/research/rollouts`. The relevant unstaged changes are already on
`main` (not on a feature branch). Create a worktree:

```
cd ~/research/rollouts
git worktree add ../rollouts-session-decoupling main
```

The files modified so far (unstaged):

```
rollouts/agents/runtime.py
rollouts/agents/session_runtime.py
rollouts/drivers/runner.py
rollouts/environments/resources.py
rollouts/eval/external_attempts.py
tests/unit/test_external_attempt_factory.py
tests/unit/test_external_attempts.py
```

New untracked files that are part of this work (the external_attempts split):

```
rollouts/eval/remote_runtime.py
rollouts/eval/openhands.py
rollouts/eval/mini_swe_agent.py
rollouts/eval/types.py
```

There are also backup files (`*.backup`, `*.bak2`, `*.bak3`) to delete.

---

## Read first: code style

These are not suggestions. Read them before writing anything.

```
~/research/docs/code_style/cheatsheet.md
~/research/docs/code_style/casey_semantic_compression.md
~/research/docs/code_style/keeping_llm_code_honest.md
~/research/docs/code_style/grugbrain_testing.md
```

Key principles that apply directly to this work:

- Assert invariants, raise at boundaries, tuple operational failures. Don't launder bugs
  into control flow.
- Write usage code first. State assumptions before implementing.
- Integration tests over unit tests. Test the real thing, not a mock.
- Friction is feedback. If the signature is awkward, the boundary is probably wrong.

---

## Read next: the problem

The TODOs in these files are the spec. Read them in order:

```
rollouts/agents/session_runtime.py   — TODO above environment_to_session_config
rollouts/agents/runtime.py           — TODO above run_agent, TODO above environment.serialize()
rollouts/eval/external_attempts.py   — TODO above ProjectedTrajectoryAdapter, TODO above del environment
```

Then read the reference article:

```
~/research/docs/references/ant-managed-agents.md
```

The article's central claim: session, harness, and sandbox should fail and be replaced
independently. A harness crash should not lose the session. The sandbox topology (local
vs remote) should not leak into the session or the harness logic.

---

## Read next: the current code shape

### External agent execution path

```
rollouts/eval/external_attempts.py      — orchestration + local driver wrappers (~700 lines)
rollouts/eval/remote_runtime.py         — remote sandbox execution (~1350 lines)
rollouts/eval/openhands.py              — openhands runner + output parsers
rollouts/eval/mini_swe_agent.py         — mini-swe-agent runner + output parser
rollouts/eval/types.py                  — ExternalAttemptArtifact (shared leaf type)
```

This was recently split from a single 2400-line file. The split is mechanical — the
structure within each file has not been cleaned up yet.

The key topology problem: `ProjectedTrajectoryAdapter` (local, takes `cwd: Path`) and
`RemoteTrajectoryAdapter` (remote, takes `SandboxWorkspaceResource`) encode topology in
the type signature. The harness branches on topology at the call site. Harbor's approach
is instructive — see `/tmp/harbor/src/harbor/environments/base.py` and
`/tmp/harbor/agents/trial.py` for how they handle `is_mounted` on the environment object
instead.

### Session store and slice machinery

```
rollouts/store.py           — SessionStore protocol + FileSessionStore
rollouts/slice.py           — slice/summarize/compact/inject DSL for trajectories
rollouts/session_index.py   — session index / search
rollouts/core/              — Trajectory, Message, TrajectorySession, EnvironmentConfig
```

`FileSessionStore` has `create(parent_id, branch_point)` on the protocol but no fork
operation. Sessions are separate files with `parent_id` pointers — the "forked files"
model.

### Drivers

```
rollouts/drivers/runner.py      — ExternalAgentDriver protocol, run_driver_to_trajectory,
                                  _make_raw_driver_line_handler, _make_external_progress_emitter
rollouts/drivers/claude.py      — ClaudeDriver (spawns claude CLI, streams stdout pipe)
rollouts/drivers/codex.py       — CodexDriver
rollouts/drivers/acp.py         — ACPDriver (local ACP protocol, _ACPEventBridge)
```

The drivers speak `StreamEvent`. `run_driver_to_trajectory` accumulates them into a
`Trajectory`. The remote execution path in `remote_runtime.py` bypasses the driver layer
entirely — it launches the CLI in a container and polls a session file. That polling loop
and the driver layer are the same concept at different topologies.

### Bifrost / sandbox interface

```
rollouts/environments/resources.py                  — RemoteSession protocol, SandboxWorkspaceResource
rollouts/environments/bifrost_workspace_resource.py — concrete bifrost implementation
```

`RemoteSession.exec` is a blocking call returning `(returncode, stdout, stderr)`. There
is no streaming exec today. The TODO added in this work explains why and what the fix
would look like. The polling loop in `remote_runtime._run_remote_external_runtime_session_file`
is the current workaround: launch CLI in background, poll the session file it writes.

---

## Read next: reference implementations

### pi-mono (TypeScript, coding agent)

```
/tmp/pi-mono/packages/coding-agent/src/core/session-manager.ts
/tmp/pi-mono/packages/coding-agent/src/core/agent-session.ts
/tmp/pi-mono/packages/coding-agent/docs/tree.md
```

Two different branching models are implemented here:

**`/branch` (fork model)** — `createBranchedSessionFromEntries(entries, branchBeforeIndex)`
copies entries up to the branch point into a new session file with a new ID and a
`branchedFrom` pointer. Simple, each branch is an independent file.

**`/tree` (leaf pointer model)** — every entry has an `id` and `parentId`. A `leaf`
pointer tracks current position. Navigating the tree moves the leaf; `buildSessionContext()`
reconstructs the linear message list by walking from leaf to root. No copying. Branches
live in one file. Abandoned branches can be summarized inline as `BranchSummaryEntry`.
The common ancestor between two positions is computed to summarize only the delta.

The `/tree` model is materially more powerful for the orchestration use case: if the
harness explores N approaches from the same branch point, they share one session file,
the tree is queryable, and summaries of abandoned paths are preserved. But it requires
restructuring how entries are stored — every entry needs `id`/`parentId` rather than
just a flat append-only list.

### Harbor (Python, multi-backend agent runner)

```
/tmp/harbor/src/harbor/environments/base.py
/tmp/harbor/src/harbor/agents/trial.py
/tmp/harbor/src/harbor/agents/claude_code.py
```

Harbor handles topology via `is_mounted` on the environment object — one execution path,
one trajectory function per agent, with a single branch at file I/O time. Compare to
rollouts' current pattern of `trajectory_from_claude_code` +
`trajectory_from_remote_claude_code` as completely separate functions.

---

## The open design questions

These are genuinely open. Don't treat the list above as a prescribed implementation.

**1. Topology collapse**

The `ProjectedTrajectoryAdapter` / `RemoteTrajectoryAdapter` split means
`trajectory_from_claude_code` and `trajectory_from_remote_claude_code` are maintained
separately despite doing the same thing. The right seam — whether it's `is_mounted` on
environment, a streaming exec on `RemoteSession`, or something else — is up to you.
A local `launch-and-poll` mode for `ClaudeDriver` (using the native session file the CLI
writes locally) would make local and remote execution structurally identical.

**2. MCP threading**

`Environment.get_mcp_servers()` exists on some environments. The external agent path
currently does `del environment` before launching. The right fix is: environment stays
alive, its MCP servers are written to a temp JSON file, `--mcp-config <file>` is passed
to the CLI. This is the mechanism by which external agents get access to arbitrary
environments without any Python-level coupling.

**3. Per-turn session persistence on the external path**

The native path (`run_agent`) writes to the session store per turn via `append_message`.
The external path writes nothing until the run completes (or crashes and loses everything).
The polling loop in `_run_remote_external_runtime_session_file` already reads turns
incrementally — it just doesn't write them anywhere. Threading `session_id` through and
calling `session_store.append_message` inside the poll loop is the mechanical fix. The
interesting question is what the right session structure looks like for a run that might
be resumed.

**4. Fork / tree**

`FileSessionStore` has no fork. The pi-mono reference shows two models: forked files
(`/branch`) and leaf-pointer-in-one-file (`/tree`). The leaf pointer model is better for
orchestration but requires changing the entry format. Think about which model serves the
actual use cases before committing to either.

---

## Tests to write before implementing

The grugbrain doc is explicit: write tests that fail against the current code and define
the target behavior. The gaps:

- External path per-turn persistence: run `execute_external_attempt` with a
  `session_store` on `run_config`, assert messages appear in the store before the
  function returns (i.e., during the run, not just at end). This test fails today.

- MCP threading: assert the MCP config file passed to the CLI contains the environment's
  servers. Also fails today (`del environment` means there's nothing to check).

- Crash recovery: partially complete an external run (fake adapter raises mid-stream),
  assert the partial trajectory is in the session store and `resume_session` picks it up.

- Fork/tree (whichever model): create a session, add N messages, fork/navigate to
  message K, assert the resulting context contains exactly messages 0..K and that the
  parent pointer is correct.

---

## What not to touch

- `charisma/` — PR #21 is in review
- `eval/`, `rl/`, `opt/` subdirectory configs in `charisma-exp` — untracked WIP
- `original_performance/` base config — has its own `WorkspaceEnvironmentConfig`
- The generated Python-in-a-string in `remote_runtime._run_remote_acp_runtime` is known
  gross but is not in scope here — it should be a real `.py` file shipped as a resource,
  but that's a separate cleanup

---

## Suggested reading order

1. Code style docs (`~/research/docs/code_style/`)
2. TODOs in `session_runtime.py`, `runtime.py`, `external_attempts.py`
3. `ant-managed-agents.md`
4. `store.py`, `slice.py`
5. `external_attempts.py`, `remote_runtime.py`
6. `drivers/runner.py`, `drivers/claude.py`, `drivers/acp.py`
7. `environments/resources.py`, `environments/bifrost_workspace_resource.py`
8. pi-mono: `session-manager.ts`, `agent-session.ts`, `docs/tree.md`
9. Harbor: `environments/base.py`, `agents/trial.py`
