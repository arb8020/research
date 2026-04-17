# Handoff: Session Decoupling for External Agent Path

## Branch / worktree

```
cd ~/research/rollouts
git worktree add ../rollouts-session-decoupling main
```

Base commit: `4cd182ad` (Remove trajectory adapter signature shim)

---

## Goal

The external agent path (`execute_external_attempt` → `_run_agent_in_workspace`) runs
a CLI agent, collects its output, and returns a completed trajectory. If the harness
crashes mid-run, everything is lost. The native path (`run_agent`) writes every message
to the session store as it arrives — crash recovery is possible because state lives
outside the harness process.

The goal is to give the external path the same guarantee: messages written to the
session store per turn, keyed by a `session_id` the caller controls.

Additionally: the CLI has its own native session ID (the file it writes to at
`~/.claude/projects/.../session.jsonl`). On resume, the harness needs to pass
`--resume <cli_session_id>` to the CLI. That CLI session ID needs to be stored in the
harness session so it's available on wake.

Read the design doc first:
```
~/research/docs/design/agent_environment_decoupling.md
```

---

## Read first: code style

```
~/research/docs/code_style/cheatsheet.md
~/research/docs/code_style/casey_semantic_compression.md
~/research/docs/code_style/keeping_llm_code_honest.md
~/research/docs/code_style/grugbrain_testing.md
```

---

## Read next: the current shape

```
rollouts/rollouts/store.py                           SessionStore protocol, FileSessionStore
rollouts/rollouts/agents/types.py                    RunConfig — has session_store, missing session_id
rollouts/rollouts/agents/runtime.py                  lines 399–401, 533–535, 800–802 — native append_message pattern
rollouts/rollouts/agents/session_runtime.py          ensure_persisted_session — how native path creates sessions
rollouts/rollouts/eval/external_attempts.py          execute_external_attempt — entry point
rollouts/rollouts/eval/remote_runtime.py             _run_agent_in_workspace — the polling loop
```

The polling loop in `_run_agent_in_workspace` already reads turns incrementally via
`_append_session_entries`. Messages go into a local `messages: list`. They need to also
go to `session_store.append_message(session_id, msg)` as they arrive.

---

## Read next: reference

```
~/research/docs/references/ant-managed-agents.md
/tmp/pi-mono/packages/coding-agent/src/core/session-manager.ts
/tmp/pi-mono/packages/coding-agent/src/core/agent-session.ts
/tmp/pi-mono/packages/coding-agent/docs/tree.md
```

Pi-mono has two branching models (`/branch` = forked files, `/tree` = leaf pointer in
one file). Read both before deciding what fork/branch support looks like here.

---

## The three concrete changes

### 1. Add `session_id` to `RunConfig`

```
rollouts/rollouts/agents/types.py
```

Add `session_id: str | None = None` to `RunConfig`. The caller creates the session
before calling `execute_external_attempt` and passes the ID in. The harness only appends
— it never calls `session_store.create()` itself.

This matches the article's model: session is independently addressable, caller controls
its lifecycle. Existing callers that don't set `session_id` are unaffected — no
persistence, same as today.

### 2. Thread `session_id` into `_run_agent_in_workspace` and write per turn

```
rollouts/rollouts/eval/remote_runtime.py    _run_agent_in_workspace
rollouts/rollouts/eval/remote_runtime.py    _append_session_entries (inner function)
```

After `messages.append(msg)` in `_append_session_entries`, call:
```python
store = getattr(run_config, "session_store", None)
sid = getattr(run_config, "session_id", None)
if store is not None and sid is not None:
    await store.append_message(sid, msg)
```

This is async inside an async inner function so it works directly.

### 3. Store the CLI's native session ID in the harness session

The CLI's native `session_id` (the Claude Code or Codex session identifier) is already
extracted from the session file JSONL in `_append_session_entries`:
```python
if session_id is None:
    session_id = entry.get("sessionId")  # claude_code
```

This `session_id` is currently only used for metadata at run-end. It needs to be written
to the harness session so it's available on resume. Use `session_store.update()`:

```python
# Once cli_session_id is first discovered:
if store is not None and sid is not None and cli_session_id is not None:
    await store.update(sid, tags={"cli_session_id": cli_session_id})
```

Check whether `store.update()` supports arbitrary tags/metadata — currently it takes
`environment_state` and `stop_reason`. It may need a `metadata: dict | None = None`
field added to support storing the CLI session ID without polluting `environment_state`.

```
rollouts/rollouts/store.py    SessionStore.update, FileSessionStore.update
```

---

## What `session_id` means for resume

`session_id` is a pointer into the store: load messages, load environment config,
continue. On resume for an external run:

1. Caller loads `session_id` from wherever they persisted it
2. `session_store.get_trajectory(session_id)` → existing messages
3. `session_store.get(session_id)` → environment config → `Environment.deserialize(config)`
4. `session_store.update(session_id, tags=...)` → `cli_session_id` for `--resume`
5. Re-call `execute_external_attempt` with same `session_id` on `run_config`

The harness appends to the existing session rather than creating a new one. The CLI
gets `--resume <cli_session_id>` so it continues from its own checkpoint.

The environment's resumability is determined by `EnvironmentResumeMode` on the stored
`TrajectoryEnvironment`. Environments that can't resume from cold state should raise
`NotImplementedError` in `deserialize` (as `TerminalBenchEnvironment` already does).

---

## Write the failing test first

```
rollouts/tests/unit/test_external_attempts.py
```

The test:
1. Create a `FileSessionStore` in a temp dir
2. Create a session: `session_id = await store.create(...)`
3. Build a `run_config` with `session_store=store, session_id=session_id`
4. Run `execute_external_attempt` with a fake adapter that yields N turns then raises
5. Assert: messages 0..K are in the store before the raise
6. Assert: `store.get_trajectory(session_id)` returns the partial trajectory

This test does not pass today. Writing it first defines the target.

---

## What not to touch

- `run_agent` (native path) — already has session persistence, don't change it
- `SessionStore.create()` — caller responsibility, not harness responsibility
- Fork/tree (`/branch`, `/tree` semantics from pi-mono) — separate work, after this lands
- MCP threading — separate track, after this lands
