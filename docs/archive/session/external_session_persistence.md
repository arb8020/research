# Handoff: Per-Turn Session Persistence for External Agents

## Branch / worktree

```
cd ~/research/rollouts
git worktree add ../rollouts-external-session main
```

Base commit: `88b71fa` (Split external_attempts.py and add session decoupling groundwork)

---

## Motivation

The native agent path (`run_agent`) writes every message to the session store as it
arrives. If the harness crashes at turn 8 of 20, the first 8 turns are in the store and
the run can resume. This is the "cattle not pets" guarantee: harness processes are
replaceable because state lives outside them.

The external agent path has no equivalent. `execute_external_attempt` runs the CLI,
waits for it to exit, and returns the complete trajectory. If the CLI crashes or the
harness is killed mid-run, the trajectory is lost entirely. There is no resume, no
partial recovery, no record of work done.

The polling loop in `remote_runtime._run_remote_external_runtime_session_file` already
reads turns incrementally as the CLI writes them — it accumulates them into a
`messages: list` that only becomes visible when the function returns. The infrastructure
for per-turn visibility exists; it just doesn't write anywhere.

---

## Read first: code style

```
~/research/docs/code_style/cheatsheet.md
~/research/docs/code_style/casey_semantic_compression.md
~/research/docs/code_style/keeping_llm_code_honest.md
~/research/docs/code_style/grugbrain_testing.md
```

---

## Read next: the gap

The native path — understand how session persistence works when it's done right:

```
rollouts/rollouts/agents/runtime.py         lines 390–410, 530–540, 795–810  (append_message call sites)
rollouts/rollouts/agents/session_runtime.py (ensure_persisted_session, state_to_persisted_trajectory)
rollouts/rollouts/store.py                  (SessionStore protocol, FileSessionStore.append_message)
```

The external path — understand where turns become visible and where they disappear:

```
rollouts/rollouts/eval/remote_runtime.py    _run_remote_external_runtime_session_file (the polling loop)
rollouts/rollouts/eval/external_attempts.py execute_external_attempt (lines 103–160)
rollouts/rollouts/eval/external_attempts.py trajectory_from_claude_code, trajectory_from_codex (local path)
```

The TODO in `runtime.py` above `run_agent` and above the `environment.serialize()` block
describes the convergence target.

---

## Read next: session / store

```
rollouts/rollouts/store.py          FileSessionStore — layout, append_message, create, get
rollouts/rollouts/slice.py          slice/summarize/compact DSL (relevant for resume semantics)
rollouts/rollouts/core/             Trajectory, TrajectorySession, Message
```

Understand: what does `session_id` look like for an external run? The native path creates
a session before the first LLM call. The external path has no equivalent — the CLI owns
its own session file. Are these the same session, or is the harness session a wrapper
around the CLI's native session?

---

## Reference

```
~/research/docs/references/ant-managed-agents.md
```

The article's framing: session, harness, and sandbox should fail and be replaced
independently. Currently the external path bundles all three — CLI crash, harness crash,
and session loss are the same event. The goal is to separate them.

---

## Observations

- `run_config` is already threaded through to the trajectory adapter. `RunConfig` already
  has `session_store` on it (used by the native path). The external path receives
  `run_config` but ignores `session_store`.

- The local trajectory functions (`trajectory_from_claude_code` etc.) go through
  `run_driver_to_trajectory` in `drivers/runner.py`. That function could be extended to
  write to a session store as events arrive — or the session writes could happen at the
  `execute_external_attempt` level. The right boundary is an open question.

- The remote polling loop (`_run_remote_external_runtime_session_file`) reads new JSONL
  lines from the CLI's native session file every `poll_interval` seconds. Each batch of
  lines is currently parsed into messages and appended to a local list. Calling
  `session_store.append_message` here would give per-poll-interval persistence, which
  is close enough to per-turn for most purposes.

- The local path streams events via subprocess pipe rather than polling a file. Per-turn
  persistence on the local path means writing to the store inside `run_driver_to_trajectory`
  or its caller as `StreamEvent`s arrive.

- The CLI's native session file (`~/.claude/projects/.../session.jsonl` for claude_code)
  and the harness session store are separate things. Think about whether they should stay
  separate, or whether the harness should treat the CLI's session file as the source of
  truth and mirror it into `FileSessionStore`.

- Resume semantics: if a run is interrupted at turn K and resumed, the external CLI
  supports `--resume <session_id>`. How the harness session and the CLI session relate
  on resume is the hard part of this work.

- Write the failing test first. The test: run `execute_external_attempt` with a
  `run_config` that has a `session_store`, interrupt it mid-run (fake adapter raises),
  assert messages up to the interruption are in the store. This test does not pass today.
