# Orchestrate Environment Kinds

Goal: keep the short-term design simple enough to ship, while leaving room for a better abstraction later.

## Core Idea

For `OrchestrateEnvironment`, treat child agents like this:

- anything **not** explicitly named `codex` or `claude_code` is a normal `rollouts` SDK environment by default
- `codex` and `claude_code` are special environment kinds backed by external runtimes/drivers

So the child kind is the top-level selector.

Examples:

```python
await system.thread(..., kind="coding")
await system.thread(..., kind="search")
await system.thread(..., kind="terminal_bench")
await system.thread(..., kind="claude_code")
await system.thread(..., kind="codex")
```

## Why This Is Good Enough For Now

This gives us:
- simple API
- working code first
- one orchestrator UX
- room to normalize all child results into one `ThreadResult`

It avoids prematurely designing a perfect runtime/provider abstraction before we
have enough concrete implementations.

## What `kind` Means

### Native kinds

Kinds like:
- `coding`
- `search`
- `terminal_bench`

should resolve to normal `rollouts` environments and run through the native agent loop.

### Special external kinds

Kinds like:
- `claude_code`
- `codex`

should resolve to external driver-backed child agents.

These are still fine to think of as environment kinds at the orchestration API
level, even if internally they use different execution plumbing.

## Resources Are A Separate Axis

Resources should not be folded into the child kind.

They are orthogonal injectable substrates that either native or external child
agents may use.

Examples:
- `WorktreeResource`
- `LocalDockerSandbox`
- `TerminalTaskResource`

This allows combinations like:

```python
await system.thread(..., kind="coding", resources=[worktree])
await system.thread(..., kind="claude_code", resources=[worktree])
await system.thread(..., kind="terminal_bench", resources=[terminal_task])
```

## Important Observation

This repo already has the beginnings of resource injection:

- [resources.py](/Users/chiraagbalu/research/rollouts/rollouts/environments/resources.py) defines `TerminalTaskResource`
- [terminal_bench.py](/Users/chiraagbalu/research/rollouts/rollouts/environments/terminal_bench.py) is built around an injected task resource
- [swe_grep.py](/Users/chiraagbalu/research/rollouts/rollouts/environments/swe_grep.py) uses injected search behavior
- [git_worktree.py](/Users/chiraagbalu/research/rollouts/rollouts/environments/git_worktree.py) and [sandboxed_worktree.py](/Users/chiraagbalu/research/rollouts/rollouts/environments/sandboxed_worktree.py) already encode isolated execution substrates

So resource injection is not a new concept here. It is already present, just not unified yet.

## Practical Short-Term Rule

For phase 1:

- `kind="claude_code"` -> run Claude Code child
- `kind="codex"` -> run Codex child
- any other `kind` -> build native `rollouts` environment

Optional resources can then be threaded into whichever branch supports them.

That means we can ship useful orchestration quickly without deciding the final
shape of:
- runtime vs provider vs driver
- environment-vs-runtime layering
- global resource registry design

## Terminal Bench

`terminal_bench` should remain a native `rollouts` kind for now.

Reason:
- it depends on a live injected `TerminalTaskResource`
- it is not just a working directory plus tools

So the intended early worker matrix is:

- `coding` -> native `rollouts`
- `search` -> native `rollouts`
- `terminal_bench` -> native `rollouts`
- `claude_code` -> external runtime
- `codex` -> external runtime

## Expected Later Refactor

If this grows successfully, the likely eventual cleanup is:

- keep `kind` as the user-facing API
- internally split kind resolution into:
  - native environment kinds
  - external runtime-backed kinds
  - optional resource injection

But we do not need that full abstraction to start shipping the feature.
