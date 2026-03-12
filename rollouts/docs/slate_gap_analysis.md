# Slate Gap Analysis For `rollouts` Environments

Goal: compare `rollouts`' current orchestration/handoff environments against the
core Slate primitives in [slate-key-primitives.md](/Users/chiraagbalu/decompilations/slate-key-primitives.md).

Focus:
- [orchestrate.py](/Users/chiraagbalu/research/rollouts/rollouts/environments/orchestrate.py)
- [handoff.py](/Users/chiraagbalu/research/rollouts/rollouts/environments/handoff.py)

## Bottom Line

`rollouts` already has a usable orchestration surface:
- orchestrator tool
- async DSL
- subthreads
- traces/docs
- structured thread results

But it is not Slate-like yet in the places that matter most:
- child threads are not durable/resumable orchestration-native sessions
- capability scoping is intended but not actually enforced
- the episode boundary is much weaker than Slate's
- there is no internal handoff-based compaction system
- the current `handoff` environment is a user-facing session transfer tool, not Slate's synthetic compaction primitive

## Primitive-by-Primitive Comparison

### 1. Orchestrator-first topology

Status: mostly present

What exists:
- `OrchestrateEnvironment` gives the parent agent a single `orchestrate` tool
- orchestration code runs against a `system` object with `thread/query/fromId/allocate/log`
- parallelism uses real `trio` concurrency

What differs from Slate:
- this is an environment plugged into a general `rollouts` runtime, not the default top-level session topology
- nothing forces the overall agent runtime to stay planner-first outside this environment choice

Assessment:
- good enough as a substrate
- not the main blocker

### 2. Durable child threads

Status: missing

What Slate needs:
- child sessions with durable history
- alias-based resumption across orchestration calls
- parent/child identity in storage

Current `rollouts` behavior:
- `system.thread(id, ...)` stores alias results only in `self._results`, an in-memory dict
- `fromId(alias)` only reads that in-memory map
- there is no orchestration-native child-session persistence or alias lookup from session storage

Consequence:
- aliases are only valid within one live orchestration call
- there is no real reusable thread workstream

This is one of the biggest structural gaps.

### 3. Episodes as synchronization artifacts

Status: partial

What exists:
- `ThreadResult(status, output, reason, trace, duration_ms, session_id)`
- `_extract_result()` scans backwards for `complete` / `abort` / `escalate`
- `_generate_transcript()` creates a minimized trace

What is missing relative to Slate:
- no evidence-linked preservation of important tool results
- no material file diff preservation
- no curated tool-response shaping
- no distinction between "full tactical history" and "episode artifact" beyond a simple text minimizer

Current transcript is closer to:
- assistant text
- assistant thinking
- tool-call names/args

It is not yet a strong reusable episode boundary.

### 4. Structured termination

Status: present

What exists:
- subthread system prompt explicitly requires `complete` / `abort` / `escalate`
- `_extract_result()` maps them into `ThreadResult`
- timeout fallback becomes aborted with failure reason

This is one of the places where `rollouts` is already aligned with Slate.

### 5. Context routing by artifact

Status: partial

What exists:
- `traces=` become rendered prior context
- `docs=` become explicit document instructions
- `fromId(alias)` returns prior results within a live orchestration

What is missing:
- durable prior-result lookup from storage
- rich episode artifacts with evidence and file state
- handoff summaries as compressed context nodes

So the routing API shape is good, but the underlying artifacts are still weak.

### 6. Capability-scoped workers

Status: intended, but currently broken

What exists:
- `_capabilities_to_allowed_tools()` computes Claude Code tool allowlists
- `system.thread(..., capabilities=...)` exposes the right conceptual surface

What is wrong:
- `allowed_tools` is computed but never passed to `run_claude()`
- `run_claude()` supports `allowed_tools`, but `orchestrate.py` does not use it

That means subthreads are not actually capability-scoped right now.

This is the clearest low-effort correctness fix.

### 7. Permission gating

Status: missing

Slate behavior:
- tool permissions are runtime-enforced and capability-aware

Current `rollouts` orchestration path:
- Claude Code is invoked with `--dangerously-skip-permissions`
- there is no Slate-like permission or policy layer around subthreads

Consequence:
- capability scoping is the only practical safety boundary here
- because that scoping is currently not enforced, the subthread boundary is weaker than intended

### 8. Shared document store

Status: partial

What exists:
- `DocumentStore` exists in `orchestrate.py`
- `system.allocate(name)` reserves document names
- docs are serialized with the environment

What is missing:
- no real document read/write API on `system`
- no document tool exposed to child threads
- docs are mostly prompt-time instructions, not a strong shared mutable coordination channel

This is weaker than Slate's shared artifact model.

### 9. Handoff-based compaction

Status: missing

Slate behavior:
- synthetic internal `role: "handoff"` checkpoint messages
- inserted automatically on prompt-token growth
- old windows summarized in place
- summaries chained forward

Current `rollouts` `HandoffEnvironment`:
- user-facing `handoff(goal)` tool
- extracts context from current session
- creates a brand new session with a focused prompt

This is not the same primitive.

Important distinction:
- current `handoff.py` is a manual session-transfer/navigation tool
- Slate handoffs are internal compaction checkpoints in the same session history

That means the current handoff environment should not be treated as "already having Slate handoffs".

### 10. Persistence / restore story

Status: mixed, with real gaps

Good:
- the main `rollouts` runtime treats `serialize()` / `deserialize()` seriously
- environment state is serialized before and after tool execution

Bad:
- `HandoffEnvironment.deserialize()` is unimplemented
- `compose.py`'s environment registry does not register `orchestrate` or `handoff`

Consequence:
- the environment protocol is stronger than these two envs currently satisfy
- session restore and composed env restore are incomplete for exactly the envs most relevant to Slate-like behavior

## Quick Wins

These are worth doing first because they improve correctness without requiring a new architecture.

### 1. Actually enforce capability scoping

Pass `allowed_tools` into `run_claude()` from `OrchestrateEnvironment.thread()`.

This is the smallest high-value fix.

### 2. Make `orchestrate` and `handoff` restorable

At minimum:
- implement a restore story for `HandoffEnvironment`
- register both envs in `compose.py`'s registry

Without that, the environment model is inconsistent exactly where orchestration starts to matter.

### 3. Strengthen `ThreadResult.trace`

Improve transcript shaping so it preserves:
- tool responses that matter
- file diffs / created-file contents
- evidence-like references for important outputs

This can happen before building full Slate-style episodes.

## Structural Gaps

These are the changes required to become genuinely Slate-like.

### 1. Durable orchestration-native child sessions

Need:
- child-session creation in storage
- parent/child linkage
- alias lookup from storage
- resume existing child thread by alias across orchestration invocations

### 2. Internal compaction handoffs

Need a new primitive, separate from the current `handoff` tool:
- synthetic checkpoint messages
- token-growth triggers
- summary replacement of old windows
- chained summaries

This likely belongs in the agent/session runtime, not as a normal environment tool.

### 3. Stronger episode artifact

Need a real parent-visible boundary object with:
- status
- output/reason
- curated trace
- preserved important evidence
- preserved material file mutations

### 4. Shared artifact store that threads can actually use

Current docs are mostly prompt text.

To match Slate more closely, the system needs:
- explicit document reads/writes
- durable orchestration-shared artifacts
- a clearer contract for passing those artifacts into later threads

## Recommended Framing

Do not try to reinterpret the existing `HandoffEnvironment` as Slate compaction.

Instead, treat the current repo as having:
- a promising `orchestrate` substrate
- a separate manual session-transfer tool

The clean path is:

1. Fix capability enforcement and restore gaps
2. Strengthen episode shaping
3. Add durable child-session storage/resumption
4. Add a new runtime-level internal handoff compaction mechanism

That preserves the good parts of the current design without confusing two very different meanings of "handoff".
