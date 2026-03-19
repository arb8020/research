# Eval / Interactive Data Cleanup

## Context

We do **not** currently have two fundamentally different products here.

- interactive sessions are agent runs where a human may:
  - provide the next input
  - reject or modify tool execution
  - inject messages or slash-command-driven control flow
- eval runs are agent runs where the harness provides:
  - the initial problem/input
  - the next input policy (usually none after the initial prompt)
  - scoring

The underlying runtime shape is the same:

```text
problem/input + endpoint/runtime + optional environment
-> live agent execution
-> durable trajectory/session state
-> optional scoring / frontend projections / exports
```

So the cleanup problem is not "unify two products".

The cleanup problem is:

> one logical run/checkpoint is currently projected into too many partially-overlapping
> types and artifacts, so it is unclear what the canonical durable state actually is.

## Existing Relevant Types

### Live runtime state

`AgentState` in `rollouts/agents/types.py` is the in-memory execution state:

- `actor.trajectory`
- `environment`
- `stop`
- `error`
- `turn_idx`
- `pending_tool_calls`
- `session_id`
- `parent_session_id`
- `branch_point`
- `confirm_tools`
- `driver_session_id`

This is the honest live state machine.

### Durable core state

`Trajectory` in `rollouts/dtypes.py` is already close to the canonical durable type:

- `messages`
- `completions`
- `metadata`
- `session`
- `environment`

Nested bundles already exist:

- `TrajectorySession`
- `TrajectoryEnvironment`

This is important: we are not starting from zero. The codebase already moved in the
right direction.

### Session control state

We explicitly dropped the old durable control sidecar for now:

- no persisted `pending_input`
- no persisted `queued_messages`
- no `--send` queue-and-resume flow

Detached sessions now stop in a resumable `pending` state and are resumed explicitly
by re-attaching or re-running with the session id.

### Eval wrapper state

`AttemptRow` in `rollouts/training/types.py` wraps:

- `problem`
- `trajectory`
- `environment_state`
- `score`
- `reward`
- `status`
- `metadata`
- training-related fields

This is useful, but it is more eval/training-colored than the core runtime/session model.

## Current Mess

### 1. Competing sources of truth

Several things can feel like the "real" persisted run:

- `Trajectory`
- historical session-sidecar machinery
- `AttemptRow`
- `samples/*.json`
- `report.json`

That is too many denotations for one logical thing.

### 2. Historical control state was an ad hoc sidecar

The old durable "waiting for input" behavior was implemented as a sidecar rather than
as an honest part of the checkpoint model.

We removed that path for now instead of preserving a misleading split state model.

### 3. Eval persistence and interactive persistence diverge

Interactive/session persistence goes through `rollouts/store.py`.

Eval persistence goes through `rollouts/eval/native.py` and writes:

- `events.jsonl`
- `samples/{sample_id}.json`
- `report.json`

These represent the same underlying run family but use different persistence stories.

### 4. `AttemptRow` mixes core state with derived annotation

`AttemptRow` currently contains:

- core execution result fields
- eval bookkeeping
- score/reward
- training-specific fields

This makes it hard to tell which fields are canonical and which are derived.

### 5. Observations are fragmented

Execution-time facts currently live in several places:

- `Trajectory.completions`
- `events.jsonl`
- per-sample JSONL logs
- request spans (`spans.jsonl`)
- workspace snapshots or reconstructed workspace state

These are all observations, but there is no clean boundary around them yet.

### 6. Frontend types are artifact-shaped adapters

The frontend consumes:

- run list items
- run reports
- trace samples
- live run state
- workspace data

but those are currently built by adapting multiple artifact formats, not by reading a
single clean domain model.

### 7. Lifecycle state is represented in too many places

Lifecycle concepts currently appear in:

- `AgentState.stop/error`
- `TrajectorySession.stop_reason`
- `AttemptRow.status`
- live frontend run status

These should be related deliberately, not drift independently.

### 8. Logs are mixed conceptually with durable state

`events.jsonl` is useful telemetry.

It should not be treated as canonical state.

Likewise frontend payloads and summary artifacts are projections, not truth.

## Working Model

The current code strongly suggests this hierarchy:

### 1. `AgentState`

Ephemeral live execution state.

### 2. `Trajectory`

Canonical durable cross-mode state.

This should be the main thing that can be serialized/deserialized across:

- interactive sessions
- eval attempts
- driver swaps
- resume / attach flows

### 3. Session control adjuncts

Currently none.

If we later need richer waiting/detached resume semantics, we should add an explicit
control-state product type rather than resurrecting ad hoc sidecars.

### 4. Eval/training wrapper

`AttemptRow` should likely become a thin wrapper around the canonical durable state plus:

- normalized problem row
- scoring outputs
- training-specific annotations

## Progress So Far

We already simplified the session/checkpoint side substantially:

- removed durable waiting / queued-input sidecars
- removed `SessionHandle`
- removed `SessionStatus` as a persisted type
- removed `TrajectoryAnnotations` and trajectory-owned reward/group/replica/advantage fields
- made `Trajectory` the canonical persisted session/checkpoint type

We also made the first eval-side denotational cut:

- introduced `AttemptResult` as the canonical execution result
- introduced `AttemptEvaluation` as the derived scoring payload
- kept `AttemptRow` only as a training/scoring bridge for now
- changed `rollouts/eval/native.py` to persist sample artifacts from `AttemptResult`
- kept the frontend server backward-compatible by flattening `AttemptResult` + `AttemptEvaluation` into the existing UI DTO shape at read time
- removed write-time `samples/{sample_id}.jsonl`; `events.jsonl` is now the only canonical operational event stream
- removed write-time `trajectories/{sample_id}.jsonl`; `samples/{sample_id}.json` now stores the full canonical attempt, including trajectory
- added write-time `report.html` and `samples/{sample_id}.html` as the default static share surface over the canonical JSON artifacts

So the current intended staging is:

```text
ProblemRow + Trajectory + environment_state + status + metadata
-> AttemptResult
-> optional AttemptEvaluation
-> optional AttemptRow / TrainingSample bridge for training code
```

This is not the end state, but it is a materially more honest split than the old
single `AttemptRow` pretending to be the canonical eval object.

### 5. Projections

Everything else should be derived:

- summary artifacts
- frontend payloads
- logs
- exports

## Cleanup Direction

### Principle 1: `Trajectory` is the durable core

We should stop letting multiple artifact types compete with `Trajectory` as the main
persisted run/checkpoint object.

Corollary:

- `Trajectory` should own durable execution/checkpoint facts
- derived eval/training outcome fields should live on `AttemptRow` and related wrappers
- `metadata` can remain as a flexible bag for now, but if it starts absorbing more
  derived semantics or cross-cutting concerns, it should be split into named fields
  instead of becoming a junk drawer

### Principle 2: add control state back only if it earns its place

If we later reintroduce waiting/queued resume semantics, they need to be modeled as an
explicit first-class control-state product type, not incidental sidecar JSON.

### Principle 3: `AttemptRow` is not the core runtime type

`AttemptRow` is useful, but it should not own the primary persistence story.

It should wrap or reference canonical trajectory/session state for eval/training use.

### Principle 4: observations are not the checkpoint

Keep a clear distinction between:

- checkpoint/durable resumable state
- observations/telemetry
- annotations/scoring
- UI/report/export projections

### Principle 5: environments own their serialized state

`TrajectoryEnvironment` is already on the right track:

- `kind`
- `config`
- `state`
- `resume_mode`

Environment internals should remain environment-owned. The core should only own the envelope.

## Progress So Far

### Landed

- removed the dead `evals_ui` path and standardized on `rollouts/frontend`
- removed the durable waiting/input queue path:
  - no persisted `pending_input`
  - no persisted `queued_messages`
  - no `--send` / `--send-file` resume flow
- removed `StopReason.NEEDS_INPUT`
- removed `SessionHandle` from live code
- removed `TrajectoryAnnotations`
- removed rollout/training scalar fields from `Trajectory`:
  - `rewards`
  - `group`
  - `replica`
  - `advantages`
- removed persisted `SessionStatus` as a first-class type
- made `TrajectorySession.stop_reason` the stored lifecycle fact
- made session/list/export status a derived string label instead of persisted truth
- normalized historical `status` / `"waiting"` session records into derived state on read
- made `Trajectory` the loaded/persisted session object throughout:
  - store load/save/create APIs
  - interactive runner session switching
  - slash-command session forks
  - slice/export/handoff/agent-trace paths
- moved the small session convenience surface onto `Trajectory` itself:
  - `session_id`
  - `parent_id`
  - `branch_point`
  - `endpoint`
  - `status`
  - `tags`
  - `created_at`
  - `updated_at`
  - `vcs`
  - `environment_config()`
  - `environment_state()`

### Consequences

- there is now one full durable session/checkpoint type in live code: `Trajectory`
- `SessionSummary` remains as a list/index projection
- detached resume is explicit by session id, not queue-driven
- `Trajectory` now carries checkpoint facts only; score/reward/grouping data no longer
  lives on the canonical durable type
- most remaining cleanup pressure is no longer in session wrappers; it is in:
  - `AttemptRow`
  - eval artifact fanout
  - frontend projection contracts

## Immediate Cleanup Passes

### Pass 1: Make the canonical durable state explicit

Document and enforce that the canonical persisted run state is:

- `Trajectory`
- plus explicit control-plane adjuncts needed for resume

Questions for this pass:

- Is `PendingInput` truly outside `Trajectory`, or should it become part of a nested
  session/control bundle?

### Pass 2: Demote eval artifacts to projections

Be explicit that:

- `report.json`
- `samples/*.json`
- `events.jsonl`

are projections/materializations, not the canonical source of a run.

### Pass 3: Thin `AttemptRow`

Clarify that `AttemptRow` is:

- problem/input
- canonical durable run state
- derived score/reward/training fields

not an independent competing persistence object.

### Pass 4: Define frontend-facing domain types

The backend should serve clean types derived from canonical state, rather than leaking
artifact layout and compatibility hacks into the frontend contract.

### Pass 5: Revisit storage backend only after the above

Only after the ownership/staging story is honest should we decide whether SQLite should
back:

- canonical trajectory/session state
- indexes/search
- projections
- or only metadata/query surfaces

Otherwise we will just pour the current mess into a database.

## Non-Goals For This Refactor

- inventing a totally new ontology when existing types already mostly capture the shape
- forcing all environments into a fake generic state schema
- making logs or frontend payloads canonical
- deciding the final storage technology before the data model is honest

## Next Step

Start with the smallest honest cleanup:

1. keep `Trajectory` as the only canonical persisted resume payload for now
2. make eval artifacts explicitly derived from that canonical state
3. only reintroduce extra control-plane state if a concrete workflow needs it
