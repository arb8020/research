# Trajectory / Session Refactor

## Context

The current agent stack has two overlapping persistence stories:

- `Trajectory` in `rollouts/dtypes.py` is the runtime/training trace type.
- `AgentSession` in `rollouts/dtypes.py` was the persisted session wrapper used by the CLI/TUI.

That split creates a lossy boundary:

- `Trajectory` carries `completions`, `metadata`, and rollout/training fields.
- The old `AgentSession` carried session identity, branching, status, endpoint, environment config, and env state.
- Session resume reconstructs `Trajectory(messages=session.messages)`, so persisted sessions are not a faithful `Trajectory`.

This document proposes the direction for a refactor where `Trajectory` becomes the canonical overcomplete type that live agent work, session persistence, and training/export can all operate over.

Current implementation direction:

- `Trajectory` is the canonical durable session object.
- full session loads use `Trajectory` directly.
- `SessionSummary` is the cheap list/index read model.
- `AgentSession` has been removed from live code. Historical docs may still mention it.

This follows the existing "overcomplete type" direction in `docs/design/unified_sample_type.md`, but moves the same idea down one level to the trace/session layer.

## Goal

Make one canonical trace type that is close enough to runtime, session persistence, and training that moving between them is a shallow transformation instead of a lossy conversion.

Desired property:

```python
live agent work -> canonical trajectory -> training sample / session resume / trace export
```

Not:

```python
live state -> session wrapper -> partial trajectory -> training sample
```

## Non-Goals

- Rewrite the whole agent loop in one pass.
- Finalize every field name up front.
- Decide now whether the store should be event-sourced forever.
- Unify provider/endpoint internals as part of this document.

## Relevant Style Constraints

From `~/research/docs/code_style/cheatsheet.md`:

- UIs are thin wrappers over data.
- Push ifs up, fors down.
- Explicit > clever.
- Make code usable before reusable.
- Boring technology.

From `~/research/docs/code_style/sean_goedecke_system_design.md`:

- Minimize stateful components.
- Keep one component owning the state.

Applied here:

- The TUI should not reconstruct business state from logs and side effects.
- The store should persist one canonical trace/session record, not a parallel lossy model.
- Runtime transitions should have one owner.

## What We Know From Current Code

### `Trajectory` is already a real hot-path type

`Trajectory` currently contains:

- `completions`
- `messages`
- `rewards`
- `group`
- `replica`
- `advantages`
- `metadata`

Important current uses:

- `messages` is the main transcript everywhere.
- `completions` is not decorative. Training code uses it for token extraction and rollout logprobs in `rollouts/training/agent_integration.py`.
- `metadata` is used by some environments to carry task-specific information.

So `Trajectory` is already more than "just chat history".

### `AgentSession` was a separate persistence envelope

`AgentSession` currently adds fields that `Trajectory` does not have:

- `session_id`
- `parent_id`
- `branch_point`
- `endpoint`
- `environment`
- `environment_state`
- `status`
- `reward`
- `tags`
- `created_at`
- `updated_at`
- `vcs`

This made `AgentSession` useful for resume and CLI session management, but it also means:

- sessions preserve `messages` but not full `Trajectory`
- session persistence and runtime trace are different concepts in code
- the store API was forced to talk in `AgentSession`, while most execution code talks in `Trajectory`

### Session/runtime ownership is split

Today state ownership is spread across:

- `cli.py`: resume vs fork, attach/send/status/ls orchestration
- `frontends/runner.py`: interactive input loop, slash commands, detached waiting behavior
- `agents.py`: turn execution, tool processing, final status mapping
- `store.py`: persistent bytes on disk

That split is the main reason the session system feels indirect.

### The current environment model is muddy

Today persistence splits environment data into:

- `environment: EnvironmentConfig`
- `environment_state: dict[str, Any] | None`

This exists for a reason:

- some env data is static setup / selection
- some env data is mutable serialized checkpoint state

But the split is currently lossy and under-specified:

- `EnvironmentConfig` often stores only a type and minimal config
- the real details live in the opaque serialized state
- some environments serialize/deserialize cleanly, some do not

Follow-up inspection:

- deserialize coverage is better than expected across environments
- but not every environment is truly cold-resumable
- `TerminalBenchEnvironment` explicitly serializes a live object reference and cannot be restored from cold storage

Implication:

- `Trajectory.environment` cannot assume "portable resumable snapshot" for every environment
- the canonical type needs to distinguish between "durable snapshot" and "best-effort live checkpoint" at least semantically, even if the final field layout stays simple

### Pending input is session control state, not transcript

Detached/waiting flows currently use `pending_input.json` as a sidecar file.

This is important because it means not all live session state belongs in the transcript itself. Some of it is control-plane state for resume/attach behavior.

## Proposed Direction

Make `Trajectory` the canonical overcomplete trace type, but organize it into explicit substructures instead of adding more flat top-level fields.

High-level shape:

```python
@dataclass(frozen=True)
class Trajectory:
    messages: list[Message]
    completions: list[ChatCompletion]
    metadata: dict[str, Any]
    annotations: ...
    session: ...
    environment: ...
```

This is a direction, not a final schema.

### 1. Keep transcript fields at the top level

These are the core of the trace:

- `messages`
- `completions`
- `metadata`

Rationale:

- They are the cross-cutting fields used by runtime, training, and export.
- They are already the main user-facing trace surface.

### 2. Move rollout/training scalars into one nested bundle

Current flat fields:

- `rewards`
- `group`
- `replica`
- `advantages`

These appear to be one concept: annotations on a trajectory for scoring, grouping, or optimization.

Do not keep growing this as unrelated flat scalars.

Working direction:

```python
@dataclass(frozen=True)
class TrajectoryAnnotations:
    reward: float | dict[str, float] | None = None
    group: int | None = None
    replica: int | None = None
    advantage: float | None = None
```

`TrajectoryAnnotations` is a placeholder name. Final naming is open.

### 3. Move session identity / branching / lifecycle into a nested session bundle

Fields that belong together:

- `session_id`
- `parent_id`
- `branch_point`
- `status`
- `created_at`
- `updated_at`
- `tags`
- `vcs`

Working direction:

```python
@dataclass(frozen=True)
class TrajectorySession:
    session_id: str | None = None
    parent_id: str | None = None
    branch_point: int | None = None
    status: SessionStatus | None = None
    created_at: str | None = None
    updated_at: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    vcs: dict[str, str] | None = None
```

This makes branching and lifecycle part of the canonical trace instead of a separate wrapper type.

### 4. Carry environment as a serializable snapshot, not a vague split by default

There are two viable models:

#### Option A: single environment snapshot

```python
@dataclass(frozen=True)
class TrajectoryEnvironment:
    kind: str
    data: dict[str, Any]
```

Use this if environments can serialize their full resumable state coherently.

#### Option B: explicit spec + snapshot

```python
@dataclass(frozen=True)
class TrajectoryEnvironment:
    spec: dict[str, Any]
    snapshot: dict[str, Any] | None = None
```

Use this if the distinction between static config and mutable checkpoint state is operationally important.

Current leaning:

- prefer a single top-level environment bundle
- only split spec vs snapshot internally if the code truly needs it

That keeps the top-level `Trajectory` simpler and avoids repeating the current `environment` vs `environment_state` confusion.

### Environment resumability must be explicit

Not all serialized environments mean the same thing.

Example:

- `TerminalBenchEnvironment` stores a live `_env_ref` and explicitly cannot be restored from cold storage.

That means the canonical trajectory model should not imply that every environment snapshot is equally resumable.

Working direction:

```python
@dataclass(frozen=True)
class TrajectoryEnvironment:
    kind: str
    data: dict[str, Any]
    resume_mode: Literal["cold", "warm", "none"] = "cold"
```

Semantics:

- `cold`: can be persisted and reconstructed in a new process from bytes alone
- `warm`: only resumable while the original live process/resources still exist
- `none`: informative only, not resumable

This keeps the top-level model honest and avoids pretending that "serialize() exists" implies durable resumability.

## What Should Probably Stay Outside `Trajectory`

Not every piece of session machinery belongs inside the transcript/trace record.

Likely keep separate:

- ephemeral input queue / pending input prompts
- attach/watcher state
- live process ownership

Reason:

- these are control-plane concerns
- they are not part of the trace you would want for training/export

So the likely boundary is:

- `Trajectory` stores durable semantic state
- `SessionSummary` owns cheap list/index metadata
- sidecars or store-owned metadata handle transient control state

## Target Ownership Model

Long term, ownership should simplify to:

- `Trajectory`: canonical durable semantic record
- `SessionSummary`: cheap listing/index view
- `agents.py`: canonical runtime transition owner
- `store.py`: persistence of canonical trajectory + small adjunct metadata only if needed
- frontends: rendering and input only

This would remove the current "session wrapper vs runtime trace" split and reduce state reconstruction in the CLI/TUI.

## Additional Findings From Code Inspection

### Driver adapters already treat messages as the canonical interchange format

`rollouts/drivers/session_adapter.py` converts between external driver formats and `list[Message]`.

That is useful because it means:

- the external-driver boundary does not currently depend on `AgentSession`
- migration can preserve `messages` as the canonical interchange layer even if the larger trajectory/session dtype changes

This lowers migration risk for `/swap` and session export/import.

### Most code that "switches session" really wants a loaded trajectory plus endpoint/env identity

Tests like `tests/test_switch_session.py` reconstruct a new `Trajectory(messages=session.messages)` after loading a session.

That is a strong signal that current callers conceptually want:

- transcript
- endpoint/model identity
- session lineage

They do not fundamentally want a separate wrapper type. They only use one because the store currently returns it.

## Migration Strategy

Do this incrementally.

### Phase 0: document and measure

- Write this design doc.
- Inventory actual field usage before changing dtypes.

### Phase 1: add nested bundles without deleting old fields

- Add nested bundles to `Trajectory`.
- Provide conversion helpers from existing flat fields.
- Keep `AgentSession` temporarily as an adapter/view type.

### Phase 2: teach the store to persist canonical trajectory fields

- Persist enough of `Trajectory` that session resume is no longer lossy.
- Decide whether `completions` are always persisted or optionally persisted.

### Phase 3: shrink or delete `AgentSession`

Options:

- convert it into a thin compatibility view over `Trajectory`
- or delete it once store/frontends no longer require it

### Phase 4: simplify ownership

- move more transition logic out of `cli.py` / `frontends/runner.py`
- make one runtime path responsible for session status transitions

## Information We Still Need Before Implementing

These are the questions to resolve first.

### 1. Which `Trajectory` fields are truly canonical vs optional caches?

Known:

- `messages` is canonical
- `completions` is important for training token/logprob extraction

Still need to decide:

- should `completions` always be persisted for interactive sessions?
- is there a size/perf threshold where we persist a reduced form?

### 2. What is the right boundary for environment persistence?

We need a concrete answer to:

- do environments have a coherent full snapshot format?
- do we need to preserve immutable setup separately from mutable state?
- which existing environments fail cleanly on serialize/deserialize today?

Known now:

- many environments implement both `serialize()` and `deserialize()`
- at least some environments are only warm-resumable, not cold-resumable
- the canonical model should capture that distinction explicitly

### 3. Should waiting / pending-input state live inside the canonical record?

The current system uses `pending_input.json`.

Need to decide:

- keep that as control-plane sidecar
- or make it part of the canonical session/session-state bundle

Current leaning: keep it outside `Trajectory`.

### 4. What is the correct persistence granularity?

Options:

- persist a fully materialized `Trajectory`
- persist a session header plus append-only events/messages
- persist both materialized snapshot and append-only log

This decision affects:

- performance
- debugging
- replayability
- how hard resume/fork/slice are

### 5. What compatibility layer do we need for training APIs?

`Sample` already treats `Trajectory` as the main execution artifact.

Need to map:

- existing training/eval code that expects current `Trajectory`
- session code that expects `AgentSession`
- trace export code that currently consumes `AgentSession`

## Immediate Investigation Tasks

Before changing the dtype definitions, inspect:

1. All live uses of `Trajectory.completions`, `metadata`, and rollout annotation fields.
2. All environments that implement `serialize()` / `deserialize()`, and which ones are lossy.
3. Whether session export / trace export should consume `Trajectory` directly.
4. Whether detached/waiting flow can stay as sidecar metadata without complicating resume.
5. Whether external drivers (`claude`, `codex`, etc.) can round-trip the canonical trajectory shape.

## Design Decisions Made Here

- `Trajectory` should become the canonical overcomplete durable record.
- Session persistence should move toward preserving a real trajectory, not a lossy wrapper around `messages`.
- Flat rollout/training scalars should become one nested bundle.
- Environment should become one explicit environment bundle, not two loosely-related top-level fields unless proven necessary.
- Pending input / attach control state is probably not part of the durable transcript.

## Open Naming Questions

Still open:

- `annotations` vs `labels` vs `rollout` for reward/group/replica/advantage
- `session` vs `lineage` for session identity/branching bundle
- `environment` vs `environment_snapshot`

Avoid locking these names too early. The shape matters more than the final labels.
