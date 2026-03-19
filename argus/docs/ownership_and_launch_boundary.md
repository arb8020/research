# Argus Ownership And Launch Boundary

This note makes the intended infrastructure split explicit so cleanup work can
be judged against one boundary instead of vibes.

## Goal

Make the stack honest:

- `rollouts` owns workload semantics and local task state
- `broker` owns procurement of runtime-ready resources
- `bifrost` owns materialization, remote execution, and artifact transport
- `argus` owns detached run lifecycle, monitoring, and supervision truth

The key compilation boundary is:

1. `rollouts` starts from mutable local task state
2. `rollouts` compiles that state into a launchable remote experiment
3. `argus` supervises execution of that launchable experiment

By the time `argus` sees a run, questions like "which files were editable" or
"what prompt did the agent see" should already be compiled away into the
workload's launchable artifact/spec.

## Layer Responsibilities

### `rollouts`

`rollouts` owns task semantics:

- datasets, samples, prompts, scoring
- local sandbox/workspace lifecycle for the agent
- which local state is eligible to be launched remotely
- translation from task state into a launchable experiment

`rollouts` does not own:

- provider search/allocation
- remote process/session transport
- detached run truth

### `broker`

`broker` owns procurement of runtime-ready resources:

- provider search and allocation
- provider identity and allocation references
- provider-native runtime knobs such as image, GPU class/count, and volumes

`broker` may allocate sandbox-shaped compute resources if a provider offers
them, but it does not own the agent's editable task workspace.

`broker` does not own:

- repo/workspace sync
- project bootstrap commands
- detached job supervision

### `bifrost`

`bifrost` owns execution on an allocated resource:

- source/workspace materialization
- remote exec / detached exec / session lifecycle
- log transport
- artifact upload/download/sync

`bifrost` does not own:

- provider search
- run identity or attempt truth
- workload semantics

### `argus`

`argus` owns detached run semantics:

- run identity
- attempt lifecycle
- allocation binding
- status, events, heartbeats, and projections
- control-plane attach / monitor / cancel flows

`argus` does not own:

- the agent's editable workspace
- task prompts or scoring semantics
- provider-specific deploy logic
- transport details for sync or execution

## What Argus Should Consume

Argus should consume a reduced launchable experiment, not a task definition and
not a mutable workspace.

That launchable experiment should be the minimum product type needed to run
something remotely:

- runtime requirement
- materialized source snapshot or workspace reference
- bootstrap commands
- run command
- artifact contract
- lifecycle policy

The exact Python type can change, but semantically it should look like:

```python
LaunchableExperiment(
    runtime=...,
    source_snapshot=...,
    bootstrap=...,
    command=...,
    artifacts=...,
    lifecycle=...,
)
```

This is intentionally narrower than today's `argus run` flag surface.

## What Argus Should Not Consume

Argus should not directly model:

- what files the agent was allowed to edit
- how a local coding sandbox is created
- per-task prompt or tool policy
- ad hoc config patching through many infra CLI flags

Those are either `rollouts` concerns or a symptom that the launchable
experiment product type is still missing.

## Current Leaks

Today the boundary still leaks in a few places:

- `argus.run` mixes allocation, image resolution, source sync, bootstrap,
  process startup, and sync behavior in one launcher path
- the CLI exposes many flags that are really run-spec fields, not control-plane
  concerns
- local mutable task state is not yet compiled into an explicit launchable
  experiment type before reaching the launcher

These leaks are operationally useful, but semantically dishonest.

## Cleanup Direction

The intended cleanup sequence is:

1. keep task/workspace semantics in `rollouts`
2. introduce a narrow launchable experiment product type at the boundary
3. make `argus` supervise that product type instead of reconstructing it from
   flags
4. push allocation details down into `broker`
5. push materialization and remote execution details down into `bifrost`
6. shrink the Argus CLI until it mostly exposes control-plane concerns

## CLI Implication

The steady-state `argus` CLI should mostly answer control-plane questions:

- what run to launch
- whether to attach, tail, or monitor
- whether to run in an explicit local-dev mode

It should not be the main place where users patch:

- provider
- GPU SKU
- volume IDs
- disk sizes
- cache layout
- deploy behavior

If changing a value materially changes execution semantics, it should usually
live in the workload config or in a named launch profile chosen by that config.
