# Runtime Contract, Materialization, and Supervision

## Intended split

- `broker` owns the runtime-ready resource contract.
- `bifrost` owns source sync, workspace materialization, and process execution on that resource.
- `argus` owns detached run truth: run/attempt/allocation/command/event/projection semantics.

`rollouts.run` should stay a thin composition layer over those packages.

## The missing distinction

The old code mixed three different concerns under words like `deps` and
`bootstrap`:

- `runtime contract`
  - what environment must exist before project-specific code runs
- `materialization plan`
  - what project/workspace setup must happen on top of that runtime
- `execution session`
  - where commands actually run

That blur made `run.py` and `modal_runner.py` duplicate policy and drift apart.

## Runtime contract

Runtime contract means:

- provider
- GPU type/count
- image/runtime contract
- cache and volume attachments
- provider-native readiness/setup guarantees
- runtime-level features such as `use_torchrun`

This is broader than "give me an SSH string", but narrower than "run my project bootstrap".

## Materialization plan

Materialization plan means:

- source sync mode
- workspace root
- project bootstrap commands

This is the layer for:

- syncing code into a remote workspace
- installing project-local overlays
- preparing the repo before launching a process

If a step is "on top of the image/runtime", it belongs here.

## Source sync policy

The most duplicated part today was the dirty-worktree policy:

- both SSH and Modal deploy committed-only source
- both need the same check and the same user-facing semantics

This now lives in one shared `SourceSyncPolicy` instead of being split between
`run.py` and `modal_runner.py`.

## Current first step in code

We introduced shared rollouts-side objects:

- `RuntimeContract`
- `MaterializationPlan`
- `SourceSyncPolicy`

They currently normalize what the runners already need, without trying to fully
redesign Broker or Bifrost in one jump.

That is deliberate. The current `DepsConfig` still mixes runtime and
materialization concerns. The new objects make the boundary explicit before we
finish pushing the lower-layer APIs into the right packages.

## Package direction

### broker

Should eventually return something closer to a runtime-ready resource than a raw
machine identifier. It should know about:

- provider-shaped resources, including Modal-style sandboxes
- images
- mounts/volumes
- runtime contract satisfaction

It should not own:

- project workspace sync
- process supervision
- run/attempt semantics

### bifrost

Should become the transport-agnostic execution/session layer for prepared
resources:

- sync source
- materialize workspace
- exec / exec_stream / detached exec
- upload/download
- process/session handles

Modal should eventually look like another execution substrate here, not a
special `rollouts` runner.

### argus

Should sit above both and own:

- durable run identity
- attempts
- authoritative append-only event journal
- `snapshot + subscribe(from_cursor)`
- command submission
- projections for UIs

The laptop is a command origin and subscriber, not the sole durable control
plane.

## Near-term implication

The next useful compression is:

1. keep provider/resource specifics in `broker`
2. move more execution/session logic into `bifrost`
3. stop duplicating deployment/source semantics in `rollouts`
4. leave detached run truth for `argus`

This is enough to move the codebase toward the intended layering without
pretending the full supervisor architecture already exists.
