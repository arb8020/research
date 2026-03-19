# Bifrost Boundary Refactor Checklist

This note turns the boundary decision in
[bifrost_miniray_boundary.md](/tmp/research-bifrost-boundary/docs/design/bifrost_miniray_boundary.md)
into pre-implementation work.

It is intentionally short. The goal is to identify the minimum design work that
must be done before a large refactor starts.

## Preconditions

The following decisions are treated as settled:

- `broker` owns provision-time runtime contract setup
- `bifrost` owns post-allocation live-resource mutation and execution
- `argus` owns durable supervisory truth
- `rollouts` owns workload topology and capability semantics
- `miniray` gets absorbed into `bifrost` instead of surviving as a competing
  execution layer

## Must Flesh Out Before Coding

### 1. First `bifrost` API

Write down the first concrete execution-layer API.

At minimum:

- `AllocationHandle`
- `SessionHandle`
- `WorkspaceSpec`
- `ProcessSpec`
- `ServiceSpec`
- `ArtifactSpec`

At minimum operations:

- `connect(allocation)`
- `materialize(session, workspace_spec)`
- `submit(session, process_spec)`
- `serve(session, service_spec)`
- `sync_back(session, artifact_spec)`

Without this, the refactor will drift file-by-file without a target boundary.

### 2. Current Leak Map

Map the current modules to target ownership.

Initial list:

- `argus/run.py`
- `rollouts/rollouts/modal_runner.py`
- `rollouts/rollouts/remote_runtime.py`
- `bifrost/bifrost/deploy.py`
- `bifrost/bifrost/provision.py`
- `miniray/cluster.py`
- `miniray/worker_server.py`
- `miniray/logs_server.py`

For each module, record one of:

- keep in place
- move to `broker`
- move to `bifrost`
- move to `argus`
- merge into another module
- delete after migration

### 3. `miniray` Migration Mechanics

Decide how `miniray` gets absorbed.

Open implementation questions:

- which primitives move first?
- does `miniray` remain as a source subtree for a while?
- do we add `bifrost` re-exports during migration?
- which APIs are frozen immediately to stop parallel growth?

Suggested priority:

1. worker/process handle ideas
2. stream/log transport
3. worker-group coordination helpers
4. NCCL/env helpers
5. any cluster bring-up code only if it still has an honest role after
   `bifrost` takes over execution

### 4. First Cut Seam

Pick the first implementation seam before broad cleanup starts.

Recommended first seam:

- extract `argus.run` away from inline execution/materialization logic
- make it call a narrower `bifrost` boundary

Rationale:

- `argus.run` currently exposes the full confusion in one place
- it is where control-plane, runtime, materialization, and execution concerns
  are most visibly mixed
- reducing that surface first prevents new leakage during later refactors

## What Should Not Be Solved First

Avoid expanding scope into:

- general cluster scheduling
- multi-tenant queueing
- fault-tolerant distributed object stores
- provider-agnostic orchestration DSLs

Those are the fastest path to accidental platform creep.

## Immediate Deliverables

Before starting the refactor, produce:

1. one API sketch for the first `bifrost` interface
2. one leak map by module
3. one migration note for `miniray` absorption
4. one chosen first seam with a small implementation plan

That is enough to begin changing code without pretending the full end-state API
is already perfect.
