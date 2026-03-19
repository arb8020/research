# Bifrost / MiniRay Boundary

## Goal

Clarify the intended ownership split between:

- `broker`
- `bifrost`
- `miniray`
- `argus`
- `rollouts`

This note is motivated by a real ambiguity in the current codebase:

- `bifrost` already wants to own remote materialization and process execution
- `miniray` already wants to own explicit distributed worker/process semantics
- `argus` already uses `miniray.logs_server`

Without a sharper boundary, `bifrost` and `miniray` will drift into competing
"run things remotely" systems.

## Decision Status

This document is no longer just exploratory. It records the current direction:

- `bifrost` is the surviving execution-layer boundary
- `miniray` should be absorbed into `bifrost` over time as the distributed
  communications/process substrate
- `argus` owns durable supervision, not live execution mechanics
- `broker` owns provision-time runtime contract setup
- `rollouts` owns workload topology and capability semantics

The naming preference is settled:

- `bifrost` survives as the public execution-layer name
- `miniray` is an implementation/source package that can be folded into it

That means we should reason in terms of **responsibility** first and package
names second.

## Core Split

The cleanest boundary is:

- `bifrost` owns **live execution state**
- `argus` owns **durable supervisory state**

This is the most important distinction in the system.

### Live execution state

Live execution state includes things that disappear when the underlying process
or session disappears:

- machine/session connections
- materialized remote workspaces
- process handles
- service handles
- worker-group handles
- log streams
- artifact transfer sessions

This belongs in `bifrost`.

### Durable supervisory state

Durable supervisory state includes facts that must remain true after processes
die:

- runs
- attempts
- allocation bindings
- stage transitions
- recorded artifacts
- metrics snapshots
- cancellation / retry history

This belongs in `argus`.

## Target Layer Responsibilities

### `broker`

`broker` owns procurement of runtime-ready resources:

- provider search and allocation
- resource contracts
- provider-native procurement knobs:
  - image
  - accelerator type/count
  - disk/volume attachments
  - placement hints

`broker` should move further toward owning **provision-time runtime contract
satisfaction**:

- image/template selection
- provider-native boot/runtime guarantees
- mounted volumes and caches
- exposed ports
- resource-level readiness constraints

If a requirement can be satisfied before the machine is handed off for live
execution, it likely belongs in `broker`.

`broker` does not own:

- workspace sync
- process/session lifecycle
- run/attempt truth
- workload semantics
- project-local mutable workspace setup

### `bifrost`

`bifrost` is the execution layer.

It owns:

- connecting to allocated resources
- materializing workspaces/snapshots on those resources
- starting processes, services, and worker groups
- log streaming
- artifact upload/download/sync
- explicit execution/session handles

`bifrost` should own **post-allocation mutation of a live resource**:

- syncing a project snapshot into a workspace
- writing config files into that workspace
- applying project-local install/overlay commands on top of the base runtime
- launching processes/services against that prepared workspace

This is the layer that should answer:

- "run this command on that resource"
- "start this service and give me its endpoint"
- "sync these artifacts back"
- "start this worker group and give me handles"

If `miniray` primitives are the best way to realize some of that, they should
be used under this layer.

This means `bifrost` should become more explicit and substrate-like, potentially
more `miniray`-shaped in implementation, while remaining the named boundary for
execution/session behavior.

### `miniray`

`miniray` is a narrow explicit distributed-process toolkit today.

It is a good fit for:

- worker-to-coordinator messaging
- worker groups
- TCP/JSON control channels
- explicit per-process handles
- simple stream/log transport
- NCCL/env bootstrapping helpers

It is **not** the right place to own:

- resource procurement
- durable job/run truth
- workload semantics
- multi-tenant scheduling
- general cloud orchestration

The intended future is stronger than "may supply execution primitives":

- `miniray`'s worker/message-passing/process-group ideas get absorbed into
  `bifrost`
- `bifrost` remains the package/API that owns the execution-layer contract
- separate top-level execution systems should not continue to grow in parallel

In other words:

- `bifrost` is the boundary
- `miniray` is the substrate that gets eaten into that boundary

### `argus`

`argus` owns detached run semantics:

- run identity
- attempts
- allocation refs
- events
- projections
- monitor / attach / cancel / retry

`argus` should supervise `bifrost` handles, not reimplement execution/session
behavior itself.

### `rollouts`

`rollouts` owns workload semantics:

- task definitions
- local sandbox/workspace semantics
- topology definitions for complex workloads
- capability injection for agent attempts
- scoring / verification semantics

`rollouts` compiles workload state into launchable work. It does not own
procurement or detached lifecycle.

## Hard Case: Parameter Golf

For the hardest `parameter-golf` shape, the workload is not "one command on one
machine." It is a topology with multiple runtime roles:

- inference pool
- training pool
- sandbox pool
- verifier/scorer pool

The right division is:

- `rollouts` describes that topology and the capability surface exposed to an
  agent sandbox
- `argus` supervises the realized topology
- `broker` procures the resources
- `bifrost` realizes the live execution on those resources

The sandbox should not know provider names, node IDs, or GPU SKUs. It should
receive capabilities like:

- inference endpoints
- launch experiment
- fetch artifacts

## Why This Is Not "Rebuilding Ray"

The non-goal is important.

This stack should not try to become:

- a general-purpose cluster manager
- a multi-tenant scheduler
- a universal distributed object store
- a generalized actor framework

That is where the codebase would start rebuilding Ray/SkyPilot/Kubernetes
poorly.

The intended scope is narrower:

- explicit resource procurement
- explicit execution/session handles
- explicit detached supervision
- workload-specific topology and scoring

The Heinrich-style lesson we want to preserve is:

- explicit processes
- explicit message channels
- explicit handles
- no giant magical actor/object-store abstraction

So "absorbing `miniray` into `bifrost`" does **not** mean turning `bifrost`
into Ray. It means making `bifrost` a cleaner explicit execution layer with
better distributed comms primitives.

## API Direction

The steady-state shape should look something like this:

### `rollouts`

Produces a launchable workload topology:

```python
topology = WorkloadTopology(...)
```

### `argus`

Turns that topology into supervised live attempts/services:

```python
run = argus.launch(topology)
```

### `broker`

Allocates resources satisfying the topology's runtime contracts.

This is the place to encode things like:

- base image or template
- accelerator shape
- volume attachments
- provider-native readiness guarantees

### `bifrost`

Creates live handles:

```python
session = bifrost.connect(allocation)
workspace = bifrost.materialize(session, snapshot)
service = bifrost.serve(session, service_spec)
job = bifrost.submit(session, process_spec)
artifacts = bifrost.sync_back(session, artifact_spec)
```

`miniray` may be one of the mechanisms behind `serve()` or `submit()` for
worker-group-shaped execution.

## Provision-Time vs Post-Allocation Setup

The word "setup" is too blurry. There are two different kinds:

### Provision-time contract setup

This belongs in `broker`.

Examples:

- choose a RunPod template
- choose a base image
- attach a network/persistent volume
- request exposed ports
- require a given accelerator class/count

### Post-allocation workspace/setup mutation

This belongs in `bifrost`.

Examples:

- sync a repo snapshot into `/workspace/project`
- write task-specific config files
- run `uv sync` for that specific workspace
- launch a job or service against that workspace

This is the key refinement:

- `broker` should own resource contract setup
- `bifrost` should own live-resource mutation and execution

## Non-goals

This design does **not** imply:

- deleting `miniray`
- renaming `miniray` to `bifrost`
- making `argus` own sandboxes or workload prompts
- moving `broker` up into job orchestration

It only says:

- `bifrost` is the surviving execution-layer boundary
- `miniray` should be narrowed to explicit distributed execution primitives
- `argus` should own durable truth, not live execution mechanics

## Migration Direction

1. Stop adding detached job/control-plane responsibilities to `miniray`
2. Stop adding execution/materialization logic directly to `argus`
3. Move toward explicit `bifrost` execution handles and operations
4. Pull the useful `miniray` process/worker/stream primitives into `bifrost`
5. Avoid sustaining `bifrost` and `miniray` as parallel execution APIs
6. Keep `rollouts` focused on workload topology and capability semantics

## What This Doc Decides

This doc settles:

- `broker` owns provision-time runtime contract setup
- `bifrost` owns post-allocation live-resource mutation and execution
- `argus` owns durable supervisory truth
- `rollouts` owns workload semantics/topology/capabilities
- `miniray` should be absorbed into `bifrost`, not kept as a competing
  execution layer

## What Remains Open

This doc does not yet settle:

- the exact public `bifrost` API surface after the refactor
- whether `miniray` remains as a source subtree, internal module, or temporary
  compatibility package during migration
- the exact sequence of extraction/merge work across `argus.run`,
  `rollouts.modal_runner`, and current `bifrost` deploy/job code
- whether project-local runtime overlays like `uv sync` stay a first-class
  `bifrost` operation or become a separate explicit sub-API

## Handoff Notes

This doc should be enough for another engineer to start cleanup work, but not
enough to implement the entire refactor without additional interface design.

What it gives them:

- the ownership boundary
- the litmus tests
- the direction that `miniray` gets folded into `bifrost`
- the non-goals that prevent platform creep

What they would still need before large implementation work:

- a concrete list of current leaks by file/module
- the first target seam to cut
- the initial `bifrost` interface to standardize around

## Litmus Test

When deciding where code belongs:

- if it disappears when the process dies, it probably belongs in `bifrost`
- if it must still be true after the process dies, it probably belongs in
  `argus`
- if it is about getting a resource in the first place, it belongs in `broker`
- if it is about what the workload means, it belongs in `rollouts`

Additional litmus test:

- if it could be baked into an image/template or decided before boot, it
  probably belongs in `broker`
- if it mutates a specific live workspace after allocation, it probably belongs
  in `bifrost`

This is the boundary we should optimize for.
