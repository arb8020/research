# Bifrost Execution Handles And Modal Folding

## Goal

Make the execution layer honest:

- `rollouts` owns workload semantics
- `argus` owns durable run and attempt truth
- `broker` owns allocation
- `bifrost` owns live execution state

The immediate target is to replace `rollouts.modal_runner` as a special execution
system with Modal as another `bifrost` backend.

This is not a training-system rewrite. It is an execution/lifecycle/observability
rewrite.

## Problem

The current code does not answer one basic question well enough:

> what exactly is a launched thing?

That missing denotation shows up as several coupled problems:

- `rollouts.modal_runner` owns app/image/sandbox/process orchestration
- `argus.run` still reconstructs execution/materialization details inline
- Modal and SSH are separate execution worlds
- process/session identity is smeared across sandboxes, tmux sessions, commands,
  log files, and health endpoints
- logs are not authoritative; they are overlapping artifacts

This breaks local reasoning. The code style docs point at the same failure from
different angles:

- `debugging_draft.md`: logs are observation, not proof; find the first dishonest
  boundary
- `logging_sucks.md`: prefer one canonical, queryable event stream over many
  ad hoc lines
- `code_philosophy_essay.md`: keep state ownership clear and push orchestration
  into visible owners

## Current Dishonest Shape

Today, the Modal path is:

1. local `argus run`
2. `rollouts.modal_runner`
3. Modal app/image/sandbox lifecycle
4. embedded supervisor Python shim
5. child `argus.run --local` inside the sandbox
6. workload process tree
7. ad hoc stdout/stderr/event parsing and failure diagnostics

The SSH path is different:

1. local `argus run`
2. `bifrost.acquire_node()`
3. `bifrost.push()`
4. `bifrost.submit()` / `bifrost.exec()`
5. tmux + log-file based job monitoring

Both paths are operationally useful. Neither gives one honest process/service
model. Modal is worse because it duplicates a whole execution substrate inside
`rollouts`.

## Decision

`bifrost` is the execution boundary.

The missing execution product types should be defined in `bifrost`, not in
`rollouts.modal_runner`, not in `argus.run`, and not in a new parallel runtime
package.

Modal should fold into those `bifrost` semantics as a backend.

`miniray` is relevant, but not the first move:

- use `miniray` ideas now: explicit handles, explicit control channels,
  explicit ownership
- use `miniray` primitives later where worker groups or richer coordinator/worker
  messaging are actually needed
- do not block the first cleanup on adopting `miniray` everywhere

For the first cut, SSH via `asyncssh`-style primitives is sufficient.

## Design Principles

1. One launched thing should have one handle.
2. Parent-owned lifecycle events are authoritative.
3. Child stdout/stderr is payload, not the control plane.
4. Readiness is explicit state plus explicit probes, not inferred from random
   log text.
5. Execution backends may differ in transport, but not in the semantics they
   expose upward.
6. The abstraction should preserve continuous granularity: higher-level helpers
   should lower into obvious lower-level operations.

## Chosen Defaults

These defaults are settled unless implementation pressure proves them wrong.

### One Public Session Interface

We want one public execution-session interface, not separate public
`SSHSession` / `ModalSession` APIs.

This interface should be async-first.

Reason:

- Modal is natively async
- observed process execution, streaming, readiness, and cancellation are
  effectful long-lived operations
- making the public session sync-first bakes SSH's shape into the abstraction

So the rule is:

- the public session surface is async-first
- sync provider conveniences may exist as adapters, not as the primary
  denotation

What this hides:

- SSH connection reuse
- Modal app/sandbox mechanics
- tmux/log-file internals for detached execution
- provider-specific output capture details

What this must not hide:

- reconnect semantics that materially differ
- readiness semantics that materially differ
- lifecycle capabilities that materially differ

And:

- hide transport mechanics
- expose semantic differences as explicit capability or handle fields

### Parent-Owned Lifecycle Truth

The parent execution layer owns lifecycle truth.

That means:

- launch/ready/exit/stop state comes from the parent handle
- child stdout/stderr is payload
- child structured events are optional semantic milestones, not authoritative
  lifecycle state

We still want an explicit parent/child event channel because the child knows
some semantic milestones first:

- model loaded
- custom routes registered
- worker extension installed
- NCCL group initialized

But those should remain typed breadcrumbs emitted into the parent-owned event
stream, not a second source of truth for process lifecycle.

## Current Migration State

The first substrate cuts are intentionally narrow:

- `bifrost` now owns the public Modal execution entrypoint
- `bifrost.modal_backend` owns sandbox creation, sandbox command execution, and
  repo materialization into the sandbox
- `argus` talks to Modal through `bifrost`, not directly through
  `rollouts.modal_runner`

The remaining dishonest pieces are narrower now:

- Modal image build/materialization lowering now lives under `broker`, but it
  still depends on Rollouts-side image/runtime contract types
- `run_modal_request()` still delegates the high-level workload state machine to
  `rollouts.modal_runner.run_modal`
- the workload inside the sandbox is still launched via the existing inner
  `argus.run --local` trampoline

This is acceptable as a staging point because the execution ownership boundary
has moved in the right direction without pretending those remaining semantics
have been cleaned up yet.

## Core Product Types

These names are illustrative. The important part is the semantics.

### `ExecutionSession`

Represents a live connection to an allocated resource.

It owns:

- provider/backend identity
- connection/session identity
- workspace/materialization operations
- process/service launch operations
- event and output sinks for launched handles

It does not own:

- run identity
- workload semantics
- provider allocation

For SSH, this is a live SSH connection.
For Modal, this is a live sandbox/app session.

### `WorkspaceHandle`

Represents a materialized project snapshot on an execution session.

It owns:

- remote path or provider-native workspace reference
- source snapshot identity
- materialization metadata

It should be the result of a single explicit materialization step, not a side
effect hidden inside process launch.

### `ProcessSpec`

Represents a one-shot process denotation.

Fields should stay small and explicit:

- command
- args
- cwd
- env
- stdin policy
- output policy
- timeout policy

This already exists in bifrost in partial form. It should remain the basic
lower-level execution request.

### `ServiceSpec`

Represents a long-lived process with readiness semantics.

At minimum:

- `process: ProcessSpec`
- `readiness: ReadinessSpec`
- `exposed_endpoints`
- `shutdown_policy`

This is the honest missing sum type today. A server is not just a process with
some extra comments.

### `ProcessHandle`

Represents a launched one-shot process.

It owns:

- backend-specific identity
- lifecycle state
- exit status
- canonical event log location/reference
- canonical stdout/stderr sink reference

Core operations:

- `wait()`
- `terminate()`
- `signal(...)`
- `poll()`

### `ServiceHandle`

Represents a launched service.

It owns everything a `ProcessHandle` owns, plus:

- readiness state
- readiness probe results
- endpoint refs

Core operations:

- `wait_ready()`
- `stop()`
- `status()`

This should be a real type, not “job + port + health endpoint” smeared across
call sites.

### `LifecycleEvent`

A canonical structured event emitted by the execution parent.

Examples:

- `workspace_materialization_started`
- `workspace_materialization_finished`
- `process_launch_started`
- `process_launch_exec`
- `process_started`
- `service_probe_started`
- `service_probe_succeeded`
- `service_ready`
- `process_exit_observed`
- `service_stop_requested`
- `service_stopped`

The parent owns these events. Child logs may still exist, but they are not the
authoritative lifecycle record.

### `ChildEvent`

Represents an optional semantic milestone emitted by the child process.

Examples:

- `model_loaded`
- `routes_registered`
- `worker_extension_installed`
- `first_training_step_entered`

These events are useful because they preserve semantic milestones without
forcing the parent to parse arbitrary log text. They should be attached to the
parent-owned event stream, but they do not override parent lifecycle truth.

### `OutputSink`

Represents owned raw stdout/stderr capture for a launched handle.

This should be explicit in the handle, not rediscovered from tmux or random log
files later.

## State Machines

### Process

`created -> launching -> running -> exited`

Optional transitions:

- `running -> termination_requested -> exited`
- `launching -> launch_failed`

### Service

`created -> launching -> starting -> ready`

Optional transitions:

- `starting -> readiness_failed`
- `ready -> stopping -> exited`
- `running_without_readiness` should not exist; that is an illegal ambiguous
  state

## Logging And Observability

The logging cleanup is mostly a consequence of the execution model.

### Canonical Truth

Authoritative:

- parent-owned `LifecycleEvent` stream
- explicit readiness probe state
- explicit handle state

Payload:

- child stdout
- child stderr

Fallback evidence:

- tail snippets attached to failure events

### What We Should Stop Doing

- treating tmux session existence as the main denotation of a process
- scraping log files to infer lifecycle truth
- replaying the same child output through multiple overlapping channels and then
  pretending they all tell the same story
- using log text instead of assertions or explicit readiness transitions

### What We Should Keep

- structured wide events
- attaching recent stdout/stderr tail on failure
- provider-specific diagnostics where the parent cannot directly observe the
  failure cause

## Backend Realizations

### SSH Backend

Realized with:

- `asyncssh` / existing SSH client machinery
- remote workspace sync
- remote process launch

Short term, SSH may still use tmux/log-file internals for detached processes,
but those details should be hidden behind `ProcessHandle` / `ServiceHandle`.

The abstraction boundary must not expose tmux as the public denotation.

### Modal Backend

Realized with:

- Modal image/app/sandbox APIs
- provider-native process launch inside the sandbox
- provider-native output capture where available

The important change is that Modal should no longer own a special execution
story in `rollouts.modal_runner`. It should satisfy the same handle semantics as
SSH.

### `miniray`

Not a top-level competing execution API.

It fits underneath `bifrost` when we need:

- worker-group handles
- explicit process-to-process control channels
- stream/log transport helpers
- richer distributed coordination than simple process/service launch

For the first cleanup, it is enough to copy the semantic lessons:

- explicit handles
- explicit ownership
- explicit channels

## Migration Plan

### Phase 0: Temporary Operational Split

Keep using:

- RunPod+Bifrost for inference smokes
- Modal runner only for the paths that still require it

This keeps training/inference validation moving while the substrate changes.

### Phase 1: Define Execution Handles In Bifrost

Add:

- `ExecutionSession`
- `WorkspaceHandle`
- `ProcessHandle`
- `ServiceHandle`
- `LifecycleEvent`
- `OutputSink`

Tradeoff:

- this adds new surface area before deleting old code
- but it creates one honest ontology for the rewrite

### Phase 2: Re-express Existing SSH Flow In Terms Of Handles

Refactor Bifrost’s current SSH path to return the new handle types without
changing the transport yet.

Tradeoff:

- we temporarily keep tmux internally
- but callers stop depending on tmux/log-file semantics directly

### Phase 3: Port Modal Runner Behind The Same Interface

Replace `rollouts.modal_runner` as the owner of execution semantics.

Likely split:

- Modal backend code under `bifrost`
- provider-specific image/materialization helpers near that backend
- no `argus.run -> rollouts.modal_runner` special case

Transitional step:

- `argus` calls a `bifrost` Modal backend entrypoint
- that backend may temporarily delegate to `rollouts.modal_runner`
- the delegation is explicit and localized instead of being a control-plane leak
  from `argus` directly into `rollouts`

Tradeoff:

- this may keep some provider-specific code in the first Modal backend
- but the provider-specific code will live at the correct layer

### Phase 4: Remove Nested `argus.run --local` Trampoline In Modal

The execution layer should launch a resolved workload entrypoint directly,
instead of invoking another control-plane dispatcher inside the sandbox.

Tradeoff:

- requires a clearer launchable experiment/product type
- removes one whole layer of indirection and duplicated lifecycle

### Phase 5: Shrink Argus To Supervision

Once SSH and Modal both speak the same execution-handle language:

- `argus` supervises handles
- `argus` records durable events and projections
- `argus` stops reconstructing runtime/materialization/process details inline

## First Concrete Cuts

1. Add handle/product-type definitions to `bifrost`.
2. Add one canonical lifecycle-event sink for launched handles.
3. Refactor current Bifrost SSH job/service APIs to return those handles.
4. Introduce a Modal backend package in `bifrost` that initially wraps current
   Modal behavior.
5. Change `argus.run` to dispatch through the execution backend interface,
   including Modal.
6. Delete execution/lifecycle ownership from `rollouts.modal_runner`.

## Non-Goals

This doc does not propose:

- rewriting GRPO or training backend semantics
- changing model/inference capability semantics
- replacing `broker`
- turning `miniray` into a public competing execution API

## Open Questions

1. Should `ExecutionSession` be one product type with backend tags, or an
   algebraic data type with `SSHSession | ModalSession` variants under one
   interface?
2. How much provider-specific image/materialization logic should live in
   `bifrost` vs a lower helper package?
3. Should child processes be allowed to emit structured lifecycle events into
   the parent-owned sink, or should all lifecycle truth remain parent-emitted?

My default answers:

1. One public session interface, backend-specific internal variants.
2. Provider-specific runtime realization belongs in `bifrost`; generic workload
   image contracts should stay above it.
3. Parent-owned events are authoritative; child structured events are optional
   semantic breadcrumbs.

## Summary

The missing abstraction is not “better Modal logging.” It is a truthful
execution model.

We should define that model in `bifrost`, using explicit handles and parent-owned
event streams, then fold Modal into it as a backend. `miniray` should inform the
semantics now and contribute implementation pieces later where richer
process-group coordination is needed.
