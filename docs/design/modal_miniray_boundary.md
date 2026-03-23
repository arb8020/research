# Modal, Bifrost, and MiniRay Boundary

## Question

If we want Heinrich-style process semantics for training and inference on Modal,
should `miniray` replace the current Modal execution path?

Short answer: no.

It is a good fit for the *inner* process/worker layer inside a sandbox. It is
not the right owner for provider/session/materialization semantics.

## The Real Layers

There are three different concerns here:

1. Provider/runtime acquisition
2. Remote execution session semantics
3. Local child-process / worker semantics inside the remote runtime

Those concerns should stay separate.

### Provider Layer

Owned by Modal + broker.

This layer is responsible for:

- creating sandboxes
- choosing images
- lifetime / cleanup policy
- provider-specific attachment and lookup

### Execution Layer

Owned by `bifrost`.

This layer is responsible for:

- materializing the workspace
- launching one-shot processes and services
- exposing handles for readiness, logs, stop, and exit
- keeping parent-owned lifecycle truth

This is where Modal and SSH should converge semantically, even if their
transports differ.

### Worker / IPC Layer

Candidate owner: `miniray`.

This layer is responsible for:

- `fork + socketpair` style worker control
- parent/child ownership semantics
- PDEATHSIG
- optional memfd/shared-memory for large local transfers
- explicit control messages instead of `multiprocessing` magic

This layer should live *inside* the sandbox or machine once the execution layer
has already launched the root process.

## What Ray Does That Is "Too Much"

Ray is not just a worker/process primitive. It is a whole distributed runtime:

- actor model
- scheduler / placement
- object store
- reference tracking
- retry / restart semantics
- cluster control plane

That is more than this training stack needs.

The thing we actually need is smaller:

- get a machine or sandbox
- sync code and deps
- launch one root process
- manage a handful of child workers explicitly
- stream logs and structured events
- handle readiness and teardown honestly

So the intended replacement is *not* "our own Ray."

The intended replacement is:

- `bifrost` for execution/session semantics
- `miniray` for explicit local worker semantics where needed

## Good Use of MiniRay Here

The promising move is:

1. `bifrost.modal_backend` launches one root workload process in the sandbox
2. that root process uses `miniray` internally for child workers
3. the root process exposes one explicit control surface upward

This would improve:

- process ownership
- teardown semantics
- separation between control messages and human logs
- Heinrich-style local reasoning

## Bad Use of MiniRay Here

The bad move is to let `miniray` pretend to be the remote execution substrate
itself.

That would blur boundaries:

- Modal still owns sandbox lifecycle
- broker still owns provider allocation
- `bifrost` still owns execution handles and workspace materialization

If `miniray` tries to absorb those concerns, we are just rebuilding another
execution runtime and lying about ownership again.

## Current Staging Constraint

Today the Modal path still launches an inner `argus.run --local` through a
supervisor trampoline. That means the current control plane is cleaner than it
was, but still not fully resolved.

The next honest step is:

- keep Modal/Bifrost as the provider + execution boundary
- replace the inner supervisor/control path with one explicit worker/control
  substrate if we need Heinrich-grade semantics

MiniRay is a plausible tool for that narrower job.

## Decision Rule

When evaluating a proposed change, ask:

Does this affect:

- provider allocation or sandbox lifecycle?
  - keep it in broker/Modal
- execution handles, readiness, workspace materialization, or remote launch?
  - keep it in `bifrost`
- local child-worker semantics inside one runtime?
  - `miniray` is a plausible owner

That is the boundary we should preserve.
