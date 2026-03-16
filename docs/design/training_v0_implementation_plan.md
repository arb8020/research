# Training V0 Implementation Plan

This note records the current v0 implementation priority for the training
track.

This is no longer the same plan as the earlier "dense offline pretraining
through an external backend" direction. That work was useful, and some of it
has already landed, but it is no longer the primary investment target.

## Why the priority changed

We are now deliberately splitting effort across two tracks:

- the RL-unblocking track will continue to exercise external backends such as
  TorchTitan and Megatron in real rollout-coupled settings
- the training track will prioritize sovereign runtime bring-up, with MoE
  semantics first-class

This means the highest-value next work for training is no longer "another
external backend witness." The highest-value work is to start owning the MoE
runtime semantics directly.

## Goal

Build a sovereign MoE-first training runtime v0.

Concretely, v0 should:

- support single-node execution only
- be architected so multinode support can be added honestly later
- make `/d` replica ownership and `/ep` expert ownership first-class
- make route / dispatch / expert compute / combine first-class
- own checkpoint semantics
- own observability semantics, especially router-health signals
- run standalone pretraining first, not RL

## Why this is the right first step

The Noumena posts and `nmoe` both make the same point:

- MoE training is not just dense training with experts glued on
- router learning, dispatch semantics, and exact resume/logging discipline are
  load-bearing
- if we do not own those semantics, we will keep renting someone else’s
  worldview

At the same time, the RL-unblocking track is already spending effort on
TorchTitan and Megatron integration. That work will continue. It just does not
need to be the main training v0 success criterion anymore.

## Scope

### In scope

- single-node runtime
- logical mesh model that can later extend to multinode
- dense trunk + MoE expert path
- standalone dense/MoE pretraining
- explicit route / dispatch / combine semantics
- exact-ish resume semantics
- router-health and systems observability core
- fail-loud unsupported states

### Out of scope

- real multinode execution in v0
- RL orchestration integration
- full external-backend lowering support
- TP/PP/CP in the sovereign runtime
- fancy precision bring-up
- full parity with upstream `nmoe`

## Runtime shape

The runtime should be:

- single-node in implementation
- multinode-compatible in architecture

That means:

- reason in terms of logical ranks and ownership, not hardcoded local GPU ids
- keep transport behind a narrow interface
- key checkpoint ownership by logical owner identity
- keep lockstep eval/generation semantics independent of host topology

What v0 should **not** do:

- pretend multinode is already supported
- bake single-host assumptions into the semantic layer

## Semantic center

V0 should be built around a small sovereign semantic core:

- internal model denotation: ours
- training-step IR: ours
- checkpoint semantics: ours
- observability semantics: ours

External runtimes remain useful, but they are not the semantic center of this
v0.

## Minimal operation set

The first useful operation set is still:

- `all_gather(spec, x)`
- `psum_scatter(spec, x)`
- `einsum_unreduced(spec, *xs)`
- `materialize(spec, x)`
- `route(spec, x, router)`
- `dispatch(spec, x, route_result, experts)`
- `expert_compute(spec, x, experts)`
- `combine(spec, expert_out, route_result)`
- `compute_loss(...)`
- `backward(loss)`
- `optimizer_step(state)`

The design rule is:

- include the semantic split
- exclude backend/runtime algorithm details like RDEP packing internals

## Concrete v0 phases

### Phase 0: keep the shared stack alive

Already useful and should remain available:

- shared runtime factory extraction
- offline dense pretraining entrypoint using external backends

This is now supporting infrastructure, not the main v0 goal.

### Phase 1: define the sovereign runtime skeleton

Add the first sovereign runtime modules for:

- logical mesh / ownership model
- runtime state
- transport abstraction
- checkpoint state
- observability core

These should be single-node implementations with multinode-compatible
interfaces.

### Phase 2: define the MoE-first step surface

Add the first authored loop/IR path for:

- routing
- dispatch
- owner-local expert compute
- combine

This is the first point where the sovereign runtime becomes meaningfully
different from a dense external-backend witness.

### Phase 3: standalone pretraining bring-up

Run standalone pretraining through the sovereign runtime:

- no RL orchestration
- no inference engine
- no live weight publication

This should be the first real runtime bring-up target.

### Phase 4: tighten trust surfaces

Before broadening capability, tighten:

- checkpoint semantics
- resume semantics
- router-health observability
- failure behavior

## Recommended file direction

Likely new modules:

- `rollouts/rollouts/training/runtime/mesh.py`
- `rollouts/rollouts/training/runtime/state.py`
- `rollouts/rollouts/training/runtime/transport.py`
- `rollouts/rollouts/training/runtime/checkpoint.py`
- `rollouts/rollouts/training/runtime/observability.py`
- `rollouts/rollouts/training/runtime/moe.py`

Likely supporting updates:

- internal model denotation surfaces
- training-step IR surfaces
- new witness loop definitions for sovereign pretraining

## Acceptance criteria for v0

V0 is complete when all of the following are true:

1. We can run standalone pretraining through a sovereign runtime path.
2. That runtime owns MoE route / dispatch / combine semantics directly.
3. The runtime is single-node only, but its ownership and transport interfaces
   are multinode-compatible.
4. Checkpoint and resume semantics are owned by the sovereign runtime.
5. Router-health and systems observability are part of the runtime contract.
6. Unsupported topology or runtime requests fail early and loudly.

## What v0 does not prove yet

Even if v0 works, it does not yet prove:

- multinode execution
- full external-backend parity
- RL integration
- TP/PP/CP support
- production-ready MoE speed

That is acceptable. The point of v0 is to own the right semantics first.
