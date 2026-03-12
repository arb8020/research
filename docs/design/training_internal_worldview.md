# Training Internal Worldview

This is a working design note for the internal training model we want to commit to before launching larger RL jobs.

The point is not to get the perfect abstraction on the first try. The point is to define an internal ontology that is good enough to prevent Megatron, TorchTitan, or other external engines from becoming our source of truth by accident.

## Core Position

We want to own the semantics and rent the machinery.

That means:

- our internal model defines what a training run means
- external engines are lowering targets and runtime substrates
- backend-specific ugliness is allowed in adapters
- backend vocabulary should not leak into the rest of the codebase

This is an anti-corruption-layer posture, not a claim that we already know everything about distributed training.

## Current Strategy

Use:

- `seqax` as inspiration for partitioning and communication semantics
- `nmoe` as inspiration for training-state ownership, checkpointing, MoE semantics, and low-precision honesty
- `Megatron` and `TorchTitan` as short-term lowering targets for velocity

Long-term direction:

- keep adapters ugly and explicit
- keep the internal model clean
- gradually build a cleaner backend whose worldview matches the internal semantics better

## Layering

The intended stack is:

1. Declarative spec
2. Internal semantic model
3. Backend lowering
4. Operational runtime

### 1. Declarative spec

This says what we want:

- model family / role
- training objective
- precision policy
- logical parallelism intent
- checkpoint and weight-sync policy
- runtime constraints

### 2. Internal semantic model

This is the real source of truth.

It should express:

- what a training step means
- what state exists and who owns it
- what the logical parallel axes are
- what precision means semantically
- what MoE routing / dispatch / expert ownership mean
- what checkpoint state means
- when inference sees updated weights

### 3. Backend lowering

This translates the internal model into backend-specific config/runtime objects.

Examples:

- Megatron config and process-group setup
- TorchTitan `ModelSpec`, `ParallelDims`, and PyTorch distributed setup

This layer is allowed to be ugly, explicit, and backend-native.

### 4. Operational runtime

This actually makes things happen:

- launching jobs
- initializing distributed runtime
- building wrapped models/optimizers
- running forward/backward/step
- checkpointing
- weight sync
- emitting events and metrics

## A More Precise Split Inside Training

The most important internal boundary is not "trainer vs backend". It is the split between:

1. objective semantics
2. distributed execution semantics
3. optimization/runtime semantics

### Objective semantics

This is what RL / SFT / distillation need to tell the training system.

Examples:

- what batch fields exist
- what model outputs are required
- what loss terms exist
- what auxiliary signals matter
- what metrics should come back for observability

### Distributed execution semantics

This is how the system realizes those requirements under a partitioned model and batch layout.

Examples:

- what tensors are local vs sharded vs transiently materialized
- what collectives are required
- what values are "at rest" sharded vs gathered on demand
- what communication must happen before and after local compute

### Optimization/runtime semantics

This is the actual training step and state evolution:

- backward
- optimizer step
- scheduler update
- checkpoint/version update
- event/metric emission

The key rule:

- objectives should specify semantic tensor requirements
- the execution/lowering layer should decide how to realize them
- objectives should not directly encode backend communication strategy

## Current Boundary Direction

The current best synthesis is:

- use a Tinker-like contract at the outer boundary
- use a seqax-like explicit language for the internal realization semantics
- be more opinionated about batch typing, objective typing, and step results

More concretely:

- the outer API should look like a clean training protocol that other systems can call
- the inner source-level training semantics should stay authorial and explicit about
  partitioning, materialization, and collectives
- backend lowerings should realize those explicit semantics using ugly rented machinery

This means we do **not** want the whole system to become either:

- a giant backend-shaped engine API, or
- a purely declarative config-driven trainer with no explicit semantic center

Instead, the shape is:

1. typed semantic batch/objective/result contract
2. explicit step-language / realization semantics
3. backend lowering
4. operational runtime

## The Training-Step Contract

The next major design task is to harden the training-step contract.

The question is:

- what precisely do RL / SFT / distillation need to tell the training code?
- what do they need to get back?

This contract is more important than the current backend protocol.

The likely common shape is:

- semantic batch inputs
- requested forward products
- loss composition
- returned metrics/state updates

This suggests eventual internal objects like:

- `TrainingBatch`
- `ForwardProducts`
- `ObjectiveSpec`
- `StepResult`
- `TrainingState`

Those names are provisional, but the distinction is important.

## Design Inspiration from seqax and nmoe

### seqax

Take from `seqax` / `shardlib`:

- named logical dimensions
- explicit partitioned layout as semantic state
- communication as a before/after layout transition
- a hard split between local compute and communication

The exact string DSL is optional, but the worldview is desirable.

### nmoe

Take from `nmoe`:

- dense vs expert state ownership as a real semantic distinction
- explicit low-precision consequences rather than cosmetic dtype flags
- checkpointing aligned to ownership boundaries
- deterministic data-plan / resume semantics

## Working Principle for External Engines

Megatron and TorchTitan are lowering targets, not the truth.

They are useful because they already contain painful machinery:

- distributed runtime setup
- process-group or mesh setup
- low-precision integration
- model wrapping / sharding
- optimizer/runtime machinery

But they should remain outside the semantic center of the codebase.

The intended posture is:

- clean internal semantics
- ugly explicit adapters
- rented machinery for velocity
- cleaner native backend later if the external engines become the main source of pain

## What We Should Model Internally

The first-pass ontology should be intentionally narrow and honest.

Likely core objects:

- `TrainingPlan`
- `ModelSpec`
- `TrainingBatch`
- `LossSpec`
- `ParallelPlan`
- `PrecisionPolicy`
- `MoESpec`
- `CheckpointPlan`
- `WeightSyncPolicy`
- `BackendCapabilities`

These names are provisional. The important thing is the semantic distinction, not the exact spelling.

## What We Should Not Model Internally Yet

Avoid dragging backend-native details into the core model too early.

Do not make these the source of truth:

- Megatron process-group algebra
- TorchTitan mesh names
- DTensor placement syntax as a required internal representation
- DeepSpeed JSON config shape
- backend-native checkpoint formats
- backend-native step APIs
- backend-specific low-precision recipe names

These belong in adapters or extension payloads.

## Design Principles

### 1. Logical semantics first

Model:

- data parallel
- tensor parallel
- pipeline parallel
- context parallel
- expert parallel

as logical axes and intentions, not as backend process groups or mesh names.

### 2. Precision is semantic

Precision is not just a dtype toggle.

Especially for low-precision MoE, the model must be able to distinguish:

- parameter precision
- compute precision
- accumulation precision
- optimizer-state precision
- router precision
- communication precision where relevant

### 3. MoE is a real subsystem

Do not collapse MoE into "just another module."

The internal model should eventually distinguish:

- routing policy
- expert ownership
- capacity/drop policy
- dispatch/communication intent
- shared experts
- observability around load balancing and token movement

### 4. Partitioning and communication should be explicit

Take inspiration from `seqax`:

- partitioned layout is semantic state
- communication transforms layout
- local compute should not secretly imply communication

We do not need to copy the exact DSL immediately, but we do want the worldview.

### 5. Checkpointing follows state ownership

Checkpoint plans should reflect the actual ownership boundaries in the system:

- replicated dense state
- expert-local state
- optimizer state
- scheduler state
- RNG / loader cursor / data-plan state
- weight-version metadata

### 6. Backends are capability sets, not identical boxes

Backends should advertise what they can actually do:

- TP / PP / CP / EP support
- low-precision mode support
- checkpoint import/export support
- async weight visibility / sync support
- supported MoE dispatch strategies

The internal model should not assume all backends are equally expressive.

## Working Rule for Incompleteness

We do not need to understand all of distributed training before modeling it.

We should:

- model the parts we understand
- keep the model explicitly provisional
- document non-goals
- revise it when real use cases prove it wrong

The failure mode to avoid is not "having an incomplete model." The failure mode is letting external engines define the ontology by default.

## Immediate Next Step

Before more RL training work:

- draft the narrow first-pass internal objects
- keep the core semantics separate from Megatron/TorchTitan lowering code
- tolerate ugly adapters
- use real pressure from backend integration to revise the internal model instead of letting backend terms leak inward

The first concrete cleanup targets are:

- `batch`
- `loss_fn`
- `StepResult`

Those are the places where the current API is still too weakly typed or too backend-shaped.
