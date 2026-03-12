# Distributed Training as a Runtime Contract

Distributed training failures are often described as "NCCL hell", "cluster cursed", or "the run hung". That language is too vague to be useful.

The better model is:

- distributed training is a stack of semantic boundaries
- startup is a state machine, not a single step
- many failures come from hidden environment state and weak observability

If we want to spend more time running experiments and less time fighting infra, we should design the runtime so that environment assumptions are normalized at the boundary, invariants are checked explicitly, and stage transitions are observable.

## Core Orientation

Treat distributed training as a program that configures and operates another distributed system, not as a single script with some environment variables.

That means:

- one explicit source of truth for launch/runtime config
- one preflight layer that normalizes environment state
- one set of invariants checked aggressively
- one clear startup state machine
- one small ladder of smoke tests

The goal is not "never fail". The goal is to fail early, locally, and informatively.

## The Semantic Boundaries

Most training failures are not one bug. They are failures at one of a few boundaries:

### 1. Python / Package Boundary

Examples:
- import failures
- wheel / ABI mismatch
- optional dependency missing
- wrong torch / flash-attn / xformers / driver combination

This is environment normalization work. It should fail before any real training work starts.

### 2. CUDA / Runtime Boundary

Examples:
- driver mismatch
- toolkit/runtime mismatch
- wrong visible devices
- unsupported compute capability
- OOM during init

This is where we establish that the local machine can do the work we think it can do.

### 3. Rendezvous Boundary

Examples:
- wrong rank
- wrong world size
- wrong master addr / port
- launcher mismatch
- processes cannot see each other

This is where "we launched some processes" becomes "we formed a distributed job".

### 4. Transport Boundary

Examples:
- NCCL picks the wrong interface
- container networking prevents peer connectivity
- IB / NVLink / PCIe assumptions are wrong
- shared memory / IPC is misconfigured
- transport silently falls back to something slower or less reliable

This is not model semantics. It is data-movement semantics.

### 5. Training Semantics Boundary

Examples:
- bad batch math
- parallelism sizes inconsistent with world size
- gradients sync at the wrong points
- checkpoint semantics unclear
- stale weights or mismatched versions

This is the boundary where the run may "work" mechanically but still be semantically wrong.

## What the Runtime Contract Should Contain

The system should have one explicit runtime contract object that captures the information the run actually depends on.

At minimum it should contain:

- package/runtime versions
- launcher / rendezvous config
- rank / local-rank / world-size mapping
- visible CUDA devices and chosen device
- parallelism sizes
- effective batch-size math
- expected transport/backend
- checkpoint / artifact paths
- timeout policy
- experiment identity and version metadata

Do not spread this across shell scripts, hidden defaults, and environment variables if you can avoid it. Normalize those inputs once, then work from one trusted internal representation.

This is a direct application of the general code-style rule:

- parse and normalize at the boundary
- run core logic on trusted data

## Startup Is a State Machine

A training job does not "start". It passes through stages.

Model it that way.

Useful stage markers:

- `BOOTSTRAP_OK`
- `IMPORTS_OK`
- `CUDA_INIT_OK`
- `RENDEZVOUS_OK`
- `PROCESS_GROUP_OK`
- `MODEL_INIT_OK`
- `FIRST_FORWARD_OK`
- `FIRST_BACKWARD_OK`
- `FIRST_ALLREDUCE_OK`
- `FIRST_STEP_OK`

These transitions should be logged explicitly.

Why:

- if a job hangs before `RENDEZVOUS_OK`, that is not the same class of problem as a hang after `FIRST_ALLREDUCE_OK`
- "job hung" is not actionable
- stage transitions make failures local and legible

## Preflight: Fail Before the Real Run

Before the real job starts, run a preflight that checks:

- required imports succeed
- runtime versions are what we expect
- CUDA devices are visible
- local rank maps to a real device
- world-size math is consistent with parallelism config
- checkpoint / output paths are writable
- rendezvous host/port config is valid
- expected transport/backend is available

This should be explicit and cheap.

If a run cannot pass preflight, it should not begin.

## Observability: Print the Run Manifest

At startup, emit a compact run manifest that records the facts we will want when the run fails.

Useful fields:

- hostname
- pid
- rank / local rank / world size
- visible devices
- selected device
- torch / cuda / nccl versions
- launcher type
- master addr / port
- backend / transport assumptions
- parallelism sizes
- microbatch / global batch math
- checkpoint path
- run / experiment identifier

Without this, many debugging sessions become archaeology.

## Debugging Ladder

Do not debug "training". Debug the smallest failing stage.

Keep a small ladder of smoke tests:

1. import / version check
2. single-GPU local run
3. 2-GPU single-node minimal process-group init
4. minimal all-reduce
5. one forward/backward step
6. one optimizer step with the real model
7. full training run

Each rung should preserve as much of the real stack as possible while shrinking the search space.

This matters because:

- if single-GPU works but 2-GPU process-group init fails, the problem is probably not model code
- if first forward works but first all-reduce hangs, that is a transport/process-group problem
- if all collectives work but training diverges, that is likely training semantics rather than infra

## How to Describe Failures Honestly

Avoid:

- "distributed training is broken"
- "NCCL is cursed"
- "the cluster is weird"

Prefer:

- all ranks launch, but process-group init hangs before first collective
- single-node works, multi-node fails at rendezvous
- forward/backward succeeds, first gradient all-reduce hangs
- NCCL transport selection is wrong for this container/network setup
- the run reaches `FIRST_STEP_OK`, then diverges numerically
- this is an environment normalization problem, not a model bug

Precise diagnosis saves time because it names the failing boundary and stage.

## Aggressive Invariants

Assert the invariants that define a valid run.

Examples:

- `world_size == dp * tp * pp`
- local rank maps to a visible CUDA device
- effective batch size matches scheduler assumptions
- sequence length and sharding assumptions are compatible
- checkpoint directory exists or can be created
- expected backend exists on this host

These checks should happen as early as possible.

Do not wait for a hang 20 minutes into the run to learn that the configuration was impossible from the start.

## Separate Infrastructure Failure from Experiment Failure

If the run never reaches `FIRST_ALLREDUCE_OK`, you do not have an experiment result.

If the run reaches `FIRST_STEP_OK` and then diverges, that is a different class of problem.

Keep these categories separate:

- infrastructure / environment failure
- runtime orchestration failure
- model / optimization / training-semantics failure

Confusing them leads to random hyperparameter fiddling when the real problem is infrastructure, or endless infra debugging when the real problem is unstable optimization.

## Design for the Future You

The future implementer should not have to rediscover how a run is supposed to work from shell history and log fragments.

They should be able to answer:

- what are the required inputs to launch this run?
- what invariants define a valid configuration?
- what stage did the run reach?
- what transport/backend did we expect to use?
- did this fail before training started, during collectives, or during optimization?

If the runtime cannot answer these questions quickly, it is under-modeled.

## PL / Type / Design Vocabulary for This Domain

Useful words for reasoning about distributed training:

- `normalization`: convert messy shell/env/container state into one trusted config
- `invariant`: a condition that must hold for the run to be valid
- `effect`: external behavior like network, filesystem, CUDA init, collectives
- `state machine`: startup and training are phases, not one step
- `capability`: what transports/backends/topologies this environment actually supports
- `local reasoning`: how much you can infer from the current stage/logs without guessing

This vocabulary matters because many infra failures are really failures of modeling, not just failures of code.

## A Good Default Principle

Prefer one explicit runtime contract, one preflight, one manifest, one set of stage markers, and one debugging ladder.

That is much cheaper than repeatedly debugging ambient environment state from scratch.
