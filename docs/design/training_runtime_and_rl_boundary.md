# Training Runtime Ownership and RL Boundary

This note captures the current synthesis from the training architecture thread,
after comparing our codebase against `nmoe`, `lingua`, Megatron, TorchTitan,
`miles`, `torchforge`, and QED-Nano, and after reading Noumena research posts
`0000` through `0003`.

The point is not to freeze the final API. The point is to record the semantic
center we are trying to protect, the boundaries we want to expose, and the ways
we are deliberately willing to be pragmatic in the near term.

## Core position

We want the training team to own the training system as much as possible.

More precisely:

- training should own runtime semantics
- training should own checkpoint and resume semantics
- training should own eval correctness semantics
- training should own observability semantics
- training should own weight publication semantics
- external engines should be rented machinery, not our ontology

This is deliberately closer in spirit to `nmoe` than to a generic
"backend-shaped" trainer.

## Why this position

The Noumena posts push in the same direction:

- `0000`: MoE training failure is often about router learning and low-precision
  semantics, not just "the model trains slowly"
- `0001`: the trust-critical invariants are resume, eval, and logging
- `0002`: loss alone is not enough; capability and router-health metrics matter
- `0003`: a useful speedrun system owns the benchmark/eval/swap contract

The main lesson is that trust-critical semantics cannot be left as
backend-specific implementation detail.

## What the system should feel like

The intended user experience is:

- training researchers write or modify witness loops
- backend integrators work on realization and lowering
- RL orchestration talks to a narrow training surface
- external runtimes are allowed for leverage, but should be progressively
  constrained into honest witnesses of our semantics

## Three stable seams

We do not have one single "training backend" seam. We have at least three.

### 1. Stable seam to RL orchestration

This should stay narrow.

RL should mostly see:

- `TrainingDatum`
- `forward_backward()`
- weight publication / weight update visibility

Some training semantics will still leak upward because they matter operationally:

- visible weight version
- publication mode
- staleness tolerance
- failure and backpressure semantics

But RL should not need to know:

- TP / PP / EP / CP layout choices
- RDEP vs all-to-all dispatch
- router health internals
- checkpoint ownership details
- backend-native runtime quirks

### 2. Stable seam to training researchers

This should be witness loops.

The important point is that researchers should be authoring code in a small
number of trusted loop shapes, not navigating callback soup and not forced
through a giant backend-shaped engine API.

The intended compromise is:

- canonical witness loop families
- direct authored code inside those loop families
- the ability to register new witness loops if necessary
- fixed trust-critical infrastructure contracts underneath

What should stay standardized even if the loop body is flexible:

- checkpoint semantics
- resume semantics
- eval correctness
- observability core
- publication/version semantics

### 3. Stable seam to backend integrators

This should be realization and lowering.

Backend integrators should work on:

- validating whether a runtime can honestly realize some internal semantics
- lowering those semantics into backend-native config and runtime objects
- failing early when the requested semantics are unsupported

They should not define the semantic center of training.

## What we mean by "own the training loop"

We do not just mean "own the loss callback" or "own `forward_backward`".

We mean the training team should own an authored training state machine.

That state machine includes:

- forward semantics
- backward and update semantics
- optimizer and scheduler evolution
- checkpoint and resume policy
- eval policy
- metrics and event emission
- weight publication lifecycle

This matters because `nmoe` makes clear that backward/update-side semantics are
load-bearing, especially for MoE and low precision.

## Why not completely free-form training programs

We explicitly do **not** want the system to devolve into arbitrary unrelated
training programs if we care about trust.

The risk is semantic drift in exactly the places Noumena treats as
non-negotiable:

- resume semantics fork
- checkpoint contents drift
- eval semantics drift
- metrics stop being comparable
- publication/version semantics drift

So the intended shape is not "everyone writes whatever". The intended shape is:

- a small witness set of loop families
- flexible code inside those families
- an escape hatch to register new families when the existing ones are not enough

## Flexibility we do want

Inside trusted witness loops, we want substantial flexibility.

Examples:

- optimizer family and parameter grouping
- clipping policy
- precision policy
- auxiliary and router losses
- schedules
- trainable-parameter policy such as LoRA vs full update

The rule is:

- be flexible where the science happens
- be rigid where validity is decided

## External backends: pragmatic now, stricter later

Near term, we are willing to be pragmatic in order to unblock RL and get a real
training loop working.

That means:

- Megatron and TorchTitan can act as pragmatic adapters for a while
- backend-native escape hatches are acceptable when they are explicit
- some behaviors may temporarily be realized approximately

Long term, the direction is stricter:

- external runtimes should be witnesses of our semantics
- unsupported requests should fail as `Unsupported`
- semantic gaps should be closed by extending our model and lowerings, not by
  silently falling back to backend-native behavior

## Current priority split

We are now deliberately splitting priorities across the training and RL tracks.

- the RL-unblocking work will continue to exercise TorchTitan and Megatron in
  real rollout-coupled contexts
- the training track should therefore stop spending its highest-value cycles on
  another external-backend witness path
- the training track should instead prioritize sovereign MoE runtime bring-up

This is not a contradiction.

It is a portfolio decision:

- external backend pressure is already being paid for by the RL work
- the highest marginal value on the training side is to start owning the MoE
  runtime semantics directly

So the intended near-term division of labor is:

- RL track: continue backend bring-up and operational unblocking
- training track: single-node, multinode-compatible sovereign runtime bring-up
  with MoE semantics first-class

The key implication is that external backends remain important, but they are no
longer the primary success criterion for training v0.

## Failure mode we want

Unsupported semantics should fail at lowering time, before launch.

This is the intended rule:

- if a backend cannot realize a request honestly, reject it early
- do not launch and discover the mismatch halfway through execution
- do not hide semantic mismatches behind fake defaults

This is especially important for distributed correctness, MoE, and resume.

## Shardlib and the internal source language

The desired internal training style is "write the training loop in shardlib",
with one important refinement.

The whole training program is not literally just shardlib layout expressions.
But shardlib-style semantics should be the internal language for the distributed
execution parts:

- layouts
- materialization
- communication
- local vs non-local compute
- sharding and replication intent

So the right model is:

- witness loop = authored training program
- shardlib-style semantics = source language for realization and distributed
  execution
- lowering = maps that language into Megatron, TorchTitan, or a future native
  runtime

## Internal model ownership

We should also own the internal model definition.

More precisely:

- we should own one internal model denotation
- witness loops and training-step semantics should talk to that denotation
- backend model construction should be derived from it

This does **not** mean we need exactly one concrete runtime implementation of
every model immediately. It means the semantic source of truth for model
structure, MoE structure, checkpoint layout, and trainable parameter families
should be ours rather than borrowed from Megatron, TorchTitan, or Hugging Face.

The intended split is:

- internal model denotation: ours
- training-step realization language: ours
- backend-native model/runtime construction: lowering targets

Inspiration:

- `lingua`: authored training system owns the program
- `nmoe`: authored training system owns MoE/runtime semantics
- `shardlib`: explicit realization language for layout and communication

## Minimal training-step IR

The current best candidate is a deliberately small IR for the training-team
authoring surface.

This is **not** meant to be a full compiler IR and **not** meant to encode
backend-native runtime details like RDEP packing internals.

The first useful operation set looks like:

- `all_gather(spec, x)`
- `psum_scatter(spec, x)`
- `einsum_unreduced(spec, *xs)`
- `materialize(spec, x)`
- `route(spec, x, router) -> RouteResult`
- `dispatch(spec, x, route_result, experts)`
- `expert_compute(spec, x, experts)`
- `combine(spec, expert_out, route_result)`
- `compute_loss(...)`
- `backward(loss)`
- `optimizer_step(state)`

The key semantic side objects are:

- `TrainState`
- `Batch`
- `StepResult`
- `RouteResult`
- `Observability`

The design rule is:

- dense FSDP-style semantics should be expressible now
- MoE route/dispatch/combine should be expressible very soon
- TP does not need to be part of the primary authoring story yet

## Pseudocode: what RL sees

The outer seam should stay thin.

```python
result = trainer.forward_backward(datum)
publication = trainer.publish_weights(...)
```

This is correct for the RL-facing/service-facing view of the system. It is not
the training-team-facing authoring model.

## Pseudocode: dense FSDP witness loop

This is the kind of authored loop we want the training team to be able to write.

```python
def dense_sft_step(state: TrainState, batch: SFTBatch) -> StepResult:
    w_embed = all_gather("vocab hidden/d -> vocab hidden", state.model.w_embed)
    w1 = all_gather("hidden ff/d -> hidden ff", state.model.w1)
    w2 = all_gather("ff hidden/d -> ff hidden", state.model.w2)

    h = einsum_unreduced("tokens/d vocab, vocab hidden -> tokens/d hidden", batch.tokens, w_embed)
    h = einsum_unreduced("tokens/d hidden, hidden ff -> tokens/d ff", h, w1)
    h = gelu(h)
    h = einsum_unreduced("tokens/d ff, ff hidden -> tokens/d hidden", h, w2)

    logits = einsum_unreduced("tokens/d hidden, hidden vocab -> tokens/d vocab", h, w_embed.T)
    loss = masked_xent(logits, batch.labels, batch.loss_mask)

    backward(loss)
    state = optimizer_step(state)

    return StepResult(
        losses={"xent": scalar(loss)},
        observability=Observability.core(),
    )
```

Important points:

- at-rest sharding over `/d` is explicit
- all-gather before use is explicit
- local compute is explicit
- update semantics still live in the authored training step

## Pseudocode: MoE witness loop

This is the kind of MoE-first authored loop we want soon.

```python
def moe_sft_step(state: TrainState, batch: SFTBatch) -> StepResult:
    h = dense_trunk(state.model.trunk, batch.tokens)

    rr = route(
        "tokens/d hidden -> tokens/d experts/k gates/k",
        h,
        state.model.router,
    )

    dispatched = dispatch(
        "tokens/d hidden -> routed/ep hidden",
        h,
        rr,
        state.model.experts,
    )

    expert_out = expert_compute(
        "routed/ep hidden -> routed/ep hidden",
        dispatched,
        state.model.experts,
    )

    combined = combine(
        "routed/ep hidden -> tokens/d hidden",
        expert_out,
        rr,
    )

    logits = state.model.head(combined)
    xent = masked_xent(logits, batch.labels, batch.loss_mask)
    total = xent + rr.router_aux

    backward(total)
    state = optimizer_step(state)

    return StepResult(
        losses={"xent": scalar(xent), "router_aux": scalar(rr.router_aux)},
        observability=Observability(
            router=rr.stats,
            systems=SystemsMetrics.required(),
        ),
    )
```

Important points:

- routing is first-class
- expert ownership is first-class
- dispatch/compute/combine is first-class
- router observability is first-class

This is the main lesson from `nmoe`: the IR should encode the semantic split,
not the RDEP implementation details.

## What `nmoe` says belongs in the semantics

The recent `nmoe` read reinforced that the following are semantic, not just
runtime implementation detail:

- router output and router bias/update semantics
- expert ownership boundaries
- dispatch/local expert compute/combine as a real split
- exact resume semantics:
  - dense/router replicated state
  - expert rank-local state
  - optimizer state
  - loader state
  - RNG state
  - config fingerprint
  - data/mixture plan fingerprint
- router-health observability:
  - CV
  - entropy
  - max load
  - experts active
  - dead experts
- lockstep distributed eval for expert-sharded models

What should stay out of the authoring IR:

- RDEP packing internals
- IPC handle setup
- fused kernel details
- alignment/padding details

Those belong in lowering/runtime implementation.

## What external backends can lower today

This is the current rough picture.

### TorchTitan

- Dense/FSDP semantics: directionally yes
- Explicit shardlib-authored `/d` IR: not yet in our code
- MoE route/dispatch/combine: not yet as our semantics; only via backend-native
  behavior if we add it

### Megatron

- Dense semantics: yes
- Explicit shardlib-authored `/d` IR: not yet in our code
- MoE semantics: only partially, and still backend-native rather than our
  authored semantics

### `nmoe`

- MoE route/dispatch/combine: yes in spirit
- Honest lowering/runtime path in our repo: not implemented yet

This is why the immediate goal should not be "support every backend perfectly."
The immediate goal should be:

1. define the internal model denotation
2. define the small training-step IR
3. make one dense FSDP witness loop and one MoE witness loop use it
4. then lower that into external runtimes as honestly as possible

## The native direction

The long-term direction is a native runtime whose worldview actually matches our
internal semantics.

That runtime does not need to exist immediately. But the internal architecture
should be designed so that it can exist without rewriting the whole system.

This is why external runtimes should be treated as lowering targets and runtime
substrates, not as the source of truth.

## The systems family split

The ecosystem comparison matters because not all systems are shaped the same.

### Authored training systems

Examples:

- `lingua`
- `nmoe`

These put the semantic center of gravity in the training program itself.

### Rented training runtimes

Examples:

- Megatron
- TorchTitan

These put the center of gravity in the backend runtime, and our code lowers into
that runtime.

### RL orchestration systems

Examples:

- `miles`
- `torchforge`
- QED-Nano

These sit above the training runtime and care about rollout/trainer/update
coordination more than training semantics directly.

This split is the main reason a single overloaded notion of "TrainingBackend" is
not enough.

## What Tinker gets right about the boundary

Tinker is useful here because it draws a much cleaner line between training and
RL than our current code does.

At the SDK boundary, Tinker exposes a narrow service-shaped surface:

- `forward_backward(data, loss_fn, loss_fn_config)`
- `optim_step(adam_params)`
- `save_state()` / `load_state_with_optimizer()`
- `save_weights_and_get_sampling_client()`

That is visible directly in the public `TrainingClient`.

The important part is not the exact method names. The important part is the
shape:

- the training service owns execution and persistence
- the caller sends datums plus objective/update requests
- the caller receives training outputs and explicit weight publication handles

Then the cookbook writes authored RL loops *outside* that service boundary.

In the minimal RL loop, the cookbook:

- builds prompts and rollouts
- computes rewards and advantages
- assembles `Datum` objects
- calls `training_client.forward_backward(...)`
- calls `training_client.optim_step(...)`
- periodically republishes weights for sampling

This is a useful proof point for our intended split:

- RL orchestration does not need to know runtime semantics
- the trainer boundary can still be narrow
- authored loops can live above the boundary without backend leakage

Tinker's limitations are also instructive:

- the training boundary is deliberately backend-owned rather than
  training-team-owned
- the loss space is partly enumerated as service-level strings
- optimizer control is narrower than what we ultimately want
- it does not support in-flight weight updates; async RL is expressed as
  "off-by-K" older-policy training instead

So the direction we want is not "become Tinker". The direction is:

- keep a similarly narrow RL-facing seam
- let the training team own the semantic center rather than a hosted service
- allow richer authored witness loops than Tinker exposes
- preserve explicit publication and resume semantics

## What should leak across the RL boundary

Training diagnostics should mostly stay with training.

For example, the RL side should not be responsible for interpreting:

- router health
- checkpoint correctness
- distributed eval correctness
- backend-specific systems failures

The RL side *should* be able to request meaningful training semantics such as:

- LoRA vs full update
- optimizer family
- objective or loss variant

Those are semantic asks, not runtime asks.

Bad RL-facing asks would be things like:

- `tp=4`
- `use_rdep`
- `sequence_parallel=true`
- `fsdp_full_shard`

Those belong below the seam.

## Observability

The training side should emit as much observability as possible, but not through
an unstructured metrics bag.

The likely shape is:

- a required core metrics/event surface
- optional extensions and loop-specific extras

The required core should cover at least:

- objective metrics
- optimizer/update metrics
- publication/version facts

When MoE is involved, the required core should also cover:

- router-health metrics

And, as a systems baseline:

- throughput
- timing
- memory

The important rule is:

- downstream consumers should be able to depend on a stable subset
- training should still be free to emit richer diagnostics

## What this implies for current code

The recent cleanup work is aligned with this direction:

- shared `ParallelIntent` was removed in favor of backend-specific provisioning
  shapes
- the stale fake `nmoe` backend was removed
- `nmoe` is now represented as an honest reserved path rather than a mislabeled
  Hugging Face wrapper

The next design work should probably focus on:

1. witness-loop shape and registration
2. the typed training-step boundary
3. realization/lowering validation
4. backend capabilities
5. observability core

## Open questions

These are not settled yet.

1. How much of the training request should higher-level callers control?
2. What exactly belongs in the typed `TrainingDatum` surface versus a separate
   train-request object?
3. How much of optimization policy should be standardized versus loop-authored?
4. What is the minimum honest `TrainingBackendCapabilities` product type?
5. Which MoE semantics belong directly in the internal realization language?
6. Which publication modes do we want to support first?
7. What is the first practical native runtime target beyond external lowerings?

## Status of the architecture questions

This is the current status of the main architecture questions from the thread.

### Mostly resolved

1. What do we mean by "our own training loop"?
- We want to own the whole training state machine.
- We likely eventually want to own the runtime too, including distributed
  execution semantics, rather than permanently renting them from external
  engines.

2. What are external runtimes allowed to do?
- Near term: act as pragmatic adapters to unblock real training.
- Long term: act as validated witnesses of our semantics.
- Unsupported requests should eventually fail early rather than silently falling
  back to backend-native behavior.

3. What should the RL-facing trainer boundary own?
- More than just gradient updates.
- Training owns checkpointing, publication, eval correctness, observability, and
  runtime semantics.
- RL mostly sends training work and consumes versioned weight updates.

4. What should RL orchestration talk to?
- A thin trainer/service interface.
- The interface should expose training requests plus weight publication and a
  small amount of operational state such as visible version / failure /
  backpressure.

6. How should MoE be represented?
- As a first-class concern soon, not as a deferred extension after dense-only
  design is settled.

7. What is the canonical notion of resume?
- Directionally: exact continuation, not just model weights.
- This should include enough state to make resume semantics trustworthy.

8. What should observability be?
- A required common core, with room for richer loop-specific or backend-specific
  extensions.
- Training should emit as much observability as possible through a stable typed
  surface.

9. What should the first native runtime be?
- Most likely a practical custom trainer with shardlib-inspired semantics,
  rather than a literal reference interpreter.

10. What matters most while iterating?
- Semantic honesty first.
- Also important: fail early, preserve API continuity, and remain pragmatic
  enough to keep real progress moving.

### Still partially open

5. How important is exact semantic equivalence between native and external
lowerings?
- The direction is that this matters a lot.
- We have not yet pinned down whether to treat exact same-plan realizability as
  a hard requirement immediately, or as a target we move toward while external
  backends are still pragmatic adapters.

### Still open in implementation shape

4b. Internally, how factored should the training runtime surface be?
- Externally, RL should see one thin trainer boundary.
- Internally, we have not fully decided how much to split publication, eval, and
  observability into separate surfaces versus keeping one runtime object.

## Current design rule

The shortest correct summary is:

- the training team owns the semantic center
- witness loops are the stable authoring seam
- realization and lowering are the backend-integrator seam
- RL sees a narrow training boundary plus weight updates
- external runtimes are rented machinery
- unsupported semantics should fail early

## Concrete references

- Tinker SDK:
  [training_client.py](/tmp/tinker/src/tinker/lib/public_interfaces/training_client.py)
- Tinker cookbook minimal RL loop:
  [rl_loop.py](/tmp/tinker-cookbook/tinker_cookbook/recipes/rl_loop.py)
- Tinker cookbook RL async/off-policy note:
  [rl-hyperparams.mdx](/tmp/tinker-cookbook/docs/rl/rl-hyperparams.mdx)
