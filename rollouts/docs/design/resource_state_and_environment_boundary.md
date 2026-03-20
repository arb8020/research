## Resource State And Environment Boundary

This note sharpens one question that is currently too mushy in the code:

- what is a `resource`?
- what is an `environment`?
- where do tools and ser/deser belong?

### Current confusion

Today `resource` can mean several different things:

- a live workspace/session handle
- an evaluator
- a terminal task handle
- a pooled lease
- sometimes also the ad hoc serialized state needed to restore one of the above

That ambiguity makes KernelBench harder to reason about because there are now
multiple denotations:

- constrained `write_kernel` environment
- projected-workspace external attempt
- future workspace-style SDK-agent environment

### Intended split

The clean split should be:

1. `ResourceState`
- serializable handle/reference for a resource
- enough information to rehydrate the live capability later
- no tool semantics

2. `ResourceHandle`
- live capability object
- examples: workspace/session, evaluator, terminal task, pool lease
- exposes effects like `run`, `read_file`, `write_file`, `score_one`
- no task semantics

3. `Environment`
- owns the task state machine
- owns visible tool calls
- owns how resource state is embedded in environment ser/deser

### What this means

A resource should not directly define model-facing tools.

A resource may optionally have a serializable state representation, but the
environment owns the real serialization boundary because the environment is the
thing that must be restorable as a task state machine.

So:

- resources provide capabilities
- environments expose tools built from those capabilities
- environment state includes serialized resource state when the resource is part
  of the environment's denotation

### KernelBench consequence

For a workspace-style KernelBench environment, the right shape is:

- injected workspace resource
- optional injected evaluator resource
- `CodingEnvironment`-style tools for file/terminal effects
- KernelBench-specific tools layered on top if needed

The external-agent KernelBench path can still exist as an `attempt_executor`,
but the internal SDK-agent path should be a normal environment over injected
resources.

### Near-term refactor direction

1. Introduce a small normalized `ResourceState` product type.
2. Adapt workspace-style resources first:
- Modal workspace
- Bifrost workspace
- evaluator later if needed
3. Keep tool exposure in the environment layer.
4. Use KernelBench as the forcing example for the new boundary.
