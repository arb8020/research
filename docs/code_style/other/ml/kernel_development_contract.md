# Kernel Development Contract

Kernel work is easy to mythologize in the same way as training instability: write some code, benchmark it, stare at profiler output, hope the fast version is also correct.

That is not a good development model.

The better model is:

- kernel optimization is a semantics-preservation problem before it is a speed problem
- the kernel has a contract
- staging, layout, precision, and work partitioning must be explicit enough to reason about

This matters even more if the implementation loop is "human + LLM", because under-modeled kernels are hard to synthesize, debug, and trust.

## Core Orientation

Before trying to make a kernel fast, define what the kernel means.

At minimum, the kernel contract should make explicit:

- input/output shapes
- layout assumptions
- dtype assumptions
- accumulation precision
- aliasing assumptions
- memory-space assumptions
- synchronization requirements
- tiling / partitioning semantics
- numerically significant differences from a reference implementation

If these are not written down, the implementation will drift and benchmarks will be hard to interpret.

## The Kernel Contract

Every kernel should have an explicit contract describing:

### 1. Shape Contract

Examples:
- supported ranks
- static vs dynamic dimensions
- divisibility requirements
- padding behavior
- sequence/block/tile assumptions

### 2. Layout Contract

Examples:
- contiguous vs strided
- row-major / column-major
- block layout
- swizzle assumptions
- expected tensor partitioning

### 3. Precision Contract

Examples:
- input dtype
- output dtype
- accumulation dtype
- cast points
- tolerance to reduced precision

### 4. Work Partition Contract

Examples:
- block / warp / warpgroup ownership
- tile ownership
- producer/consumer roles
- staging into shared memory / registers

### 5. Correctness Contract

Examples:
- exactness vs tolerance
- expected nondeterminism
- reference implementation
- tolerated numerical error envelope

### 6. Performance Contract

Examples:
- target shapes/regimes
- expected occupancy tradeoffs
- bandwidth vs compute bound expectation
- expected bottleneck

Without these, "the kernel" is not one thing. It is a vague performance artifact.

## Kernel Development as a State Machine

Do not jump directly from idea to fully-optimized kernel.

Use staged development:

1. obvious reference implementation
2. correctness harness
3. explicit contract and invariants
4. first specialized kernel
5. benchmark + profile
6. refine tiling / memory movement / synchronization
7. inspect generated PTX/SASS if needed

This matters because:

- a fast wrong kernel is worse than a slow correct one
- an unreadable kernel without a reference is hard to repair
- optimization changes often alter semantics accidentally

## What Should Be Explicit

Make these things explicit in code or adjacent notes:

- which thread/block/warp/warpgroup owns what
- where data lives at each stage
- when data moves between memory spaces
- when synchronization is required
- which values are compile-time vs runtime
- which dimensions are assumed to align or divide evenly
- what happens in edge cases

If a reader must infer these from scattered indexing arithmetic, the kernel is under-modeled.

## Staging and Local Legibility

Kernel DSLs often use staged programming:

- host/meta code chooses layouts, tiles, and lowering
- device/runtime code executes the generated program

This is not automatically bad. But it can hurt local legibility if stage boundaries are unclear.

Good staged kernel code makes it easy to answer:

- what runs now vs later?
- what is compile-time vs runtime?
- what does this call lower to?
- what hardware-level operation is being selected?

If the same syntax sometimes means "execute now" and sometimes means "emit later" without clear markers, the code becomes hard to reason about.

## Correctness First

Every optimized kernel should have:

- a slow trustworthy reference
- shape/layout precondition checks
- a correctness harness over representative inputs
- targeted edge-case inputs
- numerical checks with explicit tolerances

Do not compare only against throughput.

Correctness harnesses should vary:

- shape
- stride/layout
- dtype
- alignment/divisibility edge cases
- value distributions that expose numerical issues

## Numerical Honesty

A kernel can be "correct enough" only relative to a precision contract.

Be explicit about:

- accumulation dtype
- reduction order differences
- fused vs unfused semantics
- tolerated error bounds
- deterministic vs nondeterministic behavior

If the optimized kernel changes numerics in a meaningful way, that should be named, not discovered later during training divergence.

## Performance Work Is a Hypothesis Loop

Do not tweak kernels blindly.

Use this loop:

1. state the bottleneck hypothesis
2. change one thing
3. benchmark
4. profile
5. inspect whether the expected counters moved

Examples:

- "this kernel is bandwidth bound because shared-memory staging is insufficient"
- "occupancy is too low because register pressure is too high"
- "tensor-core utilization is poor because tile shape is misaligned with the problem"

That is much better than "I changed the tile size and it felt faster".

## What to Log or Measure

At minimum:

- latency / throughput
- achieved bandwidth
- achieved FLOP/s if relevant
- occupancy
- register usage
- shared memory usage
- launch configuration
- shape / dtype / layout regime

If using a DSL, also preserve:

- generated IR/PTX/SASS when relevant
- compile-time choices
- dispatch path chosen

Without this, performance regressions and semantic regressions are hard to localize.

## Better Language for Kernel Work

Avoid:

- "this kernel is weird"
- "the compiler is being dumb"
- "the fast version seems fine"

Prefer:

- the staging model hides which memory movement primitive is selected
- the warpgroup ownership is implicit in the indexing arithmetic
- the accumulation precision changed and the training numerics drifted
- the kernel is correct for aligned tiles but under-modeled for ragged edges
- the dispatch path depends on layout metadata that is not visible at the call site

This naming matters because kernel bugs are often semantic bugs disguised as performance work.

## The Human + LLM Workflow

For human + LLM kernel synthesis, be stricter than usual about the contract.

Before asking for implementation changes, specify:

- exact shape/layout regime
- target hardware
- precision policy
- edge-case behavior
- expected reference semantics
- benchmark regime

Then make the LLM work against a visible contract instead of vague performance goals.

This is especially important for Blackwell/Hopper-era kernels where:

- memory movement primitives matter
- staging semantics matter
- layout and tile selection matter
- small semantic mistakes can look like mysterious training problems

## Debugging Ladder

When a kernel fails or behaves strangely:

1. verify reference correctness on a tiny shape
2. verify optimized correctness on the same shape
3. expand to representative aligned shapes
4. expand to ragged / awkward edge cases
5. compare numerical error across dtypes
6. benchmark only after correctness is stable
7. inspect generated code if perf or semantics still surprise you

Do not jump straight to profile-guided tuning if the semantics are still blurry.

## PL / Type / Design Vocabulary for This Domain

Useful words:

- `contract`: what the kernel promises
- `invariant`: shape/layout/sync properties that must hold
- `staging`: compile-time vs runtime split
- `effect`: memory movement, synchronization, launch, IO with the device
- `local reasoning`: how much you can infer from the call site and nearby code
- `denotation`: what the kernel means, apart from how it is lowered

These words help because kernel development often fails when semantics are implicit and only performance is discussed.

## A Good Default Principle

Treat kernel work as contract -> reference -> instrumentation -> optimization.

Speed matters, but speed without explicit semantics is a debugging debt that often gets paid during training.

