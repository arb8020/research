# Async RL Design Decisions

## Context
We are defining the async RL training architecture interactively.

Priority order:
1. Speed
2. Debuggability / observability
3. Explicit semantics over hidden magic

Design rule:
- Default modes should maximize safe throughput.
- Sketchier behavior should exist only as explicit opt-in modes.

## Decisions

### 1. Primary Topology: Disaggregated
Decision:
- Make disaggregated inference/training the primary architecture.
- Treat colocated mode as a constrained fallback, mainly for very small setups.

Why:
- True overlap requires disaggregated inference and training.
- It keeps ownership clearer: inference generates, trainer optimizes, sync transports, buffer holds in-flight work.
- This matches the speed-first goal while keeping boundaries explicit.

Notes:
- Colocated still matters as a fallback, especially when only one GPU is available.

### 2. Buffer Semantics: Stream-Style Production
Decision:
- Prefer logically unbounded / stream-style production semantics over a producer-blocking bounded queue as the default.

Why:
- We want generation to keep flowing when possible.
- A hard bounded queue can become a hidden throughput cliff.
- We are okay with generation outrunning training as long as the system behavior is explicit and observable.

Important clarification:
- "Logically unbounded" does not mean "unbounded invisible damage".
- Overload behavior still needs explicit safety valves.

### 3. Primary Safety Valve: Version Rejection
Decision:
- Use version rejection as the primary stale-data safety valve.

Why:
- It is explicit and easy to reason about.
- It preserves the hot path better than turning producer blocking into the default overload policy.

Open follow-up:
- Define exactly what "too stale" means.
- Likely default: fixed max version lag.

### 4. API Granularity: Sample First, No Discontinuities
Decision:
- The default user-facing unit is the sample.
- The design must preserve ergonomic access to deeper control layers.

Why:
- We want oversampling and late group formation.
- We do not want API discontinuities where users must rewrite everything to gain finer control.

Required granularity levels:
- Sample level
- Group level
- Partial-sample / in-flight level
- Future token-level provenance hooks if needed

Rule:
- Higher-level APIs should be built on top of lower-level ones without hiding them.

### 5. Atomic Buffered Unit: Sample
Decision:
- The primary buffered/admission/rejection unit is the sample.

Why:
- It supports oversample-first, group-later workflows.
- It keeps the default API simple while preserving flexibility.

Implication:
- Group construction becomes an explicit downstream stage.

Open follow-up:
- Define the group assembly policy for GRPO-style training.

### 6. Backlog Location: Memory First
Decision:
- Keep backlog in memory by default.
- Add spill-to-disk only as a safety valve if needed.
- Consider a durable external stream only if we later prove we need it.

Why:
- Memory-first is simplest and fastest.
- Spill adds state and complexity, so it should be an earned escape hatch.
- Durable streams are not the baseline design.

### 7. Default Stale Rejection Location: Buffer Admission Boundary
Decision:
- Reject stale samples at the buffer admission / exposure boundary.
- Keep a trainer-side re-check as a safety assertion.

Why:
- It keeps freshness logic out of the rollout server by default.
- It makes the rejection point explicit and observable.
- It preserves a simpler rollout server contract.

Rejected object:
- The default rejected object is the completed sample.

### 8. Group Refill Policy: Request More Samples
Decision:
- If stale rejection leaves too few admissible samples to form a training group, request more samples from the rollout server.

Why:
- We want sample-level buffering and rejection while still treating group formation as an explicit downstream stage.
- This preserves oversample-first, group-later semantics.
- It avoids silently shrinking the statistical object used for training.

Implication:
- Group assembly is an active stage, not just a passive reshape.
- The assembler must be able to detect incomplete groups and trigger refill.

### 9. Overload Handling: Policy Surface, Not Hardcoded Magic
Decision:
- Overload handling should be an explicit policy surface.
- Do not hardcode one default behavior too early beyond stale rejection.

Why:
- We want the system to acknowledge overload as a real operating condition.
- Users should be able to choose within reason how the system responds.
- This preserves explicit semantics and avoids hidden behavior cliffs.

Likely policy knobs:
- stale rejection threshold
- cancel stale in-flight work
- stop admitting new prompts under pressure
- spill to disk
- block generation as a last resort

### 10. Freshness Semantics: Version Lag First
Decision:
- Define staleness primarily in terms of version lag.
- Treat wall-clock age as secondary metadata or an optional guardrail, not the primary freshness rule.

Why:
- Version lag is the semantically relevant quantity for async RL.
- Wall-clock age is useful for observability and debugging but does not define policy quality on its own.

### 11. Default Partial-Rollout Policy: Finish Under Old Version
Decision:
- When a weight update lands during an in-flight rollout, the default behavior is to let that rollout finish under the old version.

Why:
- This is the cleanest default semantics.
- It avoids silently introducing mixed-weight behavior.
- It is easier to debug and reason about than abort/resume or mixed-version continuation.

Sketchier alternatives:
- abort in-flight rollout
- prefix-resume from partial state
- mixed-weight continuation

Rule:
- These should exist only as explicit opt-in modes.

### 12. Default Sync Semantics: Drain In-Flight, Then Sync
Decision:
- The default sync semantics are: stop admitting new rollout work, let in-flight rollouts finish under the old version, then apply the new weights.

Why:
- It preserves clear per-rollout version semantics.
- It is much easier to debug than mixed-version continuation.
- It still preserves the main async benefit at the pipeline level.

Rejected as the default:
- fully blocking step/batch turn-taking
- true in-flight mixed-version sync

Rule:
- More aggressive in-flight sync modes should exist only as explicit opt-ins.

### 13. Sync API Continuity: Separate Semantics from Transport
Decision:
- Keep weight-sync semantics separate from weight-sync transport in the API.

Why:
- "When do weights become visible?" and "How are weights moved?" are different decisions.
- We want users to be able to switch transport without rewriting rollout/training control flow.
- This avoids API discontinuities as we add more aggressive modes later.

Design implication:
- A user should be able to keep the same high-level sync policy while swapping transport backends.

### 14. Default Sync Transport: NCCL First
Decision:
- Default sync transport should be NCCL.
- Fallback transports should remain available.

Preferred ordering:
- NCCL
- shared-filesystem / disk checkpoint reload
- direct HTTP weight transfer only when needed

Why:
- NCCL is the default high-performance path.
- Disk reload is slower but usually simpler and easier to reason about than full weight transfer over HTTP.
- HTTP remains useful as a control plane and as a last-resort transport, but should not define the main architecture.

Clarification:
- HTTP can still be used as the control plane even when the data plane is NCCL or disk.

### 15. Scoring / Rewarding: First-Class Separate Stage
Decision:
- Treat scoring / rewarding as a first-class pipeline stage from the beginning.

Why:
- Rewarding may need its own inference endpoint, worker management, scaling policy, and observability.
- This is especially important for LLM-as-a-judge, PRM-style scoring, and distillation-like setups.
- We do not want the architecture to hardcode a cheap synchronous verifier assumption.

Design implication:
- The core pipeline should be modeled as:
  - generation
  - scoring / reward computation
  - group assembly
  - training
  - weight sync

Rule:
- The scoring stage should be pluggable without changing the surrounding async machinery.

### 16. Pool Split: Expose It Externally
Decision:
- Expose the split between generation and scoring pools externally in the design/API rather than hiding it purely as an implementation detail.

Why:
- Users may want separate endpoint management, scaling, hardware allocation, and observability for rollout generation vs scoring.
- This keeps the architecture honest about where work is happening.
- It avoids a later API cliff when users need independent control.

Implication:
- The public model can acknowledge distinct generation and scoring resources while still offering ergonomic defaults.

### 17. Staleness Correction: Hard Rejection by Default, IS as Opt-In
Decision:
- Default to hard version-based stale rejection.
- Expose importance-sampling-style correction as an explicit opt-in policy surface.

Why:
- We do not want to over-engineer the default path.
- We do want the API to admit more advanced off-policy correction without a redesign.
- This keeps the default semantics simple while preserving future flexibility.

Rule:
- IS correction should not be required to understand or debug the default system behavior.

### 18. Environment Resource Management: Separate Lease Manager
Decision:
- Environment resource acquisition and scaling should live in a separate resource / lease manager.
- Environment implementations should consume leases, not manage provider fleets directly.

Why:
- Pooling, warm capacity, autoscaling, and provider failures are infrastructure concerns, not environment-task concerns.
- This keeps environment execution logic simpler and makes resource behavior observable in one place.
- It avoids hidden provider-specific magic inside rollout code.

Boundary:
- Environment implementation:
  - run task inside an already-acquired resource
- Resource / lease manager:
  - acquire
  - release
  - maintain warm capacity
  - scale up / down
  - report utilization, wait times, and failures

### 19. Environment Provider Abstraction: Common Protocol, Provider-Specific Implementations
Decision:
- Use a small common lease-manager protocol with provider-specific implementations behind it.

Why:
- We do not want to over-design a giant multi-provider abstraction.
- We do want to avoid baking Modal-specific assumptions into the main training architecture.
- A narrow protocol preserves API continuity while letting implementations stay concrete.

Likely minimum protocol:
- acquire(requirements, timeout)
- release(lease)
- ensure_capacity(policy or target)
- stats()

## Working Definitions

### Backlog
Backlog means work that has been produced by one stage but not yet consumed by the next stage.

Examples:
- generated samples not yet scored
- scored samples not yet grouped
- grouped samples not yet trained
- pending weight updates not yet applied

## Open Questions

### Next
- What is the minimum observability surface we consider mandatory for the first implementation?

Constraint already decided:
- preserve API continuity
- keep semantics separate from transport
- allow advanced access without forcing rewrites
