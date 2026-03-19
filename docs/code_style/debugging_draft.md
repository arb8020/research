# Debugging Draft

Debugging is not "adding prints until the code confesses." Debugging is finding the first place the program stops telling the truth about its invariants.

Most bugs fall into one of these buckets:
- A boundary admitted invalid data and failed to normalize it.
- State changed in a way the code claimed was impossible.
- Control flow hid the real cause behind indirection, mutation, or ambient effects.
- The code is modeling the wrong thing, so the bug is not local to the failing line.

The job is to identify the violated invariant, find the first dishonest transition, and then fix the model so the bug becomes harder to reintroduce.

## Core Principles

- **Reproduce before reasoning**. If you cannot make the failure happen on demand, you are not debugging yet.
- **Minimize before instrumenting**. Shrink the failing case until the state machine is small enough to reason about locally.
- **Normalize at the boundary**. Parse and validate external data once. Internal code should operate on trusted shapes.
- **Assert programmer facts**. Use `assert` for invariants that must hold if the program is correct.
- **Raise or return errors at boundaries**. Invalid input and operational failures are not invariants.
- **Prefer visible state**. Debuggable code keeps ownership clear and intermediate values inspectable.
- **Logs are observation, not proof**. Logging helps locate behavior in production, but it does not replace assertions, simplification, or a good local repro.

## The Workflow

### 1. Reproduce Reliably

Make the bug happen on demand.

- Save the exact input, config, command, seed, request body, or event sequence.
- Remove unrelated moving parts.
- If the failure is nondeterministic, narrow the nondeterminism first.

If you cannot reproduce the bug, do not start "fixing" it. At that point you are editing blind.

### 2. Minimize the Failing Case

Reduce the bug to the smallest program, test, request, or dataset that still fails.

- Delete setup until the bug disappears, then add the last necessary piece back.
- Prefer one failing integration test over debugging the whole application live.
- For pure transforms, extract the smallest input that demonstrates the wrong denotation.

Smaller failing cases make control flow honest. Large failing cases hide the first cause behind noise.

### 3. Find the First Dishonest Boundary

Most bugs begin before the observed failure.

Look for the first place where:
- external data enters the system unvalidated
- a value changes meaning without a new name
- two representations of the same fact drift apart
- a function accepts a broader state space than it can actually handle

The important question is not "where did it crash?" The important question is "where did the program first start lying?"

### 4. Instrument Invariants, Not Hunches

Add instrumentation that sharpens the search space.

Prefer:
- `assert` statements around assumptions
- named intermediate variables instead of nested expressions
- temporary probes at boundaries and state transitions
- one-off regression tests that pin down the failing case

Be careful with:
- broad `try/except` added "for safety"
- fallback paths that let execution continue dishonestly
- scattershot logging with no hypothesis

Good debug instrumentation should either:
- prove an invariant holds
- prove an invariant fails
- narrow which transition broke it

### 5. Fix the Model, Not the Symptom

Once the violated invariant is clear, choose the smallest change that makes the illegal state harder to represent.

Good fixes often look like:
- stronger normalization at the boundary
- replacing implicit state with explicit data
- splitting one ambiguous type into two honest ones
- moving branching logic into one visible owner
- deleting a fallback path that concealed the bug

Weak fixes usually look like:
- "just in case" guards with no semantics
- retrying a programmer error
- catching an exception and continuing
- adding more logs while leaving the invariant weak

## Assertions vs Errors

Use the right failure channel.

- **`assert`**: broken invariant, impossible state, programmer error
- **`raise`**: invalid input, violated precondition, unrecoverable boundary failure
- **result/error value**: expected operational failure where the caller has a real choice

Do not catch errors just to keep the program alive. Honest failure is easier to debug than dishonest continuation.

## Logging

Structured logging is useful, especially for distributed systems and production debugging. Correlation IDs, request IDs, and high-cardinality context often matter.

But logs have a narrow role:
- they help reconstruct what happened
- they do not prove the code is correct
- they do not replace local reasoning
- they do not make illegal states unrepresentable

If you are debugging locally, a minimized repro plus assertions is usually higher signal than adding more logs.

## Debugging Heuristics

- If a function is hard to step through, it probably owns too much state.
- If the same fact exists in two places, look for drift.
- If a bug disappears when you add logging, suspect timing or shared mutable state.
- If you need a lot of logs to understand a pure transform, the code shape is probably too implicit.
- If the bug fix requires a fallback, check whether the real problem is an unmodeled sum type.
- If you cannot explain the failing invariant in one sentence, you probably have not found it yet.

## What To Leave Behind

A debugging session should leave the codebase more honest than it was before.

Usually that means:
- one regression test for the minimized failing case
- one or more assertions documenting the recovered invariant
- boundary normalization if malformed input caused the issue
- deletion of temporary debug probes once the permanent shape is clear

Do not leave behind panic-driven logging or defensive branches that obscure the real contract.

## Checklist

Before calling a bug fixed, ask:

1. Can I reproduce the original failure?
2. Did I minimize it?
3. What invariant was violated?
4. Where did the program first become dishonest?
5. Did I fix the model, not just the symptom?
6. What test or assertion now prevents regression?
