# Tests

## Correctness hierarchy

1. **Types** — constrain what values *can* be. Illegal states should be unrepresentable.
2. **Assertions** — constrain what values *should* be at runtime. Pre/postconditions, invariants, impossible internal states. Co-located with the code they protect. This is the primary correctness mechanism.
3. **Tests** — crystallize behavior you've already proven manually. Not how you discover correctness; how you lock it in. Manual proof means running a real config or entrypoint and verifying the output — terminal output, logs, or span contents you could show someone.

## When to write a test

Ask: do types and assertions alone make it impossible for this to be wrong?

If yes — no test needed. The invariant is already enforced at runtime.

If no — write a test. The canonical case is non-trivial pure logic where the input/output relationship is surprising enough that a future reader would not trust it without proof.

## Test tiers

**`unit/`** — non-trivial pure logic where types and assertions alone cannot guarantee correctness. The bar: would a future reader be surprised by what this function does with edge-case inputs, in a way that assertions can't fully express? If the assertion in the function already documents the invariant, a unit test is redundant.

**`integration/`** — boundary behavior you've run manually and want to stay fixed. Tests a real subsystem interaction. No mocks of external systems. This is the main line of defense.

**`regression/`** — bug reproductions. Write the failing test first, then fix. The test must fail without the fix.

**`live/`** — real APIs, real providers, real hardware. Not in default CI. Run manually or in slow CI to verify that an external system behaves as assumed. These tests own "does the real provider return the fields we expect" — not the unit tier.

**`manual/`** — scripts for humans. Not pytest.

## What we don't do

- Mock external systems to hit a coverage number
- Write unit tests that restate what the code already says
- Test only the happy path — the negative space (invalid input, boundary values) is where bugs live
- Treat coverage % as a target — it incentivizes testing the wrong things (Goodhart's law)

## On mocks

Mocks encode assumptions about external systems in a parallel universe of test code that only runs sometimes. They're a second place where the assumption lives, and the second place drifts.

Mock only at coarse, honest cut points where you're replacing a real external system or nondeterministic boundary — and only when that boundary is already covered by live or integration tests. Never mock to make a unit test possible.
