# Testing Philosophy — Handoff

## What this is

A record of the testing and correctness philosophy we converged on, so future work stays coherent with it. Written after the `_parse_usage` cache accounting fix as a concrete example.

---

## The hierarchy

1. **Types** — illegal states unrepresentable at the type level.
2. **Assertions** — invariants, pre/postconditions, impossible internal states. Co-located with the code. Tiger style: assert pre/postconditions, split compound assertions, assert both positive and negative space, pair assertions where the same property can be checked at two points.
3. **Tests** — crystallize behavior proven manually. Last step, not first.

The question before writing a test: *do types and assertions alone make it impossible for this to be wrong?* If yes, no test needed.

## Unit tests

Reserved for non-trivial pure logic where types and assertions cannot fully guarantee correctness. The bar is: would a future reader be surprised by edge-case behavior in a way that in-code assertions can't express?

If the assertion in the function already documents the invariant, a unit test is redundant — it just restates the code in a separate file that drifts.

## Integration tests

Main line of defense. Boundary behavior run manually first, then crystallized. No mocks of external systems.

## Live tests

Own "does the real provider return the fields we expect." The `_parse_usage` normalization logic lives in assertions in the function itself. The corresponding live test (not yet written) would send two requests to OpenRouter with a stable prefix and assert nonzero `cache_write_tokens` in the returned span.

## Concrete example: `_parse_usage`

**The fix:** OpenRouter reports `prompt_tokens_details.cached_tokens` as `(hits + writes)`. We were only reading `cached_tokens` into `cache_read_tokens` and never reading `cache_write_tokens`, so cache write accounting was always zero and cache read accounting was inflated.

**Where correctness lives:** In assertions inside `_parse_usage` in `rollouts/providers/openai_completions.py`:
- Non-negative preconditions on raw fields
- `cache_write_tokens <= reported_cached_tokens` (the overlapping representation invariant)
- Non-negative postconditions on all output fields

**What we deleted:** A unit test file (`tests/unit/test_parse_usage.py`) that tested the same normalization math the assertions already enforce. The comment in the function documents the OpenRouter behavior; the assertions enforce it.

**What's still missing:** A live test in `tests/live/` that hits the real OpenRouter API with a stable prefix and asserts nonzero cache fields come back in the span. That test owns the boundary claim the unit test was pretending to own.

## On mocks

Mocks encode assumptions about external systems in a separate file that only runs sometimes. They drift. Mock only at coarse cut points where the boundary is already covered by live/integration tests. Never mock to make a unit test possible.

## References

- `docs/code_style/grugbrain_testing.md`
- `docs/code_style/why_not_coverage.md`
- `docs/code_style/codex_codestyle_interview.md` — "Testing seams" section
- `docs/code_style/tiger_style.md` — assertion density and pairing rules
