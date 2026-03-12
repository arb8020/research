# Earned Compression

Much of this code style guidance is written as a guardrail against premature or dishonest abstraction. That can make it sound more anti-abstraction than it really is.

The real goal is not "avoid abstraction." The real goal is:

- avoid unearned compression
- seek earned compression

Good semantic compression is one of the main things we want from good code.

## What Compression Means

Compression is when a representation says more with less.

In code, that can mean:

- a better function boundary
- a more honest type
- a small state machine instead of flag soup
- a helper that captures a real repeated pattern
- an API that reflects the actual usage shape
- a declarative spec that lowers into messy runtime machinery

Compression is not automatically good. It is only good when it preserves or sharpens the meaning we care about.

## Unearned Compression

Compression is unearned when it hides uncertainty, mixes distinct cases, or guesses at the wrong seams.

Examples:

- abstracting after one example
- using one nullable field to represent several semantic cases
- hiding failure in ambient exceptions
- creating a generic base class before the real invariants are clear
- wrapping several backend-specific concepts in a fake "neutral" abstraction that explains nothing

Unearned compression launders complexity into a neater-looking surface.

## Earned Compression

Compression is earned when the abstraction has been forced into existence by the problem and makes the important semantics clearer.

Signs that compression is earned:

- there are multiple real examples, not just speculation
- the abstraction exposes a true invariant or repeated structure
- the compressed form is easier to expand mentally without surprises
- the important control-flow and failure facts remain visible
- the abstraction reduces semantic surface area instead of merely hiding mechanics
- invalid states become harder to represent
- the underlying examples still make sense through the abstraction

Good compression can be shorter and more honest than explicit boilerplate.

## Compression vs Cleverness

Compression is not the same as cleverness.

Clever code often makes the surface smaller by increasing the amount of hidden inference the reader has to do.

Good compression does the opposite:

- it removes incidental detail
- it keeps the important structure visible
- it improves local reasoning or makes the global model cleaner

If the abstraction makes the code look simpler while making the semantics harder to see, it is probably not earned.

## What We Actually Want

We want code that:

- says the truest thing first
- compresses real structure
- keeps semantics visible
- uses explicitness as a tool, not as an aesthetic

This is why so much of the surrounding style advice focuses on:

- usage code first
- don't abstract until the pattern is real
- make control flow obvious
- make invariants explicit
- normalize at boundaries
- keep backend-specific machinery from infecting the core model

Those are not anti-compression rules. They are how we earn the right to compress.

## A Good Test

Before introducing an abstraction, ask:

1. What real structure is this compressing?
2. What semantics become clearer because of it?
3. What distinct cases am I coupling together?
4. Can I still explain the underlying examples cleanly?
5. Is this hiding mechanics, or hiding meaning?

If you do not have good answers, the compression is probably not earned yet.
