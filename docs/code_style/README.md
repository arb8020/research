# Code Style Guide

## Start Here

These are my handwritten docs on how I think about code:

- **[code_philosophy.md](code_philosophy.md)** - The full philosophy: usage code first, state is the enemy, parse at boundaries, classes vs functions
- **[keeping_llm_code_honest.md](keeping_llm_code_honest.md)** - How to work with LLMs: compression over working code, friction is feedback, verification loops

---

## Quick Reference

| Need | Go To |
|------|-------|
| General principles & patterns | [core-ai/CHEATSHEET.md](core-ai/CHEATSHEET.md) |
| Error handling decision | [core-ai/ERROR_HANDLING.md](core-ai/ERROR_HANDLING.md) |
| Class vs function? | [core-ai/CLASSES_VS_FUNCTIONAL.md](core-ai/CLASSES_VS_FUNCTIONAL.md) |
| ML experiment configs | [core-ai/experiment_config.md](core-ai/experiment_config.md) |
| Tensor shape naming | [domain/shape_suffixes_shazeer.md](domain/shape_suffixes_shazeer.md) |
| Frontend patterns | [frontend/](frontend/) |
| Async Python | [domain/anyio_advice.md](domain/anyio_advice.md) |

---

## Directory Structure

### Core
- **[core-ai/](core-ai/)** - Synthesized guides (AI-assisted, under review)
  - FAVORITES.md - Top 10 patterns
  - CHEATSHEET.md - Consolidated reference
  - ERROR_HANDLING.md - When to use exceptions vs tuples vs assertions
  - CLASSES_VS_FUNCTIONAL.md - Decision framework
  - IMMUTABILITY_AND_FP.md - Frozen dataclasses, pure functions
  - experiment_config.md - ML config patterns
  - LOGGING_LEVELS_RECOMMENDATIONS.md - INFO vs DEBUG
  - CURSOR_AI_SLOP_DRAFT.md - Cleaning AI-generated code

### External Sources
- **[references/](references/)** - Notes on foundational sources
  - Casey Muratori: semantic compression, granularity, worst API
  - Tiger Style: safety-critical code rules
  - Sean Goedecke: system design, API design
  - CodeAesthetic: abstraction = coupling

- **[review/](review/)** - Code review in the LLM era
  - ezyang, simon willison, sean goedecke on reviewing AI code

### Domain-Specific
- **[frontend/](frontend/)** - UI patterns, React, design philosophy
- **[logging/](logging/)** - Observability, logging dysfunction
- **[testing/](testing/)** - Testing philosophy, error patterns
- **[domain/](domain/)** - Async Python, multiprocessing, tensor naming, Midori

### Other
- **[templates/](templates/)** - Design doc template, commit conventions
- **[miscellaneous/](miscellaneous/)** - Rough notes, raw transcripts, drafts

---

## Key Principles (TL;DR)

1. **Write usage code first** - What do you *want* to write?
2. **Don't abstract until 2+ examples** - Make it work before reusable
3. **State is the enemy** - Minimize owners, make it explicit
4. **Parse at boundaries, assert internally** - Guards at the gates
5. **Push ifs up, fors down** - Parent has control flow, helpers compute
6. **Single assignment** - Name each transformation
7. **Friction is feedback** - If it's hard to write, something's wrong
