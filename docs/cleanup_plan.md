# Codebase Cleanup Plan

> Goal: Fix pre-commit hooks so we don't need `--no-verify`. Apply semantic compression principles from FAVORITES.md.

## Current State

- **ruff**: 2262 errors (1422 are ANN type hints)
- **ty**: 250 diagnostics (mostly None handling)
- Pre-commit runs: ruff lint, ruff format, ty check

## Priority Order (by structural impact)

### Phase 1: Critical Bugs (must fix)
- [ ] **F821**: 36 undefined names (actual bugs)
- [ ] **F841**: 55 unused variables (dead code)
- [ ] **F401**: 12 unused imports

### Phase 2: Structural - Function Decomposition (Tiger Style)
These force the architectural changes FAVORITES emphasizes.

- [ ] **PLR0915**: 56 functions exceed 70 statement limit
  - Worst: `rollouts/cli.py:main()` at 346 statements (5x limit)
  - Strategy: Extract helpers, push ifs up, move computation down

- [ ] **PLR1702**: 41 too-many-nested-blocks violations
  - Strategy: Early returns, extract nested logic to helpers

- [ ] **PLR0913**: 49 too-many-arguments violations
  - Strategy: Group related args into dataclasses/configs

### Phase 3: Structural - SSA & Clarity (Carmack Style)
- [ ] **PLW2901**: 9 loop variable reassignment violations
  - Pattern: `for line in f: line = line.strip()`
  - Fix: `for raw_line in f: line = raw_line.strip()`

- [ ] **A001/A002**: 32 shadowing builtins
  - `ConnectionError` class → rename to `SSHConnectionError`
  - `def list()` CLI command → add `# noqa: A001` (CLI name is intentional)
  - `def exec()` CLI command → add `# noqa: A001` (CLI name is intentional)
  - `input` in PyTorch hooks → rename to `inputs` (matches convention)

### Phase 4: Type Safety (ty errors)
- [ ] Fix None handling patterns (add guards or assertions)
- [ ] Fix `Path` vs `str` mismatches in refactor tools

### Phase 5: Tune Lint Config
Decide what to keep vs disable:

```toml
# pyproject.toml changes to consider
[tool.ruff.lint]
ignore = [
    "ANN",     # Disable all type annotations? Or fix gradually?
    # Keep TRY003 ignored (already is)
]
```

Options for ANN rules:
1. **Disable entirely** - focus on structural quality, not annotations
2. **Per-directory ignores** - enforce in core packages, skip dev/
3. **Fix everything** - significant effort, unclear value

## Recommended Approach

1. **Quick wins first**: Run `ruff check --fix` for auto-fixable issues (48 safe, 762 unsafe)
2. **Decompose the monsters**: Start with `rollouts/cli.py:main()` - it's 5x over limit
3. **Rename shadowed builtins**: `ConnectionError` → `SSHConnectionError`
4. **Tune config last**: Once structural issues fixed, decide on ANN rules

## Files to Prioritize

By error density and importance:

| File | Issues | Notes |
|------|--------|-------|
| `rollouts/rollouts/cli.py` | PLR0915 (346 stmts) | Main CLI, needs major decomposition |
| `rollouts/rollouts/training/grpo.py` | PLR0915 (182 stmts) | Core training loop |
| `broker/broker/cli.py` | PLR0915 (146 stmts), A001 | `info()` and `list()` commands |
| `bifrost/bifrost/types.py` | A001 | `ConnectionError` shadowing |
| `bifrost/bifrost/client.py` | PLR1702 | Deeply nested SSH streaming |

## Success Criteria

- [ ] `git commit` works without `--no-verify`
- [ ] No functions over 70 statements
- [ ] No shadowed builtins (except intentional CLI names with noqa)
- [ ] No undefined names or unused variables
