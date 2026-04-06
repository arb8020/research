# Codebase Cleanup Plan

> Goal: Fix pre-commit hooks so we don't need `--no-verify`. Apply semantic compression principles from FAVORITES.md.

## Current State (2024-12-22, after Phase 2 partial)

- **ruff**: 1270 errors remaining
  - ANN (type annotations): 760
  - TRY (exception handling): 186
  - PLR (structural): 97
  - Other: 227
- **ty**: 623 diagnostics (mostly None handling, unresolved attributes)
- Pre-commit runs: ruff lint, ruff format, ty check

### Top 10 Error Codes
| Code | Count | Description |
|------|-------|-------------|
| ANN001 | 274 | Missing type annotation for function argument |
| ANN201 | 248 | Missing return type annotation for public function |
| ANN202 | 129 | Missing return type annotation for private function |
| ANN204 | 93 | Missing return type annotation for special method |
| TRY003 | 80 | Avoid specifying long messages outside exception class |
| TRY400 | 69 | Use `logging.exception` instead of `logging.error` |
| B904 | 45 | Raise without `from` inside exception handler |
| TRY300 | 37 | `return` in `try` block (prefer `else`) |
| PLR0915 | 37 | Too many statements |
| PLR0913 | 34 | Too many arguments |

### Config Changes Made
```toml
# Added to pyproject.toml [tool.ruff]
exclude = [
    "dev/**",           # Experimental scripts with sys.path hacks
    "scripts/**",       # One-off scripts with hardcoded paths
    "cutlass/**",       # External vendored code
    "cutlass_repo/**",  # External vendored code
    "worktrees/**",     # Git worktrees
]
```

## Priority Order (by structural impact)

### Phase 1: Critical Bugs (must fix) ✅ COMPLETE

- [x] **F821**: 30 undefined names → **FIXED**
- [x] **F841**: 32 unused variables → **FIXED**
- [x] **F401**: 11 unused imports → **FIXED**

### Phase 2: Structural - Function Decomposition (Tiger Style) 🔄 IN PROGRESS

- [x] **PLR0915**: Major offenders decomposed (56 → 37 remaining)
  - ✅ `rollouts/cli.py:main()` - 345 → <70 statements
  - ✅ `rollouts/frontends/tui/interactive_agent.py:run()` - 188 → <70 statements
  - ✅ `rollouts/training/grpo.py:_grpo_train_async()` - 182 → <70 statements

- [ ] **PLR1702**: 26 too-many-nested-blocks violations (was 41)
  - Strategy: Early returns, extract nested logic to helpers

- [ ] **PLR0913**: 34 too-many-arguments violations (was 49)
  - Strategy: Group related args into dataclasses/configs

### Phase 3: Structural - SSA & Clarity (Carmack Style)
- [ ] **PLW2901**: 8 loop variable reassignment violations
  - Pattern: `for line in f: line = line.strip()`
  - Fix: `for raw_line in f: line = raw_line.strip()`

- [ ] **A002**: 18 shadowing builtins (was 32)
  - `input` in PyTorch hooks → rename to `inputs`
  - Various `id`, `type`, `format` parameter names

### Phase 4: Exception Handling
- [ ] **TRY003**: 80 - long messages in exceptions (consider ignoring)
- [ ] **TRY400**: 69 - use `logging.exception` instead of `logging.error`
- [ ] **TRY300**: 37 - `return` in `try` block
- [ ] **B904**: 45 - `raise` without `from` in exception handler

### Phase 5: Type Safety (ty errors)
- [ ] 623 ty diagnostics (mostly None handling, unresolved attributes)
- [ ] Consider tuning ty config or ignoring some rules

### Phase 6: Tune Lint Config
Decide what to keep vs disable:

```toml
# pyproject.toml changes to consider
[tool.ruff.lint]
ignore = [
    "ANN",     # Disable all type annotations? (760 errors)
    "TRY003",  # Long exception messages (80 errors)
]
```

Options for ANN rules:
1. **Disable entirely** - focus on structural quality, not annotations
2. **Per-directory ignores** - enforce in core packages, skip dev/
3. **Fix gradually** - add types as code is touched

## Files to Prioritize

By error density and importance:

| File | Issues | Notes |
|------|--------|-------|
| ✅ `rollouts/rollouts/cli.py` | PLR0915 | DONE - decomposed main() |
| ✅ `rollouts/rollouts/training/grpo.py` | PLR0915 | DONE - decomposed _grpo_train_async() |
| ✅ `rollouts/frontends/tui/interactive_agent.py` | PLR0915 | DONE - decomposed run() |
| `broker/broker/cli.py` | PLR0915 (146 stmts) | `info()` and `list()` commands |
| `bifrost/bifrost/client.py` | PLR1702 | Deeply nested SSH streaming |

## Success Criteria

- [ ] `git commit` works without `--no-verify`
- [x] No functions over 200 statements (worst offenders fixed)
- [ ] No functions over 70 statements (37 remaining)
- [ ] No undefined names or unused variables
