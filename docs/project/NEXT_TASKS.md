# Next Tasks - Handoff Document

## Context

The midas/kerbal refactor is complete. Midas has been deleted entirely, and kerbal is now a self-contained package for remote script execution.

**Current state:**
- `kerbal/` - Single package with all remote execution utilities
- `dev/integration_training/` - Example deployment that uses kerbal
- Need to review and fix code that may still reference old midas API

## Tasks

### 1. Review wafer_stuff/qwen3_next

**Goal:** Understand what this code does and check if it needs updating for the new kerbal API.

**Actions:**
- [ ] Read and understand wafer_stuff/qwen3_next code
- [ ] Check if it imports from midas (needs update) or kerbal (might be okay)
- [ ] Document what this code does and its purpose
- [ ] Update to use new kerbal API if needed

**Questions to answer:**
- What is qwen3_next?
- Does it use midas or kerbal?
- Does it need to be updated?

### 2. Review and Fix wafer_stuff/clicker

**Goal:** Review clicker code and fix any issues with midas/kerbal usage.

**Actions:**
- [ ] Read and understand wafer_stuff/clicker code
- [ ] Check for midas imports or old API usage
- [ ] Update to use new kerbal API (`from kerbal import DependencyConfig, setup_script_deps`)
- [ ] Test that clicker still works after updates
- [ ] Document what clicker does

**Expected issues:**
- May import from midas (needs to change to kerbal)
- May use old `UvBackend` or `optional_dependencies` API
- May have `ensure_uv()` calls (now handled automatically by `setup_script_deps()`)

### 3. Review and Fix dev/integration_training

**Goal:** Ensure integration_training works correctly with the refactored kerbal.

**Actions:**
- [ ] Review all files in dev/integration_training/
- [ ] Check deploy.py for any remaining issues
- [ ] Check train.py and config files for midas references
- [ ] Verify the deployment flow works end-to-end
- [ ] Update any documentation in the directory

**Current status:**
- deploy.py has been updated to inline uv installation
- May have other files that need checking
- Should verify the full deployment workflow

## New Kerbal API Reference

For updating code:

```python
# Old (midas + kerbal)
from midas import ensure_uv, DependencyConfig
from kerbal import setup_script_deps

ensure_uv(client)
deps = DependencyConfig(
    project_name="training",
    dependencies=["torch>=2.0"],
    optional_dependencies={"dev": ["pytest"]},  # old name
)
setup_script_deps(client, workspace, deps, install_extras=["dev"])

# New (kerbal only)
from kerbal import DependencyConfig, setup_script_deps

deps = DependencyConfig(
    project_name="training",
    dependencies=["torch>=2.0"],
    extras={"dev": ["pytest"]},  # renamed
)
setup_script_deps(client, workspace, deps, install_extras=["dev"])  # uv auto-installed
```

**Key changes:**
- Import from `kerbal` instead of `midas`
- Remove `ensure_uv()` calls (handled automatically)
- Rename `optional_dependencies` → `extras` in `DependencyConfig`
- `install_extras` is a list, not a single string

## Tiger Style Reminders

When updating code:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
- No complex conditionals
- TYPE_CHECKING for imports

## Priority Order

1. **wafer_stuff/qwen3_next** - Review first to understand context
2. **wafer_stuff/clicker** - Fix next (likely has midas imports)
3. **dev/integration_training** - Final verification and cleanup

Start with qwen3_next to understand the codebase context.
