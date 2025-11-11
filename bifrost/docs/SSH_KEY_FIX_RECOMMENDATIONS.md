# Bifrost SSH Key Configuration - Fix Recommendations

## Executive Summary

The bifrost CLI has a UX issue where the error message for missing SSH keys suggests using `--ssh-key` flag, but this only works when placed BEFORE the command name due to Typer framework design. This document provides recommendations for fixing this issue.

## Problem Statement

### User Experience Issue

1. User runs: `bifrost exec user@host "ls"` (no SSH key configured)
2. Error message suggests: "Or use: --ssh-key /path/to/key"
3. User tries: `bifrost exec --ssh-key /path/to/key user@host "ls"`
4. Result: "No such option: --ssh-key" error
5. Expected: `bifrost --ssh-key /path/to/key exec user@host "ls"` (unintuitive flag placement)

### Root Cause

The `--ssh-key` flag is defined in `@app.callback()` (global scope), not in individual command decorators. Typer/Click processes global options before routing to subcommands, requiring flag placement before the command name.

## Recommended Solutions

### Solution 1: Fix the Error Message (EASIEST - LOW EFFORT)

**Location:** `/Users/chiraagbalu/research/bifrost/bifrost/cli.py` line 78

**Current Code:**
```python
logger.info(f"Or use: --ssh-key {found_keys[0]}")
```

**Fixed Code:**
```python
logger.info(f"Or use: bifrost --ssh-key {found_keys[0]} <command> ...")
```

**Benefits:**
- Minimal change (single line)
- Eliminates user confusion about flag placement
- Shows correct Typer syntax

**Effort:** 5 minutes

---

### Solution 2: Move --ssh-key to Command Level (MEDIUM - EFFORT)

Move the `--ssh-key` flag from `@app.callback()` to each command decorator. This allows the natural `bifrost exec --ssh-key PATH ...` syntax.

**Implementation:**
1. Remove `--ssh-key` from `main()` callback
2. Add to each command that needs it: `push`, `exec`, `deploy`, `run`, `jobs`, `logs`, `download`, `upload`
3. Pass via `ctx.obj` or directly to `resolve_ssh_key()`

**Example Refactoring:**

```python
# Before (in callback):
@app.callback()
def main(
    ctx: typer.Context,
    ssh_key: Optional[str] = typer.Option(None, "--ssh-key", ...),
    ...
):
    ctx.obj = {"ssh_key": ssh_key, ...}

# After (in exec command):
@app.command()
def exec(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(...),
    command: str = typer.Argument(...),
    env: Optional[List[str]] = typer.Option(None, "--env", ...),
    ssh_key: Optional[str] = typer.Option(None, "--ssh-key", ...),
):
    ctx.obj = {"ssh_key": ssh_key}  # Store in context
    ssh_key = resolve_ssh_key(ctx)
    # ... rest of function ...
```

**Benefits:**
- Intuitive flag placement: `bifrost exec --ssh-key PATH ...`
- Eliminates "No such option" errors
- No need to adjust error messages
- More discoverable with `bifrost exec --help`

**Drawbacks:**
- Requires adding `--ssh-key` to 8 command decorators
- Slightly more code duplication
- Need to test all 8 commands

**Effort:** 30-45 minutes (including testing)

---

### Solution 3: Support Both Syntaxes (BEST UX - HIGHER EFFORT)

Combine solutions 1 and 2: support both global and command-level `--ssh-key` flags.

**Implementation:**
1. Keep `--ssh-key` in `@app.callback()` for backward compatibility
2. Add `--ssh-key` to each command decorator as alternative
3. Merge both sources in `resolve_ssh_key()`

**Benefits:**
- Fully backward compatible
- Users get the intuitive syntax they expect
- Global flag still works for power users
- Best user experience

**Drawbacks:**
- More code duplication
- Slight complexity in merge logic

**Effort:** 1 hour (including testing)

---

## Recommended Approach: Hybrid (Solution 1 + 2)

**Short-term (immediate):** Implement Solution 1 - Fix error message
- Single line change
- Eliminates immediate confusion
- Can be deployed in next patch release

**Long-term (next minor release):** Implement Solution 2
- Provides intuitive syntax users expect
- More discoverable through `--help`
- Plan in next development cycle

## Implementation Details for Solution 2

### Step 1: Remove from Callback

**File:** `bifrost/cli.py` lines 106-129

```python
# BEFORE:
@app.callback()
def main(
    ctx: typer.Context,
    ssh_key: Optional[str] = typer.Option(None, "--ssh-key", ...),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
    json_output: bool = typer.Option(False, "--json"),
    debug: bool = typer.Option(False, "--debug"),
):
    # Setup logging...
    ctx.obj = {"ssh_key": ssh_key, "json": json_output}

# AFTER:
@app.callback()
def main(
    ctx: typer.Context,
    quiet: bool = typer.Option(False, "-q", "--quiet"),
    json_output: bool = typer.Option(False, "--json"),
    debug: bool = typer.Option(False, "--debug"),
):
    # Setup logging...
    ctx.obj = {"json": json_output}  # Only json, not ssh_key
```

### Step 2: Add to Each Command

**Example for exec (lines 214-265):**

```python
@app.command()
def exec(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(...),
    command: str = typer.Argument(..., help="Command to execute"),
    env: Optional[List[str]] = typer.Option(None, "--env", help="KEY=VALUE"),
    ssh_key: Optional[str] = typer.Option(None, "--ssh-key", help="Path to SSH private key"),  # ADD THIS
):
    """Execute command on remote instance"""
    # Store ssh_key in context for resolve_ssh_key() to find
    if not ctx.obj:
        ctx.obj = {}
    ctx.obj["ssh_key"] = ssh_key
    
    ssh_key = resolve_ssh_key(ctx)  # Function already handles this
    # ... rest unchanged ...
```

**Repeat for all 8 commands:**
- push (line 148)
- exec (line 214)
- deploy (line 268)
- run (line 336)
- jobs (line 408)
- logs (line 474)
- download (line 508)
- upload (line 540)

### Step 3: Update resolve_ssh_key()

**File:** `bifrost/cli.py` lines 58-85

No changes needed! The function already checks `ctx.obj.get("ssh_key")` first, which will work for both global and command-level flags.

### Step 4: Test All Commands

Create test cases for each command with `--ssh-key` flag:

```bash
# Syntax verification
bifrost --help                    # Should show global flags only
bifrost exec --help              # Should show --ssh-key
bifrost push --help              # Should show --ssh-key
bifrost deploy --help            # Should show --ssh-key
bifrost run --help               # Should show --ssh-key
bifrost jobs --help              # Should show --ssh-key
bifrost logs --help              # Should show --ssh-key
bifrost download --help          # Should show --ssh-key
bifrost upload --help            # Should show --ssh-key

# Functional tests (with dummy key)
bifrost exec --ssh-key ~/.ssh/test user@host "ls"      # Should work
bifrost --ssh-key ~/.ssh/test exec user@host "ls"      # Should still work (backward compat)
```

## Error Message Improvements

### Current Messages (Confusing)

```
✗ No SSH key specified

Found keys at:
  /Users/user/.ssh/id_ed25519

Or use: --ssh-key /Users/user/.ssh/id_ed25519
Set SSH_KEY_PATH in .env (run: bifrost init)
```

### Improved Messages

#### Version A (Minimal - for Solution 1):
```
✗ No SSH key specified

Found keys at:
  /Users/user/.ssh/id_ed25519

Options:
1. Use: bifrost --ssh-key /Users/user/.ssh/id_ed25519 <command> ...
2. Set SSH_KEY_PATH in .env (run: bifrost init)
3. Export: export SSH_KEY_PATH=/Users/user/.ssh/id_ed25519
```

#### Version B (After Solution 2):
```
✗ No SSH key specified

Found keys at:
  /Users/user/.ssh/id_ed25519

Options:
1. Use: bifrost exec --ssh-key /Users/user/.ssh/id_ed25519 ...
2. Set SSH_KEY_PATH in .env (run: bifrost init)
3. Export: export SSH_KEY_PATH=/Users/user/.ssh/id_ed25519
```

## Backward Compatibility

**Solution 1 (Error message fix):** No breaking changes

**Solution 2 (Move to command level):** 
- Keep `@app.callback()` as fallback for 2 releases to maintain backward compatibility
- Document in changelog
- Update README with new syntax

**Solution 3 (Support both):** 
- 100% backward compatible
- No deprecation needed
- Users can use either syntax

## Testing Checklist

- [ ] Fix error message is clear and actionable
- [ ] All 8 commands accept `--ssh-key` flag (for Solution 2+)
- [ ] Global flag still works (for Solution 3)
- [ ] `--help` shows flag for each command
- [ ] SSH connection fails gracefully with helpful error if key doesn't exist
- [ ] Auto-discovery still works when no flag/env var specified
- [ ] .env file loading still works
- [ ] Test with mock SSH (smoke test already exists)

## Priority

1. **HIGH (Immediate):** Fix error message (Solution 1)
2. **MEDIUM (Next release):** Move to command-level flags (Solution 2)

## Files to Modify

1. `bifrost/cli.py` (main changes)
   - `main()` callback (lines 106-129)
   - `resolve_ssh_key()` function (lines 58-85) - may need error message updates
   - 8 command functions (push, exec, deploy, run, jobs, logs, download, upload)

2. `README.md` (documentation)
   - Update CLI examples to show new flag syntax
   - Add troubleshooting section

3. `tests/smoke_exec_stream.py` (verification)
   - Add tests for `--ssh-key` flag

## References

- Typer documentation on callback options: https://typer.tiangolo.com/
- Current implementation: `/Users/chiraagbalu/research/bifrost/bifrost/cli.py`
- Helper functions: `/Users/chiraagbalu/research/shared/shared/config.py`
