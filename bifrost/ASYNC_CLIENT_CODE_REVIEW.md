# AsyncBifrostClient Code Style Review

## Summary
The `async_client.py` implementation follows most Tiger Style principles but has room for improvement in assertions, SSA, and documentation.

## ✅ What's Good

### 1. **Class Design - Correct Use Case**
- ✅ `AsyncBifrostClient` is a legitimate use of a class (manages stateful SSH connection)
- ✅ Owns resources (SSH connection, needs cleanup)
- ✅ Has lifecycle (`__init__`, `close()`, async context manager)
- ✅ Coordinates mutation across multiple methods

### 2. **Function Lengths**
- ✅ Most functions under 70 lines (only 2 minor violations):
  - `copy_files()`: 76 lines (6 over)
  - `upload_files()`: 74 lines (4 over)
- Both are SFTP operations with similar structure - acceptable

### 3. **Control Flow**
- ✅ Explicit control flow throughout
- ✅ No clever tricks or hidden magic
- ✅ Clear error handling with try/except
- ✅ Proper use of Trio's structured concurrency

### 4. **Documentation**
- ✅ Good docstrings on all public methods
- ✅ Explains trio-asyncio bridge in comments
- ✅ Usage examples in docstrings

### 5. **Limits on Everything**
- ✅ Connection retry has `max_attempts = 3`
- ✅ Timeouts enforced (`timeout` parameter)
- ✅ All `while True` loops have exit conditions

## ⚠️ Areas for Improvement

### 1. **Insufficient Assertions (High Priority)**
**Issue:** Most functions have 0-1 assertions, goal is 2+

**Functions lacking assertions:**
- `exec()` - 0 assertions
- `exec_stream()` - 0 assertions
- `get_all_jobs()` - 0 assertions
- `get_job_status()` - 0 assertions
- `get_logs()` - 0 assertions
- `_establish_connection()` - 0 assertions
- `_get_connection()` - 1 assertion (needs 1 more)

**Recommendation:** Add precondition/postcondition assertions:

```python
async def exec(self, command: str, ...) -> ExecResult:
    # Preconditions
    assert command, "command must be non-empty"
    assert isinstance(command, str), "command must be string"

    conn = await self._get_connection()
    # ... execution

    result = await _trio_wrap(conn.run)(full_command, check=False)

    # Postconditions
    assert result.exit_status is not None, "exit_status must be set"
    assert isinstance(result.stdout, str), "stdout must be string"
    assert isinstance(result.stderr, str), "stderr must be string"

    return ExecResult(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_status or 0
    )
```

**Why assertions matter:**
- Document assumptions (what must be true)
- Catch bugs early (fail fast)
- Help debugging (know exactly what violated)
- Self-documenting code (assertions = inline specs)

### 2. **Single Assignment (SSA) Violations (Medium Priority)**
**Issue:** ~10 variable reassignments hurt debuggability

**Examples:**
```python
# Line 139-145: needs_reconnect reassigned 3 times
needs_reconnect = False
if self._ssh_conn is None:
    needs_reconnect = True
else:
    transport = self._ssh_conn.get_transport()
    if transport is None:
        needs_reconnect = True
    elif not transport.is_active():
        needs_reconnect = True

# BETTER: Use early returns (single assignment to _ssh_conn)
if self._ssh_conn is None:
    self._ssh_conn = await self._establish_connection()
    return self._ssh_conn

transport = self._ssh_conn.get_transport()
if transport is None or not transport.is_active():
    self._ssh_conn = await self._establish_connection()
    return self._ssh_conn

return self._ssh_conn
```

```python
# Line 220-223: working_dir reassigned
if working_dir is None:
    result = await _trio_wrap(conn.run)("test -d ~/.bifrost/workspace", check=False)
    if result.exit_status == 0:
        working_dir = "~/.bifrost/workspace"
    else:
        working_dir = "~"

# BETTER: Single assignment with explicit decision
if working_dir is None:
    result = await _trio_wrap(conn.run)("test -d ~/.bifrost/workspace", check=False)
    workspace_exists = (result.exit_status == 0)
    default_dir = "~/.bifrost/workspace" if workspace_exists else "~"
    working_dir = default_dir
```

**Why SSA matters:**
- When debugging, you can inspect `raw_data`, `filtered_data`, `sorted_data` separately
- No confusion about "which version of this variable am I looking at?"
- Easier to add logging/breakpoints between transformations

**Exception:** Iterative calculations are OK (e.g., `files_copied += 1`)

### 3. **Missing "Why" Comments (Low Priority)**
**Issue:** Magic numbers and non-obvious decisions lack explanation

**Examples needing "why":**
```python
# Line 98: max_attempts = 3
max_attempts = 3
delay = 2
backoff = 2

# BETTER:
# Retry 3 times with exponential backoff (2s, 4s, 8s = 14s total)
# Enough to handle transient network issues but fail fast on real problems
max_attempts = 3
delay = 2
backoff = 2
```

```python
# Line 111: keepalive_interval=30
keepalive_interval=30,

# BETTER:
# Send keepalive every 30s to prevent idle timeout
# Most SSH servers drop idle connections after 60s, so 30s = 2x safety margin
keepalive_interval=30,
```

```python
# Line 303-304: Manual readline instead of async for
while True:
    try:
        line = await _trio_wrap(process.stdout.readline)()

# BETTER (add comment):
# Can't use 'async for' because asyncssh's async iterator doesn't work with
# trio-asyncio's event loop shim. Manual readline() calls work correctly.
while True:
    try:
        line = await _trio_wrap(process.stdout.readline)()
```

**Why "why" comments matter:**
- 6 months later you'll ask "why did I do this?"
- Prevents well-meaning "cleanup" that breaks things
- Documents tradeoffs and constraints

### 4. **Positive Invariants**
**Status:** ✅ Mostly good!

The code generally states conditions positively:
```python
# GOOD (line 219)
if result.exit_status == 0:
    working_dir = "~/.bifrost/workspace"
```

Minor improvement opportunity:
```python
# Line 145: Could be more positive
elif not transport.is_active():

# SLIGHTLY BETTER:
elif transport.is_closing():  # More direct/positive statement
```

### 5. **Split Compound Assertions**
**Status:** ✅ No compound assertions found!

All assertions are already split into separate statements.

## 📊 Metrics

| Metric | Status | Details |
|--------|--------|---------|
| **Total Lines** | 1015 | Reasonable for a full async client |
| **Functions > 70 lines** | 2 | `copy_files` (76), `upload_files` (74) - both acceptable |
| **Functions with < 2 assertions** | 16/25 | **Main issue** - needs more assertions |
| **SSA violations** | ~10 | Variable reassignments, fixable |
| **Unbounded loops** | 0 | ✅ All have exits |
| **Compound assertions** | 0 | ✅ None found |
| **Missing "why" comments** | ~5 | Magic numbers, design decisions |

## 🔧 Recommended Fixes (Priority Order)

### Priority 1: Add Assertions (High Impact)
Add 2+ assertions to each function, focusing on:
- **Preconditions** - Validate all inputs
- **Postconditions** - Validate outputs before return
- **Invariants** - Check assumptions mid-function

**Example:**
```python
async def _establish_connection(self) -> asyncssh.SSHClientConnection:
    # Preconditions
    assert self.ssh.host, "host must be non-empty"
    assert self.ssh.port > 0, "port must be positive"
    assert self.ssh_key_path, "ssh_key_path must be set"

    # ... connection logic

    # Postconditions
    assert conn is not None, "connection must succeed or raise"
    assert conn._transport is not None, "transport must exist"
    return conn
```

**Estimate:** 2-3 hours to add ~40 assertions across all functions

### Priority 2: Fix SSA Violations (Medium Impact)
Reduce variable reassignments:
- Use early returns instead of flag variables (`needs_reconnect`)
- Create new names for transformed values (`default_dir` instead of reusing `working_dir`)
- Use single-expression assignments where possible

**Estimate:** 1-2 hours

### Priority 3: Add "Why" Comments (Low Impact)
Document rationale for:
- Magic numbers (timeouts, retry counts)
- Non-obvious design decisions (manual readline vs async for)
- Workarounds for trio-asyncio quirks

**Estimate:** 30 minutes

### Priority 4: Minor Cleanups (Optional)
- Replace `not transport.is_active()` with more positive phrasing
- Consider extracting `copy_files` and `upload_files` into smaller helpers (currently 76/74 lines)

**Estimate:** 1 hour

## 🎯 Overall Grade: B+

**Strengths:**
- ✅ Correct use of class for stateful resource
- ✅ Clean async/await patterns with Trio
- ✅ Good structure and decomposition
- ✅ Explicit control flow throughout
- ✅ Proper limits on loops and retries

**Main Weakness:**
- ❌ Insufficient assertions (16/25 functions have < 2)

**Secondary Issues:**
- ⚠️ SSA violations (~10 reassignments)
- ⚠️ Missing "why" comments (~5 places)

**Recommendation:**
- **Ship it** - The code is functional and well-structured
- **Follow up** - Add assertions incrementally (Priority 1)
- Address SSA violations during next refactor (Priority 2)

The lack of assertions is the main issue. Everything else is solid. Adding ~40 assertions (2-3 per function) would bring this to an A.

## 📚 References

Applied principles from:
- `/Users/chiraagbalu/.claude/skills/code-style/SKILL.md`
- Tiger Style: 70-line limit, SSA, assertions
- Explicit control flow over clever tricks
- Classes for state, functions for computation
