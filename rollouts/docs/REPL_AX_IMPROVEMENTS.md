# REPL Environment AX Improvements

> **Based on:** Analysis of rollouts `repl.py` vs rlm-minimal `repl.py`  
> **Issue observed:** Sub-agent/REPL not returning results properly (interrupted, no result)

---

## Root Cause Analysis

### Issue 1: Sub-agent `final_answer` not being captured

When `_async_agent` spawns a child agent, it relies on the child calling `final_answer` to set `child_env._final_answer`. But there are failure modes:

```python
# Current code (repl.py line 797-803)
await run_agent(child_state, child_run_config)

# Return final answer from child
if child_env._final_answer:
    return child_env._final_answer
else:
    return "(sub-agent did not provide a final answer)"
```

**Problems:**
1. If child hits `max_agent_turns` (15) without calling `final_answer`, returns generic message
2. If child errors, returns generic message
3. **No extraction of partial answer from trajectory**

**Fix:** Extract the last assistant message as fallback:
```python
states = await run_agent(child_state, child_run_config)

if child_env._final_answer:
    return child_env._final_answer

# Fallback: extract last assistant response
if states:
    last_state = states[-1]
    last_msg = last_state.actor.trajectory.messages[-1]
    if last_msg.role == "assistant":
        content = last_msg.content
        if isinstance(content, str):
            return f"(sub-agent partial): {content[:500]}"
        # Handle list content...

return "(sub-agent did not provide a final answer)"
```

### Issue 2: Child RunConfig missing critical handlers

The child agent's RunConfig only sets a few fields:

```python
# Current (repl.py lines 788-795)
child_run_config = RunConfig(
    on_chunk=run_config.on_chunk,  # Inherited - could cause TUI issues!
    handle_stop=compose_handlers([...]),
    session_store=None,
)
```

**Missing handlers use dangerous defaults:**
- `on_input` → `default_stdin_handler` which calls `input()` → **BLOCKS IF TRIGGERED**
- `confirm_tool` → OK (auto-confirms)
- `handle_no_tool` → OK (does nothing)

**Fix:** Provide safe handlers:
```python
async def noop_input_handler(prompt: str) -> str:
    """Sub-agents should not request input."""
    raise RuntimeError("Sub-agent attempted to request user input")

child_run_config = RunConfig(
    on_chunk=run_config.on_chunk,  # Or: lambda e: None to silence
    on_input=noop_input_handler,   # Prevent blocking
    handle_stop=compose_handlers([...]),
    session_store=None,
)
```

### Issue 3: Inherited `on_chunk` may cause TUI conflicts

When sub-agent inherits parent's `on_chunk`, its streaming events go to the same frontend. This can cause:
- Interleaved output
- TUI state corruption
- Unexpected render behavior

**Fix options:**
1. Silence sub-agent streaming: `on_chunk=lambda e: None`
2. Prefix events with agent ID: wrap callback
3. Buffer sub-agent output until complete

---

## Identified Issues (Additional)

### 1. Thread-bridging complexity
The current implementation uses `trio.from_thread.run` to bridge sync code execution back to async. This is correct but has edge cases:

```python
# Current approach in rollouts
def sync_llm_query(prompt: str) -> str:
    try:
        return trio.from_thread.run(
            self._async_llm_query,
            prompt,
            trio_token=trio_token,  # Must be captured before thread
        )
    except Exception as e:
        return f"[llm_query error: {e}]"
```

**Problem:** If the trio token is stale or the event loop state changes, this can hang or fail silently.

### 2. Missing output capture isolation
The rlm-minimal version uses thread locks and proper context managers:

```python
# rlm-minimal approach
@contextmanager
def _capture_output(self):
    """Thread-safe context manager to capture stdout/stderr"""
    with self._lock:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            yield stdout_buffer, stderr_buffer
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
```

**Our current approach** uses `contextlib.redirect_stdout` which works but doesn't capture stderr or use thread locking.

### 3. Expression evaluation not robust
Current code tries eval then falls back to exec:

```python
# Current
try:
    result = eval(main_code, namespace)
    if result is not None:
        print(repr(result))
except SyntaxError:
    exec(main_code, namespace)
```

**rlm-minimal** handles this more carefully by:
- Splitting out non-comment lines
- Checking if last line "looks like an expression"
- Only evaluating the last line, executing prior lines as statements

### 4. No temp directory isolation
rlm-minimal creates a temp directory for each REPL session:

```python
self.temp_dir = tempfile.mkdtemp(prefix="repl_env_")

@contextmanager
def _temp_working_directory(self):
    """Temporarily change working directory for REPL execution"""
    old_cwd = os.getcwd()
    try:
        os.chdir(self.temp_dir)
        yield
    finally:
        os.chdir(old_cwd)
```

We don't do this, which means file operations could affect the actual working directory.

### 5. No execution timing
rlm-minimal tracks execution time:

```python
start_time = time.time()
# ... execution ...
end_time = time.time()
execution_time = end_time - start_time
```

Useful for debugging slow operations and potential infinite loops.

---

## Proposed Fixes

### Fix 1: Better Output Capture

```python
import sys
import io
import threading
from contextlib import contextmanager

class REPLEnvironment:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    @contextmanager
    def _capture_output(self):
        """Thread-safe stdout/stderr capture."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
```

### Fix 2: Smarter Expression Evaluation

```python
def _exec_code(code: str, namespace: dict) -> tuple[str, str, bool]:
    """Execute code, return (stdout, stderr, had_error)."""
    lines = code.strip().split('\n')
    
    # Separate imports
    import_lines = [l for l in lines if l.strip().startswith(('import ', 'from '))]
    other_lines = [l for l in lines if not l.strip().startswith(('import ', 'from '))]
    
    # Execute imports in globals
    if import_lines:
        exec('\n'.join(import_lines), namespace, namespace)
    
    if not other_lines:
        return "", "", False
    
    # Check if last non-comment line is an expression
    non_comment = [l for l in other_lines if l.strip() and not l.strip().startswith('#')]
    if non_comment:
        last_line = non_comment[-1]
        is_expr = not any([
            last_line.strip().startswith(('import ', 'from ', 'def ', 'class ', 'if ', 
                                          'for ', 'while ', 'try:', 'with ', 'return ')),
            '=' in last_line.split('#')[0],  # Not assignment
            last_line.strip().endswith(':'),  # Not control structure
        ])
        
        if is_expr:
            # Execute all but last as statements
            if len(non_comment) > 1:
                exec('\n'.join(other_lines[:-1]), namespace, namespace)
            # Eval last line and print result
            result = eval(last_line, namespace, namespace)
            if result is not None:
                print(repr(result))
            return
    
    # Fall back to full exec
    exec('\n'.join(other_lines), namespace, namespace)
```

### Fix 3: Add Timeout/Execution Limits

```python
import signal

MAX_EXEC_TIME = 30  # seconds

def _exec_code_with_timeout(code: str, namespace: dict, timeout: int = MAX_EXEC_TIME):
    """Execute with timeout to prevent infinite loops."""
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution exceeded {timeout}s limit")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        return _exec_code(code, namespace)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
```

### Fix 4: Track Variables for Debugging

```python
@dataclass
class REPLEnvironment:
    _execution_count: int = 0
    _last_execution_time: float = 0.0
    
    async def _exec_repl(self, tool_call: ToolCall, run_config: RunConfig) -> ToolResult:
        self._execution_count += 1
        start = time.time()
        
        # ... execution ...
        
        self._last_execution_time = time.time() - start
        
        # Log for debugging
        logger.debug(f"REPL exec #{self._execution_count}: {self._last_execution_time:.2f}s")
```

### Fix 5: Better Error Messages

```python
def _exec_code(code: str, namespace: dict) -> tuple[str, bool]:
    try:
        # ... execution ...
    except Exception as e:
        # Include line number and context
        import traceback
        tb = traceback.format_exc()
        
        # Find the line in user code that failed
        for line in tb.split('\n'):
            if '<string>' in line or 'exec(' in line:
                continue
            if 'File' in line and 'line' in line:
                return f"Error on {line}\n{type(e).__name__}: {e}", True
        
        return f"{type(e).__name__}: {e}\n{tb}", True
```

---

## Quick Win: Add Stderr Capture

The simplest improvement is capturing stderr alongside stdout:

```python
def _exec_code(code: str, namespace: dict[str, Any]) -> tuple[str, str, bool]:
    """Execute code in namespace, return (stdout, stderr, had_error)."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    had_error = False

    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            # ... existing logic ...
    except Exception as e:
        stderr.write(f"{type(e).__name__}: {e}\n")
        had_error = True

    return stdout.getvalue(), stderr.getvalue(), had_error
```

Then in ToolResult, include both:

```python
output = stdout_output
if stderr_output:
    output += f"\n[stderr]\n{stderr_output}"
```

---

## Testing the REPL

Before/after test:

```python
# Test 1: Basic execution
repl(code="x = 1 + 1; print(x)")  # Should print: 2

# Test 2: Expression result
repl(code="len(context)")  # Should print: <number>

# Test 3: llm_query from within repl
repl(code='answer = llm_query("What is 2+2?"); print(answer)')

# Test 4: Error handling
repl(code="1/0")  # Should return error message, not crash

# Test 5: Nested agent
repl(code='result = agent("Count words", context[:1000]); print(result)')
```

---

## Summary

| Issue | Impact | Fix Priority |
|-------|--------|--------------|
| **Missing `on_input` handler** | Sub-agent blocks on `input()` | **Critical** |
| **No fallback for missing final_answer** | Returns useless message | **High** |
| Inherited `on_chunk` conflicts | TUI interleaving | High |
| No stderr capture | Missing error context | Medium |
| Thread-bridging edge cases | Hangs/silent failures | Medium |
| No execution timeout | Infinite loop risk | Medium |
| Expression eval not robust | Inconsistent output | Low |
| No temp directory | Side effects | Low |
| No execution timing | Hard to debug | Low |

## Priority Fix Order

1. **Add `on_input` handler that errors instead of blocking** - prevents silent hangs
2. **Extract partial answer from trajectory as fallback** - always returns something useful
3. **Consider silencing sub-agent streaming** - cleaner TUI behavior
4. Add stderr capture
5. Add timeout

---

## Fixes Applied (2024-12-25)

### Fixed in `_async_agent`:

1. **Safe `on_input` handler** - Returns error message instead of blocking on `input()`:
   ```python
   async def noop_input_handler(prompt: str) -> str:
       return "[sub-agent cannot request user input]"
   ```

2. **Silenced sub-agent streaming** - Prevents TUI conflicts:
   ```python
   on_chunk=lambda e: None,  # Silence sub-agent streaming
   ```

3. **Explicit `confirm_tool`** - Auto-confirms all tools:
   ```python
   async def auto_confirm_tool(...) -> tuple[AgentState, ToolConfirmResult]:
       return state, ToolConfirmResult(proceed=True)
   ```

4. **Partial answer fallback** - Extracts last assistant message if no `final_answer`:
   ```python
   if states:
       last_state = states[-1]
       for msg in reversed(messages):
           if msg.role == "assistant" and msg.content:
               return f"[sub-agent partial response]: {content[:1000]}"
   ```

### Files Modified
- `rollouts/environments/repl.py` - `_async_agent` method
- `rollouts/docs/REPL_AX_IMPROVEMENTS.md` - This document
