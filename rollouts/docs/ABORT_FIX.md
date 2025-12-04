# Fixing Abort Support in Rollouts

## Current Problem

The abort implementation in commit `e4b6b87` uses polling-based checks that don't actually cancel in-flight HTTP requests:

```python
# Current approach - WRONG
if rcfg.cancel_scope and rcfg.cancel_scope.cancelled_caught:
    return replace(current_state, stop=StopReason.ABORTED)
```

**Issues:**
1. `cancelled_caught` is only `True` AFTER the scope exits, not during
2. Polling only checks at discrete points, not during I/O
3. HTTP requests continue running even after "abort"

---

## How pi-ai Does It (JavaScript)

pi-ai uses the Web API `AbortController`:

```typescript
// agent.ts
this.abortController = new AbortController();

// Pass signal to HTTP client
const anthropicStream = client.messages.stream(
    { ...params },
    { signal: options?.signal }  // <-- Signal passed to fetch()
);

// Abort method
abort() {
    this.abortController?.abort();  // Immediately cancels HTTP request
}
```

The signal is passed all the way down to `fetch()`, which cancels the TCP connection.

---

## How Trio Cancellation Works

Trio's cancellation is **automatic and superior** - no need to pass signals:

```python
async def example():
    with trio.CancelScope() as scope:
        await some_http_request()  # <-- Automatically cancelled when scope.cancel() called

    # scope.cancelled_caught is True here (AFTER scope exits)
```

Key differences from JavaScript:
- **No signal passing needed** - Trio cancels any `await` inside the scope
- **Works with httpx** - httpx uses anyio which respects Trio's cancellation
- **Structured concurrency** - Cancellation flows through the call stack

---

## The Fix

### Design Principles

Following the code style guidelines:

1. **Explicit control flow** - Don't hide cancellation in polling checks
2. **Assertions for programmer errors** - Use `assert` to document invariants
3. **Exceptions at boundaries** - Convert `trio.Cancelled` to `StopReason.ABORTED` at the agent boundary
4. **Immutable state** - Use `replace()` to create new state with stop reason

### 1. Remove Polling Checks

Delete all the manual `cancelled_caught` checks:

```python
# DELETE these patterns throughout agents.py (lines 386, 427, 561, 569, 701)
if rcfg.cancel_scope and rcfg.cancel_scope.cancelled_caught:
    return replace(current_state, stop=StopReason.ABORTED)
```

### 2. Wrap Agent Loop with try/except

The agent boundary is where we convert the Trio exception to our domain:

```python
async def run_agent(
    state: AgentState,
    run_config: RunConfig,
    session_id: str | None = None,
) -> list[AgentState]:
    """Run agent until stop condition, with cancellation support.

    If run_config.cancel_scope is provided and cancelled, raises trio.Cancelled.
    Caller is responsible for handling cancellation at their boundary.
    """
    assert state is not None
    assert isinstance(state, AgentState)
    assert run_config is not None

    states = [state]
    current_state = state

    try:
        while not current_state.stop:
            # Check stop condition via handle_stop callback
            current_state = run_config.handle_stop(current_state)
            if current_state.stop:
                break

            # Checkpoint events
            await handle_checkpoint_event(current_state, "turn_start", run_config, session_id)

            # Run one step - this is where HTTP calls happen
            # Trio will raise Cancelled if cancel_scope.cancel() was called
            next_state = await run_agent_step(current_state, run_config)

            current_state = next_state
            states.append(current_state)

            await handle_checkpoint_event(current_state, "turn_end", run_config, session_id)

    except trio.Cancelled:
        # Convert Trio's cancellation to our domain
        # This is the boundary where we translate external signals to internal state
        aborted_state = replace(current_state, stop=StopReason.ABORTED)
        states.append(aborted_state)
        # Re-raise to let Trio handle cleanup properly
        raise

    # Save final state
    await handle_checkpoint_event(current_state, "final", run_config, session_id)
    return states
```

### 3. Simplify run_agent_step and process_pending_tools

Remove the polling checks - Trio handles cancellation automatically:

```python
async def run_agent_step(
    state: AgentState,
    rcfg: RunConfig,
) -> AgentState:
    """Execute one complete turn: LLM call -> ALL tool executions -> next turn.

    Cancellation is handled automatically by Trio - any await will raise
    trio.Cancelled if the cancel_scope was cancelled.
    """
    assert state is not None
    assert rcfg is not None

    # No polling needed - remove these lines:
    # if rcfg.cancel_scope and rcfg.cancel_scope.cancelled_caught:
    #     return replace(state, stop=StopReason.ABORTED)

    state = rcfg.handle_stop(state)
    if state.stop:
        return state

    # ... rest of the function unchanged
    # Trio automatically cancels the HTTP call if cancel_scope.cancel() is called
    next_actor = await rollout(
        updated_actor,
        rcfg.on_chunk,
        rcfg.user_message_for_thinking,
        state.turn_idx,
        rcfg.inline_thinking,
    )
    # ...
```

---

## Usage Patterns

### Pattern 1: External Abort (User Cancellation)

```python
async def run_with_abort_support():
    """Run agent with ability to abort from another task."""

    cancel_scope = trio.CancelScope()
    run_config = RunConfig(
        on_chunk=my_handler,
        handle_stop=handle_stop_max_turns(10),
        cancel_scope=cancel_scope,
    )

    async with trio.open_nursery() as nursery:
        # Track result
        result_states = []

        async def agent_task():
            nonlocal result_states
            with cancel_scope:
                result_states = await run_agent(state, run_config)

        nursery.start_soon(agent_task)

        # Abort after user input or timeout
        await user_abort_signal()  # Or trio.sleep(30)
        cancel_scope.cancel()  # <-- Cancels HTTP request mid-stream!
```

### Pattern 2: Timeout with move_on_after

```python
async def run_with_timeout():
    """Run agent with automatic timeout."""

    with trio.move_on_after(60) as cancel_scope:  # 60 second timeout
        run_config = RunConfig(
            on_chunk=my_handler,
            cancel_scope=cancel_scope,
        )
        states = await run_agent(state, run_config)

    if cancel_scope.cancelled_caught:
        print("Agent timed out!")
        # states contains partial progress up to cancellation
```

### Pattern 3: Graceful Shutdown

```python
async def run_with_graceful_shutdown():
    """Handle SIGINT/SIGTERM gracefully."""

    cancel_scope = trio.CancelScope()

    # Set up signal handler
    def handle_signal(signum, frame):
        print("Received shutdown signal, aborting...")
        cancel_scope.cancel()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    run_config = RunConfig(
        on_chunk=my_handler,
        cancel_scope=cancel_scope,
    )

    try:
        with cancel_scope:
            states = await run_agent(state, run_config)
    except trio.Cancelled:
        # Save partial progress
        save_checkpoint(states[-1])
        print("Agent aborted, checkpoint saved")
```

---

## Files to Modify

### `rollouts/agents.py`

1. **Remove polling checks** at lines:
   - 386 (rollout function)
   - 427 (run_agent_step)
   - 561, 569 (process_pending_tools)
   - 701 (run_agent main loop)

2. **Add try/except in run_agent():**
   ```python
   try:
       while not current_state.stop:
           # ... existing loop ...
   except trio.Cancelled:
       aborted_state = replace(current_state, stop=StopReason.ABORTED)
       states.append(aborted_state)
       raise
   ```

3. **Remove cancel_scope parameter from rollout()** - not needed since Trio handles it

### `rollouts/dtypes.py`

Keep `cancel_scope` in `RunConfig` (line 918) but update docstring:

```python
@dataclass(frozen=True)
class RunConfig:
    # ... other fields ...

    # Optional Trio cancel scope for graceful cancellation.
    # When cancel_scope.cancel() is called, any in-flight HTTP request
    # is immediately cancelled and trio.Cancelled is raised.
    # The agent loop catches this and sets stop=StopReason.ABORTED.
    cancel_scope: trio.CancelScope | None = None
```

---

## Testing

### Test 1: Verify Cancellation During HTTP

```python
@pytest.mark.trio
async def test_abort_cancels_http_request():
    """Verify abort actually cancels in-flight HTTP."""

    cancel_scope = trio.CancelScope()
    aborted = False

    async with trio.open_nursery() as nursery:
        async def agent_task():
            nonlocal aborted
            run_config = RunConfig(
                on_chunk=lambda e: None,
                cancel_scope=cancel_scope,
            )
            try:
                with cancel_scope:
                    await run_agent(state, run_config)
            except trio.Cancelled:
                aborted = True
                raise

        nursery.start_soon(agent_task)

        # Cancel almost immediately - should interrupt HTTP
        await trio.sleep(0.1)
        cancel_scope.cancel()
        nursery.cancel_scope.cancel()  # Clean up nursery

    assert aborted, "Agent should have been cancelled"
```

### Test 2: Verify State is ABORTED

```python
@pytest.mark.trio
async def test_abort_sets_stop_reason():
    """Verify final state has StopReason.ABORTED."""

    states = []
    cancel_scope = trio.CancelScope()

    async with trio.open_nursery() as nursery:
        async def agent_task():
            nonlocal states
            run_config = RunConfig(
                on_chunk=lambda e: None,
                cancel_scope=cancel_scope,
            )
            try:
                with cancel_scope:
                    states = await run_agent(state, run_config)
            except trio.Cancelled:
                pass  # Expected

        nursery.start_soon(agent_task)
        await trio.sleep(0.5)
        cancel_scope.cancel()
        nursery.cancel_scope.cancel()

    assert len(states) > 0
    assert states[-1].stop == StopReason.ABORTED
```

---

## Summary

| Aspect | Current (Wrong) | Fixed (Correct) |
|--------|----------------|-----------------|
| Check method | Poll `cancelled_caught` | Catch `trio.Cancelled` |
| HTTP cancellation | Never happens | Automatic via Trio |
| When it works | Only at checkpoints | Immediately, mid-request |
| Signal passing | Unused | Not needed |
| Code complexity | More code (polling) | Less code (let Trio work) |

The fix follows the code style principles:
- **Explicit control flow**: try/except at the boundary, not polling
- **Assertions**: Document invariants with assert
- **Immutable state**: `replace(current_state, stop=StopReason.ABORTED)`
- **Boundary exceptions**: Convert `trio.Cancelled` to domain at agent boundary

We're removing code, not adding it. Trio's structured concurrency handles everything automatically.
