# AnyIO: Pragmatic Async in Python

> **Core Advice:** Use anyio as the default choice for async Python. It provides trio's structured concurrency model while maintaining asyncio ecosystem compatibility.

---

## Decision Framework

### When to Use What

| Choice | When |
|--------|------|
| **anyio** | Libraries, production code, when ecosystem compatibility matters |
| **trio** | Personal projects, when correctness is paramount, signal handling critical |
| **raw asyncio** | Legacy codebases, extreme performance requirements, no structured concurrency needed |

### The Pragmatic Default: anyio

```python
# anyio gives you trio's safety with asyncio's ecosystem
import anyio

async def main():
    async with anyio.create_task_group() as tg:
        tg.start_soon(task_a)
        tg.start_soon(task_b)
        # If either task fails, the other is cancelled automatically
```

---

## Why Not Raw asyncio?

### 1. Task Groups That Actually Work

The stdlib `asyncio.TaskGroup` (3.11+) is a step in the right direction but still has edge cases. trio/anyio's task groups are battle-tested:

```python
# asyncio: manually handling partial failures is error-prone
# anyio: automatic cleanup on any error

async with anyio.create_task_group() as tg:
    tg.start_soon(connect_to_db)
    tg.start_soon(connect_to_cache)
    # If DB fails, cache connection is automatically cancelled
    # No zombie tasks, no resource leaks
```

### 2. Level-Triggered Cancellation

This is the killer feature. In asyncio, cancellation is "edge-triggered" - you get one shot to handle it. In trio/anyio, cancellation is "level-triggered" - the cancelled state persists:

```python
# Edge-triggered (asyncio): easy to miss cancellation
try:
    await some_operation()
except asyncio.CancelledError:
    # If you don't re-raise, cancellation is "consumed"
    # Bug: task continues running when it shouldn't
    pass  # Oops

# Level-triggered (trio/anyio): cancellation persists
# Even in finally blocks, you're still cancelled
# Forces you to explicitly shield if you need cleanup time
async with anyio.CancelScope(shield=True):
    await async_cleanup()  # Protected from cancellation
```

### 3. Proper Signal Handling

trio invested enormous effort in getting Ctrl+C and SIGTERM right:

```python
# asyncio: "ehhh, mostly works"
# - Multiple Ctrl+C often needed
# - SIGINT sometimes ignored in prod
# - Cleanup may not run

# trio/anyio: rock solid
# - Single Ctrl+C always works
# - Signals handled correctly in all contexts
# - Cleanup guaranteed to run (with shield)
```

---

## Migration Path: asyncio → anyio

### Step 1: Replace Imports

```python
# Before
import asyncio

# After
import anyio
```

### Step 2: Replace Primitives

| asyncio | anyio |
|---------|-------|
| `asyncio.sleep()` | `anyio.sleep()` |
| `asyncio.create_task()` | Use task groups instead |
| `asyncio.gather()` | Use task groups instead |
| `asyncio.wait_for(coro, timeout)` | `anyio.fail_after(timeout)` or `anyio.move_on_after(timeout)` |
| `asyncio.Event()` | `anyio.Event()` |
| `asyncio.Lock()` | `anyio.Lock()` |
| `asyncio.Semaphore()` | `anyio.Semaphore()` |
| `asyncio.Queue()` | `anyio.create_memory_object_stream()` |

### Step 3: Replace gather with Task Groups

```python
# Before: asyncio.gather
results = await asyncio.gather(task_a(), task_b(), task_c())

# After: anyio task group with result collection
async def collect_results():
    results = []

    async def run_and_store(coro, index):
        results.append((index, await coro))

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_and_store, task_a(), 0)
        tg.start_soon(run_and_store, task_b(), 1)
        tg.start_soon(run_and_store, task_c(), 2)

    return [r for _, r in sorted(results)]
```

### Step 4: Replace wait_for with Cancel Scopes

```python
# Before: asyncio.wait_for
try:
    result = await asyncio.wait_for(operation(), timeout=5.0)
except asyncio.TimeoutError:
    handle_timeout()

# After: anyio cancel scope (fail_after)
with anyio.fail_after(5.0):
    result = await operation()
# Raises TimeoutError on timeout

# Or: move_on_after (doesn't raise)
with anyio.move_on_after(5.0) as scope:
    result = await operation()

if scope.cancelled_caught:
    handle_timeout()
```

### Step 5: Replace run_in_executor

```python
# Before: asyncio run_in_executor
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, blocking_function, arg1, arg2)

# After: anyio.to_thread.run_sync
result = await anyio.to_thread.run_sync(blocking_function, arg1, arg2)
```

---

## Patterns

### Streaming with Cancellation

```python
async def stream_with_timeout(
    source: AsyncIterable[str],
    timeout_seconds: float,
) -> AsyncGenerator[str, None]:
    """Stream items with overall timeout and proper cancellation."""

    async with anyio.create_task_group() as tg:
        send_stream, receive_stream = anyio.create_memory_object_stream()

        async def producer():
            async with send_stream:
                async for item in source:
                    await send_stream.send(item)

        tg.start_soon(producer)

        with anyio.move_on_after(timeout_seconds):
            async with receive_stream:
                async for item in receive_stream:
                    yield item

        # Task group ensures producer is cancelled if we timeout
```

### Graceful Shutdown

```python
async def run_server():
    """Server with graceful shutdown on signals."""

    async with anyio.create_task_group() as tg:
        # Start server
        tg.start_soon(serve_requests)

        # Wait for shutdown signal
        with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
            async for sig in signals:
                print(f"Received {sig}, shutting down...")
                tg.cancel_scope.cancel()
                break

        # Cleanup happens automatically as task group exits
```

### Concurrent Operations with Partial Failure Handling

```python
async def fetch_all_with_fallbacks(urls: list[str]) -> list[Response | None]:
    """Fetch URLs concurrently, None for failures."""
    results: dict[int, Response | None] = {}

    async def fetch_one(index: int, url: str):
        try:
            results[index] = await http_client.get(url)
        except Exception:
            results[index] = None

    async with anyio.create_task_group() as tg:
        for i, url in enumerate(urls):
            tg.start_soon(fetch_one, i, url)

    return [results[i] for i in range(len(urls))]
```

---

## Gotchas

### 1. finally Blocks Are Still Cancelled

```python
async def operation():
    try:
        await risky_thing()
    finally:
        # This is STILL in cancelled state!
        # Any await here will raise CancelledError
        await cleanup()  # Will fail!

        # Fix: shield the cleanup
        with anyio.CancelScope(shield=True):
            await cleanup()  # Now works
```

### 2. Ecosystem Gaps

Some popular libraries don't support anyio/trio:

- **asyncpg**: No anyio support (use asyncpg with asyncio backend)
- **aiohttp**: asyncio-only (use httpx with anyio instead)
- **motor**: asyncio-only

Workaround: Use anyio with asyncio backend for these cases:

```python
# anyio defaults to asyncio backend anyway
anyio.run(main)  # Uses asyncio under the hood

# Explicit:
anyio.run(main, backend="asyncio")
```

### 3. Performance

uvloop doesn't support trio backend. If you need maximum performance:

```python
# Use anyio with asyncio backend + uvloop
import uvloop
uvloop.install()
anyio.run(main, backend="asyncio")
```

---

## When Raw asyncio is Fine

Don't over-engineer. Raw asyncio is acceptable for:

1. **Simple scripts** - No complex cancellation, no task groups needed
2. **Performance-critical hot paths** - Minimal abstraction overhead
3. **Heavy asyncpg usage** - Wrapping adds complexity
4. **Team familiarity** - If team knows asyncio well and code is working

The goal is correctness and maintainability, not purity. anyio is a tool, not a religion.

---

## References

- [anyio documentation](https://anyio.readthedocs.io/)
- [trio documentation](https://trio.readthedocs.io/)
- [Notes on structured concurrency](https://vorpus.org/blog/notes-on-structured-concurrency-or-go-statement-considered-harmful/)
- [Timeouts and cancellation for humans](https://vorpus.org/blog/timeouts-and-cancellation-for-humans/)
