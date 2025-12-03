# `on_chunk` vs `emit_event` in Rollouts: Analysis & Recommendations

## TL;DR

**You're right** - there's overlap between `on_chunk` and `emit_event`. Here's what each does:

- **`on_chunk`** - Stream **token-level** and **tool-level** events (fine-grained, high-frequency)
- **`emit_event`** - Emit **turn-level** and **sample-level** events (coarse-grained, low-frequency)

**Recommendation:** **Keep both**, but clarify their purposes. They serve different layers of granularity.

---

## Current Implementation

### **1. `on_chunk` - Fine-Grained Streaming**

**Signature:**
```python
# rollouts/dtypes.py:477
on_chunk: Callable[[StreamChunk], Awaitable[None]]
```

**Used by:** `rollout_*` functions (providers.py)

**Emits:**
```python
# High-frequency events during LLM API streaming
StreamChunk(kind="token", data={"text": "Hello"})
StreamChunk(kind="thinking", data={"text": "..."})
StreamChunk(kind="tool_call_complete", data={"name": "bash", "args": {...}})
StreamChunk(kind="tool_result", data={"ok": True, "content": "..."})
```

**Purpose:** Real-time token streaming for live UI updates

**Call sites:**
- `rollout_openai()` - Emits tokens as they arrive from OpenAI
- `rollout_anthropic()` - Emits tokens + thinking content
- `rollout_sglang()` - Emits tokens from SGLang

**Example:**
```python
# providers.py:473
async for chunk in stream:
    if chunk.choices[0].delta.content:
        await on_chunk(StreamChunk("token", {
            "text": chunk.choices[0].delta.content
        }))
```

**Frequency:** **Very high** - One event per token (~50-500 events per turn)

---

### **2. `emit_event` - Coarse-Grained Events**

**Signature:**
```python
# rollouts/dtypes.py:492
emit_event: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None
```

**Used by:** `evaluation.py` and `agents.py`

**Emits:**
```python
# Low-frequency events for turn boundaries and sample lifecycle
await emit_event("sample_start", {"sample_id": "001", "data": {...}})
await emit_event("message", {"role": "user", "content": "..."})
await emit_event("tool_complete", {"turn": 2, "session_id": "abc"})
await emit_event("sample_end", {"sample_id": "001", "reward": 1.0})
```

**Purpose:** Track evaluation lifecycle and turn boundaries

**Call sites:**

1. **evaluation.py:273-285** - Sample lifecycle:
   ```python
   # Start of sample
   await run_config.emit_event("sample_start", {
       "sample_id": sample_id,
       "data": sample_data,
   })

   # Initial messages (before agent runs)
   for msg in initial_messages:
       await run_config.emit_event("message", {
           "role": msg.role,
           "content": msg.content,
       })
   ```

2. **agents.py:65-69** - Checkpoint events:
   ```python
   async def handle_checkpoint_event(state, event, run_config):
       if run_config.emit_event is not None:
           await run_config.emit_event(event, {
               "turn": state.turn_idx,
               "session_id": session_id,
           })
   ```

   Called with:
   - `"tool_complete"` - After tool execution (agents.py:548)

**Frequency:** **Very low** - One event per turn or sample (~2-10 events per sample)

---

## Comparison

| Aspect | `on_chunk` | `emit_event` |
|--------|-----------|--------------|
| **Granularity** | Token-level | Turn/sample-level |
| **Frequency** | Very high (~500/turn) | Very low (~5/sample) |
| **Data type** | `StreamChunk` (typed) | `(str, Dict)` (untyped) |
| **Used by** | Providers (rollout_*) | Evaluation, agents |
| **Purpose** | Real-time UI streaming | Lifecycle tracking |
| **Examples** | "token", "tool_call_complete" | "sample_start", "tool_complete" |
| **Optional** | ❌ Required | ✅ Optional |
| **Type safety** | ✅ StreamChunk dataclass | ❌ Generic (str, dict) |

---

## Event Hierarchy

```
Sample Lifecycle (emit_event)
├── "sample_start"                    # emit_event
│   ├── Turn 0
│   │   ├── "token"                   # on_chunk (×50)
│   │   ├── "tool_call_complete"      # on_chunk
│   │   ├── "tool_result"             # on_chunk
│   │   └── "tool_complete"           # emit_event
│   ├── Turn 1
│   │   ├── "token"                   # on_chunk (×80)
│   │   ├── "thinking"                # on_chunk (×20)
│   │   ├── "tool_call_complete"      # on_chunk
│   │   ├── "tool_result"             # on_chunk
│   │   └── "tool_complete"           # emit_event
│   └── Turn 2
│       ├── "token"                   # on_chunk (×30)
│       └── (no tools)
└── "sample_end"                      # emit_event (hypothetical - not implemented)
```

---

## Current Problems

### **1. `emit_event` is Barely Used**

**Currently only emits:**
- `"sample_start"` (evaluation.py:274)
- `"message"` (evaluation.py:282) - Initial messages only
- `"tool_complete"` (agents.py:548)

**Missing events:**
- ❌ No `"sample_end"` event
- ❌ No `"turn_start"` event
- ❌ No `"turn_end"` event
- ❌ No `"usage_update"` event
- ❌ No `"status_update"` event

### **2. Overlap with `on_chunk`**

`emit_event("message", ...)` duplicates what `on_chunk` already does:
```python
# evaluation.py:282
await emit_event("message", {"role": "user", "content": "..."})

# But on_chunk already emits tokens that construct this message!
await on_chunk(StreamChunk("token", {"text": "..."}))  # Many times
```

**Redundant!**

### **3. Type Safety**

```python
# emit_event is untyped
emit_event: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]]

# on_chunk is typed
on_chunk: Callable[[StreamChunk], Awaitable[None]]
```

No compile-time safety for event types in `emit_event`.

### **4. Confusion About Purpose**

**From dtypes.py:488-491:**
```python
# Event emission for frontend live streaming. Generic emit_event(type, **data)
# more flexible than specific on_start/on_end hooks.
# TODO: Revisit this design choice - could use structured callbacks
#       (on_turn_start/end) for type safety.
# TODO: Consider semantic compression of events - currently emits every
#       token separately to events.jsonl.
```

**Comments show uncertainty about design!**

---

## Design Options

### **Option 1: Remove `emit_event` (Use `on_chunk` for Everything)**

**Merge into `on_chunk`:**

```python
@dataclass(frozen=True)
class StreamChunk:
    kind: str  # Expand to include lifecycle events
    data: Mapping[str, Any]
    timestamp: float = field(default_factory=time.time)
    turn: int | None = None

# New kinds:
# - "sample_start"
# - "sample_end"
# - "turn_start"
# - "turn_end"
# - "tool_complete"
```

**Pros:**
- ✅ Single callback interface
- ✅ Type-safe (StreamChunk)
- ✅ Simpler API

**Cons:**
- ❌ High-frequency callback also handles low-frequency events (mixing concerns)
- ❌ Frontend must filter events by frequency
- ❌ Unclear separation of layers

---

### **Option 2: Keep Both, Clarify Purpose**

**`on_chunk`** - Token/tool streaming (high-frequency)
**`emit_event`** - Lifecycle events (low-frequency)

**Enhance `emit_event` to actually emit lifecycle events:**

```python
# evaluation.py
async def evaluate_sample(...):
    # Sample start
    if run_config.emit_event:
        await run_config.emit_event("sample_start", {
            "sample_id": sample_id,
            "timestamp": time.time(),
        })

    # Run agent (emits on_chunk events internally)
    states = await run_agent(initial_state, run_config)

    # Sample end
    if run_config.emit_event:
        await run_config.emit_event("sample_end", {
            "sample_id": sample_id,
            "reward": scored_trajectory.rewards,
            "turns": len(states),
            "duration_seconds": time.time() - start_time,
        })

# agents.py
async def agent_loop(state, run_config):
    while state.turn_idx < state.max_turns:
        # Turn start
        if run_config.emit_event:
            await run_config.emit_event("turn_start", {
                "turn": state.turn_idx,
                "timestamp": time.time(),
            })

        # Execute turn (emits on_chunk events)
        next_actor = await rollout(state.actor, run_config.on_chunk)

        # Turn end
        if run_config.emit_event:
            await run_config.emit_event("turn_end", {
                "turn": state.turn_idx,
                "stop_reason": state.stop,
                "timestamp": time.time(),
            })
```

**Pros:**
- ✅ Clear separation of concerns
- ✅ Low-frequency events don't spam high-frequency channel
- ✅ Frontend can optimize differently for each

**Cons:**
- ❌ Two callback interfaces to implement
- ❌ Less type safety for `emit_event`

---

### **Option 3: Structured Callbacks (Type-Safe Events)**

**Replace `emit_event` with structured callbacks:**

```python
@dataclass(frozen=True)
class RunConfig:
    # Fine-grained streaming (keep as-is)
    on_chunk: Callable[[StreamChunk], Awaitable[None]]

    # Structured lifecycle callbacks (type-safe!)
    on_sample_start: Optional[Callable[[str, Dict], Awaitable[None]]] = None
    on_sample_end: Optional[Callable[[str, float, int], Awaitable[None]]] = None
    on_turn_start: Optional[Callable[[int], Awaitable[None]]] = None
    on_turn_end: Optional[Callable[[int, StopReason | None], Awaitable[None]]] = None
```

**Pros:**
- ✅ Type-safe!
- ✅ Explicit which events are supported
- ✅ Clear separation

**Cons:**
- ❌ More callbacks to implement
- ❌ Less flexible (can't add new event types without changing RunConfig)

---

## Recommendation

### **Option 2: Keep Both, But Enhance `emit_event`**

**Why:**
1. ✅ Already have both in codebase
2. ✅ Clear separation: `on_chunk` = streaming, `emit_event` = lifecycle
3. ✅ Can optimize frontends differently for each
4. ✅ Minimal breaking changes

**Action items:**

#### **1. Clarify Purpose in Documentation**

```python
@dataclass(frozen=True)
class RunConfig:
    """Configuration for agent execution.

    Callbacks:
    - on_chunk: HIGH-FREQUENCY streaming callback for tokens and tool events.
                Called hundreds of times per turn. Use for real-time UI updates.

    - emit_event: LOW-FREQUENCY lifecycle callback for turn/sample boundaries.
                  Called ~5-10 times per sample. Use for progress tracking and
                  structured logging.
    """

    # High-frequency: Token/tool streaming
    on_chunk: Callable[[StreamChunk], Awaitable[None]]

    # Low-frequency: Lifecycle events
    emit_event: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None
```

#### **2. Add Missing Lifecycle Events**

```python
# agents.py - Add turn boundaries
async def agent_loop(state: AgentState, run_config: RunConfig) -> AgentState:
    while state.turn_idx < state.max_turns:
        # Emit turn start
        if run_config.emit_event:
            await run_config.emit_event("turn_start", {
                "turn": state.turn_idx,
                "timestamp": time.time(),
            })

        # Execute turn (emits on_chunk)
        next_state = await execute_turn(state, run_config)

        # Emit turn end
        if run_config.emit_event:
            await run_config.emit_event("turn_end", {
                "turn": state.turn_idx,
                "stop_reason": str(next_state.stop) if next_state.stop else None,
                "timestamp": time.time(),
            })

        state = next_state

        if state.stop:
            break

    return state

# evaluation.py - Add sample end
async def evaluate_sample(...):
    start_time = time.time()

    # Sample start (already exists)
    if run_config.emit_event:
        await run_config.emit_event("sample_start", {...})

    # Run agent
    states = await run_agent(initial_state, run_config)
    scored_trajectory = reward_fn(states[-1].actor.trajectory)

    # Sample end (NEW!)
    if run_config.emit_event:
        await run_config.emit_event("sample_end", {
            "sample_id": sample_id,
            "reward": scored_trajectory.rewards,
            "turns_used": states[-1].turn_idx,
            "duration_seconds": time.time() - start_time,
            "status": "success" if not error_message else "failed",
        })
```

#### **3. Remove Redundant `emit_event("message", ...)` Call**

```python
# evaluation.py:279-285 - Remove this block
# BEFORE (redundant):
for msg in initial_messages:
    await run_config.emit_event("message", {
        "role": msg.role,
        "content": msg.content,
    })

# AFTER (removed - on_chunk already handles this)
# Just rely on on_chunk during agent execution
```

**Why remove:** `on_chunk` already emits tokens that construct these messages. No need to duplicate.

#### **4. Document Event Catalog**

```python
# rollouts/events.md (new file)

# Event Catalog

## High-Frequency Events (on_chunk)

Emitted by providers during LLM streaming.

- **token** - Text token streamed
  ```python
  StreamChunk("token", {"text": "Hello"})
  ```

- **thinking** - Extended thinking token (Anthropic)
  ```python
  StreamChunk("thinking", {"text": "Let me think..."})
  ```

- **tool_call_complete** - Tool call ready for execution
  ```python
  StreamChunk("tool_call_complete", {
      "name": "bash",
      "args": {"command": "ls"}
  })
  ```

- **tool_result** - Tool execution result
  ```python
  StreamChunk("tool_result", {
      "ok": True,
      "content": "file1.py\nfile2.py"
  })
  ```

## Low-Frequency Events (emit_event)

Emitted by evaluation/agent loop for lifecycle tracking.

- **sample_start** - Evaluation sample began
  ```python
  await emit_event("sample_start", {
      "sample_id": "001",
      "timestamp": 1705318800.0,
  })
  ```

- **sample_end** - Evaluation sample completed
  ```python
  await emit_event("sample_end", {
      "sample_id": "001",
      "reward": 1.0,
      "turns_used": 3,
      "duration_seconds": 15.2,
      "status": "success",
  })
  ```

- **turn_start** - Agent turn began
  ```python
  await emit_event("turn_start", {
      "turn": 2,
      "timestamp": 1705318805.0,
  })
  ```

- **turn_end** - Agent turn completed
  ```python
  await emit_event("turn_end", {
      "turn": 2,
      "stop_reason": None,
      "timestamp": 1705318810.0,
  })
  ```

- **tool_complete** - Tool execution finished (checkpoint)
  ```python
  await emit_event("tool_complete", {
      "turn": 2,
      "session_id": "abc123",
  })
  ```
```

---

## Frontend Usage Patterns

### **Web Frontend (Separate Handlers)**

```python
class WebFrontend:
    async def on_chunk(self, chunk: StreamChunk):
        """Handle high-frequency streaming events."""
        if chunk.kind == "token":
            # Stream to websocket immediately
            await self.websocket.send(json.dumps({
                "type": "token",
                "text": chunk.data["text"]
            }))

    async def emit_event(self, event_type: str, data: Dict):
        """Handle low-frequency lifecycle events."""
        # Log to structured event file
        with open("events.jsonl", "a") as f:
            f.write(json.dumps({
                "type": event_type,
                "data": data,
                "timestamp": time.time(),
            }) + "\n")

        # Update progress bar
        if event_type == "turn_end":
            self.update_progress(data["turn"])

# Usage
run_config = RunConfig(
    on_chunk=frontend.on_chunk,
    emit_event=frontend.emit_event,
)
```

### **CLI Frontend (Ignore Lifecycle)**

```python
class CLIFrontend:
    async def on_chunk(self, chunk: StreamChunk):
        """Only handle streaming - ignore lifecycle."""
        if chunk.kind == "token":
            print(chunk.data["text"], end="", flush=True)

# Usage (no emit_event needed)
run_config = RunConfig(
    on_chunk=frontend.on_chunk,
    # emit_event=None  # CLI doesn't need lifecycle events
)
```

---

## Summary

### **Current State:**

- ✅ `on_chunk` - Well-used for token/tool streaming
- ⚠️ `emit_event` - Barely used, unclear purpose

### **Your Instinct:**

**✅ Correct!** There is overlap and confusion.

### **Recommendation:**

**Keep both, but:**

1. ✅ **Clarify purposes** - `on_chunk` = streaming, `emit_event` = lifecycle
2. ✅ **Add missing events** - `turn_start`, `turn_end`, `sample_end`
3. ✅ **Remove redundancy** - Delete `emit_event("message", ...)` call
4. ✅ **Document event catalog** - Clear spec for what each emits

### **Should You Remove `emit_event`?**

**No!** It serves a different purpose:
- `on_chunk` = Real-time UI updates (high-frequency)
- `emit_event` = Progress tracking, logging (low-frequency)

**Frontends can use both for different purposes:**
- Web UI: Use both (streaming + progress bars)
- CLI: Use only `on_chunk` (just show tokens)
- Training pipeline: Use only `emit_event` (track progress, ignore tokens)

---

**TL;DR:**
- ✅ Keep `on_chunk` for token streaming
- ✅ Keep `emit_event` for lifecycle events
- ✅ Add missing `turn_start/end`, `sample_end` events
- ✅ Document clear separation of concerns
- ❌ Don't merge - they serve different layers

Does this match your thinking?