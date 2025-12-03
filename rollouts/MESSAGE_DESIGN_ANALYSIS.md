# Message Design Analysis: Rollouts vs Agent-Runner

## Question 1: Do we need to revise the core Message design?

**Short answer: No major revision needed. Rollouts' Message is actually MORE feature-rich than agent-runner's.**

---

## Message Comparison

### **Rollouts Message (dtypes.py:68-76)**

```python
@dataclass(frozen=True)
class Message(JsonSerializable):
    role: str
    content: Optional[str | List[Dict[str, Any]]]  # Vision support!
    reasoning_content: Optional[Any] = None         # O1 reasoning 🆕
    thinking_content: Optional[str] = None          # Extended thinking 🆕
    thinking_signature: Optional[str] = None        # Thinking metadata 🆕
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
```

### **Agent-Runner Message (messages.py:14-26)**

```python
@dataclass
class Message:
    id: str                                          # Message ID
    role: str
    content: str                                     # No vision support
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)  # Metadata 🆕
```

---

## Feature Comparison Matrix

| Feature | Rollouts | Agent-Runner | Winner | Notes |
|---------|----------|--------------|--------|-------|
| **Vision support** | ✅ `str \| List[Dict]` | ❌ `str` only | Rollouts | Critical for multimodal |
| **O1 reasoning** | ✅ `reasoning_content` | ❌ Missing | Rollouts | Required for O1 models |
| **Extended thinking** | ✅ `thinking_content` | ❌ Missing | Rollouts | Anthropic extended thinking |
| **Thinking metadata** | ✅ `thinking_signature` | ❌ Missing | Rollouts | Tracking thinking mode |
| **Message ID** | ❌ Missing | ✅ `id: str` | Agent-Runner | Useful for tracing |
| **Metadata** | ❌ Missing | ✅ `meta: dict` | Agent-Runner | Extensibility |
| **Immutability** | ✅ `frozen=True` | ❌ Mutable | Rollouts | Functional purity |
| **Validation** | ❌ No validation | ✅ `__post_init__` | Agent-Runner | Type safety |

---

## Analysis

### **What Rollouts has that Agent-Runner doesn't:**

1. **Vision support** - `content: str | List[Dict[str, Any]]`
   - Critical for multimodal models
   - Agent-runner is text-only

2. **O1 reasoning** - `reasoning_content: Optional[Any]`
   - Stores O1 model's internal reasoning
   - Agent-runner can't capture this

3. **Extended thinking** - `thinking_content: str` + `thinking_signature: str`
   - Anthropic's extended thinking feature
   - Agent-runner has no equivalent

4. **Immutability** - `@dataclass(frozen=True)`
   - Functional purity
   - Agent-runner's Message is mutable (bug-prone)

### **What Agent-Runner has that Rollouts doesn't:**

1. **Message ID** - `id: str`
   - Unique identifier for each message
   - Useful for tracing, debugging, referencing
   - **This is valuable for rollouts**

2. **Metadata dict** - `meta: dict[str, Any]`
   - Extensible metadata (timestamps, tool names, etc.)
   - Currently rollouts has no way to attach arbitrary metadata
   - **This is valuable for rollouts**

3. **Validation** - `__post_init__` checks
   - Validates role is valid
   - Validates tool messages have tool_call_id
   - Validates only assistant can have tool_calls
   - **This is valuable for rollouts**

---

## Recommendations

### ✅ **Add to Rollouts Message (High Priority)**

#### **1. Message ID**

```python
@dataclass(frozen=True)
class Message(JsonSerializable):
    role: str
    content: Optional[str | List[Dict[str, Any]]]
    reasoning_content: Optional[Any] = None
    thinking_content: Optional[str] = None
    thinking_signature: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None

    # 🆕 Add message ID
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
```

**Why:**
- Enables message-level tracing
- Can reference specific messages in logs/analysis
- Debugging: "Error occurred in message abc123"
- No breaking change (has default)

#### **2. Metadata dict**

```python
@dataclass(frozen=True)
class Message(JsonSerializable):
    # ... existing fields ...

    # 🆕 Add metadata
    meta: Dict[str, Any] = field(default_factory=dict)
```

**Why:**
- Extensible: Can add timestamps, model info, latency, etc. without changing schema
- Training: Attach metadata like `{"generated_at": timestamp, "temperature": 0.7}`
- Debugging: Store provider-specific info
- No breaking change (has default)

**Use cases:**

```python
# Store timestamp
Message(
    role="assistant",
    content="Hello",
    meta={"timestamp": time.time(), "model": "gpt-4o"}
)

# Store tool execution metadata
Message(
    role="tool",
    content="File created",
    tool_call_id="123",
    meta={"tool_name": "write_file", "latency_ms": 45.2, "files_changed": ["main.py"]}
)

# Store reasoning metadata
Message(
    role="assistant",
    content="...",
    reasoning_content="...",
    meta={"reasoning_tokens": 1542, "reasoning_effort": "high"}
)
```

#### **3. Validation (Optional but recommended)**

```python
@dataclass(frozen=True)
class Message(JsonSerializable):
    # ... all fields ...

    def __post_init__(self):
        """Validate message constraints."""
        # Use object.__setattr__ because frozen=True

        # Validate role
        valid_roles = ("system", "user", "assistant", "tool")
        if self.role not in valid_roles:
            raise ValueError(f"Invalid role: {self.role}. Must be one of {valid_roles}")

        # Tool messages must have tool_call_id
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages MUST have tool_call_id")

        # Only tool messages can have tool_call_id
        if self.role != "tool" and self.tool_call_id:
            raise ValueError("Only tool messages may have tool_call_id")

        # Only assistant can have tool_calls
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("Only assistant messages may have tool_calls")
```

**Why:**
- Catch bugs early (invalid messages)
- Self-documenting (enforces protocol)
- Training: Ensures clean data (no malformed messages)

---

### ❌ **Don't Add (Rollouts already better)**

1. **Mutable Message** - Keep `frozen=True`
2. **String-only content** - Keep vision support
3. **Remove reasoning/thinking fields** - These are valuable

---

## Question 2: How to add tracing and other prod features?

### **Tracing Architecture Options**

There are **3 architectural approaches** for tracing:

---

### **Option 1: Metadata-Based Tracing (Recommended for Rollouts)**

**Pattern:** Store trace data in `Message.meta` and `Trajectory.metadata`

**Pros:**
- ✅ No new data structures
- ✅ Backward compatible
- ✅ Functional (no side effects)
- ✅ Fits rollouts' design

**Cons:**
- ❌ Less structured (dict instead of typed events)
- ❌ No real-time streaming

**Implementation:**

```python
# Enhanced Message with tracing metadata
@dataclass(frozen=True)
class Message(JsonSerializable):
    role: str
    content: Optional[str | List[Dict[str, Any]]]
    reasoning_content: Optional[Any] = None
    thinking_content: Optional[str] = None
    thinking_signature: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None

    # 🆕 Tracing fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    meta: Dict[str, Any] = field(default_factory=dict)

# Usage in providers
async def rollout_openai(actor: Actor, on_chunk: Callable) -> Actor:
    start_time = time.time()

    # ... API call ...

    assistant_message = Message(
        role="assistant",
        content=response_text,
        tool_calls=tool_calls,
        meta={
            "timestamp": time.time(),
            "latency_ms": (time.time() - start_time) * 1000,
            "model": actor.endpoint.model,
            "provider": "openai",
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "finish_reason": response.finish_reason,
        }
    )

    return replace(actor, messages=actor.messages + (assistant_message,))

# Enhanced Trajectory with session metadata
@dataclass(frozen=True)
class Trajectory(JsonSerializable):
    # ... existing fields ...
    metadata: Dict[str, Any] = field(default_factory=dict)

# Usage
trajectory = Trajectory(
    messages=messages,
    rewards=0.0,
    metadata={
        "session_id": "abc123",
        "created_at": datetime.now().isoformat(),
        "environment": "calculator",
        "total_tokens": sum(m.meta.get("completion_tokens", 0) for m in messages),
        "total_cost": calculate_cost(messages),
    }
)
```

**Analysis:**

```python
# Post-hoc analysis of trajectories
def analyze_trajectories(trajectories: List[Trajectory]) -> Dict[str, Any]:
    """Analyze trajectory metadata."""
    total_tokens = 0
    total_cost = 0.0
    latencies = []

    for traj in trajectories:
        for msg in traj.messages:
            if msg.role == "assistant":
                total_tokens += msg.meta.get("completion_tokens", 0)
                latencies.append(msg.meta.get("latency_ms", 0))

        total_cost += traj.metadata.get("total_cost", 0.0)

    return {
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "num_trajectories": len(trajectories),
    }
```

---

### **Option 2: Event Stream (Agent-Runner Style)**

**Pattern:** Separate event stream alongside messages

**Pros:**
- ✅ Real-time streaming
- ✅ Structured events (typed)
- ✅ Can emit events during execution

**Cons:**
- ❌ More complex (two parallel data structures)
- ❌ Less functional (side effects via emit_event)
- ❌ Doesn't fit rollouts' pure function style

**Implementation:**

```python
# New event types
@dataclass(frozen=True)
class TraceEvent(JsonSerializable):
    event_id: str
    event_type: str  # "tool_call_started", "api_call", "token_usage"
    timestamp: float
    data: Dict[str, Any]

# Trajectory with events
@dataclass(frozen=True)
class Trajectory(JsonSerializable):
    messages: Tuple[Message, ...]
    rewards: float | List[float]
    completions: List[ChatCompletion]
    events: List[TraceEvent] = field(default_factory=list)  # 🆕

# Usage (requires changing function signatures)
async def rollout_openai(
    actor: Actor,
    on_chunk: Callable,
    emit_event: Callable[[TraceEvent], Awaitable[None]]  # 🆕
) -> Tuple[Actor, List[TraceEvent]]:  # 🆕 Return events

    events = []

    # Emit API call start
    events.append(TraceEvent(
        event_id=str(uuid.uuid4()),
        event_type="api_call_started",
        timestamp=time.time(),
        data={"model": actor.endpoint.model, "messages": len(actor.messages)}
    ))
    await emit_event(events[-1])

    # ... API call ...

    # Emit token usage
    events.append(TraceEvent(
        event_id=str(uuid.uuid4()),
        event_type="token_usage",
        timestamp=time.time(),
        data={"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
    ))
    await emit_event(events[-1])

    return updated_actor, events
```

**This breaks rollouts' functional design** - not recommended unless you need real-time event streaming.

---

### **Option 3: Hybrid (Best of Both Worlds)**

**Pattern:** Store trace data in `meta`, but also support optional event emission

**Pros:**
- ✅ Functional by default (metadata only)
- ✅ Optional real-time streaming (events if needed)
- ✅ Backward compatible

**Cons:**
- ❌ More complex implementation

**Implementation:**

```python
# Message with metadata (always)
@dataclass(frozen=True)
class Message(JsonSerializable):
    # ... fields ...
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    meta: Dict[str, Any] = field(default_factory=dict)

# Optional event emission (already exists!)
@dataclass(frozen=True)
class RunConfig:
    on_chunk: Callable[[StreamChunk], Awaitable[None]]
    emit_event: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None  # Already exists!

# Usage
async def rollout_openai(actor: Actor, on_chunk: Callable) -> Actor:
    start_time = time.time()

    # Optional: Emit event if configured
    if actor.run_config and actor.run_config.emit_event:
        await actor.run_config.emit_event("api_call_started", {
            "model": actor.endpoint.model,
            "timestamp": start_time,
        })

    # ... API call ...

    # Always: Store in message metadata
    assistant_message = Message(
        role="assistant",
        content=response_text,
        meta={
            "timestamp": time.time(),
            "latency_ms": (time.time() - start_time) * 1000,
            "model": actor.endpoint.model,
        }
    )

    # Optional: Emit event if configured
    if actor.run_config and actor.run_config.emit_event:
        await actor.run_config.emit_event("message_created", {
            "message_id": assistant_message.id,
            "role": "assistant",
            "latency_ms": assistant_message.meta["latency_ms"],
        })

    return replace(actor, messages=actor.messages + (assistant_message,))
```

**This is the rollouts way!** You already have `emit_event` in RunConfig - just start using it more.

---

## Recommended Implementation Plan

### **Phase 1: Enhance Message (Week 1)**

**Day 1-2: Add ID and metadata**

```python
# rollouts/dtypes.py

import uuid

@dataclass(frozen=True)
class Message(JsonSerializable):
    role: str
    content: Optional[str | List[Dict[str, Any]]]
    reasoning_content: Optional[Any] = None
    thinking_content: Optional[str] = None
    thinking_signature: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None

    # 🆕 Tracing fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate message constraints."""
        # Note: Must use object.__setattr__ because frozen=True
        valid_roles = ("system", "user", "assistant", "tool")
        if self.role not in valid_roles:
            raise ValueError(f"Invalid role: {self.role}. Must be one of {valid_roles}")

        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages MUST have tool_call_id")

        if self.role != "tool" and self.tool_call_id:
            raise ValueError("Only tool messages may have tool_call_id")

        if self.role != "assistant" and self.tool_calls:
            raise ValueError("Only assistant messages may have tool_calls")
```

**Day 3: Add Trajectory.metadata**

```python
@dataclass(frozen=True)
class Trajectory(JsonSerializable):
    messages: Tuple[Message, ...]
    rewards: float | List[float]
    completions: List[ChatCompletion]

    # 🆕 Session metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Day 4-5: Update providers to populate metadata**

```python
# rollouts/providers.py

async def rollout_openai(actor: Actor, on_chunk: Callable) -> Actor:
    start_time = time.time()

    # ... existing API call ...

    # 🆕 Populate message metadata
    assistant_message = Message(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        meta={
            "timestamp": time.time(),
            "latency_ms": (time.time() - start_time) * 1000,
            "model": actor.endpoint.model,
            "provider": "openai",
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "finish_reason": completion.choices[0].finish_reason,
        }
    )

    return replace(actor, messages=actor.messages + (assistant_message,))
```

**Tests:**
- Verify backward compatibility (existing code still works)
- Verify metadata is populated correctly
- Verify validation catches invalid messages

---

### **Phase 2: Add Tracing Utilities (Week 2)**

**Day 1-2: Add trajectory analysis utilities**

```python
# rollouts/trace_analysis.py

from typing import List, Dict, Any
from rollouts.dtypes import Trajectory, Message

def get_total_tokens(trajectory: Trajectory) -> int:
    """Get total tokens used in trajectory."""
    total = 0
    for msg in trajectory.messages:
        if msg.role == "assistant":
            total += msg.meta.get("prompt_tokens", 0)
            total += msg.meta.get("completion_tokens", 0)
    return total

def get_latency_stats(trajectory: Trajectory) -> Dict[str, float]:
    """Get latency statistics for trajectory."""
    latencies = [
        msg.meta["latency_ms"]
        for msg in trajectory.messages
        if msg.role == "assistant" and "latency_ms" in msg.meta
    ]

    if not latencies:
        return {"min": 0, "max": 0, "avg": 0, "total": 0}

    return {
        "min": min(latencies),
        "max": max(latencies),
        "avg": sum(latencies) / len(latencies),
        "total": sum(latencies),
    }

def get_tool_usage(trajectory: Trajectory) -> Dict[str, int]:
    """Count tool calls by tool name."""
    counts = {}
    for msg in trajectory.messages:
        if msg.role == "tool":
            tool_name = msg.meta.get("tool_name", "unknown")
            counts[tool_name] = counts.get(tool_name, 0) + 1
    return counts

def filter_by_token_count(
    trajectories: List[Trajectory],
    min_tokens: int = 0,
    max_tokens: int = float('inf')
) -> List[Trajectory]:
    """Filter trajectories by token count."""
    return [
        traj for traj in trajectories
        if min_tokens <= get_total_tokens(traj) <= max_tokens
    ]

def filter_by_latency(
    trajectories: List[Trajectory],
    max_avg_latency_ms: float
) -> List[Trajectory]:
    """Filter trajectories by average latency."""
    return [
        traj for traj in trajectories
        if get_latency_stats(traj)["avg"] <= max_avg_latency_ms
    ]
```

**Day 3-4: Add session trace export**

```python
# rollouts/session_trace.py

import json
from pathlib import Path
from typing import List
from rollouts.dtypes import Trajectory, Message

def export_session_trace_jsonl(trajectory: Trajectory, filepath: str) -> None:
    """Export trajectory as interleaved JSONL (messages + metadata events).

    Format:
        {"_type": "metadata", "session_id": "...", ...}
        {"_type": "message", "id": "...", "role": "user", ...}
        {"_type": "event", "type": "token_usage", "data": {...}}
        {"_type": "message", "id": "...", "role": "assistant", ...}
    """
    lines = []

    # Session metadata
    if trajectory.metadata:
        lines.append({
            "_type": "metadata",
            **trajectory.metadata
        })

    # Interleave messages and extract events from message.meta
    for msg in trajectory.messages:
        # Message
        msg_dict = {
            "_type": "message",
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "tool_calls": msg.tool_calls,
            "tool_call_id": msg.tool_call_id,
            "meta": msg.meta,
        }
        lines.append(msg_dict)

        # Extract events from meta (e.g., token_usage, latency)
        if msg.role == "assistant" and msg.meta:
            if "prompt_tokens" in msg.meta:
                lines.append({
                    "_type": "event",
                    "event_type": "token_usage",
                    "message_id": msg.id,
                    "timestamp": msg.meta.get("timestamp"),
                    "data": {
                        "prompt_tokens": msg.meta["prompt_tokens"],
                        "completion_tokens": msg.meta.get("completion_tokens", 0),
                    }
                })

            if "latency_ms" in msg.meta:
                lines.append({
                    "_type": "event",
                    "event_type": "latency",
                    "message_id": msg.id,
                    "timestamp": msg.meta.get("timestamp"),
                    "data": {"latency_ms": msg.meta["latency_ms"]}
                })

    # Write JSONL
    Path(filepath).write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in lines),
        encoding="utf-8"
    )

def load_session_trace_jsonl(filepath: str) -> Trajectory:
    """Load trajectory from JSONL trace file."""
    # Implementation: Parse JSONL and reconstruct Trajectory
    pass
```

**Day 5: Documentation**

Write docs on:
- How to use message metadata
- How to analyze trajectories
- How to export/import session traces

---

### **Phase 3: Enhanced Event Emission (Optional - Week 3)**

Only do this if you need real-time streaming for debugging/monitoring.

```python
# Enhance existing emit_event usage in agents.py

async def agent_loop(state: AgentState, run_config: RunConfig) -> AgentState:
    """Enhanced agent loop with tracing events."""

    # Start of turn
    if run_config.emit_event:
        await run_config.emit_event("turn_started", {
            "turn": state.turn_idx,
            "timestamp": time.time(),
        })

    # API call
    start_time = time.time()
    next_actor = await rollout(state.actor, run_config.on_chunk)
    latency = time.time() - start_time

    if run_config.emit_event:
        await run_config.emit_event("api_call_completed", {
            "turn": state.turn_idx,
            "latency_ms": latency * 1000,
            "timestamp": time.time(),
        })

    # Tool execution
    for tool_call in next_actor.pending_tool_calls:
        if run_config.emit_event:
            await run_config.emit_event("tool_call_started", {
                "tool_name": tool_call.name,
                "tool_call_id": tool_call.id,
                "timestamp": time.time(),
            })

        result = state.environment.execute_tool(tool_call, state)

        if run_config.emit_event:
            await run_config.emit_event("tool_call_completed", {
                "tool_name": tool_call.name,
                "tool_call_id": tool_call.id,
                "success": result.ok,
                "timestamp": time.time(),
            })

    return updated_state
```

---

## Summary

### **Message Design: Minor Enhancements**

**Add to Rollouts Message:**
1. ✅ `id: str` - Message identifier (backward compatible)
2. ✅ `meta: Dict[str, Any]` - Extensible metadata (backward compatible)
3. ✅ `__post_init__` validation - Type safety (backward compatible)

**Keep from Rollouts:**
- ✅ Vision support (`str | List[Dict]`)
- ✅ O1 reasoning (`reasoning_content`)
- ✅ Extended thinking (`thinking_content`, `thinking_signature`)
- ✅ Immutability (`frozen=True`)

### **Tracing: Metadata-Based Approach (Recommended)**

**Use existing functional patterns:**
1. Store trace data in `Message.meta` (timestamps, latency, tokens)
2. Store session data in `Trajectory.metadata` (session_id, environment)
3. Use existing `emit_event` callback for optional real-time streaming
4. Add post-hoc analysis utilities

**Don't adopt:**
- ❌ Separate event stream (breaks functional design)
- ❌ Stateful session management (rollouts has checkpoints)

### **Implementation Effort**

- **Week 1:** Enhance Message (50 lines) + update providers (100 lines) = **2-3 days**
- **Week 2:** Add tracing utilities (200 lines) = **2-3 days**
- **Week 3:** Enhanced event emission (optional) = **2-3 days**

**Total: 5-9 days for full production tracing**

---

## Final Recommendation

1. **Do enhance Message** - Add `id` and `meta` fields (backward compatible)
2. **Do use metadata-based tracing** - Fits rollouts' functional style perfectly
3. **Don't add separate event stream** - You already have `emit_event` callback
4. **Do add analysis utilities** - Make it easy to query/filter trajectories

This gives you production-grade tracing without compromising rollouts' clean functional architecture!
