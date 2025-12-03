# Agent-Runner Tracing Implementation: Deep Dive

## Overview

Agent-runner implements tracing through an **EventBus pub-sub system** that emits `StreamEvent` objects during agent execution. This document explains how it works and compares it to rollouts' approach.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  AgentRunnerAgent                           │
│                                                             │
│  process_message() {                                       │
│    // At every step...                                     │
│    if (event_bus):                                         │
│        event_bus.publish(StreamEvent(...))                 │
│  }                                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ Publishes StreamEvents
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     EventBus                                │
│                                                             │
│  - _queue: asyncio.Queue[StreamEvent]                      │
│  - _subscribers: List[Callable]                            │
│                                                             │
│  publish(event) {                                          │
│    _queue.put_nowait(event)    // For async iteration     │
│    for handler in _subscribers:                           │
│        handler(event)           // Sync callbacks         │
│  }                                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
┌──────────────────┐      ┌──────────────────┐
│ CLI Frontend     │      │ Web Frontend     │
│                  │      │                  │
│ subscribe()      │      │ async iter       │
│ handler(event)   │      │ for event in bus │
└──────────────────┘      └──────────────────┘
```

---

## Core Components

### **1. StreamEvent Data Structure**

```python
# agent-runner/src/agentrunner/core/events.py:67-90

@dataclass
class StreamEvent:
    """Event emitted during agent execution."""

    type: EventType          # Event type (literal type-safe)
    data: dict[str, Any]     # Event payload
    model_id: str            # Which model created this event
    ts: str                  # ISO timestamp
    id: str                  # Unique event ID for deduplication
```

**Event Types (events.py:25-64):**
```python
EventType = Literal[
    # Agent execution
    "token_delta",           # Streaming token
    "assistant_delta",       # Assistant message chunk
    "user_message",          # User input
    "assistant_message",     # Final assistant message
    "tool_call_started",     # Tool execution began
    "tool_output",           # Tool output
    "tool_call_completed",   # Tool execution finished
    "status_update",         # Agent status change
    "usage_update",          # Token usage stats
    "error",                 # Error occurred
    "compaction",            # Context window compaction

    # File system
    "file_created",          # File created
    "file_modified",         # File modified
    "file_tree_update",      # File tree changed

    # Deployment
    "scaffold_complete",     # Scaffold generated
    "preview_update",        # Preview URL update
    "preview_ready",         # Preview server ready
    "deployment_ready",      # Vercel deployment complete

    # Server/Screenshot
    "server_starting",       # Dev server starting
    "server_ready",          # Dev server ready
    "screenshot_taken",      # Screenshot captured
    "server_stopped",        # Dev server stopped

    # Bash execution
    "bash_started",          # Bash command started
    "bash_executed",         # Bash command completed

    # Session
    "session_created",       # New session created
    "session_restored",      # Session resumed
    "workspace_updated",     # Workspace changed

    # Multi-agent
    "execution_summary",     # Multi-agent summary
    "agent_error",           # Agent error
]
```

---

### **2. EventBus Implementation**

```python
# agent-runner/src/agentrunner/core/events.py:92-223

class EventBus:
    """Thread-safe event bus for streaming agent updates."""

    def __init__(self, max_history: int = 100):
        self._subscribers: list[Callable[[StreamEvent], None]] = []
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._closed = False
        self._event_history: list[StreamEvent] = []
        self._max_history = max_history

    def publish(self, event: StreamEvent) -> None:
        """Publish event to all subscribers and queue."""
        # Add to queue (for async iteration)
        self._queue.put_nowait(event)

        # Call all sync subscribers
        for handler in self._subscribers:
            try:
                handler(event)
            except Exception as e:
                logger.error("Handler failed", error=str(e))

    def subscribe(self, handler: Callable[[StreamEvent], None]) -> None:
        """Subscribe to events with sync callback."""
        self._subscribers.append(handler)

    def unsubscribe(self, handler: Callable[[StreamEvent], None]) -> None:
        """Unsubscribe from events."""
        self._subscribers.remove(handler)

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """Async iteration over events."""
        while not self._closed or not self._queue.empty():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                yield event
            except TimeoutError:
                continue
```

**Key design choices:**

1. **Dual consumption model:**
   - Sync callbacks via `subscribe()` - for CLI handlers
   - Async iteration via `async for event in bus` - for web/streaming

2. **Global state:**
   - EventBus is passed to Agent constructor
   - Agent holds reference and publishes throughout execution
   - Not functional - relies on shared mutable state

3. **Buffering:**
   - Queue stores events for async consumers
   - History buffer for late subscribers (not shown in code)

---

### **3. Agent Integration**

Agent emits events at **every significant step**:

```python
# agent-runner/src/agentrunner/core/agent.py

class AgentRunnerAgent:
    def __init__(self, ..., event_bus: EventBus | None = None):
        self.event_bus = event_bus
        # ...

    async def process_message(self, user_message: str) -> AssistantResult:
        # 1. Emit user message event
        if self.event_bus:
            self.event_bus.publish(StreamEvent(
                type="user_message",
                data={"message_id": msg.id, "content": user_message},
                model_id=self.provider.config.model,
                ts=datetime.now(UTC).isoformat(),
            ))

        # Agentic loop
        while rounds < max_rounds:
            # 2. Emit status: thinking
            if self.event_bus:
                self.event_bus.publish(StreamEvent(
                    type="status_update",
                    data={"status": "thinking", "detail": "Waiting for LLM response"},
                    model_id=self.provider.config.model,
                    ts=datetime.now(UTC).isoformat(),
                ))

            # 3. Get LLM response
            response = await self.provider.chat(messages, tools)

            # 4. If no tool calls, emit final message
            if not response.tool_calls:
                if self.event_bus:
                    self.event_bus.publish(StreamEvent(
                        type="assistant_message",
                        data={
                            "message_id": uuid.uuid4(),
                            "content": final_content,
                            "metadata": {
                                "rounds": rounds + 1,
                                "input_tokens": usage["prompt_tokens"],
                                "output_tokens": usage["completion_tokens"],
                                "total_tokens": usage["total_tokens"],
                                "cost": cost,
                            },
                        },
                        model_id=self.provider.config.model,
                        ts=datetime.now(UTC).isoformat(),
                    ))

                    # Emit status: completed
                    self.event_bus.publish(StreamEvent(
                        type="status_update",
                        data={"status": "idle", "detail": "Completed"},
                        model_id=self.provider.config.model,
                        ts=datetime.now(UTC).isoformat(),
                    ))

                return AssistantResult(...)

            # 5. Execute tools
            for tool_call in response.tool_calls:
                # Emit tool start
                if self.event_bus:
                    self.event_bus.publish(StreamEvent(
                        type="tool_call_started",
                        data={
                            "tool_name": tool_call.name,
                            "call_id": tool_call.id,
                            "arguments": tool_call.arguments,
                        },
                        model_id=self.provider.config.model,
                        ts=datetime.now(UTC).isoformat(),
                    ))

                # Execute tool
                start_time = time.time()
                result = await self.tools.execute(tool_call)
                duration_ms = int((time.time() - start_time) * 1000)

                # Emit tool complete
                if self.event_bus:
                    self.event_bus.publish(StreamEvent(
                        type="tool_call_completed",
                        data={
                            "tool_name": tool_call.name,
                            "call_id": tool_call.id,
                            "success": result.success,
                            "output": result.output or result.error,
                            "duration_ms": duration_ms,
                            "files_changed": result.files_changed or [],
                        },
                        model_id=self.provider.config.model,
                        ts=datetime.now(UTC).isoformat(),
                    ))
```

**Events emitted in typical execution:**

```
user_message          → User input received
status_update (thinking) → Waiting for LLM
assistant_delta       → (Optional) Streaming tokens
tool_call_started     → Tool execution begins
tool_call_completed   → Tool execution ends
status_update (thinking) → Back to LLM
assistant_message     → Final response
status_update (idle)  → Completed
usage_update          → Token usage stats
```

---

### **4. Provider Integration (Streaming)**

Providers emit `token_delta` events during streaming:

```python
# agent-runner/src/agentrunner/providers/openai_provider.py

async def _chat_stream_completions_api(
    self,
    messages: list[Message],
    tools: list[ToolDefinition] | None,
) -> AsyncIterator[StreamChunk]:
    """Stream using Chat Completions API."""

    stream = await self.async_client.chat.completions.create(
        model=self.config.model,
        messages=openai_messages,
        tools=openai_tools,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta

        # Token delta
        if delta.content:
            yield StreamChunk(
                type="token_delta",
                data={"text": delta.content},
                model_id=self.config.model,
            )

        # Tool call delta
        if delta.tool_calls:
            # ... accumulate tool calls ...
            yield StreamChunk(
                type="tool_call_delta",
                data={"tool_calls": accumulated},
                model_id=self.config.model,
            )
```

**Note:** Provider emits `StreamChunk`, not `StreamEvent`. Agent converts these to events.

---

## Frontend Consumption

### **CLI Frontend (Sync Subscription)**

```python
# agent-runner/src/agentrunner/cli/main.py (conceptual)

def handle_event(event: StreamEvent):
    """Sync event handler for CLI."""
    if event.type == "token_delta":
        print(event.data["text"], end="", flush=True)

    elif event.type == "tool_call_started":
        print(f"\n🔧 {event.data['tool_name']}...")

    elif event.type == "tool_call_completed":
        status = "✓" if event.data["success"] else "✗"
        print(f"{status} {event.data['output'][:50]}")

    elif event.type == "status_update":
        print(f"\n[{event.data['status']}]")

# Subscribe to event bus
event_bus = EventBus()
event_bus.subscribe(handle_event)

# Run agent (publishes events to bus)
agent = AgentRunnerAgent(provider, workspace, config, event_bus=event_bus)
result = await agent.process_message("Write code")
```

### **Web Frontend (Async Iteration)**

```python
# agent-runner web server (conceptual)

async def stream_handler(request):
    """SSE endpoint for web UI."""
    response = web.StreamResponse()
    response.headers['Content-Type'] = 'text/event-stream'
    await response.prepare(request)

    # Create event bus for this request
    event_bus = EventBus()

    # Run agent in background
    asyncio.create_task(agent.process_message("Write code"))

    # Stream events to client
    async for event in event_bus:
        data = json.dumps({
            "type": event.type,
            "data": event.data,
            "timestamp": event.ts,
        })
        await response.write(f"data: {data}\n\n".encode())

    return response
```

---

## Comparison: Agent-Runner vs Rollouts

| Aspect | Agent-Runner | Rollouts |
|--------|-------------|----------|
| **Mechanism** | EventBus (pub-sub) | Callbacks (`on_chunk`) |
| **Coupling** | Global state (EventBus) | Functional (injected callbacks) |
| **Event types** | 25+ typed events | 4 StreamChunk kinds |
| **Granularity** | Very fine (token_delta, status_update) | Coarse (token, tool_call, result) |
| **Consumption** | Dual (sync callbacks + async iteration) | Async callbacks only |
| **State** | Mutable (queue, subscribers) | Stateless (pure functions) |
| **Tracing** | Built-in (EventBus history) | Manual (Message.meta) |
| **Session storage** | SessionManager + EventBus | Checkpointing only |

---

## Detailed Tracing Flow Example

### **Agent-Runner: Full Execution Trace**

```python
# User executes: agent.process_message("Calculate 2+2")

# Events emitted (in order):

StreamEvent(
    type="user_message",
    data={"message_id": "msg1", "content": "Calculate 2+2"},
    model_id="gpt-4o",
    ts="2025-01-15T10:00:00.000Z"
)

StreamEvent(
    type="status_update",
    data={"status": "thinking", "detail": "Waiting for LLM response"},
    model_id="gpt-4o",
    ts="2025-01-15T10:00:00.100Z"
)

StreamEvent(
    type="token_delta",
    data={"text": "I'll"},
    model_id="gpt-4o",
    ts="2025-01-15T10:00:01.000Z"
)

StreamEvent(
    type="token_delta",
    data={"text": " use"},
    model_id="gpt-4o",
    ts="2025-01-15T10:00:01.050Z"
)

StreamEvent(
    type="token_delta",
    data={"text": " the"},
    model_id="gpt-4o",
    ts="2025-01-15T10:00:01.100Z"
)

StreamEvent(
    type="token_delta",
    data={"text": " calculator"},
    model_id="gpt-4o",
    ts="2025-01-15T10:00:01.150Z"
)

StreamEvent(
    type="tool_call_started",
    data={
        "tool_name": "calculator",
        "call_id": "tc1",
        "arguments": {"expr": "2+2"}
    },
    model_id="gpt-4o",
    ts="2025-01-15T10:00:02.000Z"
)

StreamEvent(
    type="tool_call_completed",
    data={
        "tool_name": "calculator",
        "call_id": "tc1",
        "success": True,
        "output": "4",
        "duration_ms": 15,
        "files_changed": []
    },
    model_id="gpt-4o",
    ts="2025-01-15T10:00:02.015Z"
)

StreamEvent(
    type="status_update",
    data={"status": "thinking", "detail": "Waiting for LLM response"},
    model_id="gpt-4o",
    ts="2025-01-15T10:00:02.020Z"
)

StreamEvent(
    type="token_delta",
    data={"text": "The"},
    model_id="gpt-4o",
    ts="2025-01-15T10:00:03.000Z"
)

StreamEvent(
    type="token_delta",
    data={"text": " answer"},
    model_id="gpt-4o",
    ts="2025-01-15T10:00:03.050Z"
)

StreamEvent(
    type="token_delta",
    data={"text": " is"},
    model_id="gpt-4o",
    ts="2025-01-15T10:00:03.100Z"
)

StreamEvent(
    type="token_delta",
    data={"text": " 4"},
    model_id="gpt-4o",
    ts="2025-01-15T10:00:03.150Z"
)

StreamEvent(
    type="assistant_message",
    data={
        "message_id": "msg2",
        "content": "The answer is 4",
        "metadata": {
            "rounds": 2,
            "input_tokens": 150,
            "output_tokens": 25,
            "total_tokens": 175,
            "cost": 0.00035
        }
    },
    model_id="gpt-4o",
    ts="2025-01-15T10:00:03.200Z"
)

StreamEvent(
    type="status_update",
    data={"status": "idle", "detail": "Completed"},
    model_id="gpt-4o",
    ts="2025-01-15T10:00:03.205Z"
)

StreamEvent(
    type="usage_update",
    data={
        "usage": {
            "prompt": 150,
            "completion": 25,
            "total": 175
        }
    },
    model_id="gpt-4o",
    ts="2025-01-15T10:00:03.210Z"
)
```

**Total: 16 events for single interaction!**

---

### **Rollouts: Same Execution**

```python
# User executes: agent_loop(state, run_config)

# Callbacks invoked:

await on_chunk(StreamChunk(
    kind="token",
    data={"text": "I'll use the calculator"}
))

await on_chunk(StreamChunk(
    kind="tool_call_complete",
    data={"name": "calculator", "args": {"expr": "2+2"}}
))

await on_chunk(StreamChunk(
    kind="tool_result",
    data={"ok": True, "content": "4"}
))

await on_chunk(StreamChunk(
    kind="token",
    data={"text": "The answer is 4"}
))
```

**Total: 4 callbacks for same interaction.**

---

## Pros and Cons

### **Agent-Runner's EventBus Approach**

**Pros:**
- ✅ **Very detailed tracing** - 25+ event types capture everything
- ✅ **Dual consumption** - Sync callbacks + async iteration
- ✅ **Built-in history** - Event buffering for late subscribers
- ✅ **Type-safe events** - Literal types for all event kinds
- ✅ **Rich metadata** - Token usage, costs, durations embedded in events

**Cons:**
- ❌ **Global state** - EventBus passed everywhere, breaks functional design
- ❌ **Tight coupling** - Agent knows about EventBus, not composable
- ❌ **Verbose** - 16 events for simple interaction
- ❌ **Hard to test** - Need to mock EventBus
- ❌ **Single-frontend bias** - Despite dual consumption, still CLI-centric

### **Rollouts' Callback Approach**

**Pros:**
- ✅ **Functional** - Pure functions, callbacks injected
- ✅ **Composable** - Easy to add new frontends (just implement callback)
- ✅ **Simple** - 4 StreamChunk kinds cover most needs
- ✅ **Testable** - Pass no-op callback for tests
- ✅ **Multi-frontend ready** - Backend has no UI knowledge

**Cons:**
- ❌ **Less detailed** - No built-in status updates, usage events
- ❌ **No history** - No buffering for late subscribers
- ❌ **Async only** - No sync callback support
- ❌ **Manual metadata** - Need to add Message.meta yourself

---

## What Rollouts Should Adopt

### ✅ **DO Adopt: Event Granularity**

Add more StreamChunk kinds:

```python
# rollouts/dtypes.py

@dataclass(frozen=True)
class StreamChunk(JsonSerializable):
    kind: str  # Expand this!
    data: Mapping[str, Any]

# Current kinds: "token", "tool_call_complete", "tool_result", "thinking"

# Add these:
# - "status_update" - Agent status changes
# - "usage_update" - Token usage per turn
# - "turn_start" - New turn began
# - "turn_end" - Turn completed
# - "file_changed" - File modification
```

**Implementation:**

```python
# rollouts/agents.py

async def agent_loop(state: AgentState, run_config: RunConfig) -> AgentState:
    while state.turn_idx < state.max_turns:
        # Emit turn start
        await run_config.on_chunk(StreamChunk(
            kind="turn_start",
            data={"turn": state.turn_idx}
        ))

        # Emit status
        await run_config.on_chunk(StreamChunk(
            kind="status_update",
            data={"status": "thinking"}
        ))

        # ... execute turn ...

        # Emit usage
        await run_config.on_chunk(StreamChunk(
            kind="usage_update",
            data={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
        ))

        # Emit turn end
        await run_config.on_chunk(StreamChunk(
            kind="turn_end",
            data={"turn": state.turn_idx, "stop_reason": state.stop}
        ))
```

### ✅ **DO Adopt: Structured Event Data**

Use consistent event data structure:

```python
@dataclass(frozen=True)
class StreamChunk(JsonSerializable):
    kind: str
    data: Mapping[str, Any]
    timestamp: float = field(default_factory=time.time)  # Add timestamp
    turn: int | None = None                              # Add turn number
```

### ❌ **DON'T Adopt: EventBus Architecture**

**Don't add EventBus** - rollouts' callback design is superior:

```python
# ❌ Don't do this (agent-runner style)
class Agent:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus  # Global state!

    async def run(self):
        self.event_bus.publish(...)  # Coupled to EventBus

# ✅ Do this (rollouts style)
async def agent_loop(
    state: AgentState,
    run_config: RunConfig  # Callback injected
) -> AgentState:
    await run_config.on_chunk(...)  # Pure function!
```

### ✅ **DO Adopt: Helper Functions**

Create event helper functions like agent-runner:

```python
# rollouts/stream_events.py

def create_usage_event(
    prompt_tokens: int,
    completion_tokens: int
) -> StreamChunk:
    """Create usage update event."""
    return StreamChunk(
        kind="usage_update",
        data={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        timestamp=time.time(),
    )

def create_status_event(status: str, detail: str = "") -> StreamChunk:
    """Create status update event."""
    return StreamChunk(
        kind="status_update",
        data={"status": status, "detail": detail},
        timestamp=time.time(),
    )
```

---

## Recommended Enhancement for Rollouts

### **Enhanced StreamChunk System**

```python
# rollouts/dtypes.py

@dataclass(frozen=True)
class StreamChunk(JsonSerializable):
    """Enhanced streaming chunk with more event types."""

    kind: str  # Event kind
    data: Mapping[str, Any]  # Event payload
    timestamp: float = field(default_factory=time.time)  # When emitted
    turn: int | None = None  # Turn number (optional)

    # Event kinds:
    # - "token" - Text token streamed
    # - "thinking" - Extended thinking token
    # - "tool_call_complete" - Tool call ready
    # - "tool_result" - Tool execution result
    # - "turn_start" - New turn began
    # - "turn_end" - Turn completed
    # - "status_update" - Status change (thinking, executing, idle)
    # - "usage_update" - Token usage stats
    # - "file_changed" - File modified
    # - "error" - Error occurred

# Usage in agents.py
async def agent_loop(state: AgentState, run_config: RunConfig) -> AgentState:
    for turn in range(state.max_turns):
        # Turn start
        await run_config.on_chunk(StreamChunk(
            kind="turn_start",
            data={"turn": turn, "max_turns": state.max_turns},
            turn=turn,
        ))

        # Status: thinking
        await run_config.on_chunk(StreamChunk(
            kind="status_update",
            data={"status": "thinking"},
            turn=turn,
        ))

        # Execute rollout (emits tokens)
        next_actor = await rollout(state.actor, run_config.on_chunk)

        # Execute tools (emits tool events)
        for tool_call in next_actor.pending_tool_calls:
            result = state.environment.execute_tool(tool_call, state)

            await run_config.on_chunk(StreamChunk(
                kind="tool_result",
                data={"name": result.name, "ok": result.ok, "content": result.content},
                turn=turn,
            ))

        # Usage update
        usage = next_actor.messages[-1].meta.get("usage", {})
        await run_config.on_chunk(StreamChunk(
            kind="usage_update",
            data=usage,
            turn=turn,
        ))

        # Turn end
        await run_config.on_chunk(StreamChunk(
            kind="turn_end",
            data={"turn": turn, "stop_reason": state.stop},
            turn=turn,
        ))
```

---

## Summary

### **Agent-Runner's Tracing:**

- **Mechanism:** EventBus pub-sub with global state
- **Granularity:** Very fine (25+ event types)
- **Coupling:** Tight (Agent holds EventBus reference)
- **Best for:** Detailed tracing, rich debugging
- **Architecture:** ❌ Breaks functional design

### **Rollouts' Tracing:**

- **Mechanism:** Callback-based with injected `on_chunk`
- **Granularity:** Coarse (4 StreamChunk kinds)
- **Coupling:** None (pure functions)
- **Best for:** Multi-frontend flexibility
- **Architecture:** ✅ Clean functional design

### **Recommendation:**

1. **Keep** rollouts' callback architecture (superior!)
2. **Add** more StreamChunk kinds (from agent-runner's event types)
3. **Add** timestamp and turn to StreamChunk
4. **Add** helper functions for creating events
5. **Don't** add EventBus (breaks functional design)

**Result:** Best of both worlds - detailed tracing with functional architecture!

Would you like me to implement the enhanced StreamChunk system for rollouts?