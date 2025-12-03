# Agent-Runner vs Rollouts: Feature Comparison & Adoption Analysis

## Executive Summary

**Your instinct is correct** - rollouts is missing several important data model features from agent-runner, particularly:

1. **OpenAI Responses API support** (GPT-5 Codex, O1 models)
2. **Enhanced tool result metadata** (diffs, files_changed tracking)
3. **Structured tracing/session history** (JSONL-based execution traces)

This document analyzes what's worth adopting from agent-runner into rollouts.

---

## Part 1: Provider Implementation Gaps

### 🚨 **CRITICAL: OpenAI Responses API Missing**

**What rollouts is missing:**

Agent-runner supports **two OpenAI APIs**:
1. **Chat Completions API** - Standard GPT-4o, GPT-4, etc.
2. **Responses API** - Required for O1, GPT-5 Codex, GPT-5.1 models

**Key differences:**

| Feature | Chat Completions | Responses API |
|---------|-----------------|---------------|
| Models | GPT-4o, GPT-4, GPT-3.5 | O1, GPT-5 Codex, GPT-5.1 |
| Temperature | ✅ Supported | ❌ Not supported |
| Message format | `messages` array | `instructions` + `input` items |
| Tool format | Nested `function.name/args` | Flat `function_call` items |
| Token field | `max_tokens` | `max_output_tokens` |
| Reasoning effort | N/A | `reasoning.effort` (low/med/high) |

**Agent-runner implementation:**

```python
# openai_provider.py:63-76
def _requires_responses_api(self, model: str) -> bool:
    """O1 and Codex models ONLY support Responses API."""
    responses_only_models = ["o1", "o1-preview", "o1-mini", "gpt-5-codex", "gpt-5.1"]
    return any(m in model.lower() for m in responses_only_models)

# Routing logic:
if self._requires_responses_api(self.config.model):
    return await self._chat_responses_api(messages, tools)
else:
    return await self._chat_completions_api(messages, tools)
```

**Message format conversion (openai_provider.py:621-683):**

```python
def _convert_messages_to_responses_format(messages):
    """
    Chat Completions format:
        {"role": "system", "content": "You are..."}
        {"role": "user", "content": "Hello"}
        {"role": "assistant", "tool_calls": [...]}
        {"role": "tool", "tool_call_id": "123", "content": "result"}

    Responses API format:
        instructions: "You are..."  # System message extracted
        input: [
            {"role": "user", "content": "Hello"},
            {"type": "function_call", "call_id": "123", "name": "bash", "arguments": "..."},
            {"type": "function_call_output", "call_id": "123", "output": "result"}
        ]
    """
    instructions = None
    input_items = []

    for msg in messages:
        if msg.role == "system":
            instructions = msg.content  # Extract as instructions
        elif msg.role == "user":
            input_items.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant" and msg.tool_calls:
            # Assistant tool calls → function_call items
            for tc in msg.tool_calls:
                input_items.append({
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                })
        elif msg.role == "tool":
            # Tool results → function_call_output items
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.tool_call_id,
                "output": msg.content
            })

    return instructions, input_items
```

**Reasoning effort extraction (for GPT-5.1):**

```python
# openai_provider.py:78-100
def _get_reasoning_effort(self, model: str) -> str | None:
    """
    GPT-5.1 models with reasoning suffixes:
        - gpt-5.1-2025-11-13-low → "low"
        - gpt-5.1-2025-11-13-medium → "medium"
        - gpt-5.1-2025-11-13-high → "high"
    """
    if "gpt-5.1" not in model.lower():
        return None

    if model.endswith("-low"):
        return "low"
    elif model.endswith("-medium"):
        return "medium"
    elif model.endswith("-high"):
        return "high"
    return None

# Used in API call:
api_params = {
    "model": normalized_model,
    "input": input_items,
    "instructions": instructions,
    "tools": openai_tools,
    "max_output_tokens": self.config.max_tokens,
}

if reasoning_effort:
    api_params["reasoning"] = {"effort": reasoning_effort}

response = await self.async_client.responses.create(**api_params)
```

**What rollouts needs to add:**

1. Detection logic for Responses API models
2. Message format converter (Chat → Responses format)
3. Tool format converter (nested → flat structure)
4. Reasoning effort extraction for GPT-5.1
5. Separate streaming handlers for both APIs

**Recommended approach for rollouts:**

```python
# rollouts/providers.py

def _requires_responses_api(endpoint: Endpoint) -> bool:
    """Pure function - check if model requires Responses API."""
    responses_models = ["o1", "o1-preview", "o1-mini", "gpt-5-codex", "gpt-5.1"]
    return any(m in endpoint.model.lower() for m in responses_models)

async def rollout_openai(actor: Actor, on_chunk: Callable) -> Actor:
    """Route to appropriate OpenAI API."""
    if _requires_responses_api(actor.endpoint):
        return await _rollout_openai_responses(actor, on_chunk)
    else:
        return await _rollout_openai_completions(actor, on_chunk)

def _convert_to_responses_format(messages: List[Message]) -> Tuple[Optional[str], List[Dict]]:
    """Convert rollouts messages to Responses API format.

    Returns: (instructions, input_items)
    """
    # Implementation similar to agent-runner
    pass

async def _rollout_openai_responses(actor: Actor, on_chunk: Callable) -> Actor:
    """Rollout using OpenAI Responses API (for O1, Codex)."""
    instructions, input_items = _convert_to_responses_format(actor.messages)

    # Convert tools to flat format
    tools = _convert_tools_to_responses_format(actor.tools) if actor.tools else None

    # Extract reasoning effort
    reasoning_effort = _extract_reasoning_effort(actor.endpoint.model)

    api_params = {
        "model": actor.endpoint.model,
        "input": input_items,
    }

    if instructions:
        api_params["instructions"] = instructions
    if tools:
        api_params["tools"] = tools
    if actor.endpoint.max_tokens:
        api_params["max_output_tokens"] = actor.endpoint.max_tokens
    if reasoning_effort:
        api_params["reasoning"] = {"effort": reasoning_effort}

    # NOTE: No temperature for Responses API!

    response = await client.responses.create(**api_params)
    # Parse response and return updated actor
```

---

## Part 2: Data Model Gaps

### **1. Enhanced ToolResult Metadata**

**Agent-runner's ToolResult (tool_protocol.py:27-36):**

```python
@dataclass
class ToolResult:
    success: bool
    output: str | None = None
    error: str | None = None
    error_code: str | None = None
    data: dict[str, Any] | None = None
    diffs: list[dict[str, Any]] | None = None      # 🆕 File diffs
    files_changed: list[str] | None = None          # 🆕 Changed file paths
```

**Rollouts' ToolResult (dtypes.py:30-37):**

```python
@dataclass(frozen=True)
class ToolResult(JsonSerializable):
    ok: bool
    name: str
    content: str
    tool_call_id: str
    # ❌ Missing: diffs, files_changed, error_code, data
```

**What's missing in rollouts:**

1. **`diffs`** - Structured file change diffs (useful for training on code edits)
2. **`files_changed`** - List of affected files (useful for filtering/metrics)
3. **`error_code`** - Structured error codes (E_NOT_FOUND, E_PERMISSIONS, etc.)
4. **`data`** - Arbitrary metadata (e.g., test results, search hits count)

**Why this matters for rollouts:**

```python
# Training use case: Filter samples by code change size
def filter_large_changes(trajectory: Trajectory) -> bool:
    """Filter out trajectories with massive diffs."""
    for msg in trajectory.messages:
        if msg.role == "tool" and msg.tool_result:
            if msg.tool_result.diffs:
                total_lines = sum(len(d["diff"]) for d in msg.tool_result.diffs)
                if total_lines > 500:  # Too large for training
                    return False
    return True

# Metrics use case: Track files edited per trajectory
def count_files_edited(trajectory: Trajectory) -> int:
    """Count unique files changed during trajectory."""
    files = set()
    for msg in trajectory.messages:
        if msg.role == "tool" and msg.tool_result:
            if msg.tool_result.files_changed:
                files.update(msg.tool_result.files_changed)
    return len(files)
```

**Recommendation:**

Add optional fields to rollouts' ToolResult:

```python
@dataclass(frozen=True)
class ToolResult(JsonSerializable):
    ok: bool
    name: str
    content: str
    tool_call_id: str
    error_code: Optional[str] = None                # Structured error codes
    diffs: Optional[List[Dict[str, Any]]] = None    # File diffs for code edits
    files_changed: Optional[List[str]] = None       # Changed file paths
    data: Optional[Dict[str, Any]] = None           # Arbitrary metadata
```

---

### **2. Structured Session Tracing**

**Agent-runner's CLISession (cli_session.py:17-100):**

Stores JSONL trace with interleaved messages and events:

```jsonl
{"_type": "metadata", "session_id": "cli_abc123", "created_at": "2025-01-15T10:00:00"}
{"_type": "message", "role": "user", "content": "Write a function", "timestamp": "..."}
{"_type": "event", "type": "tool_call_started", "data": {"name": "write_file", ...}, "timestamp": "..."}
{"_type": "event", "type": "file_created", "data": {"path": "main.py", "size": 142}, "timestamp": "..."}
{"_type": "message", "role": "assistant", "content": "Done!", "timestamp": "..."}
{"_type": "event", "type": "usage_update", "data": {"tokens": 350, "cost": 0.0012}, "timestamp": "..."}
```

**Benefits:**

1. **Full execution trace** - Every tool call, file change, token usage
2. **Debugging** - Replay exactly what happened at each step
3. **Analytics** - Post-hoc analysis of agent behavior
4. **Reproducibility** - Can reconstruct entire session state

**What rollouts has:**

- `Trajectory` - Complete conversation history with rewards
- `checkpoint.py` - Save/load trajectories
- `emit_event` callback - Optional event emission

**What rollouts is missing:**

- Structured JSONL trace format for sessions
- Interleaved events (tool calls, file changes, token usage)
- Session metadata (created_at, workspace, model)

**Recommendation:**

Rollouts doesn't need CLISession (too CLI-specific), but could benefit from:

```python
# rollouts/session_trace.py

@dataclass(frozen=True)
class TraceEvent(JsonSerializable):
    """Event emitted during trajectory execution."""
    event_type: str  # "tool_call_started", "file_changed", "usage_update"
    timestamp: float
    data: Dict[str, Any]

@dataclass(frozen=True)
class SessionTrace(JsonSerializable):
    """Complete execution trace with messages + events."""
    session_id: str
    trajectory: Trajectory
    events: List[TraceEvent]
    metadata: Dict[str, Any]

    def to_jsonl(self) -> str:
        """Export as interleaved JSONL (messages + events chronologically)."""
        # Merge messages and events by timestamp
        items = []
        for msg in self.trajectory.messages:
            items.append({"_type": "message", "timestamp": msg.timestamp, **asdict(msg)})
        for event in self.events:
            items.append({"_type": "event", **asdict(event)})

        # Sort by timestamp
        items.sort(key=lambda x: x["timestamp"])

        return "\n".join(json.dumps(item) for item in items)
```

**Use case for training:**

```python
# Analyze tool usage patterns across all sessions
def analyze_tool_patterns(session_traces: List[SessionTrace]) -> Dict[str, int]:
    """Count tool calls across all sessions."""
    tool_counts = {}
    for trace in session_traces:
        for event in trace.events:
            if event.event_type == "tool_call_started":
                tool_name = event.data["name"]
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
    return tool_counts
```

---

## Part 3: Tool System Comparison

### **Agent-Runner's Tool System**

**Key features:**

1. **ToolContext** - Rich execution context passed to every tool
2. **ToolRegistry** - Central registry with native/external tool separation
3. **Safety flags** - `requires_read_first`, `requires_confirmation` in ToolDefinition
4. **Error codes** - Standardized error codes (E_NOT_FOUND, E_VALIDATION, etc.)

**ToolContext (tools/base.py:21-40):**

```python
@dataclass
class ToolContext:
    workspace: Workspace           # Sandboxed file operations
    logger: AgentRunnerLogger      # Structured logging
    model_id: str                  # Which model is executing
    event_bus: EventBus | None     # Real-time event streaming
    config: dict[str, Any]         # Tool-specific config
    session_uid: int | None        # Unix UID for isolation
    session_gid: int | None        # Unix GID for isolation
    deployment_context: dict       # Deployment state, DB connections
```

**ToolDefinition with safety (tool_protocol.py:40-62):**

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    safety: dict[str, bool] = field(default_factory=dict)

    # Safety flags:
    # - requires_read_first: Must read file before editing
    # - requires_confirmation: User confirmation needed
```

**Example usage:**

```python
# tools/edit.py - Edit tool with safety
class EditTool(BaseTool):
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="edit",
            description="Edit files with search/replace",
            parameters={...},
            safety={
                "requires_read_first": True,      # Must Read first
                "requires_confirmation": False,   # Auto-approve
            }
        )
```

### **Rollouts' Tool System**

**Key features:**

1. **Environment Protocol** - Tool execution via environment.execute_tool()
2. **Tool confirmation** - Via confirm_tool callback in RunConfig
3. **Functional design** - Tools are pure functions, not classes

**Environment Protocol (dtypes.py:352-398):**

```python
@runtime_checkable
class Environment(Protocol):
    """Protocol for agent environments."""

    def execute_tool(self, tc: ToolCall, state: AgentState) -> Tuple[AgentState, ToolResult]:
        """Execute tool and return updated state + result."""
        ...

    def requires_confirmation(self, tc: ToolCall) -> bool:
        """Check if tool requires user confirmation."""
        ...

    def get_tools(self) -> List[Tool]:
        """Get available tools."""
        ...
```

**Tool definition (dtypes.py:150-159):**

```python
@dataclass(frozen=True)
class Tool(JsonSerializable):
    type: str  # "function"
    function: Dict[str, Any]  # {"name": "...", "description": "...", "parameters": {...}}
```

### **Comparison Summary**

| Feature | Agent-Runner | Rollouts |
|---------|-------------|----------|
| Tool abstraction | Class-based (BaseTool) | Protocol-based (Environment) |
| Execution context | ToolContext (rich) | AgentState (minimal) |
| Tool registry | Centralized ToolRegistry | Per-environment |
| Safety flags | In ToolDefinition | In Environment.requires_confirmation |
| Error codes | Standardized (E_*) | Freeform strings |
| Logging | Structured logger in context | Basic logger |
| Event emission | EventBus in context | on_chunk callback |
| Sandboxing | Workspace in context | Environment-specific |

**Winner: Rollouts** (functional design is cleaner)

**What rollouts could adopt:**

1. **Standardized error codes** - Add E_NOT_FOUND, E_VALIDATION constants
2. **Enhanced ToolResult** - Add diffs, files_changed fields (already discussed)
3. **ToolContext concept** - Not the class, but passing richer context to execute_tool

**Recommended enhancement:**

```python
# rollouts/dtypes.py

# Standardized error codes
E_NOT_FOUND = "E_NOT_FOUND"
E_VALIDATION = "E_VALIDATION"
E_PERMISSIONS = "E_PERMISSIONS"
E_TIMEOUT = "E_TIMEOUT"
E_UNSAFE = "E_UNSAFE"

@dataclass(frozen=True)
class ToolResult(JsonSerializable):
    ok: bool
    name: str
    content: str
    tool_call_id: str
    error_code: Optional[str] = None  # Use E_* constants
    diffs: Optional[List[Dict[str, Any]]] = None
    files_changed: Optional[List[str]] = None
    data: Optional[Dict[str, Any]] = None
```

---

## Part 4: What NOT to Adopt

### ❌ **Don't Adopt These Agent-Runner Patterns:**

1. **Class-based tools** - Rollouts' Environment protocol is cleaner
2. **ToolRegistry** - Rollouts' per-environment tools are more flexible
3. **EventBus** - Rollouts' callback-based design is superior
4. **Stateful providers** - Rollouts' pure functions are better
5. **Workspace class** - Rollouts' environment isolation is sufficient

---

## Part 5: Adoption Recommendations

### 🎯 **High Priority (Adopt These)**

#### 1. **OpenAI Responses API Support** ⭐⭐⭐⭐⭐

**Why:** Required for O1, GPT-5 Codex, GPT-5.1 models (latest cutting-edge models)

**What to adopt:**
- Model detection logic (`_requires_responses_api`)
- Message format converter (Chat → Responses)
- Tool format converter (nested → flat)
- Reasoning effort extraction (GPT-5.1)
- Streaming support for Responses API

**Estimated effort:** 300-400 lines of code in `rollouts/providers.py`

**Files to reference:**
- `agent-runner/src/agentrunner/providers/openai_provider.py:63-76` - Detection
- `agent-runner/src/agentrunner/providers/openai_provider.py:621-683` - Conversion
- `agent-runner/src/agentrunner/providers/openai_provider.py:229-321` - API call
- `agent-runner/src/agentrunner/providers/openai_provider.py:451-550` - Streaming

#### 2. **Enhanced ToolResult Metadata** ⭐⭐⭐⭐

**Why:** Critical for training analysis (filter by code change size, track files edited)

**What to adopt:**
- `diffs: Optional[List[Dict]]` - File change diffs
- `files_changed: Optional[List[str]]` - Changed file paths
- `error_code: Optional[str]` - Standardized error codes
- `data: Optional[Dict]` - Arbitrary metadata

**Estimated effort:** 50 lines (update dtypes.py + update environments)

**Files to reference:**
- `agent-runner/src/agentrunner/core/tool_protocol.py:27-36` - ToolResult definition
- `agent-runner/src/agentrunner/tools/edit.py` - Example of diffs usage

#### 3. **Standardized Error Codes** ⭐⭐⭐

**Why:** Better error handling and filtering in training pipelines

**What to adopt:**
```python
E_NOT_FOUND = "E_NOT_FOUND"
E_VALIDATION = "E_VALIDATION"
E_PERMISSIONS = "E_PERMISSIONS"
E_TIMEOUT = "E_TIMEOUT"
E_UNSAFE = "E_UNSAFE"
```

**Estimated effort:** 10 lines (add constants to dtypes.py)

**Files to reference:**
- `agent-runner/src/agentrunner/core/tool_protocol.py:6-14` - Error codes

### 🤔 **Medium Priority (Consider These)**

#### 4. **Session Trace Format** ⭐⭐⭐

**Why:** Useful for debugging and post-hoc analysis of training runs

**What to adopt:**
- JSONL trace format with interleaved messages + events
- TraceEvent dataclass
- SessionTrace container

**Estimated effort:** 100-150 lines (new file `rollouts/session_trace.py`)

**Files to reference:**
- `agent-runner/src/agentrunner/core/cli_session.py:17-100` - Session format

**Note:** This is more useful for interactive debugging than training pipelines.

### ⬇️ **Low Priority (Optional)**

#### 5. **ToolContext Pattern** ⭐⭐

**Why:** Richer context for tool execution (logger, event_bus, config)

**What to adopt:** Concept only - pass more context to Environment.execute_tool()

**Estimated effort:** Refactor existing code

**Decision:** Probably not worth it - rollouts' AgentState is sufficient

---

## Part 6: Implementation Plan

If you want to adopt the high-priority features:

### **Phase 1: Data Model Enhancements (Week 1)**

**Day 1-2: Enhanced ToolResult**
```python
# rollouts/dtypes.py

# Add error code constants
E_NOT_FOUND = "E_NOT_FOUND"
E_VALIDATION = "E_VALIDATION"
E_PERMISSIONS = "E_PERMISSIONS"
E_TIMEOUT = "E_TIMEOUT"
E_UNSAFE = "E_UNSAFE"

# Update ToolResult
@dataclass(frozen=True)
class ToolResult(JsonSerializable):
    ok: bool
    name: str
    content: str
    tool_call_id: str
    error_code: Optional[str] = None
    diffs: Optional[List[Dict[str, Any]]] = None
    files_changed: Optional[List[str]] = None
    data: Optional[Dict[str, Any]] = None
```

**Day 3-5: Update Environments**
- Update all environments to populate new ToolResult fields
- Add diffs to file edit tools
- Add files_changed tracking
- Use standardized error codes

**Tests:**
- Verify backward compatibility (optional fields)
- Test serialization/deserialization
- Test training pipeline still works

### **Phase 2: OpenAI Responses API (Week 2)**

**Day 1-2: Detection & Routing**
```python
# rollouts/providers.py

def _requires_responses_api(endpoint: Endpoint) -> bool:
    responses_models = ["o1", "o1-preview", "o1-mini", "gpt-5-codex", "gpt-5.1"]
    return any(m in endpoint.model.lower() for m in responses_models)

async def rollout_openai(actor: Actor, on_chunk: Callable) -> Actor:
    if _requires_responses_api(actor.endpoint):
        return await _rollout_openai_responses(actor, on_chunk)
    else:
        return await _rollout_openai_completions(actor, on_chunk)
```

**Day 3-4: Message Format Conversion**
```python
def _convert_to_responses_format(messages: List[Message]) -> Tuple[Optional[str], List[Dict]]:
    """Convert rollouts messages to Responses API format."""
    # Implementation from agent-runner
    pass

def _convert_tools_to_responses_format(tools: List[Tool]) -> List[Dict]:
    """Convert tool definitions to flat Responses API format."""
    pass
```

**Day 5: Non-Streaming Implementation**
```python
async def _rollout_openai_responses(actor: Actor, on_chunk: Callable) -> Actor:
    """Rollout using OpenAI Responses API."""
    # Implementation from agent-runner
    pass
```

**Week 3: Streaming Support**
```python
async def _rollout_openai_responses_stream(actor: Actor, on_chunk: Callable) -> Actor:
    """Stream using OpenAI Responses API."""
    # Implementation from agent-runner
    pass
```

**Tests:**
- Test O1 model execution
- Test GPT-5 Codex execution
- Test GPT-5.1 with reasoning effort
- Test tool calling with Responses API
- Test streaming

### **Phase 3: Optional Enhancements (If Needed)**

**Session Trace Format:**
- Only implement if you need debugging support
- Can skip for production training pipelines

---

## Part 7: Final Recommendation

### **Must Adopt:**

1. ✅ **OpenAI Responses API** - Critical for latest models (O1, Codex, GPT-5.1)
2. ✅ **Enhanced ToolResult** - Important for training analysis and metrics

### **Should Adopt:**

3. ✅ **Standardized Error Codes** - Low effort, high value for error handling

### **Consider Later:**

4. 🤔 **Session Trace Format** - Useful for debugging, not critical for training

### **Don't Adopt:**

5. ❌ Agent-runner's class-based tool system
6. ❌ Agent-runner's EventBus (you have better callbacks)
7. ❌ Agent-runner's stateful provider classes

---

## Conclusion

**Your instinct was correct** - rollouts is missing important data model features:

1. **Responses API support** - Blocking you from using O1/Codex/GPT-5.1
2. **ToolResult metadata** - Missing diffs/files_changed for training analysis
3. **Error codes** - Unstructured errors make filtering harder

The good news: These are **additive changes** that don't compromise rollouts' superior functional architecture. You can adopt the data model improvements while keeping rollouts' clean design.

**Estimated total effort:** 2-3 weeks for full implementation of high-priority features.

Would you like me to start implementing any of these features?
