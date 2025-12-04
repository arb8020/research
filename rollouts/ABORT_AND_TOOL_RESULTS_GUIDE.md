# Abort Support & Structured Tool Results Implementation Guide

## Overview

This document guides implementation of two key features from pi-ai:
1. **Abort support** - Graceful cancellation throughout the pipeline
2. **Structured split tool results** - Dual output streams (LLM vs UI)

## Prerequisites Reading

**Code Style & Philosophy:**
1. `~/research/docs/code_style/FAVORITES.md` - Tiger Style, Push Ifs Up, assertions
2. `rollouts/config/README.md` - Rollouts design philosophy
3. `rollouts/PI_AI_INTEGRATION_PLAN.md` - Current migration status

**Pi-AI Reference Implementation:**
- Location: `/private/tmp/pi-mono/packages/ai/src/`
- Blog post: https://mariozechner.at/posts/2025-11-30-pi-coding-agent/#toc_3

---

## Feature 1: Abort Support

### What It Is

Graceful cancellation using Trio's cancellation primitives. When aborted:
- Kill long-running processes (bash commands, file operations)
- Stop LLM streaming mid-response
- Return partial results instead of discarding work
- Clean up resources (file handles, subprocesses)

### Reference Implementation

**Pi-AI Files:**
- `/private/tmp/pi-mono/packages/ai/src/agent/agent-loop.ts` lines 67-76 (abort handling)
- `/private/tmp/pi-mono/packages/coding-agent/src/tools/bash.ts` lines 172-185 (cleanup pattern)
- `/private/tmp/pi-mono/packages/coding-agent/src/tools/grep.ts` lines 64-70 (early abort check)

**Key Pattern from bash.ts:**
```typescript
const onAbort = () => {
    if (child.pid) {
        killProcessTree(child.pid);  // Cleanup!
    }
};

if (signal) {
    if (signal.aborted) {
        onAbort();  // Already aborted
    } else {
        signal.addEventListener("abort", onAbort, { once: true });
    }
}
```

### Implementation in Rollouts

#### 1. Add CancelScope Parameter to Tool Protocol

**File:** `rollouts/dtypes.py`

**Current:**
```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: ToolFunctionParameter
    # execute function signature TBD
```

**Change to:**
```python
from typing import Protocol
import trio

class ToolExecutor(Protocol):
    async def __call__(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        ...

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: ToolFunctionParameter
    execute: ToolExecutor
```

#### 2. Update Tool Execution in Agent Loop

**File:** `rollouts/agents.py`

**Search for:** Tool execution code (likely around line 200-300)

**Add abort checks:**
```python
async def execute_tool(
    tool: Tool,
    tool_call: ToolCall,
    cancel_scope: trio.CancelScope | None = None,
) -> ToolResult:
    """Execute tool with cancellation support."""
    assert tool is not None
    assert tool_call is not None

    # Early abort check
    if cancel_scope and cancel_scope.cancelled_caught:
        raise trio.Cancelled("Tool execution aborted before start")

    try:
        result = await tool.execute(
            tool_call.id,
            tool_call.args,
            cancel_scope=cancel_scope,
        )
        return result
    except trio.Cancelled:
        # Return partial result if available
        return ToolResult(
            tool_call_id=tool_call.id,
            content="Tool execution cancelled",
            is_error=True,
        )
```

#### 3. Thread CancelScope Through run_agent

**File:** `rollouts/agents.py`

**Find:** `async def run_agent(...)` function signature

**Add parameter:**
```python
async def run_agent(
    state: AgentState,
    run_config: RunConfig,
    cancel_scope: trio.CancelScope | None = None,  # Add this
) -> AgentState:
    """Run agent with cancellation support."""
    ...
    # Pass cancel_scope to tool execution
    for tool_call in tool_calls:
        result = await execute_tool(tool, tool_call, cancel_scope=cancel_scope)
    ...
```

#### 4. Add StopReason for Aborts

**File:** `rollouts/dtypes.py`

**Current:**
```python
@dataclass(frozen=True)
class StopReason:
    reason: str  # "stop", "length", "tool_use", "error"
```

**Add:**
```python
@dataclass(frozen=True)
class StopReason:
    reason: Literal["stop", "length", "tool_use", "error", "aborted"]  # Add "aborted"
```

#### 5. Example Tool Implementation

**Create:** `rollouts/environments/tools/bash_tool.py`

**Reference:** `/private/tmp/pi-mono/packages/coding-agent/src/tools/bash.ts` (full file)

**Pattern:**
```python
import trio
import subprocess
import signal

async def bash_execute(
    tool_call_id: str,
    args: dict[str, Any],
    cancel_scope: trio.CancelScope | None = None,
) -> ToolResult:
    """Execute bash command with cancellation support."""
    assert "command" in args
    command = args["command"]
    timeout = args.get("timeout")

    # Early abort check
    if cancel_scope and cancel_scope.cancelled_caught:
        raise trio.Cancelled("Bash execution aborted")

    proc = await trio.lowlevel.open_process(
        ["bash", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    async def kill_on_cancel():
        """Kill process when cancelled."""
        await trio.sleep_forever()  # Wait for cancellation

    try:
        async with trio.open_nursery() as nursery:
            if cancel_scope:
                nursery.start_soon(kill_on_cancel)

            # Set timeout if provided
            if timeout:
                with trio.move_on_after(timeout) as timeout_scope:
                    stdout, stderr = await proc.communicate()
            else:
                stdout, stderr = await proc.communicate()

            if timeout and timeout_scope.cancelled_caught:
                proc.kill()
                raise ToolExecutionError(f"Command timed out after {timeout}s")

            output = stdout.decode() + stderr.decode()
            return ToolResult(
                tool_call_id=tool_call_id,
                content=output or "(no output)",
                is_error=proc.returncode != 0,
            )
    except trio.Cancelled:
        # Kill process tree on abort
        proc.kill()
        await proc.wait()
        raise  # Re-raise to propagate cancellation
```

---

## Feature 2: Structured Split Tool Results

### What It Is

Tools return **two separate outputs**:
1. **`content`** - Concise text/images for LLM context (token-efficient)
2. **`details`** - Rich structured data for UI display (JSON, not parsed)

### Reference Implementation

**Pi-AI Files:**
- `/private/tmp/pi-mono/packages/ai/src/agent/types.ts` lines 14-19 (AgentToolResult)
- `/private/tmp/pi-mono/packages/ai/src/types.ts` lines 114-122 (ToolResultMessage)
- `/private/tmp/pi-mono/packages/ai/src/agent/agent-loop.ts` lines 117-123 (stripping details before LLM)

**Key Pattern from agent-loop.ts:**
```typescript
// Strip 'details' before sending to LLM
messages: [...processedMessages].map((m) => {
    if (m.role === "toolResult") {
        const { details, ...rest } = m;  // Remove details!
        return rest;
    } else {
        return m;
    }
}),
```

### Implementation in Rollouts

#### 1. Update ToolResult Dataclass

**File:** `rollouts/dtypes.py`

**Current:**
```python
@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str
```

**Change to:**
```python
@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    # What LLM sees (token-efficient summary)
    content: str | list[ContentBlock]
    # What UI sees (rich structured data)
    details: dict[str, Any] | None = None
    is_error: bool = False
```

#### 2. Strip Details Before Sending to LLM

**File:** `rollouts/providers.py`

**Find:** Message conversion functions:
- `_message_to_openai()` (~line 348)
- `_message_to_anthropic()` (~line 750)
- `_messages_to_openai_responses()` (~line 1200)

**Add filtering:**
```python
def _prepare_messages_for_llm(messages: list[Message]) -> list[Message]:
    """Strip tool result details before sending to LLM.

    Tiger Style: Explicit filtering, no magic.
    """
    assert messages is not None
    assert isinstance(messages, list)

    filtered = []
    for msg in messages:
        if msg.role == "tool":
            # Strip details field - UI-only data
            # Keep only content for LLM context
            filtered_msg = replace(msg, details=None)  # Remove details
            filtered.append(filtered_msg)
        else:
            filtered.append(msg)

    assert len(filtered) == len(messages)  # No messages dropped
    return filtered
```

**Then call before provider conversion:**
```python
async def rollout_openai(...):
    ...
    # Strip details before converting to OpenAI format
    llm_messages = _prepare_messages_for_llm(state.actor.trajectory.messages)
    openai_messages = [_message_to_openai(m) for m in llm_messages]
    ...
```

#### 3. Update Message Type for Tool Results

**File:** `rollouts/dtypes.py`

**Add details field:**
```python
@dataclass(frozen=True)
class Message:
    role: str  # "user", "assistant", "tool"
    content: str | list[ContentBlock] | None
    provider: str | None = None
    api: str | None = None
    model: str | None = None
    tool_call_id: str | None = None  # For tool role only

    # New: UI-only structured data (stripped before LLM)
    details: dict[str, Any] | None = None
```

#### 4. Example Tool with Split Results

**Create:** `rollouts/environments/tools/grep_tool.py`

**Reference:** `/private/tmp/pi-mono/packages/coding-agent/src/tools/grep.ts`

**Pattern:**
```python
async def grep_execute(
    tool_call_id: str,
    args: dict[str, Any],
    cancel_scope: trio.CancelScope | None = None,
) -> ToolResult:
    """Search files with split LLM/UI results."""
    pattern = args["pattern"]
    path = args.get("path", ".")

    # Execute ripgrep
    matches = await run_ripgrep(pattern, path, cancel_scope)

    # LLM gets concise summary
    llm_content = f"Found {len(matches)} matches for '{pattern}' in {path}"

    # UI gets rich structured data
    ui_details = {
        "pattern": pattern,
        "path": path,
        "total_matches": len(matches),
        "matches": [
            {
                "file": m.file,
                "line_number": m.line_num,
                "line_content": m.line,
                "context_before": m.context_before,
                "context_after": m.context_after,
            }
            for m in matches[:100]  # Limit for UI
        ],
        "truncated": len(matches) > 100,
    }

    return ToolResult(
        tool_call_id=tool_call_id,
        content=llm_content,  # Token-efficient
        details=ui_details,   # Rich data for UI
        is_error=False,
    )
```

**Another Example - Edit Tool:**
```python
async def edit_execute(...) -> ToolResult:
    # Apply file edit
    diff = apply_edit(file_path, old_string, new_string)

    # LLM sees summary
    llm_content = f"Edited {file_path}: replaced {len(old_string)} chars with {len(new_string)} chars"

    # UI sees full diff with metadata
    ui_details = {
        "file_path": file_path,
        "diff": diff,  # Full unified diff
        "old_length": len(old_string),
        "new_length": len(new_string),
        "lines_changed": count_changed_lines(diff),
    }

    return ToolResult(
        tool_call_id=tool_call_id,
        content=llm_content,
        details=ui_details,
    )
```

---

## Integration Testing

### Test Abort Flow

**Create:** `rollouts/tests/test_abort_support.py`

```python
import trio
import pytest

async def test_tool_abort_cleanup():
    """Test that aborted tools clean up resources."""
    async with trio.open_nursery() as nursery:
        cancel_scope = nursery.cancel_scope

        # Start long-running tool
        async def run_tool():
            result = await bash_execute(
                "tool_1",
                {"command": "sleep 100"},
                cancel_scope=cancel_scope,
            )

        nursery.start_soon(run_tool)

        # Cancel after 100ms
        await trio.sleep(0.1)
        cancel_scope.cancel()

    # Verify process was killed (not hanging)
    # Verify partial result returned
```

### Test Split Results

**Create:** `rollouts/tests/test_split_tool_results.py`

```python
async def test_details_stripped_from_llm():
    """Test that UI details are stripped before LLM sees messages."""
    # Create tool result with details
    tool_result = Message(
        role="tool",
        content="Found 5 matches",
        tool_call_id="call_1",
        details={
            "matches": [...],  # Rich UI data
            "total": 5,
        }
    )

    messages = [tool_result]

    # Filter for LLM
    llm_messages = _prepare_messages_for_llm(messages)

    # Verify details stripped
    assert llm_messages[0].details is None
    assert llm_messages[0].content == "Found 5 matches"
```

---

## Success Criteria

### Aborts
- ✅ Long-running bash commands can be cancelled mid-execution
- ✅ Process trees are killed (no zombie processes)
- ✅ Partial results returned when possible
- ✅ Agent loop stops gracefully on abort
- ✅ `StopReason.aborted` set correctly

### Split Tool Results
- ✅ Tools return both `content` and `details`
- ✅ Details stripped before LLM API calls
- ✅ UI/frontend can access rich `details` data
- ✅ Token usage reduced (LLM sees summaries, not full data)
- ✅ All existing tests pass with new ToolResult structure

---

## Migration Strategy

**Phase 1: Abort Support (2-3 days)**
1. Add `cancel_scope` parameter to tool protocol
2. Update `run_agent()` to accept and propagate cancel_scope
3. Implement bash tool with abort support (reference implementation)
4. Add abort handling to agent loop
5. Update StopReason enum
6. Write integration tests

**Phase 2: Split Tool Results (2-3 days)**
1. Update `ToolResult` dataclass with `details` field
2. Add `details` to `Message` dataclass
3. Implement `_prepare_messages_for_llm()` filtering
4. Update all provider conversion functions to use filtering
5. Port 2-3 example tools (bash, grep, edit) with split results
6. Write tests for detail stripping
7. Update frontend to consume `details` field

**Phase 3: Tool Library (1 week)**
- Port remaining pi-ai tools with abort + split results
- Document tool implementation patterns
- Create tool template/scaffold

**Total Estimate:** 2-3 weeks for full implementation

---

## Code Style Compliance

✅ **Tiger Style**
- Explicit control flow (if/else for abort checks, no hidden magic)
- Assertions everywhere (2+ per function)
- Crash loud (raise on invariant violations)

✅ **Push Ifs Up**
- Abort checks at function entry
- Early returns on cancellation

✅ **Full State Passing**
- CancelScope passed explicitly (no global state)
- ToolResult immutable (frozen dataclass)

✅ **Semantic Compression**
- Extracted from pi-ai (battle-tested in production)
- Only abstracting after seeing pattern in 6+ tools

---

## Questions?

- **Why Trio over asyncio?** Rollouts already uses Trio. Structured concurrency makes cancellation cleaner.
- **Why not use context managers for cleanup?** Trio's cancel scopes ARE context managers - this is idiomatic.
- **Do all tools need split results?** No. Simple tools can return `details=None`. Use when UI needs rich data.
- **What if a tool doesn't support abort?** Mark as non-cancellable in tool metadata. User gets warning.

---

## References

- **Pi-AI repo**: https://github.com/badlogic/pi-mono
- **Mario's blog**: https://mariozechner.at/posts/2025-11-30-pi-coding-agent/#toc_3
- **Trio docs**: https://trio.readthedocs.io/en/stable/reference-core.html#cancellation-and-timeouts
- **Current rollouts**: `rollouts/dtypes.py`, `rollouts/agents.py`, `rollouts/providers.py`
