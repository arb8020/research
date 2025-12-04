# TUI Interactive Agent Bugs

## Replication

```bash
python -m rollouts.frontends.tui.cli --model claude-sonnet-4-5 --provider anthropic
```

Type "hi", press Enter. Claude responds "Hello! How can I help you today?" then immediately crashes with:

```
❌ Empty message content detected! Role: assistant
Message object: Message(role='assistant', content='[0 blocks: ]')
```

## Debug Output

Added print statements in `aggregate_anthropic_stream` show:

```
🔍 DEBUG: Building final message from 1 content blocks   # First call - works
  Block 0: type=text, accumulated='Hello! How can I help you today?'

🔍 DEBUG: Building final message from 0 content blocks   # Second call - empty!
  accumulated_content=''
  thinking_content=''
  tool_calls=0
```

---

## Issue 1: Control Flow - Agent Doesn't Wait for User Input

**Location:** `agents.py:508-516`

```python
# If no tools, we're done with this turn
if not tool_calls:
    current_state = await rcfg.handle_no_tool(current_state, rcfg)
    # Check if handler added a stop reason
    if current_state.stop:
        return current_state
    # Otherwise increment turn and continue  <-- BUG: continues without user input!
    return replace(current_state,
                  turn_idx=current_state.turn_idx + 1,
                  pending_tool_calls=[])
```

**Problem:** When LLM responds without tool calls, the default `handle_no_tool` does nothing (`dtypes.py:896-898`). The agent loop then immediately starts another turn without waiting for user input.

**Expected behavior for interactive chat:** After LLM responds (no tools), wait for user to type next message before continuing.

**Fix:** Implement a custom `handle_no_tool` callback that:
1. Calls `on_input` to wait for user message
2. Appends user message to trajectory
3. Returns updated state

---

## Issue 2: Empty Message Not Caught Early (Graceful Failure)

**Location:** `providers.py:2306-2341` in `aggregate_anthropic_stream`

```python
final_content_blocks: list = []

# Reconstruct content blocks in order from the content_blocks tracking dict
for index in sorted(content_blocks.keys()):
    # ... builds final_content_blocks ...

final_message = Message(
    role="assistant",
    content=final_content_blocks,  # Can be empty list!
)
```

**Problem:** If `content_blocks` dict is empty (no streaming events received), `final_content_blocks` will be an empty list. This creates an invalid `Message(content=[])` that later causes an assertion failure in `_message_to_anthropic` (line 2058).

**Why this happens:** On the second (erroneous) API call, something prevents streaming events from populating `content_blocks`. The function returns an empty message instead of failing gracefully.

**Expected behavior:** `aggregate_anthropic_stream` should validate its output and raise a clear error if the message is empty, rather than returning invalid data.

**Fix:** Add validation before returning:
```python
if not final_content_blocks:
    raise ValueError(
        "aggregate_anthropic_stream produced empty message. "
        "No content blocks were received from the stream."
    )
```

---

## Root Cause Chain

1. User sends "hi"
2. `run_agent` calls `run_agent_step` → `rollout_anthropic` → Claude responds ✓
3. No tool calls, so `handle_no_tool` is called (does nothing)
4. `run_agent_step` returns, `run_agent` loop continues (no `stop` set)
5. `run_agent` calls `run_agent_step` again immediately
6. `rollout_anthropic` is called again, but now the trajectory has assistant message at end
7. API call happens, but something goes wrong - no streaming events received
8. `aggregate_anthropic_stream` returns empty message
9. Loop continues, tries to prepare messages for THIRD call
10. `_message_to_anthropic` sees empty assistant message → assertion fails

## Priority

**Fix Issue 2 first** - Graceful failure will give better error messages and prevent cascade. ✅ FIXED in commit 32ea4fc

**Fix Issue 1 second** - Proper control flow for interactive chat. ✅ FIXED - Added `handle_no_tool_interactive` callback that waits for user input before continuing the agent loop.
