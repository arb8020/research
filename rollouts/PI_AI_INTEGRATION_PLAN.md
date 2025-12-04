# Pi-AI Integration Plan for Rollouts

## Quick Start for New Developer

**Read these first**:
1. `~/research/docs/code_style/FAVORITES.md` - Core coding principles (Tiger Style, Push Ifs Up, etc.)
2. `~/research/docs/code_style/grugbrain_testing.md` - Testing philosophy (integration tests are sweet spot)
3. This document's "Code Style & Design Principles" section below

**Current task**: Complete ContentBlock migration by fixing legacy field usage.

**Context**: We removed `tool_calls`, `thinking_content`, `reasoning_content` from Message class to force proper ContentBlock usage. Several files need updates (see "Immediate Next Steps" below).

---

## Current Status (2025-12-03 Evening)

**Phase 2 In Progress** 🚧 - Cross-Provider Context Handoff & Legacy Cleanup:

### ✅ Completed
- ✅ ContentBlock types defined (TextContent, ThinkingContent, ToolCallContent, ImageContent)
- ✅ `transform_messages()` function created with two-pass algorithm (supports both calling conventions)
- ✅ Provider metadata fields added to Message (provider/api/model)
- ✅ Endpoint validation for Claude thinking budget (>= 1024 tokens)
- ✅ Integration test suite created (`tests/test_provider_switching.py`)
- ✅ **DECISION**: Keep unified Message type (not split like pi-ai)
- ✅ **DECISION**: Remove legacy fields immediately (force proper migration)
- ✅ Message.get_tool_calls() helper method added

### 🚧 In Progress - Legacy Field Removal
**Current blocker**: Removed `tool_calls`, `thinking_content`, `reasoning_content` from Message class.

**Breaking changes** - These files need updates to use ContentBlocks:

**High Priority** (blocks all tests):
1. `rollouts/providers.py`:
   - Line 331: `if not m.content and not (hasattr(m, 'tool_calls') and m.tool_calls)`
   - Line 348-360: `_message_to_openai()` - reads `m.tool_calls`, needs to read ContentBlocks
   - Line 425-426: Streaming aggregation - reads `c.message.tool_calls`
   - Line 512-513: Delta processing - reads `delta.tool_calls`

2. `rollouts/agents.py`:
   - ✅ Line 276: Fixed to use `get_tool_calls()`
   - ✅ Line 438: Fixed to use `get_tool_calls()`

3. `rollouts/environments/binary_search.py`:
   - Line 179: Counts tool calls - needs `get_tool_calls()`

**Medium Priority** (tests will run but may fail):
4. OpenAI provider needs to **create** ContentBlocks (not just read):
   - Currently creates Messages with `tool_calls=[]` field
   - Needs to create ToolCallContent blocks in `content`

5. `_messages_to_openai_responses()` - partially fixed, needs completion:
   - ✅ User messages: handles string content
   - ✅ Assistant messages: handles string content
   - ❌ Tool messages: Line 1241 assumes `msg.content` is list - needs string handling

### ⏳ Not Started
- OpenAI Completions API migration to ContentBlocks (output)
- Google Gemini provider ContentBlock support check
- Integration tests (blocked by breaking changes)

---

## Immediate Next Steps (For Next Developer)

### Step 1: Fix `_message_to_openai()` to Read ContentBlocks
**File**: `rollouts/providers.py` lines 321-368

**Current code**:
```python
def _message_to_openai(m: Message) -> ChatCompletionMessageParam:
    # Line 348: Reads m.tool_calls (doesn't exist anymore)
    if m.tool_calls and m.role == "assistant":
        msg["tool_calls"] = [...]
```

**Fix needed**:
```python
def _message_to_openai(m: Message) -> ChatCompletionMessageParam:
    # Use helper method
    tool_calls = m.get_tool_calls()
    if tool_calls and m.role == "assistant":
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
            }
            for tc in tool_calls
        ]
```

**Also fix line 331**:
```python
# OLD: if not m.content and not (hasattr(m, 'tool_calls') and m.tool_calls)
# NEW:
if not m.content and not m.get_tool_calls() and m.role != "tool":
```

### Step 2: Fix OpenAI Streaming to Create ContentBlocks
**File**: `rollouts/providers.py` lines 425-426, 512-513

**Problem**: When building Messages from OpenAI streaming responses, code creates `tool_calls=[]` field.

**Fix**: Create ToolCallContent blocks instead:
```python
# In streaming aggregation (around line 512-540):
from .dtypes import ToolCallContent

# OLD: Build tool_calls list
# NEW: Build content blocks list
final_content_blocks = []
if text_content:
    final_content_blocks.append(TextContent(text=text_content))

for tc in tool_calls_list:  # from delta.tool_calls aggregation
    final_content_blocks.append(ToolCallContent(
        id=tc.id,
        name=tc.function.name,
        arguments=json.loads(tc.function.arguments)
    ))

final_message = Message(role="assistant", content=final_content_blocks)
```

### Step 3: Fix `_messages_to_openai_responses()` Tool Handling
**File**: `rollouts/providers.py` line 1239-1250

**Current code** (line 1241):
```python
elif msg.role == "tool":
    text_blocks = [b for b in msg.content if isinstance(b, TextContent)]  # Crashes if string!
```

**Fix needed**:
```python
elif msg.role == "tool":
    # Handle string content (simple tool result)
    if isinstance(msg.content, str):
        tool_result_text = msg.content
    # Handle ContentBlock list
    elif isinstance(msg.content, list):
        text_blocks = [b for b in msg.content if isinstance(b, TextContent)]
        tool_result_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""
    else:
        tool_result_text = ""
```

### Step 4: Fix Environment Tool Call Counter
**File**: `rollouts/environments/binary_search.py` line 179

**Change**:
```python
# OLD: tool_calls = sum(len(msg.tool_calls) for msg in final_state.actor.trajectory.messages)
# NEW:
tool_calls = sum(len(msg.get_tool_calls()) for msg in final_state.actor.trajectory.messages)
```

### Step 5: Run Tests
```bash
cd /Users/chiraagbalu/research/rollouts
source .venv/bin/activate
PYTHONPATH=/Users/chiraagbalu/research/rollouts python tests/test_provider_switching.py
```

**Expected**: Tests should pass once all fixes complete.

---

**Phase 0.3 Complete** ✅ - All 4 API types implemented and tested:
- ✅ `openai-completions` - OpenAI, Groq (tested)
- ✅ `openai-responses` - o1, o3, GPT-5 Codex (tested with GPT-5.1-Codex-Mini)
- ✅ `anthropic-messages` - Claude (tested)
- ✅ `google-generative-ai` - Gemini (implemented, ready for testing)

**Recent Updates**:
- Recovered from beads git mishap (lesson: commit early and often!)
- Re-applied all provider updates from session history
- All providers now build messages using ContentBlocks (TextContent, ThinkingContent, ToolCallContent, ImageContent)

**Implementation Philosophy**: Port [pi-ai TypeScript](https://github.com/badlogic/pi-mono/tree/main/packages/ai/src/providers) to Python. Pi-ai is battle-tested and provides clean patterns for streaming, event handling, and provider abstraction.

---

## Code Style & Design Principles

Following **Tiger Style** and patterns from `~/research/docs/code_style/FAVORITES.md`:

### 1. Explicit Control Flow (Push Ifs Up, Fors Down)
All provider conversion functions (`_message_to_anthropic`, `_messages_to_openai_responses`, etc.) **must** handle both content types explicitly:

```python
# GOOD - Explicit branching at top of function
if isinstance(msg.content, str):
    # Handle string content
    text = msg.content
elif isinstance(msg.content, list):
    # Handle ContentBlock list
    text = extract_text(msg.content)
else:
    assert False, f"Invalid content type: {type(msg.content)}"
```

**Why**: Makes invisible assumptions visible. No mental simulation required.

### 2. Assertions Everywhere (Crash Loud)
- Minimum 2 assertions per function
- Split compound assertions: `assert a; assert b;` not `assert a and b`
- Document invariants: `assert embedding_dim % num_heads == 0  # Must be evenly divisible`

```python
# GOOD - Clear failure messages
assert isinstance(msg.content, (str, list)), f"content must be str or list[ContentBlock], got {type(msg.content)}"
```

### 3. Unified Message Type (Design Decision)
**Decision**: Keep single `Message` class, not split into UserMessage/AssistantMessage/ToolResultMessage like pi-ai.

**Rationale**:
- Python doesn't benefit from TypeScript's compile-time type safety
- Simpler to work with (no type juggling)
- Lower coupling (not forced to share contracts)
- Runtime assertions catch errors during tests

**Message structure**:
```python
@dataclass(frozen=True)
class Message:
    role: str  # "user", "assistant", "tool"
    content: str | list[ContentBlock] | None
    provider: str | None
    api: str | None
    model: str | None
    tool_call_id: str | None  # For tool role only
```

**No legacy fields** - Removed `tool_calls`, `thinking_content`, `reasoning_content`. Use ContentBlocks only.

### 4. Support Both String and ContentBlock Content
**Pi-AI design**: UserMessage content is `string | (TextContent | ImageContent)[]`

Our providers must handle **both**:
- String: Simple text messages (most common case)
- ContentBlock list: Structured messages with thinking/tools/images

**Never assume** content is always a list. This is the #1 bug we're fixing.

### 5. Continuous Granularity
Provide both high-level and low-level functions:
- High-level: `transform_messages(msgs, target_provider="openai", target_api="...")`
- Low-level: `transform_messages(msgs, from_provider="anthropic", from_api="...", to_provider="openai", to_api="...")`

Each level uses the lower level. Don't delete low-level functions.

### 6. Migration Strategy
**Phase 1** (Current): Provider input functions handle BOTH formats
- ✅ `_message_to_anthropic` handles string and ContentBlocks
- 🚧 `_messages_to_openai_responses` handles string and ContentBlocks
- ⏳ `_message_to_openai` needs ContentBlock support

**Phase 2** (After 2+ providers migrated): Evaluate whether to:
- Keep unified Message type OR
- Split into UserMessage/AssistantMessage/ToolResultMessage (like pi-ai)

**Decision deferred** until we have 2+ examples to compress (Semantic Compression principle).

---

## Next Steps: Testing & Validation (Phase 2 Completion)

**Goal**: Enable switching models mid-conversation with automatic message transformation.

### Why This Matters
- Cost optimization (start with Claude, switch to GPT-4o for complex tasks)
- Fallback strategies (retry with different provider on failure)
- Multi-model workflows (use o1 for reasoning, GPT-4o for execution)

### Key Challenges

#### 1. Message Format Transformation
Different providers have incompatible message formats:

**Anthropic thinking**:
```python
Message(
    role="assistant",
    content=[
        {"type": "thinking", "thinking": "Let me analyze..."},
        {"type": "text", "text": "The answer is 42"}
    ]
)
```

**OpenAI reasoning** (o1/o3):
```python
Message(
    role="assistant",
    reasoning_content="Let me analyze...",  # Separate field
    content="The answer is 42"
)
```

**OpenAI Responses** (GPT-5 Codex):
```python
Message(
    role="assistant",
    thinking_signature='{"type": "reasoning", "id": "rs_...", "summary": [...]}',  # JSON blob
    content="The answer is 42"
)
```

**Need**: Transform between these formats when switching providers.

#### 2. Tool Call Format Differences

**OpenAI Completions**:
- Tool calls on message: `tool_calls=[{id, name, args}]`
- Tool results: separate messages with `role="tool"`

**OpenAI Responses**:
- Tool calls: separate `function_call` objects in conversation
- Must include associated `reasoning` item before each `function_call`

**Anthropic**:
- Tool calls embedded in content blocks
- Tool results in user message with `tool_result` content

**Need**: Bidirectional conversion between these formats.

#### 3. Provider-Specific Metadata

Some data only exists for certain providers:
- `thinking_signature` (GPT-5 Codex reasoning items)
- `reasoning_content` (o1/o3 reasoning)
- Vision content (multimodal messages)

**Need**: Preserve or safely discard metadata when switching.

### Implementation Plan

#### Task 1: Define Message Transformation Protocol (1 day)

Create `message_transform.py` with:

```python
def transform_messages(
    messages: list[Message],
    from_provider: str,
    to_provider: str,
) -> list[Message]:
    """Transform messages from one provider format to another.

    Handles:
    - Thinking/reasoning content conversion
    - Tool call format transformation
    - Provider-specific metadata preservation/cleanup
    """
```

**Reference**: [pi-ai message transformation](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers)

**Key transformations**:
1. **Anthropic → OpenAI**: Extract `thinking` blocks, convert to `reasoning_content`
2. **OpenAI → Anthropic**: Convert `reasoning_content` to `thinking` content blocks
3. **GPT-5 Codex ↔ Other**: Handle `thinking_signature` JSON blobs
4. **Tool calls**: Normalize all formats to internal representation, then convert to target

#### Task 2: Implement Core Transformations (2 days)

**Priority order**:
1. Thinking/reasoning content (most common)
2. Tool call formats
3. Metadata cleanup

**Test strategy**:
- Unit tests for each transformation direction
- Integration test: OpenAI → Anthropic → OpenAI roundtrip
- Real API test: Start conversation with Claude, continue with GPT-4o

#### Task 3: Add Switch Helpers (1 day)

```python
async def continue_with_provider(
    state: AgentState,
    new_endpoint: Endpoint,
) -> AgentState:
    """Continue conversation with different provider.

    Transforms message history automatically.
    """
```

**Use cases**:
- Fallback on rate limit/error
- Cost optimization (cheap model → expensive model for hard tasks)
- Capability routing (vision → text-only)

#### Success Criteria
- ✅ Messages transform correctly between all 4 API types
- ✅ Tool calls work across provider switches
- ✅ Real API test: Multi-turn conversation switching providers works
- ✅ Thinking/reasoning content preserved or safely converted

**Estimated effort**: 4-5 days

---

## Phase 1: Additional Providers (Optional - Easy Wins)

**Low priority** - Can add these quickly if needed:

### Cerebras (1 hour)
- Add to `models.py` registry
- Maps to `openai-completions`
- No code changes needed

### xAI (1 hour)
- Add Grok models to registry
- Maps to `openai-completions`
- No code changes needed

### OpenRouter (1 hour)
- Multi-model proxy
- Maps to `openai-completions`
- No code changes needed

---

## Completed Phases (Summary)

**Phase 0.1** - Granular streaming events ✅
- 13 event types (start/delta/end lifecycle)
- Progressive tool argument parsing
- Updated all providers

**Phase 0.2** - Unified provider API ✅
- Model registry system
- API type abstraction (4 types)
- Provider selection via registry
- Simplified test suite (3 end-to-end tests)

**Phase 0.3** - Remaining API types ✅
- OpenAI Responses API (o1/o3/GPT-5 Codex)
- Google Generative AI (Gemini)
- Fixed GPT-5 Codex reasoning bug

---

## Reference

- **Pi-AI repo**: https://github.com/badlogic/pi-mono/tree/main/packages/ai
- **Provider implementations**: https://github.com/badlogic/pi-mono/tree/main/packages/ai/src/providers
- **Key files**:
  - `rollouts/providers.py` - All provider implementations
  - `rollouts/models.py` - Model registry
  - `rollouts/dtypes.py` - Message and event types
  - `tests/test_real_agent.py` - End-to-end tests
