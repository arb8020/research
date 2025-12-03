# Pi-AI Integration Plan for Rollouts

## Current Status (2025-12-03)

**Phase 0.3 Complete** ✅ - All 4 API types implemented and tested:
- ✅ `openai-completions` - OpenAI, Groq (tested)
- ✅ `openai-responses` - o1, o3, GPT-5 Codex (tested with GPT-5.1-Codex-Mini)
- ✅ `anthropic-messages` - Claude (tested)
- ✅ `google-generative-ai` - Gemini (implemented, ready for testing)

**Recent Fix**: GPT-5 Codex reasoning item persistence bug - models now correctly re-submit reasoning items in conversation history.

**Implementation Philosophy**: Port [pi-ai TypeScript](https://github.com/badlogic/pi-mono/tree/main/packages/ai/src/providers) to Python. Pi-ai is battle-tested and provides clean patterns for streaming, event handling, and provider abstraction.

---

## Next Phase: Cross-Provider Context Handoff (Phase 2)

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
