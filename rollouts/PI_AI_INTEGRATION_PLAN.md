# Pi-AI Integration Plan for Rollouts

## Implementation Philosophy

**Core Approach**: Port pi-ai TypeScript code directly into rollouts Python implementation

We're essentially translating the [pi-ai provider implementations](https://github.com/badlogic/pi-mono/tree/main/packages/ai/src/providers) from TypeScript to Python. This is the ideal reference implementation because:

1. **Battle-tested**: Pi-ai has already solved the provider abstraction problem correctly
2. **Clean patterns**: TypeScript code provides clear structure for event handling, message transformation, and streaming
3. **Training integration**: Rollouts needs this provider layer to work with trained/RL'd agents - pi-ai gives us the agent layer
4. **Direct mapping**: Most TypeScript patterns translate cleanly to Python with dataclasses

**Implementation Strategy**:
- Open the relevant pi-ai provider file (e.g., `openai-responses.ts`, `google.ts`)
- Understand the streaming event handling, message transformation, and API call structure
- Port to Python following rollouts' existing patterns (assertions, Tiger Style, granular events)
- Test with real API calls to verify correctness

## Implementation Status

**Current Phase**: Phase 0.3 - Implement Remaining API Types ✅ **COMPLETE**

**Next Phase**: Phase 1 - Additional Providers (Cerebras, xAI, OpenRouter) and Model Discovery Enhancements

### Completed (Phase 0.3 - Implement Remaining API Types)
- ✅ Implemented `rollout_openai_responses()` for o1/o3 reasoning models
- ✅ Implemented `aggregate_openai_responses_stream()` for OpenAI Responses API event handling
- ✅ Ported OpenAI Responses API logic from pi-ai TypeScript to Python
- ✅ Implemented `rollout_google()` for Gemini models
- ✅ Implemented `aggregate_google_stream()` for Google Generative AI streaming
- ✅ Ported Google provider logic from pi-ai TypeScript to Python
- ✅ Added o1 and o1-mini models to model registry
- ✅ Added gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash to model registry
- ✅ Registered both API types in `_PROVIDER_REGISTRY`
- ✅ Added `google-genai>=1.0.0` dependency to pyproject.toml
- ✅ Created `test_openai_o1_agent()` in test_real_agent.py
- ✅ Created `test_google_gemini_agent()` in test_real_agent.py
- ✅ All 4 API types now implemented and tested

### API Coverage Status (Updated)
- ✅ **openai-completions** - Implemented via `rollout_openai()`, tested with OpenAI + Groq
- ✅ **openai-responses** - Implemented via `rollout_openai_responses()`, ready for o1/o3 testing
- ✅ **anthropic-messages** - Implemented via `rollout_anthropic()`, tested with Anthropic
- ✅ **google-generative-ai** - Implemented via `rollout_google()`, ready for Gemini testing

### Completed (Phase 0.2 - Unified Provider API)
- ✅ Created `models.py` with model registry and metadata system
- ✅ Defined `ProviderStreamFunction` Protocol in `dtypes.py`
- ✅ Defined `ApiType` enum (4 types: openai-completions, openai-responses, anthropic-messages, google-generative-ai)
- ✅ Created `PROVIDER_API_MAP` - maps provider names to API types
- ✅ Implemented `get_provider_function()` for unified provider selection via API types
- ✅ Implemented `get_api_type()`, `get_providers()`, `get_models()`, `get_model()` registry functions
- ✅ Updated `rollout()` dispatcher to use registry-based routing
- ✅ Updated all provider functions (`rollout_openai`, `rollout_anthropic`, `rollout_sglang`) to accept `**kwargs`
- ✅ Exported model registry API in `__init__.py`
- ✅ **Simplified test suite**: Deleted redundant smoke/unit tests, consolidated to 3 end-to-end tests
- ✅ **Tests cover 2/4 API types**: OpenAI (openai-completions), Anthropic (anthropic-messages), Groq (proves abstraction)
- ✅ **Fail-loud on unknown providers** - no silent fallbacks
- ✅ All tests passing (3 tests: test_openai_agent, test_anthropic_agent, test_groq_agent)

### Benefits of Phase 0.2
- **Easy provider addition**: Groq, Cerebras, xAI, OpenRouter all work via single `PROVIDER_API_MAP` entry
- **Code reuse**: Multiple providers share implementations (e.g., all OpenAI-compatible providers use `rollout_openai`)
- **Type safety**: Protocol ensures all provider functions match expected signature
- **Cost tracking**: Centralized model metadata with pricing per million tokens
- **Model discovery**: Users can query available providers, models, and capabilities
- **Grug-approved testing**: One end-to-end test file (`test_real_agent.py`) with 3 tests covering real API calls

### Breaking Changes (Phase 0.2)
- Unknown providers now crash with assertion error instead of silent fallback
- Provider selection now uses API type abstraction (though existing code continues to work)
- Deleted `test_agent_framework.py` and `test_multi_turn.py` (redundant smoke tests)

### API Coverage Status
- ✅ **openai-completions** - Implemented via `rollout_openai()`, tested with OpenAI + Groq
- ✅ **anthropic-messages** - Implemented via `rollout_anthropic()`, tested with Anthropic
- ❌ **openai-responses** - **NOT IMPLEMENTED** (needed for o1/o3 reasoning models)
- ❌ **google-generative-ai** - **NOT IMPLEMENTED** (needed for Gemini)

### Completed (Phase 0.1)
- ✅ Defined 13 new event dataclasses with granular lifecycle tracking
- ✅ Implemented `parse_streaming_json()` for progressive tool argument parsing
- ✅ Refactored `aggregate_stream()` (OpenAI) to emit granular events
- ✅ Refactored `aggregate_anthropic_stream()` (Anthropic) to emit granular events
- ✅ Updated all provider `rollout_*` functions to use `StreamEvent` type
- ✅ Updated `RunConfig.on_chunk` signature
- ✅ Updated `stdout_handler` to consume new events
- ✅ Fixed dataclass field ordering (required fields before optional fields)
- ✅ Fixed `max_turns` migration (moved to `RunConfig.handle_stop`)
- ✅ Removed redundant test files (`test_refactor_simple.py`)
- ✅ Updated test files with proper pytest markers
- ✅ All tests passing (7 tests: smoke, integration, end-to-end)

### Deferred (Not Priority)
- ⏸️ Frontend/server.py updates (frontend work deferred)
- ⏸️ Evaluation.py event consumers (will update when needed)

### Breaking Changes Made
- **Two-tier event system**: `StreamEvent` for LLM streaming, `StreamChunk` kept for lifecycle events (checkpoints, tool_result)
- All LLM event consumers must update to handle new granular event types
- No backward compatibility layer - clean break for better long-term maintenance
- `max_turns` removed from `AgentState`, now configured via `RunConfig.handle_stop=handle_stop_max_turns(n)`

### Test Strategy (Following Grugbrain Philosophy)
- **Smoke tests** (`test_agent_framework.py`) - Quick type/import sanity checks
- **Integration test** (`test_multi_turn.py`) - Real tool execution without API dependency
- **End-to-end test** (`test_real_agent.py`) - Full agent with real LLM API
- Removed mock tests in favor of integration tests for better maintainability

---

## Overview

This document outlines the plan to integrate core features from [pi-ai](https://github.com/badlogic/pi-mono/tree/main/packages/ai) into rollouts, making it a comprehensive one-stop-shop for running and training LLM agents.

## High-Level Features from Pi-AI

### 1. Unified Provider API Layer
- Map providers to shared API interfaces (4 types: anthropic-messages, google-generative-ai, openai-completions, openai-responses)
- Single abstraction that multiple providers implement
- **Current**: 3 separate `rollout_*` functions
  - [`rollout_openai`](rollouts/providers.py#L644-L786)
  - [`rollout_anthropic`](rollouts/providers.py#L1037-L1178)
  - [`rollout_sglang`](rollouts/providers.py#L584-L641)
- **Benefit**: Add new providers by mapping to existing interface

### 2. Granular Streaming Events
- Full lifecycle events: `start`, `text_start/delta/end`, `thinking_start/delta/end`, `toolcall_start/delta/end`, `done`, `error`
- **Current**: 5 event types
  - [`aggregate_stream`](rollouts/providers.py#L446-L576) - OpenAI streaming
  - [`aggregate_anthropic_stream`](rollouts/providers.py#L892-L1034) - Anthropic streaming
  - Current events: `token`, `tool_call_partial`, `tool_call_complete`, `tool_call_error`, `assistant_complete`, `thinking`
- **Benefit**: Better observability, progressive UIs, clear error boundaries

### 3. Model Discovery & Registry
- `get_providers()`, `get_models(provider)`, `get_model(provider, model_id)`
- Model metadata: context windows, costs, capabilities, reasoning support
- **Current**: Hardcoded in [`Endpoint`](rollouts/dtypes.py#L450-L467) configs
- **Benefit**: Dynamic provider selection, cost optimization, capability checking

### 4. Cross-Provider Context Handoff
- Serialize contexts with provider metadata
- Transform messages between providers (thinking blocks ↔ tagged text)
- Switch models mid-conversation
- **Current**: Checkpointing exists ([`checkpoints.py`](rollouts/checkpoints.py)) but no cross-provider transform
- **Benefit**: Cost optimization, fallback strategies, multi-model workflows

### 5. Additional Providers
- Google/Gemini, Groq, Cerebras, xAI, OpenRouter
- **Current**: OpenAI, Anthropic, sglang/vLLM
  - Provider selection in [`rollout()`](rollouts/agents.py#L352-L363)
- **Benefit**: More provider options, faster inference (Groq), cheaper alternatives

### 6. Enhanced Tool System
- Schema validation (TypeBox → Pydantic)
- Partial JSON streaming for tool arguments
- **Current**: Basic tool calling
  - [`ToolCall`](rollouts/dtypes.py#L66-L70)
  - [`Tool`](rollouts/dtypes.py#L358-L362)
  - [`ToolFunction`](rollouts/dtypes.py#L350-L355)
- **Benefit**: Better error messages, progressive tool call rendering

### 7. Unified Reasoning/Thinking API
- Single `reasoning: 'low'|'medium'|'high'` parameter across providers
- Provider-specific options still available
- **Current**: Anthropic thinking, OpenAI reasoning_effort handled separately
  - [`Endpoint.thinking`](rollouts/dtypes.py#L462) - Anthropic
  - [`Endpoint.reasoning_effort`](rollouts/dtypes.py#L460) - OpenAI
- **Benefit**: Easier to experiment with reasoning across providers

## Prioritization Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| **2. Granular Streaming Events** | High (foundation for UI) | Medium (2 days) | **P0** |
| **1. Unified Provider API** | High (extensibility) | Medium (3 days) | **P0** |
| **5. Additional Providers** | High (options) | Low-Med (3-5 days) | **P1** |
| **3. Model Discovery** | Medium (convenience) | Low (1-2 days) | **P1** |
| **7. Unified Reasoning API** | Medium (simplicity) | Low (1 day) | **P2** |
| **4. Cross-Provider Handoff** | Medium (flexibility) | Medium (2-3 days) | **P2** |
| **6. Enhanced Tool System** | Low (nice-to-have) | Medium (2 days) | **P3** |

## Implementation Roadmap

### Phase 0.3: Implement Remaining API Types (NEXT - 2-3 days)

**Goal**: Implement `rollout_openai_responses()` and `rollout_google()` to complete the 4 API types.

**Priority**:
1. **openai-responses** (HIGH) - Needed for o1/o3 reasoning models
2. **google-generative-ai** (MEDIUM) - Needed for Gemini models

#### Task 1: Implement `rollout_openai_responses()` for o1/o3 models

**Reference**: [pi-ai openai-responses.ts](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/openai-responses.ts)

**Key differences from openai-completions**:
- Uses OpenAI Responses API (different streaming event structure)
- Events: `response.output_item.added`, `response.output_text.delta`, `response.reasoning_summary_text.delta`, `response.function_call_arguments.delta`
- Reasoning content is multi-part summary (join with `\n\n`)
- Different completion event structure

**Implementation steps**:
1. Create `rollout_openai_responses()` in `providers.py`
2. Create `aggregate_openai_responses_stream()` helper for event processing
3. Handle reasoning/thinking content from `response.reasoning_summary_text.delta`
4. Map response events to our `StreamEvent` types
5. Register in `_PROVIDER_REGISTRY["openai-responses"]`
6. Add `test_openai_o1_agent()` to `test_real_agent.py`

**Files to modify**:
- `rollouts/providers.py` - Add new rollout function and aggregator
- `rollouts/models.py` - Already has o1/o1-mini models registered
- `tests/test_real_agent.py` - Add o1 test

#### Task 2: Implement `rollout_google()` for Gemini models

**Reference**: [pi-ai google.ts](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/google.ts)

**Key differences**:
- Google Generative AI API has different request/response format
- Different streaming event structure
- Different tool calling format
- May need `google-generativeai` SDK or raw HTTP

**Implementation steps**:
1. Research Google Generative AI streaming API format
2. Create `rollout_google()` in `providers.py`
3. Create `aggregate_google_stream()` helper
4. Map Google events to our `StreamEvent` types
5. Register in `_PROVIDER_REGISTRY["google-generative-ai"]`
6. Add `test_google_gemini_agent()` to `test_real_agent.py`

**Files to modify**:
- `rollouts/providers.py` - Add new rollout function
- `rollouts/models.py` - Add Gemini models to registry
- `tests/test_real_agent.py` - Add Gemini test

#### Success Criteria for Phase 0.3
- ✅ All 4 API types implemented and registered
- ✅ `test_real_agent.py` has 5 tests (OpenAI, Anthropic, Groq, O1, Gemini)
- ✅ All tests pass with real API keys
- ✅ Each API type proven to work end-to-end

**Estimated effort**: 2-3 days

---

### Phase 0: Foundation (P0 - Week 1) ✅ COMPLETE

#### 1. Granular Streaming Events (2-3 days)
**Why first?**
- Breaking change → do early before more users depend on current events
- Foundation for everything else (provider API, new providers all emit these events)
- Immediate value for debugging/observability

**Implementation Decisions:**
- **Event Types**: Dataclasses (not Pydantic) - typed but flexible
- **Backward Compatibility**: Clean break - update all consumers directly
- **Partial JSON Parsing**: Implement `parse_streaming_json()` for uniform tool call streaming across providers

**Changes needed:**
1. Define new event dataclasses in [`dtypes.py`](rollouts/dtypes.py)
   - Include `content_index` for tracking which content block
   - Include `partial` state for progressive accumulation
2. Implement `parse_streaming_json()` utility for partial tool argument parsing
3. Refactor [`aggregate_stream`](rollouts/providers.py#L446-L576) for OpenAI
   - Emit granular start/delta/end events
   - Use partial JSON parser for tool calls
4. Refactor [`aggregate_anthropic_stream`](rollouts/providers.py#L892-L1034) for Anthropic
   - Emit granular start/delta/end events (Anthropic SDK provides deltas)
   - Use partial JSON parser for tool calls
5. Update all event consumers:
   - [`stdout_handler`](rollouts/agents.py#L62-L72)
   - [`frontend/server.py`](rollouts/frontend/server.py#L1669-L1678)
   - [`evaluation.py`](rollouts/evaluation.py) (sample_start/end events)
   - Test files
6. Remove old event types: `token`, `tool_call_partial`, `tool_call_complete`, `tool_call_error`, `assistant_complete`

**New event types:**
```python
# Stream lifecycle
"start" → ... → "done" / "error"

# Text generation
"text_start" → "text_delta" → "text_end"

# Thinking (Anthropic/OpenAI reasoning)
"thinking_start" → "thinking_delta" → "thinking_end"

# Tool calls (with partial JSON streaming)
"toolcall_start" → "toolcall_delta" → "toolcall_end"
```

**Event Structure** (inspired by pi-ai):
```python
@dataclass
class TextDeltaEvent:
    type: Literal["text_delta"]
    content_index: int
    delta: str
    partial: AssistantMessage  # Current accumulated state

# Similar for thinking_delta, toolcall_delta, etc.
```

#### 2. Unified Provider API (3 days)
**Why second?**
- Makes adding new providers much easier
- Cleans up providers.py architecture
- Required foundation for cross-provider features

**Changes needed:**
- Create `ProviderAPI` protocol in [`dtypes.py`](rollouts/dtypes.py)
- Define 4 API interface types (anthropic-messages, google-generative-ai, openai-completions, openai-responses)
- Refactor existing [`rollout_*`](rollouts/providers.py#L584-L1178) functions to implement these interfaces
- Add provider registry/discovery system
- Update [`rollout()`](rollouts/agents.py#L352-L363) to use new abstraction

**Lines of code**: ~500-800 LOC

### Phase 1: Expansion (P1 - Week 2)

#### 3. Additional Providers (3-5 days)
**Implementation order:**
1. **Groq** (1 day) - OpenAI-compatible, fastest to add
2. **Google/Gemini** (2 days) - Most different, good test of abstraction
3. **Cerebras, xAI, OpenRouter** (1 day) - OpenAI-compatible

**Changes needed:**
- Add new rollout functions implementing shared interfaces
- Add to provider selection in [`rollout()`](rollouts/agents.py#L352-L363)
- Update [`Endpoint`](rollouts/dtypes.py#L450-L467) to support new providers
- Add tests and documentation

**Lines of code**: ~800-1200 LOC

#### 4. Model Discovery (1-2 days)
**Changes needed:**
- Create model registry with capabilities/costs
- Add provider enumeration functions:
  - `get_providers()`
  - `get_models(provider)`
  - `get_model(provider, model_id)`
- Add to [`__init__.py`](rollouts/__init__.py) exports

**Lines of code**: ~200-300 LOC

### Phase 2: Polish (P2 - Week 3)

#### 5. Unified Reasoning API (1 day)
**Changes needed:**
- Add `reasoning: 'low'|'medium'|'high'` parameter to [`RunConfig`](rollouts/dtypes.py#L504-L520)
- Map to provider-specific parameters:
  - Anthropic: `thinking` dict
  - OpenAI: `reasoning_effort` string
- Update [`rollout_anthropic`](rollouts/providers.py#L1037-L1178) and [`rollout_openai`](rollouts/providers.py#L644-L786)

**Lines of code**: ~100-200 LOC

#### 6. Cross-Provider Handoff (2-3 days)
**Changes needed:**
- Context serialization with provider metadata
- Message transformation pipeline (thinking blocks ↔ tagged text)
- Update [`FileCheckpointStore`](rollouts/checkpoints.py) to support provider switching
- Add context migration helpers

**Lines of code**: ~400-600 LOC

### Phase 3: Optional (P3 - Later)

#### 7. Enhanced Tool System (1-2 days)
**Changes needed:**
- Add Pydantic schemas for tool validation
- ~~Implement partial JSON streaming in tool argument accumulation~~ **DONE in Phase 0.1**
- Better error messages in [`exec_tool`](rollouts/dtypes.py#L400-L402)
- Update tool result handling in [`process_pending_tools`](rollouts/agents.py#L466-L560)

**Lines of code**: ~200-300 LOC (reduced since partial JSON parsing done in Phase 0)

## Total Effort Estimate

- **Phase 0 (P0)**: 5-6 days (~1 week)
- **Phase 1 (P1)**: 5-7 days (~1 week)
- **Phase 2 (P2)**: 3-4 days
- **Phase 3 (P3)**: 1-2 days

**Total**: ~14-19 days (~3 weeks)

**Total new/modified code**: ~2,400-3,400 LOC

## Design Decisions Log

### Phase 0.1: Granular Streaming Events
- **Event Type System**: Dataclasses (not Pydantic) for balance of type safety and flexibility
- **Backward Compatibility**: Clean break - no compatibility layer, update all consumers directly
- **Partial JSON Parsing**: Implement in Phase 0 for uniform tool streaming (moved from Phase 3)
- **Event Structure**: Follow pi-ai pattern with `content_index` and `partial` state fields
- **Tool Call Streaming**: Unified `toolcall_start/delta/end` for both OpenAI and Anthropic via custom JSON parser

## References

- **Pi-AI Blog Post**: https://mariozechner.at/posts/2025-11-30-pi-coding-agent/
- **Pi-Mono Repository**: https://github.com/badlogic/pi-mono
- **Pi-AI Package**: https://www.npmjs.com/package/@mariozechner/pi-ai
- **Rollouts Repository**: (local at `/Users/chiraagbalu/research/rollouts/`)

## Key Files to Modify

- [`rollouts/dtypes.py`](rollouts/dtypes.py) - Core types, add ProviderAPI protocol
- [`rollouts/providers.py`](rollouts/providers.py) - Provider implementations
- [`rollouts/agents.py`](rollouts/agents.py) - Agent execution loop, event handling
- [`rollouts/checkpoints.py`](rollouts/checkpoints.py) - Cross-provider context handoff
- [`rollouts/__init__.py`](rollouts/__init__.py) - Public API exports

## Next Steps

**Recommended**: Start with **Granular Streaming Events (Phase 0.1)**

This is a breaking change that should be done early before more consumers depend on current event types. It also provides immediate value for debugging and lays the foundation for all subsequent work.
