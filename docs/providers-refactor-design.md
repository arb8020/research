# Providers Refactor: Usage/Cost Tracking & Module Split

**DRI:** chiraag
**Claude:** [this conversation]

## Context

The `rollouts/providers.py` file is a ~2700 line monolith containing all provider implementations (Anthropic, OpenAI completions, OpenAI responses, Google, vLLM/sglang). This makes it hard to maintain and extend.

Additionally, usage tracking is incomplete:
- Current `Usage` dataclass only has `prompt_tokens`, `completion_tokens`, `total_tokens`
- No cache token tracking (critical for Anthropic/OpenAI cost savings)
- No cost calculation wired into provider responses
- `models.py` has `ModelCost` and `calculate_cost()` but they're not connected to providers

**Reference:** [pi-mono](https://github.com/badlogic/pi-mono) has a clean `providers/` structure with integrated usage/cost tracking that we're drawing inspiration from.

## Design Principles (from code_style/)

Following the codebase style guides:

1. **Frozen dataclasses for data** (IMMUTABILITY_AND_FP) - `Cost` and `Usage` are immutable data, not stateful objects
2. **Pure functions for computation** (CLASSES_VS_FUNCTIONAL) - `calculate_cost()` is pure math, no class needed
3. **Split assertions** (FAVORITES/Tiger Style) - separate asserts for each invariant
4. **Continuous granularity** (FAVORITES/Casey) - keep low-level provider functions accessible
5. **Don't abstract until 2+ examples** (FAVORITES/Casey) - we have 4 providers, split is justified
6. **SSA style** (FAVORITES/Carmack) - named intermediate values, no variable reassignment

## Goals

1. **Split providers.py** into separate modules per API type
2. **Enhance Usage** with cache tokens and cost breakdown
3. **Wire cost calculation** into each provider after API responses
4. **Maintain API compatibility** - no changes to how providers are called

## Out of Scope

- Adding new providers (can be done after refactor)
- Changing the `Actor`/`Trajectory` abstraction
- Billing/quota enforcement (separate concern)
- Provider-specific retry logic changes

## Solution

### New Directory Structure

```
rollouts/
├── providers/
│   ├── __init__.py              # Re-exports, provider registry
│   ├── base.py                  # Shared types, helpers (NonRetryableError, etc.)
│   ├── anthropic.py             # rollout_anthropic + aggregate_anthropic_stream
│   ├── openai_completions.py    # rollout_openai + aggregate_stream
│   ├── openai_responses.py      # rollout_openai_responses + aggregate_openai_responses_stream
│   ├── google.py                # rollout_google + aggregate_google_stream
│   └── sglang.py                # rollout_sglang + vLLM helpers
├── transform_messages.py        # Already exists (cross-provider message transforms)
├── dtypes.py                    # Enhanced Usage, Cost dataclasses
├── models.py                    # Model registry (unchanged API)
└── ...
```

### Enhanced Usage & Cost Types

**Current (`dtypes.py`):**
```python
@dataclass(frozen=True)
class Usage(JsonSerializable):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Any | None = None
```

**New (`dtypes.py`):**
```python
@dataclass(frozen=True)
class Cost:
    """Cost breakdown in USD. Immutable - create new instance to change.

    Following IMMUTABILITY_AND_FP: frozen dataclass for data that doesn't change.
    """
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0

    @property
    def total(self) -> float:
        return self.input + self.output + self.cache_read + self.cache_write


@dataclass(frozen=True)
class Usage(JsonSerializable):
    """Token usage with cost tracking. Immutable.

    Following IMMUTABILITY_AND_FP: state changes are explicit via replace().
    Following SSA: each transformation creates a new binding.

    Example:
        # SSA style - named intermediate values
        raw_usage = Usage(input_tokens=100, output_tokens=50)
        usage_with_cost = replace(raw_usage, cost=calculated_cost)
    """
    # Token counts (primary fields)
    input_tokens: int = 0           # Non-cached input tokens
    output_tokens: int = 0          # Output/completion tokens (excludes reasoning)
    reasoning_tokens: int = 0       # Reasoning/thinking tokens (OpenAI o1/o3, Anthropic thinking)
    cache_read_tokens: int = 0      # Tokens read from cache (Anthropic/OpenAI)
    cache_write_tokens: int = 0     # Tokens written to cache (Anthropic)

    # Cost breakdown (computed by provider after API response)
    cost: Cost = field(default_factory=Cost)

    # Computed properties
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens + self.cache_read_tokens + self.cache_write_tokens

    # Legacy aliases for backwards compatibility (FAVORITES: don't break userspace)
    @property
    def prompt_tokens(self) -> int:
        """Legacy alias: input_tokens + cache_read_tokens"""
        return self.input_tokens + self.cache_read_tokens

    @property
    def completion_tokens(self) -> int:
        """Legacy alias: output_tokens + reasoning_tokens (rolled together for compat)"""
        return self.output_tokens + self.reasoning_tokens
```

### Provider Cost Calculation

**Pure function for cost calculation** (CLASSES_VS_FUNCTIONAL: functions for computation):

```python
# In providers/base.py (shared helper)

def calculate_cost_from_usage(usage: Usage, model_cost: ModelCost | None) -> Cost:
    """Pure function: Usage + ModelCost -> Cost.

    Following CLASSES_VS_FUNCTIONAL: pure math, no class needed.
    Following FAVORITES: explicit inputs/outputs, no hidden state.
    """
    if model_cost is None:
        return Cost()

    return Cost(
        input=(usage.input_tokens / 1_000_000) * model_cost.input,
        output=(usage.output_tokens / 1_000_000) * model_cost.output,
        cache_read=(usage.cache_read_tokens / 1_000_000) * model_cost.cache_read,
        cache_write=(usage.cache_write_tokens / 1_000_000) * model_cost.cache_write,
    )
```

**SSA-style usage in providers:**

```python
# In providers/anthropic.py

from rollouts.models import get_model
from rollouts.providers.base import calculate_cost_from_usage

async def rollout_anthropic(...) -> Actor:
    # ... existing streaming code ...

    # SSA style: named intermediate values, no reassignment
    raw_completion = await aggregate_anthropic_stream(stream, on_chunk)

    # Calculate cost (pure function call)
    model_meta = get_model(actor.endpoint.provider, actor.endpoint.model)
    cost = calculate_cost_from_usage(raw_completion.usage, model_meta.cost if model_meta else None)

    # Create new completion with cost (immutable update)
    usage_with_cost = replace(raw_completion.usage, cost=cost)
    completion = replace(raw_completion, usage=usage_with_cost)

    # ... rest of existing code ...
```

### Usage Extraction Per Provider

**Note:** Since `Usage` is frozen, stream aggregators accumulate into local variables during streaming, then construct the final `Usage` at the end:

```python
# Pattern: accumulate during streaming, construct frozen Usage at end
input_tokens = 0
output_tokens = 0
reasoning_tokens = 0
cache_read_tokens = 0
cache_write_tokens = 0

# ... accumulate from stream events ...

# Construct frozen Usage once at end
usage = Usage(
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    reasoning_tokens=reasoning_tokens,
    cache_read_tokens=cache_read_tokens,
    cache_write_tokens=cache_write_tokens,
)
```

**Anthropic** (`message_start` + `message_delta` events):
```python
# In aggregate_anthropic_stream:
if event_type == "message_start":
    input_tokens = event.message.usage.input_tokens or 0
    cache_read_tokens = event.message.usage.cache_read_input_tokens or 0
    cache_write_tokens = event.message.usage.cache_creation_input_tokens or 0

elif event_type == "message_delta":
    output_tokens = event.usage.output_tokens or 0
    # Note: Anthropic thinking tokens come from content blocks, not usage
```

**OpenAI Completions** (final chunk with `usage`):
```python
# In aggregate_stream:
if chunk.usage:
    cached = getattr(chunk.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
    reasoning = getattr(chunk.usage.completion_tokens_details, 'reasoning_tokens', 0) or 0

    input_tokens = (chunk.usage.prompt_tokens or 0) - cached
    output_tokens = (chunk.usage.completion_tokens or 0) - reasoning  # exclude reasoning
    reasoning_tokens = reasoning  # track separately
    cache_read_tokens = cached
```

**OpenAI Responses** (`response.completed` event):
```python
# In aggregate_openai_responses_stream:
if event_type == "response.completed":
    cached = getattr(response.usage.input_tokens_details, 'cached_tokens', 0) or 0
    reasoning = getattr(response.usage.output_tokens_details, 'reasoning_tokens', 0) or 0

    input_tokens = (response.usage.input_tokens or 0) - cached
    output_tokens = (response.usage.output_tokens or 0) - reasoning
    reasoning_tokens = reasoning
    cache_read_tokens = cached
```

**Google** (`usageMetadata` in chunks):
```python
# In aggregate_google_stream:
if chunk.usageMetadata:
    input_tokens = chunk.usageMetadata.promptTokenCount or 0
    output_tokens = chunk.usageMetadata.candidatesTokenCount or 0
    reasoning_tokens = chunk.usageMetadata.thoughtsTokenCount or 0  # separate!
    cache_read_tokens = chunk.usageMetadata.cachedContentTokenCount or 0
```

---

## Details

### Provider Registry

The `providers/__init__.py` will maintain the registry pattern:

```python
# providers/__init__.py

from .anthropic import rollout_anthropic
from .openai_completions import rollout_openai
from .openai_responses import rollout_openai_responses
from .google import rollout_google
from .sglang import rollout_sglang

from rollouts.models import ApiType

_PROVIDER_REGISTRY: dict[ApiType, ProviderStreamFunction] = {
    "openai-completions": rollout_openai,
    "openai-responses": rollout_openai_responses,
    "anthropic-messages": rollout_anthropic,
    "google-generative-ai": rollout_google,
}

def get_provider_function(provider: str, model_id: str | None = None) -> ProviderStreamFunction:
    """Get streaming function for a provider/model combination."""
    from rollouts.models import get_api_type

    api_type = get_api_type(provider, model_id)
    func = _PROVIDER_REGISTRY.get(api_type)
    assert func is not None, f"No provider for API type: {api_type}"
    return func

# Re-export for backwards compatibility
__all__ = [
    "rollout_anthropic",
    "rollout_openai",
    "rollout_openai_responses",
    "rollout_google",
    "rollout_sglang",
    "get_provider_function",
]
```

### Shared Base Module

Common code extracted to `providers/base.py`:

```python
# providers/base.py

class NonRetryableError(Exception):
    """Exception for errors that should not be retried."""
    pass

def sanitize_request_for_logging(params: dict) -> dict:
    """Sanitize request parameters to remove large base64 image data."""
    # ... existing implementation ...

def add_cache_control_to_last_content(messages, cache_control={"type": "ephemeral"}, max_cache_controls: int = 4):
    """Adds cache control metadata to the final content block."""
    # ... existing implementation ...
```

### Migration Path

1. **Phase 1:** Create `providers/` directory, move code without changing behavior
   - Create base.py with shared code
   - Split each provider into its own file
   - Update imports in `__init__.py`
   - Deprecate direct imports from `providers.py`

2. **Phase 2:** Enhance Usage/Cost
   - Add `Cost` dataclass to dtypes.py
   - Update `Usage` with new fields (keep old as properties for compat)
   - Update each provider's stream aggregator to populate cache tokens

3. **Phase 3:** Wire cost calculation
   - Add cost calculation to each provider after API response
   - Update ChatCompletion to include enriched usage
   - Add tests for cost calculation

### Backwards Compatibility

**Preserved:**
- `Usage.prompt_tokens` → property alias for `input_tokens + cache_read_tokens`
- `Usage.completion_tokens` → property alias for `output_tokens`
- `Usage.total_tokens` → property (still computed)
- `from rollouts.providers import rollout_anthropic` → still works
- `get_provider_function()` API unchanged

**Changed:**
- `Usage` is now a richer dataclass with more fields
- New `Cost` dataclass for cost breakdown
- Internal imports will shift to `rollouts.providers.*`

### Testing

```python
# tests/test_providers/test_usage.py

def test_usage_backwards_compat():
    """Verify old field names still work."""
    usage = Usage(input_tokens=100, output_tokens=50, cache_read_tokens=25)

    assert usage.prompt_tokens == 125  # input + cache_read
    assert usage.completion_tokens == 50  # output
    assert usage.total_tokens == 175  # all

def test_cost_calculation():
    """Verify cost is calculated correctly."""
    from rollouts.models import ModelCost

    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=500_000,
        cache_read_tokens=200_000,
    )
    cost = ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)

    # Manual calculation
    expected_input = 1.0 * 3.0  # 1M tokens * $3/M
    expected_output = 0.5 * 15.0  # 0.5M tokens * $15/M
    expected_cache = 0.2 * 0.3  # 0.2M tokens * $0.3/M

    calculated = Cost(
        input=(usage.input_tokens / 1_000_000) * cost.input,
        output=(usage.output_tokens / 1_000_000) * cost.output,
        cache_read=(usage.cache_read_tokens / 1_000_000) * cost.cache_read,
    )

    assert calculated.input == expected_input
    assert calculated.output == expected_output
    assert calculated.cache_read == expected_cache
```

---

## Open Questions

- [x] Should `Cost` be frozen/immutable or mutable? → **Frozen** (IMMUTABILITY_AND_FP)
- [x] Add reasoning token tracking separately? → **Yes**, `reasoning_tokens` field, rolled into `completion_tokens` for compat
- [ ] Track latency/timing in Usage as well?
- [ ] Add provider-specific usage fields (e.g., Anthropic's `cache_creation_input_tokens`)?

---

## Files

**Create:**
- `rollouts/providers/__init__.py` - registry + re-exports
- `rollouts/providers/base.py` - shared helpers
- `rollouts/providers/anthropic.py` - Anthropic provider
- `rollouts/providers/openai_completions.py` - OpenAI completions provider
- `rollouts/providers/openai_responses.py` - OpenAI responses provider
- `rollouts/providers/google.py` - Google provider
- `rollouts/providers/sglang.py` - vLLM/sglang provider

**Modify:**
- `rollouts/dtypes.py` - add Cost, enhance Usage
- `rollouts/providers.py` - deprecate, re-export from providers/

**Delete (after migration):**
- `rollouts/providers.py` (eventually, keep as re-export shim initially)
