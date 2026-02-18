# Extensibility Design

## Context

Comparison with [pi-mono](https://github.com/badlogicgames/pi-mono) revealed different approaches to extensibility. Pi-mono uses a top-down plugin architecture with 25+ lifecycle hooks, jiti-based extension loading, and event subscriptions. Rollouts uses bottom-up protocols and callback injection.

This doc outlines how to improve rollouts' extensibility while staying aligned with compression-oriented programming principles.

## Design Principles

From the codebase style guides:

1. **Semantic compression**: Don't abstract until you have 2+ instances. Build bottom-up.
2. **Continuous granularity**: Never create holes in the API. Layer on top, don't modify underneath.
3. **Write usage code first**: The API should be obvious from what you want to write.
4. **Simple control flow**: No callbacks when a simple loop would do.
5. **Assertions over fallbacks**: Fail fast, crash loud.

## Current State

### What's Pluggable

| Component | Pattern | Add Without Modifying Core? |
|-----------|---------|----------------------------|
| Environment | Protocol | ✅ Yes (but must add to registry for deserialize) |
| Frontend | Protocol | ✅ Yes |
| SessionStore | Protocol | ✅ Yes |
| RunConfig callbacks | Dataclass | ✅ Yes |
| Provider | Registry dict | ❌ Must modify `_PROVIDER_REGISTRY` |
| Driver | Function | ❌ Must modify CLI dispatch |
| Model | `register_model()` | ✅ Yes |

### Granularity Gaps

**Gap 1: Tool execution is monolithic**

`process_pending_tools()` in `agents.py` does confirmation → execution → error handling in one blob. Can't replace just execution logic.

**Gap 2: Provider registry exists but isn't exported**

`_PROVIDER_REGISTRY` in `providers/__init__.py` is private. Can't register providers from outside the package.

**Gap 3: Driver dispatch is if/elif chain**

`cli.py:2058-2071` requires editing to add drivers. Fine for 3 drivers, but doesn't scale.

**Gap 4: Driver parsing is coupled to process spawning**

`ClaudeDriver.run()` mixes subprocess management with output parsing. Can't reuse parser for testing.

## Target Usage Code

What extending rollouts should look like:

```python
# Adding a new driver
from rollouts import register_driver

@register_driver("my-agent")
async def run_my_agent(state: AgentState, config: RunConfig) -> list[AgentState]:
    ...

# Adding a guardrail (only if we have 2+ use cases)
from rollouts import add_hook

@add_hook("before_tool")
def block_dangerous(tool_call: ToolCall) -> bool:
    return "rm -rf" not in tool_call.arguments.get("command", "")

# Adding a provider
from rollouts import register_provider

register_provider("my-api", my_stream_function)
```

## Recommendations

### Level 0: Expose Existing Registries (Immediate)

No new code. Just export what exists.

```python
# rollouts/__init__.py
from .providers import register_provider
from .environments.compose import register_environment
from .models import register_model
```

Files to change:
- `rollouts/__init__.py` - add exports
- `rollouts/providers/__init__.py` - add `register_provider()` function
- `rollouts/environments/compose.py` - add `register_environment()` function

### Level 1: Factor Monolithic Functions (Short-term)

Break `process_pending_tools()` into composable pieces:

```python
# agents.py - new functions

async def confirm_tool_call(
    tool_call: ToolCall,
    state: AgentState,
    config: RunConfig,
) -> tuple[AgentState, ToolConfirmResult]:
    """Confirm a single tool call. Returns updated state and confirmation result."""
    ...

async def execute_tool_call(
    tool_call: ToolCall,
    environment: Environment,
    state: AgentState,
    config: RunConfig,
) -> ToolResult:
    """Execute a single tool call. Returns tool result."""
    ...

def finalize_tool_result(
    result: ToolResult,
    state: AgentState,
    config: RunConfig,
) -> AgentState:
    """Handle tool result (including errors). Returns updated state."""
    ...

# process_pending_tools() becomes composition of above
```

This creates continuous granularity: use the composed function for normal cases, use individual functions when you need custom behavior.

### Level 1.5: Factor Driver Parsing (Short-term)

Decouple parsing from subprocess management:

```python
# drivers/claude.py

def parse_claude_event(line: str) -> StreamEvent | None:
    """Parse a single line of Claude Code stream-json output."""
    ...

class ClaudeDriver:
    async def run(self, prompt: str) -> AsyncIterator[StreamEvent]:
        # Thin wrapper: spawn process, call parse_claude_event per line
        ...
```

Same for Codex. Enables testing parsers without spawning processes.

### Level 2: Single Hook Point (Medium-term, Only If Needed)

If you find 2+ places wanting to intercept before LLM calls (e.g., cost tracking AND guardrails), add ONE hook:

```python
# dtypes.py - add to RunConfig

@dataclass
class RunConfig:
    # ... existing fields ...

    # Optional hook: modify messages before LLM call
    # Return modified messages or same messages
    transform_context: Callable[[list[Message], AgentState], list[Message]] | None = None
```

Integration in `agents.py`:

```python
# Before calling provider
if config.transform_context:
    messages = config.transform_context(messages, state)
actor = await provider_fn(actor, messages, ...)
```

No event system. No subscription. No ceremony. Just a function that transforms messages.

### Not Now: Full Plugin System

Don't add:
- **Event hooks with 25 lifecycle points** - No evidence of need
- **jiti-style extension loading** - Not shipping to third parties who can't modify source
- **UI integration API** - Rollouts is batch/eval oriented, not TUI-heavy
- **Driver registry** - Only 3 drivers. Add registry when you hit 5+

## Comparison with Pi-mono

| Category | Pi-mono | Rollouts Current | Rollouts Target |
|----------|---------|------------------|-----------------|
| Philosophy | Top-down plugin architecture | Bottom-up protocols | Bottom-up with exposed registries |
| Third-party extension | Yes (jiti + events) | No | Not a goal |
| Granularity | Fine (25 hooks) | Coarse (3 protocols) | Medium (factored functions) |
| Control flow | Callbacks everywhere | Simple functions | Simple functions |

Pi-mono's architecture is designed for a plugin ecosystem where third parties extend without forking. Rollouts doesn't have that requirement. The target is **continuous granularity** so any piece can be replaced, not a plugin system.

## Implementation Order

1. **Export registries** - 30 min, zero risk
2. **Factor `process_pending_tools()`** - 2 hours, low risk (existing tests cover it)
3. **Factor driver parsers** - 1 hour per driver, enables better testing
4. **Add `transform_context` hook** - Only when there are 2+ use cases

## References

- Casey Muratori, "Semantic Compression" (2014)
- Casey Muratori, "Complexity and Granularity" (2014)
- Casey Muratori, "The Worst API Ever Made" (2014)
- TigerBeetle, "Tiger Style" safety guidelines
- Sean Goedecke, "Everything I Know About Good System Design"
