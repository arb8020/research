# Hooks System Design

**Status:** Draft  
**Priority:** Low (no external users yet)  
**Inspiration:** [pi-mono hooks](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/hooks.md)

---

## Overview

Hooks are Python modules that subscribe to agent lifecycle events and can intercept/modify behavior. They enable:

- Permission gates (confirm before dangerous commands)
- Git checkpointing (stash before each turn, restore on branch)
- Protected paths (block writes to `.env`, `node_modules/`)
- External triggers (file watchers, webhooks inject messages)
- Custom compaction (use different model for summarization)

## Current Architecture

### RunConfig (the junk drawer)

```python
@dataclass
class RunConfig:
    # Event callbacks
    on_chunk: Callable[[StreamEvent], Awaitable[None]]
    on_input: Callable[[str], Awaitable[str]]
    
    # Lifecycle handlers
    on_step_start: Callable[[AgentState], AgentState]
    handle_stop: Callable[[AgentState], AgentState]
    handle_no_tool: Callable[[AgentState, RunConfig], Awaitable[AgentState]]
    
    # Tool policies
    confirm_tool: Callable[[ToolCall, AgentState, RunConfig], Awaitable[tuple[AgentState, ToolConfirmResult]]]
    handle_tool_error: Callable[[ToolResult, AgentState], AgentState]
    
    # Provider-specific
    user_message_for_thinking: str | None
    inline_thinking: str | None
    
    # Cancellation & persistence
    cancel_scope: trio.CancelScope | None
    session_store: Any | None
    
    # RL training
    use_tito: bool
    tokenizer: Any | None
    suffix_ids: tuple[int, ...] | None
```

### Agent Loop Phases

```
run_agent(state, config)
  └─► run_agent_step(state, config)           # One turn
        ├─► handle_stop(state)                # Check stop conditions
        ├─► on_step_start(state)              # Pre-turn hook point
        ├─► rollout(actor, on_chunk, ...)     # LLM call
        ├─► environment.on_assistant_message  # Env response
        └─► process_pending_tools(state, config)
              └─► for each tool_call:
                    ├─► confirm_tool(...)     # ← HOOK POINT: tool_call
                    ├─► env.exec_tool(...)    # Execute
                    ├─► on_chunk(ToolResultReceived)  # ← HOOK POINT: tool_result
                    └─► handle_tool_error(...)
```

## Proposed Design

### Option 1: Extend RunConfig (not recommended)

Add more callbacks to RunConfig. Simple but makes the junk drawer worse.

### Option 2: Tool Wrapper (recommended)

A `HookRunner` wraps the environment's tools before passing to agent. Keeps hooks separate from core agent logic.

```python
# rollouts/hooks/runner.py

class HookRunner:
    def __init__(self, hooks: list[LoadedHook], ctx: HookContext):
        self.hooks = hooks
        self.ctx = ctx
        self._tool_call_handlers: list[ToolCallHandler] = []
        self._tool_result_handlers: list[ToolResultHandler] = []
        self._session_handlers: list[SessionHandler] = []
        # ... collect handlers from hooks
    
    def wrap_environment(self, env: Environment) -> Environment:
        """Return environment with wrapped exec_tool."""
        return WrappedEnvironment(env, self)
    
    async def emit_tool_call(self, event: ToolCallEvent) -> BlockToolResult | None:
        """Emit tool_call event, return block result if any handler blocks."""
        for handler in self._tool_call_handlers:
            result = await handler(event, self.ctx)
            if result and result.block:
                return result
        return None
    
    async def emit_tool_result(self, event: ToolResultEvent) -> ModifyResultResult | None:
        """Emit tool_result event, return modifications if any."""
        # Last handler wins for modifications
        ...
```

```python
# Integration in InteractiveRunner or CLI

hooks = discover_and_load_hooks(cwd)
hook_runner = HookRunner(hooks, ctx)

# Wrap environment before passing to agent
wrapped_env = hook_runner.wrap_environment(environment)
state = AgentState(actor=actor, environment=wrapped_env, ...)
```

### Option 3: Separate HookConfig

New config object passed alongside RunConfig. Cleaner but more plumbing.

## Event Types

```python
@dataclass
class ToolCallEvent:
    tool_name: str
    tool_call_id: str
    input: dict[str, Any]

@dataclass
class ToolResultEvent:
    tool_name: str
    tool_call_id: str
    input: dict[str, Any]
    content: str
    is_error: bool
    details: dict[str, Any] | None

@dataclass
class SessionEvent:
    reason: Literal["start", "shutdown", "before_branch", "branch", "before_switch", "switch"]
    session_file: Path | None
    entries: list[Message]
    target_turn_index: int | None = None  # for branch events

@dataclass
class TurnStartEvent:
    turn_index: int

@dataclass
class TurnEndEvent:
    turn_index: int
    message: Message
    tool_results: list[ToolResult]
```

## Hook Context

```python
@dataclass
class HookContext:
    cwd: Path
    session_file: Path | None
    has_ui: bool
    
    # UI helpers (no-op if has_ui=False)
    async def select(self, title: str, options: list[str]) -> str | None: ...
    async def confirm(self, title: str, message: str) -> bool: ...
    async def input(self, title: str, placeholder: str = "") -> str | None: ...
    def notify(self, message: str, level: Literal["info", "warning", "error"] = "info") -> None: ...
    
    # Shell execution
    async def exec(self, command: str, args: list[str] = [], timeout: float | None = None) -> ExecResult: ...
```

## Hook File Format

```python
# ~/.rollouts/hooks/permission_gate.py

from rollouts.hooks import HookAPI, ToolCallEvent, HookContext, BlockToolResult
import re

DANGEROUS = [r"\brm\s+(-rf?|--recursive)", r"\bsudo\b"]

async def check_dangerous(event: ToolCallEvent, ctx: HookContext) -> BlockToolResult | None:
    if event.tool_name != "bash":
        return None
    
    command = event.input.get("command", "")
    if any(re.search(p, command, re.I) for p in DANGEROUS):
        if not ctx.has_ui:
            return BlockToolResult(reason="Dangerous command blocked")
        if not await ctx.confirm("⚠️ Dangerous command", command):
            return BlockToolResult(reason="Blocked by user")
    return None

def register(hooks: HookAPI) -> None:
    hooks.on_tool_call(check_dangerous)
```

## Discovery

```
~/.rollouts/hooks/*.py      # Global hooks
.rollouts/hooks/*.py        # Project hooks  
--hook path/to/hook.py      # CLI flag
```

Each file must export `register(hooks: HookAPI) -> None`.

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Define event and result types
- [ ] Implement HookContext with UI helpers
- [ ] Implement HookRunner with tool wrapping
- [ ] Hook discovery and loading

### Phase 2: Integration
- [ ] Wire into InteractiveRunner for session events
- [ ] Wire into process_pending_tools for tool events
- [ ] Add `--hook` CLI flag
- [ ] Add `--no-hooks` CLI flag

### Phase 3: Examples
- [ ] permission_gate.py - confirm dangerous commands
- [ ] protected_paths.py - block writes to sensitive files
- [ ] git_checkpoint.py - stash before each turn

## Open Questions

1. **Timeout for hooks?** Pi-mono uses 30s default, no timeout for tool_call (user prompts). Same?

2. **Error handling?** If hook throws, log and continue? Or block the operation?

3. **Hook ordering?** First registered wins? Last wins? Explicit priority?

4. **Async loading?** Hooks are Python files - import synchronously or use importlib async?

5. **send() for message injection?** Pi-mono lets hooks inject messages. Complex - defer?

---

## Why Deprioritized

Hooks are an extensibility feature for power users. Current state:
- No external users
- Core features still evolving
- Would need to maintain hook API stability

Better to focus on:
- Features that make the core experience better
- Things that help with internal experimentation
- Foundation work that hooks would build on anyway
