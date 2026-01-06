# InteractiveAgentRunner Refactor Design

## Goal

Replace implicit flag-based control flow with explicit return types.

Current pattern (flags):
```python
# Slash command sets flag
self._session_switched = True

# Later, caller checks flag
if self._session_switched:
    self._session_switched = False
    state = rebuild_from_self_properties()
```

Proposed pattern (return values):
```python
# Slash command returns what happened
result = await handle_slash_command(...)
match result:
    case SessionSwitch(new_state):
        state = new_state
```

---

## New Types

```python
from dataclasses import dataclass

# ─── Input Phase Results ───────────────────────────────────────────────────────
# What happened when we waited for user input?

@dataclass(frozen=True)
class InputExit:
    """User wants to exit entirely (Ctrl+C)."""
    pass

@dataclass(frozen=True)
class InputContinue:
    """Slash command handled, loop back for more input."""
    message: str | None = None

@dataclass(frozen=True)
class InputNewState:
    """Slash command changed state (model/env/session). Use this state going forward."""
    state: AgentState
    message: str | None = None

@dataclass(frozen=True)
class InputMessage:
    """User entered a message to send to LLM."""
    text: str

InputResult = InputExit | InputContinue | InputNewState | InputMessage


# ─── Agent Phase Results ───────────────────────────────────────────────────────
# What happened when we ran the agent?

@dataclass(frozen=True)
class AgentCompleted:
    """Agent finished normally (task complete or no tools)."""
    states: list[AgentState]

@dataclass(frozen=True)
class AgentInterrupted:
    """User pressed Escape to interrupt."""
    states: list[AgentState]
    partial_response: str | None = None

@dataclass(frozen=True)
class AgentExited:
    """User pressed Ctrl+C to exit."""
    states: list[AgentState]

@dataclass(frozen=True)
class AgentError:
    """Recoverable error (context too long, OAuth expired, etc.)."""
    states: list[AgentState]
    error: Exception
    error_kind: str  # "context_too_long", "oauth_expired", etc.

AgentOutcome = AgentCompleted | AgentInterrupted | AgentExited | AgentError
```

---

## New Main Loop Shape

```python
async def _run_agent_loop(self) -> list[AgentState]:
    """Main loop with explicit control flow."""
    self.input_send, self.input_receive = trio.open_memory_channel[str](10)
    
    if self.initial_prompt:
        self.input_send.send_nowait(self.initial_prompt)
    
    all_states: list[AgentState] = []
    state: AgentState | None = None
    
    async with trio.open_nursery() as nursery:
        self.cancel_scope = nursery.cancel_scope
        nursery.start_soon(self._input_reading_loop)
        nursery.start_soon(self.tui.run_animation_loop)
        
        self._focus_input()
        
        # ══════════════════════════════════════════════════════════════════
        # MAIN LOOP - two phases: get input, run agent
        # ══════════════════════════════════════════════════════════════════
        while True:
            
            # ─── PHASE 1: Get user input ──────────────────────────────────
            # This handles slash commands and returns what happened.
            # Loops internally until we get a message or state change.
            
            input_result = await self._get_input_result(state)
            
            match input_result:
                case InputExit():
                    break
                    
                case InputContinue(message):
                    self._show_ghost_message(message)
                    continue  # loop back to get more input
                    
                case InputNewState(new_state, message):
                    state = new_state
                    self._show_ghost_message(message)
                    continue  # loop back to get more input (state changed, but no message to send)
                    
                case InputMessage(text):
                    # Create or update state with user message
                    if state is None:
                        state = self._create_initial_state(text)
                    else:
                        state = self._add_user_message(state, text)
            
            # ─── PHASE 2: Run agent ───────────────────────────────────────
            # Runs until completion, interruption, or error.
            
            outcome = await self._run_agent_with_outcome(state)
            all_states.extend(outcome.states)
            
            match outcome:
                case AgentCompleted(states):
                    state = states[-1] if states else state
                    # Loop back to get next input
                    
                case AgentInterrupted(states, partial_response):
                    state = states[-1] if states else state
                    state = self._finalize_interrupt(state, partial_response)
                    # Loop back to get next input
                    
                case AgentExited(states):
                    break  # Exit the loop entirely
                    
                case AgentError(states, error, error_kind):
                    self._show_error(error, error_kind)
                    state = states[-1] if states else state
                    # Loop back to get next input (let user try again)
    
    return all_states
```

---

## Changes to Slash Command Handling

Current (`slash_commands.py`):
```python
async def _handle_model(runner: InteractiveAgentRunner, args: str) -> SlashCommandResult:
    # ... validation ...
    runner.endpoint = dc_replace(runner.endpoint, provider=provider, model=model_id)
    # Persists to session
    return SlashCommandResult(message=f"Switched to: {provider}/{model_id}")
```

The problem: mutates `runner.endpoint`, caller has to know to rebuild state.

Proposed:
```python
async def _handle_model(runner: InteractiveAgentRunner, args: str) -> SlashCommandResult:
    # ... validation ...
    new_endpoint = dc_replace(runner.endpoint, provider=provider, model=model_id)
    # Return the new endpoint, let caller update state
    return SlashCommandResult(
        message=f"Switched to: {provider}/{model_id}",
        new_endpoint=new_endpoint,  # NEW: explicit return
    )
```

Then in `_get_input_result`:
```python
result = await handle_slash_command(self, command)
if result.new_endpoint:
    new_state = self._with_endpoint(current_state, result.new_endpoint)
    return InputNewState(new_state, result.message)
if result.new_session_id:
    new_state = await self._load_session(result.new_session_id)
    return InputNewState(new_state, result.message)
# etc.
```

---

## Flags to Eliminate

| Flag | Current Use | Replacement |
|------|-------------|-------------|
| `_session_switched` | Signal main loop to rebuild state from `self.initial_trajectory` | Return `InputNewState` with loaded state |
| `_environment_changed` | Signal to update environment in state | Return `InputNewState` with new environment |
| `escape_pressed` | Distinguish Escape from Ctrl+C | `AgentInterrupted` vs `AgentExited` |
| `input_pending` | Track if waiting for input | Local to `_get_input_result` |

---

## What Each Slash Command Actually Mutates

### /model
```python
# Mutates:
runner.endpoint = dc_replace(runner.endpoint, provider=provider, model=model_id)

# Persists:
await runner.session_store.update(runner.session_id, endpoint=runner.endpoint)

# No flags set - but next agent run needs updated endpoint in state
```

### /thinking
```python
# Mutates:
runner.endpoint = dc_replace(runner.endpoint, thinking=..., max_tokens=..., temperature=...)

# Persists:
await runner.session_store.update(runner.session_id, endpoint=runner.endpoint)

# No flags set - same as /model
```

### /slice
```python
# Creates child session via run_slice_command()
# Then calls runner.switch_session(child.session_id) which:
#   - runner.session_id = new_session_id
#   - runner.endpoint = session.endpoint (from storage)
#   - runner.initial_trajectory = Trajectory(messages=session.messages)
#   - runner._current_trajectory = runner.initial_trajectory
#   - runner._session_switched = True  # <-- FLAG
#   - Updates status line, re-renders chat

# Also saves/restores endpoint to keep OAuth token
```

### /env
```python
# Creates new environment
# Creates child session
# Mutates:
runner.environment = new_env
runner._environment_changed = True  # <-- FLAG

# Then calls runner.switch_session() which sets:
runner._session_switched = True  # <-- FLAG (but then immediately reset!)

# Explicitly resets _session_switched because /env wants different rebuild path:
runner._session_switched = False

# So only _environment_changed remains set
```

### Summary of What Commands Return Now vs What They Should Return

| Command | Currently Returns | Should Return |
|---------|------------------|---------------|
| /model | `SlashCommandResult(message=...)` | `SlashCommandResult(new_endpoint=...)` |
| /thinking | `SlashCommandResult(message=...)` | `SlashCommandResult(new_endpoint=...)` |
| /slice | `SlashCommandResult(message=...)` | `SlashCommandResult(new_session=..., new_trajectory=...)` |
| /env | `SlashCommandResult(message=...)` | `SlashCommandResult(new_session=..., new_environment=...)` |

---

## Migration Plan

1. **Add new types** to a new file `control_flow_types.py`
2. **Add `_get_input_result`** method that wraps current input handling
3. **Add `_run_agent_with_outcome`** method that wraps current agent running
4. **Rewrite `_run_agent_loop`** to use the new shape
5. **Update slash commands** to return changes instead of mutating
6. **Remove flags** once nothing uses them

Each step should be independently testable.

---

---

## Minimal State That Flows Through the Loop

Looking at what actually changes during the loop:

```
LOOP STATE (changes each iteration):
├── trajectory    # Messages so far - grows with each turn
├── endpoint      # Can change via /model, /thinking  
├── environment   # Can change via /env
└── session_id    # Can change via /slice, /env

TUI STATE (set once, doesn't change):
├── terminal, tui, renderer, input_component, status_line, loader_container
├── input_channel (send/receive)
├── session_store (dependency)
└── cancel_scope (outer nursery scope)
```

The key insight: **we only need to pass around the things that change**.

### Option A: Pass individual pieces

```python
# State that changes
endpoint: Endpoint
environment: Environment | None
session_id: str | None
trajectory: Trajectory  # or just messages

# InputNewState returns what changed
@dataclass(frozen=True)
class InputNewState:
    endpoint: Endpoint | None = None      # if /model or /thinking
    environment: Environment | None = None # if /env
    session_id: str | None = None         # if /slice or /env
    trajectory: Trajectory | None = None  # if /slice (new messages)
    message: str | None = None
```

### Option B: Pass AgentState (current approach with run_agent)

```python
# AgentState already bundles these:
@dataclass(frozen=True)
class AgentState:
    actor: Actor           # has trajectory + endpoint + tools
    environment: Environment | None
    session_id: str | None
    ...
```

**Recommendation: Option B** - keeps compatibility with `run_agent()` which takes `AgentState`.

The loop variable is just `state: AgentState | None`, and slash commands return new states:

```python
while True:
    input_result = await get_input_result(state)
    match input_result:
        case InputNewState(new_state, message):
            state = new_state  # Just swap in the new state
            continue
        case InputMessage(text):
            state = with_user_message(state, text)
    
    outcome = await run_agent_with_outcome(state)
    state = outcome.final_state
```

---

## Where switch_session Logic Goes

Currently `switch_session` does too much:
1. Loads session from store
2. Updates `self.endpoint`, `self.session_id`, `self.initial_trajectory`, `self._current_trajectory`
3. Sets `self._session_switched = True`
4. Updates TUI (status line, re-renders chat)

Split into:
1. **Pure function**: Load session and build new AgentState
2. **TUI update**: Separate method to update UI (status line, re-render)

```python
# Pure function - returns new state
async def load_session_state(
    session_store: SessionStore,
    session_id: str,
    current_endpoint: Endpoint,  # to preserve OAuth token
) -> AgentState:
    session, err = await session_store.get(session_id)
    # ... build and return AgentState

# TUI update - side effect
def update_tui_for_session(tui: TUI, renderer: AgentRenderer, session: AgentSession):
    renderer.clear_chat()
    renderer.render_history(session.messages)
    tui.reset_render_state()
    tui.request_render()
```

---

## Questions / TODOs

- [ ] Where should the new types live? `control_flow_types.py`? `dtypes.py`?
- [ ] Should `InputNewState` carry the full `AgentState` or just the changed parts? → **Full AgentState**
- [ ] How to handle the `_pending_user_messages` batching? (might be separate cleanup)
- [ ] Tab completion state - move into Input component?
- [ ] Should we introduce a `LoopState` that's smaller than `AgentState`? (probably not - run_agent needs AgentState)
