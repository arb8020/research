# Runtime Config Change: Agent Restart Design

## Problem

Currently, slash commands (`/model`, `/thinking`, `/tools`) update mutable state (`self.endpoint`, `self.environment`) and try to apply changes mid-run by updating the AgentState between turns. This feels hacky and fights against the immutable dataclass design.

## Better Approach: Agent Restart on Config Change

When config changes, **stop the current agent run and start a new one** with the new config.

### Why This Is Better

1. **Clean separation** - Each agent run has a fixed, immutable endpoint/environment
2. **Aligns with trajectory model** - Each trajectory has consistent execution context
3. **Matches "handoff" mental model** - Config change = handoff to new agent with same conversation history
4. **Session already tracks this** - We log ConfigChangeEntry events for replay

### Design

#### 1. Add StopReason for config changes

```python
# In dtypes.py
class StopReason(str, Enum):
    MAX_TURNS = "max_turns"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    USER_CANCELLED = "user_cancelled"
    CONFIG_CHANGE = "config_change"  # NEW
```

#### 2. Slash commands set a flag

```python
# In interactive_agent.py
async def _handle_slash_command(self, command: str) -> bool:
    if cmd == "/model":
        # ... update self.endpoint ...
        self.config_changed = True  # Set flag
        return True
```

#### 3. Return sentinel value from input handler

```python
async def _tui_input_handler(self, prompt: str) -> str:
    user_input = await self.input_receive.receive()

    if user_input.startswith("/"):
        handled = await self._handle_slash_command(user_input)
        if handled:
            if self.config_changed:
                return "__CONFIG_CHANGE__"  # Sentinel
            else:
                return await self._tui_input_handler(prompt)  # Retry input
```

#### 4. Detect sentinel in handle_no_tool

```python
async def handle_no_tool_interactive(state: AgentState, rcfg: RunConfig) -> AgentState:
    user_input = await rcfg.on_input("Enter your message: ")

    if user_input == "__CONFIG_CHANGE__":
        # Signal agent to stop for config change
        return dc_replace(state, stop=StopReason.CONFIG_CHANGE)

    # Normal path: add message and continue
    new_trajectory = Trajectory(messages=state.actor.trajectory.messages + [
        Message(role="user", content=user_input)
    ])
    new_actor = dc_replace(state.actor, trajectory=new_trajectory)
    return dc_replace(state, actor=new_actor)
```

#### 5. Outer loop restarts agent

```python
async def run(self) -> list[AgentState]:
    while True:
        agent_states = await run_agent(initial_state, run_config)

        if not agent_states:
            break

        last_state = agent_states[-1]

        if last_state.stop == StopReason.CONFIG_CHANGE:
            # Config changed! Restart agent with new config
            self.config_changed = False

            # Create new initial state with updated endpoint/environment
            initial_state = AgentState(
                actor=Actor(
                    trajectory=Trajectory(messages=last_state.actor.trajectory.messages),
                    endpoint=self.endpoint,  # Updated config
                    tools=self.environment.get_tools(),
                ),
                environment=self.environment,  # Updated environment
            )
            continue  # Restart loop with new config
        else:
            break  # Agent finished normally

    return agent_states
```

## Benefits

1. **Each agent run has immutable config** - No more updating frozen dataclasses mid-run
2. **Clean semantics** - Config change = explicit handoff point
3. **Session replay works** - Can reconstruct exact trajectory boundaries from ConfigChangeEntry events
4. **Training data is clean** - Each trajectory segment has consistent execution context

## Session File Format

```jsonl
{"type": "session", "id": "...", "provider": "anthropic", "model": "claude-haiku-4-5"}
{"type": "message", "role": "user", "content": "hello"}
{"type": "message", "role": "assistant", "content": "hi there"}
{"type": "config_change", "endpoint": {...}, "environment_type": "coding"}
{"type": "message", "role": "user", "content": "test"}
...
```

The `config_change` event marks where one agent run ended and a new one began.

## Implementation Checklist

- [ ] Add `StopReason.CONFIG_CHANGE` to dtypes
- [ ] Add `self.config_changed` flag to InteractiveAgentRunner
- [ ] Update slash command handlers to set flag
- [ ] Return sentinel from `_tui_input_handler` when config changed
- [ ] Detect sentinel in `handle_no_tool_interactive` and set stop reason
- [ ] Add outer loop in `run()` to restart agent on config change
- [ ] Test: `/model`, `/thinking`, `/tools` all restart agent cleanly
- [ ] Verify: Conversation history preserved across restarts
- [ ] Verify: Session logs config changes correctly
