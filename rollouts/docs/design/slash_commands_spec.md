# Slash Commands for TUI

**Status:** Draft (Exploratory, Near-term)  
**Author:** @chiraagbalu  
**Date:** 2024-12-29

## Summary

Add slash commands as an in-band control plane for the TUI, allowing users to fluidly switch models, toggle thinking, manipulate sessions via slice, and invoke custom commands without leaving the conversation flow.

## Motivation

Currently, changing settings mid-session requires:
1. `Ctrl+C` to exit
2. Re-run with new flags (`rollouts -c --model opus`)
3. Lose flow and mental context

Slash commands provide a **parallel control plane** - frequent, fluid switching between talking to the LLM and controlling the system, while keeping the nice underlying unix-like CLI primitives for scripting/automation.

**Design principle:** Slash commands are syntactic sugar over "exit + restart with new flags" - they persist to the session and behave identically to CLI flag changes.

---

## Design Decisions

### Mental Model
- **Parallel control plane**, not escape hatches
- Frequent use expected - constantly switching between LLM conversation and system control
- Slash commands = hot reload of settings that would otherwise require restart

### Persistence
- Changes made via slash commands **persist to the session**
- `--continue` remembers `/model` switches, `/thinking` toggles, etc.
- Equivalent to `Ctrl+C` then re-running with new flags

### Effect Timing
- Changes take effect on **next message**, not retroactively
- `/model opus` doesn't re-run the previous exchange

### Arguments
- **Simple positional arguments**: `/model anthropic/claude-opus-4`, `/thinking 10000`
- No complex CLI-style flags
- Goal: cover 80% of common modifications

### Error Handling (Typos)
- **Suggest correction**: Show "Did you mean /model?" but require confirmation
- Don't silently auto-correct, don't pass unknown commands to LLM

### Model Name Resolution
- **Exact match only** with tab autocomplete
- Require full provider/model: `/model anthropic/claude-opus-4-20250514`
- Tab completion makes this ergonomic

### Custom Commands
- Support file-based commands like pi-mono
- `~/.rollouts/commands/` (user-level)
- `.rollouts/commands/` (project-level)
- YAML frontmatter for description, `$1 $2 $@` argument substitution
- **LLM prompts only** - file commands expand to text sent to LLM
- Built-in commands do local actions (clean separation)

### Tab Completion
- **Ghost text** for single match (minimal, fast)
- Show faded completion inline, Tab accepts

### Output Display
- Command output appears **in chat area** but is **not part of conversation**
- Visual "ghost" messages that don't get sent to LLM
- Keeps UI clean while maintaining context

### Command History
- No separate command history needed
- Slash commands are short enough to retype

### Streaming Behavior
- Commands typed during streaming are **queued**
- Execute after stream completes

---

## V1 Commands (Minimal)

### `/model [provider/model]`
Switch model mid-conversation.

```
/model                                    # Show current model
/model anthropic/claude-opus-4-20250514   # Switch to Opus 4
/model openai/gpt-5.1-codex               # Switch to GPT-5 Codex
```

Tab completion for model names from registry. Change persists to session.

### `/thinking [on|off|budget]`
Toggle extended thinking mode.

```
/thinking               # Show current status
/thinking on            # Enable with default budget (10000)
/thinking off           # Disable
/thinking 10000         # Enable with specific token budget
```

**Error handling:** If current model doesn't support thinking (check `ModelMetadata.reasoning`), show error and block: "Cannot enable thinking for gpt-4o - model does not support extended thinking."

### `/slice [spec]`
Interactive session slicing. Runs the existing `--slice` CLI functionality in-band.

```
/slice                  # Show message count (same as /slice count)
/slice count            # Show message count
/slice 0:4              # Keep messages 0-3, create child session
/slice 0:4, summarize:4:18, 18:
                        # Full slice spec with summarization
```

When a slice spec is provided:
1. Parse and validate the spec
2. Show preview of operations
3. Execute slice, creating child session
4. **Auto-switch** to child session
5. Show new session ID in TUI (no parent lineage indicator)

---

## File-Based Custom Commands

### Location
- `~/.rollouts/commands/*.md` - user-level (available in all sessions)
- `.rollouts/commands/*.md` - project-level (available in that directory)

### Format
```markdown
---
description: Review a file for issues
---
Please review the following file for potential issues:

$1

Focus on:
- Logic errors
- Edge cases  
- Performance concerns
```

### Usage
```
/review src/main.py
```

Expands to the markdown content with `$1` replaced by `src/main.py`, then sent to LLM as user message.

### Argument Substitution
- `$1`, `$2`, ... - positional arguments
- `$@` - all arguments joined with spaces

---

## Implementation

### Architecture

```
rollouts/frontends/tui/
├── slash_commands.py      # Core slash command handling
│   ├── SlashCommand       # Command definition dataclass
│   ├── SlashCommandResult # Execution result
│   ├── handle_builtin()   # Built-in command dispatch
│   ├── load_file_commands() # Load custom .md commands
│   └── expand_file_command() # Argument substitution
└── interactive_agent.py   # Integration point
    └── _handle_slash_command() # Entry point (currently stub)
```

### Entry Point

In `interactive_agent.py`, the stub already exists:

```python
async def _handle_slash_command(self, command: str) -> bool:
    """Handle slash commands.
    
    Returns:
        True if command was handled, False if it should be passed to LLM
    """
    # Currently returns False (passes to LLM)
```

### Result Types

```python
@dataclass
class SlashCommandResult:
    handled: bool = True          # If False, pass original text to LLM
    message: str | None = None    # Display to user (ghost message)
    expanded_text: str | None = None  # For file commands, send this to LLM
    persist_changes: dict | None = None  # Session metadata to persist
```

### Ghost Messages

Need new renderer method to display output that isn't part of conversation:

```python
class AgentRenderer:
    def add_ghost_message(self, content: str) -> None:
        """Add message that displays but isn't in conversation history."""
        # Render with distinct styling (dimmed? different border?)
        # Don't add to trajectory
```

### Model Switching

```python
async def handle_model_command(runner: InteractiveAgentRunner, args: str) -> SlashCommandResult:
    if not args:
        # Show current model
        return SlashCommandResult(
            message=f"Current model: {runner.endpoint.provider}/{runner.endpoint.model}"
        )
    
    # Parse provider/model
    if "/" not in args:
        return SlashCommandResult(
            message=f"Invalid model format. Use: /model provider/model-id"
        )
    
    provider, model_id = args.split("/", 1)
    
    # Validate model exists
    from rollouts.models import get_model
    model_meta = get_model(provider, model_id)
    if not model_meta:
        return SlashCommandResult(
            message=f"Unknown model: {args}\nUse Tab to autocomplete available models."
        )
    
    # Update endpoint (persists to session via RunConfig)
    runner.endpoint = replace(runner.endpoint, provider=provider, model=model_id)
    
    return SlashCommandResult(
        message=f"Switched to: {provider}/{model_id}",
        persist_changes={"provider": provider, "model": model_id}
    )
```

### Thinking Toggle

```python
async def handle_thinking_command(runner: InteractiveAgentRunner, args: str) -> SlashCommandResult:
    from rollouts.models import get_model
    
    model_meta = get_model(runner.endpoint.provider, runner.endpoint.model)
    
    if not args:
        # Show current status
        budget = runner.endpoint.thinking_budget
        status = f"on (budget: {budget})" if budget else "off"
        return SlashCommandResult(message=f"Thinking: {status}")
    
    # Parse argument
    if args == "off":
        new_budget = None
    elif args == "on":
        new_budget = 10000  # Default
    else:
        try:
            new_budget = int(args)
        except ValueError:
            return SlashCommandResult(
                message=f"Invalid thinking argument: {args}\nUse: /thinking [on|off|budget]"
            )
    
    # Check model supports thinking
    if new_budget and model_meta and not model_meta.reasoning:
        return SlashCommandResult(
            message=f"Cannot enable thinking for {runner.endpoint.model} - model does not support extended thinking."
        )
    
    runner.endpoint = replace(runner.endpoint, thinking_budget=new_budget)
    
    status = f"on (budget: {new_budget})" if new_budget else "off"
    return SlashCommandResult(
        message=f"Thinking: {status}",
        persist_changes={"thinking_budget": new_budget}
    )
```

### Slice Command

```python
async def handle_slice_command(runner: InteractiveAgentRunner, args: str) -> SlashCommandResult:
    from rollouts.slice import parse_slice_spec, apply_slice, slice_session
    
    messages = runner.trajectory.messages
    
    if not args or args == "count":
        return SlashCommandResult(
            message=f"Session has {len(messages)} messages"
        )
    
    # Parse spec
    try:
        segments = parse_slice_spec(args)
    except ValueError as e:
        return SlashCommandResult(message=f"Invalid slice spec: {e}")
    
    # Execute slice
    if runner.session_store is None:
        return SlashCommandResult(
            message="Cannot slice: no session store configured"
        )
    
    child, err = await run_slice_command(
        session=...,  # Need to get AgentSession
        spec=args,
        endpoint=runner.endpoint,
        session_store=runner.session_store,
    )
    
    if err:
        return SlashCommandResult(message=f"Slice failed: {err}")
    
    # Switch to child session
    runner.switch_session(child.session_id)
    
    return SlashCommandResult(
        message=f"Created child session: {child.session_id}"
    )
```

---

## Tab Completion

### Ghost Text Implementation

When user types `/mod`, show faded `el` completion:

```python
class Input:
    def _get_completion(self, text: str) -> str | None:
        if not text.startswith("/"):
            return None
        
        prefix = text[1:]  # Remove /
        
        # Get all commands
        commands = get_all_slash_commands()  # built-in + file
        
        matches = [c.name for c in commands if c.name.startswith(prefix)]
        
        if len(matches) == 1:
            # Single match - return completion suffix
            return matches[0][len(prefix):]
        
        return None
    
    def render(self) -> str:
        completion = self._get_completion(self.text)
        if completion:
            # Render text + faded completion
            return self.text + dim(completion)
        return self.text
```

### Model Autocomplete

For `/model anth`, complete to `/model anthropic/`:

```python
def _get_model_completion(self, text: str) -> str | None:
    if not text.startswith("/model "):
        return None
    
    arg = text[7:]  # After "/model "
    
    # Get all provider/model combinations
    from rollouts.models import get_providers, get_models
    
    all_models = []
    for provider in get_providers():
        for model in get_models(provider):
            all_models.append(f"{provider}/{model.id}")
    
    matches = [m for m in all_models if m.startswith(arg)]
    
    if len(matches) == 1:
        return matches[0][len(arg):]
    
    return None
```

---

## Resolved Design Questions

### 1. Session State Access for /slice

**Decision:** Load from store on-demand.

`/slice` needs `AgentSession` but `InteractiveAgentRunner` only has `session_id` + `session_store`. Solution:

```python
async def handle_slice_command(runner, args):
    session, err = await runner.session_store.get(runner.session_id)
    if err or not session:
        return SlashCommandResult(message=f"Cannot load session: {err}")
    child, err = await run_slice_command(session, args, runner.endpoint, runner.session_store)
    ...
```

This matches how the CLI does it in `cmd_slice()` (cli.py:960). The store's `get()` method (store.py:234) loads full session with messages.

### 2. Persistence Mechanism

**Decision:** Extend `store.update()` to accept endpoint.

Currently `store.update()` (store.py:255) only accepts status/environment_state/reward/tags. Need to add `endpoint` parameter so `/model` changes persist to session.json.

When user does `rollouts -c`, the CLI inherits model from session config (cli.py:1203-1206) unless explicitly overridden. So persisting endpoint to session.json gives the expected behavior.

### 3. Endpoint Propagation

**Decision:** Update `handle_no_tool_interactive` to use `self.endpoint`.

`Endpoint` is frozen - `/model` creates new instance stored in `runner.endpoint`. The issue is `handle_no_tool_interactive` (interactive_agent.py:648) creates new actor from `state.actor`, which has the OLD endpoint.

**Fix:** One-line change in `handle_no_tool_interactive`:

```python
# Before (line 648):
new_actor = dc_replace(state.actor, trajectory=new_trajectory)

# After:
new_actor = dc_replace(state.actor, trajectory=new_trajectory, endpoint=self.endpoint)
```

This ensures any `/model` changes are picked up when the next user message is processed. The flow:

1. User types `/model opus` → `self.endpoint` updated
2. User types "hello" → `handle_no_tool_interactive` creates new actor with `endpoint=self.endpoint`
3. `run_agent` uses new endpoint for LLM call

---

## Reference Implementations

### pi-mono (TypeScript)

Primary reference for slash command design.

| Component | File | Key Lines |
|-----------|------|-----------|
| Slash command types | `/tmp/pi-mono/packages/coding-agent/src/core/slash-commands.ts` | L6-14: `FileSlashCommand` interface |
| Parse frontmatter | same | L19-43: `parseFrontmatter()` |
| Arg substitution | same | L76-90: `substituteArgs()` with `$1`, `$@` |
| Load from dirs | same | L102-175: `loadCommandsFromDir()` recursive |
| Expand command | same | L202-218: `expandSlashCommand()` |
| Built-in handlers | `/tmp/pi-mono/packages/coding-agent/src/modes/interactive/interactive-mode.ts` | L651-730: `/settings`, `/model`, `/export`, etc. |
| Command list | same | L157-170: Built-in command definitions |
| Model resolver | `/tmp/pi-mono/packages/coding-agent/src/core/model-resolver.ts` | L49-93: `tryMatchModel()` fuzzy matching |
| Agent session | `/tmp/pi-mono/packages/coding-agent/src/core/agent-session.ts` | L451-488: `prompt()` with slash expansion |

### mistral-vibe (Python)

Reference for Textual-based completion UI.

| Component | File | Key Lines |
|-----------|------|-----------|
| Slash controller | `/tmp/mistral-vibe/vibe/cli/autocompletion/slash_command.py` | L11-90: `SlashCommandController` |
| Key handling | same | L45-70: Tab/Enter/Up/Down handling |
| Completion popup | `/tmp/mistral-vibe/vibe/cli/textual_ui/widgets/chat_input/container.py` | L49-56: Command entries setup |

### rollouts (current)

| Component | File | Notes |
|-----------|------|-------|
| Stub entry point | `rollouts/frontends/tui/interactive_agent.py:160` | `_handle_slash_command()` returns False |
| Slash detection | same:222-226 | `if user_input.startswith("/")` |
| Slice logic | `rollouts/slice.py` | Full slice spec parsing and execution |
| Model registry | `rollouts/models.py` | `get_model()`, `get_providers()` |
| Session store | `rollouts/store.py:255` | `update()` - needs endpoint support |
| Session config | `rollouts/cli.py:1180-1230` | `_apply_session_config()` - model inheritance |

---

## Future Commands (V2+)

From pi-mono for reference:
- `/compact` - Manual context compression
- `/session` - Show session info/stats
- `/branch` - Branch from previous message
- `/new` - Start fresh session
- `/resume` - Switch to different session
- `/export` - Export session to file
- `/copy` - Copy last message to clipboard
