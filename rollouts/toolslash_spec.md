# /env Slash Command Specification

Hot-swap environments/tools mid-conversation, similar to how `/model` and `/thinking` hot-swap model configuration.

## Overview

The `/env` command is a TUI convenience wrapper around the existing CLI primitive: `rollouts -s <session_id> --env <env_spec>`. It creates a child session branched from the current point with the new environment configuration.

## Syntax

```
/env                     # Show current environment
/env list                # List available environments
/env <env_spec>          # Swap to new environment (creates child session)
```

### Environment Spec Format

Environments are combined using `+` syntax:
```
/env coding              # Single environment
/env coding+ask_user     # Combined environments
/env coding+ask_user+browsing  # Multiple combinations
```

This mirrors the CLI: `rollouts --env coding+ask_user`

## Behavior

### `/env` (no arguments)
Display the currently active environment(s) and their names.

Example output:
```
Current environment: coding+ask_user
```

### `/env list`
Display all available environments that can be used. This is TUI-only (no CLI equivalent).

Output format: names only
```
Available environments:
  coding
  ask_user
  git_worktree
  browsing
  repl
  calculator
  ...
```

**Configuration**: Available envs are configurable in `~/.rollouts/config`. If no config exists, falls back to all registered environments from the environment registry.

### `/env <env_spec>`
Create a child session with the new environment and switch to it.

**Flow**:
1. Parse the env spec (e.g., `coding+ask_user`)
2. Check if already on the same env → show "Already using env: coding+ask_user" and no-op
3. Check for tool name collisions between combined envs → show error if collision exists
4. Create child session branched from current point
5. Child session has: same messages up to branch point, new EnvironmentConfig.type, fresh environment_state
6. Child session naturally gets the new env's system prompt
7. Auto-switch to the child session
8. Show confirmation: `Switched to session <session_id> with env coding+ask_user`

**Note**: v1 executes immediately without Y/n confirmation (matching /model, /thinking, /slice behavior). Confirmation can be added later if needed.

## Session Semantics

`/env` follows the session branching model:
- Creates a **child session** (not mutation of current session)
- Inherits conversation history up to the branch point
- New session has fresh environment with correct system prompt
- Parent session remains unchanged (can switch back)
- Equivalent to Ctrl+C then `rollouts -s <parent> --env <new_env>`

## Tool Composition

Uses existing `ComposedEnvironment` from `rollouts/environments/compose.py`:
- Multiple environments are combined via the `compose()` function
- Tool name collisions raise an error (keep existing behavior)
- Error message: `Cannot combine X+Y: both define tool Z`

## Scope & Timing

- **Interactive only**: Slash commands are a TUI/REPL feature
- **Future-only**: Swap affects only the new session going forward
- **No stale reference concerns**: Tool outputs are just text in context; Claude adapts naturally

## Error Handling

- **Unknown environment**: Show error with available alternatives
- **Tool collision**: Show which tools conflict between which envs, do not proceed
- **Instantiation failure**: Full rollback - show error, remain on current session
- **Same env as current**: No-op with message "Already using env: X"

## Tab Completion

Environment names only (no `+` syntax completion in v1):
```
/env cod<tab>  →  /env coding
/env ask<tab>  →  /env ask_user
```

## Persistence

Child session stores:
- `EnvironmentConfig.type`: The env spec string (e.g., "coding+ask_user")
- `environment_state`: Serialized environment state via `env.serialize()`

On session resume (`rollouts -s <id>`), environment is restored from these fields.

## Future Considerations (Deferred)

- **Partial modification syntax**: `/env +ask_user` (add to current), `/env -bash` (remove from current)
- **Preview mode**: `/env coding+ask_user --preview` to see tool diff before confirming

## Implementation Notes

### Files Modified

1. **`rollouts/frontends/tui/slash_commands.py`**
   - Added `env` to `BUILTIN_COMMANDS` with arg_hint `[env_spec|list]`
   - Added `_get_available_envs()` - returns sorted list from environment registry
   - Added `_get_current_env_name(runner)` - extracts env name(s) from runner
   - Added `_create_environment_from_spec(spec, working_dir)` - parses spec and creates env
   - Added `_handle_env(runner, args)` - main command handler

2. **`rollouts/frontends/tui/interactive_agent.py`**
   - Added tab completion for `/env` arguments (env names + "list")

### Key Functions Used

- `compose(*environments)` from `rollouts/environments/compose.py`
- `_get_environment_registry()` for discovering available envs
- `runner.switch_session()` for auto-switching after creating child session
- `session_store.create()` for creating child session
- `session_store.append_message()` for copying messages to child
- `session_store.update()` for persisting environment state

### Testing with Headless JSON Frontend

A new `--frontend json` option was added for programmatic testing:

```bash
# Test slash commands
echo '{"type": "user", "text": "/env list"}' | rollouts --frontend json --env coding --no-session

# Multiple commands
echo '{"type": "user", "text": "/env"}
{"type": "user", "text": "/env list"}
{"type": "user", "text": "hello"}' | rollouts --frontend json --env coding --no-session
```

Output format (NDJSON):
```json
{"type": "system", "subtype": "init", "session_id": "", "tools": [], "environment": "coding"}
{"type": "slash_command", "command": "/env list", "handled": true, "result": "Available environments:..."}
{"type": "assistant", "message": {"content": [...]}}
{"type": "result", "subtype": "success", "session_id": "", "num_turns": 1}
```

Implementation: `rollouts/frontends/headless_json.py`

### Future Work

- `~/.rollouts/config` support for `allowed_envs` (configurable env list)
- Y/n confirmation before creating child session
- Partial modification syntax (`/env +ask_user`, `/env -bash`)
