# Tmux-Style Session Management

## Problem

Currently rollouts runs interactively in TUI or streams to completion in batch mode. No middle ground for:
- Starting a task detached, checking on it later
- Agent asking a question while you're not watching
- Multiple agents running in parallel (different terminals/worktrees)

## Design

Sessions are long-running, you attach/detach. Like tmux.

### States

```
running     → agent executing (in some terminal)
waiting     → agent exited, needs input to continue
completed   → agent finished successfully
failed      → agent errored out
```

### Commands

```bash
# Start detached (blocks until done or waiting)
rollouts implement spec.md
# → prints session_id, exits when done/waiting

# Start attached (TUI from the start)
rollouts implement spec.md --attach

# Send message to waiting session (resumes, blocks until done/waiting)
rollouts send <session_id> "message"
rollouts send <session_id> --file answer.md

# Attach TUI to any session
rollouts attach <session_id>
# → if waiting: answer inline, continues
# → if running: watch live
# → if completed/failed: view history

# Check status (never blocks)
rollouts status <session_id>

# List sessions
rollouts ls              # active only (running/waiting)
rollouts ls --all        # include completed/failed
```

### State on Disk

```
~/.rollouts/sessions/<session_id>/
├── session.json        # existing: endpoint, env config, status
├── messages.jsonl      # existing: append-only trajectory
├── environment.json    # existing: optional env state
└── pending_input.json  # NEW: when status=waiting
```

**pending_input.json:**
```json
{
  "type": "ask_user",
  "question": "JWT or session cookies?",
  "options": ["JWT", "Session cookies"],
  "context": "Implementing auth...",
  "timestamp": "2026-01-14T12:00:00Z"
}
```

Or for implicit wait (no tools):
```json
{
  "type": "no_tools",
  "last_message": "I've finished the initial implementation...",
  "timestamp": "2026-01-14T12:00:00Z"
}
```

### Behavior Matrix

| State | `send` | `attach` | `status` |
|-------|--------|----------|----------|
| running | queues message | TUI watch | "running" |
| waiting | delivers + resumes | TUI answer | shows question |
| completed | error | TUI view history | "completed" |
| failed | error | TUI view history | shows error |

## Implementation

### Changes Required

1. **SessionStatus enum** (dtypes.py)
   - Add `WAITING = "waiting"`

2. **FileSessionStore** (store.py)
   - `write_pending_input(session_id, data)`
   - `read_pending_input(session_id) -> data | None`
   - `clear_pending_input(session_id)`

3. **Runner exit on NEEDS_INPUT** (frontends/runner.py)
   - When `StopReason.NEEDS_INPUT`, write pending_input.json and exit
   - Don't loop back to get_input() in detached mode

4. **AskUser handler for detached mode** (environments/ask_user.py)
   - Instead of blocking on stdin, write question to pending_input.json
   - Return special result that triggers NEEDS_INPUT

5. **handle_no_tool for detached mode** (agents.py or runner)
   - When no tools and detached, write pending_input.json (type=no_tools)
   - Set StopReason.NEEDS_INPUT

6. **CLI commands** (cli.py)
   - `--send <session_id> <message>` or `--send <session_id> --file <path>`
   - `--attach <session_id>`
   - `--status [session_id]`
   - `--ls [--all]`

### No Daemon

Agent runs in foreground. "Background" = run in another terminal.
"Waiting" = process exited, state on disk, resume with `send` or `attach`.

File-based state over process management. Boring > clever.

## Non-Goals

- Process management / daemonization
- Message queuing for running agents (just attach if you want to interact)
- Automatic retry of failed sessions (explicit `retry` command if needed later)
