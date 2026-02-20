# Message Queue UX: Porting from pi-mono to rollouts

## Overview

pi-mono has a sophisticated message queue system that distinguishes between two types of queued messages when the agent is busy:

1. **Steering messages** — interrupt the agent mid-run, delivered after current tool execution
2. **Follow-up messages** — wait until the agent finishes completely, then delivered

rollouts currently has a single undifferentiated queue. This document scopes the work to port pi-mono's richer model.

---

## Current State

### pi-mono (`packages/coding-agent`)

**Keybindings** (`src/core/keybindings.ts:66-67`):
- `Enter` while agent is busy → steering (interrupt)
- `Alt+Enter` → follow-up (wait for completion)
- `Alt+Up` → dequeue (restore all queued messages to editor)

**Agent-level queues** (`src/core/agent-session.ts`):
- `_steeringMessages: string[]` — pending steering messages
- `_followUpMessages: string[]` — pending follow-up messages
- `steer(text)` — queues message that interrupts at tool boundaries
- `followUp(text)` — queues message that waits for turn to complete
- `clearQueue()` — returns `{ steering: string[], followUp: string[] }`

**UI display** (`src/modes/interactive/interactive-mode.ts:2860-2876`):
```typescript
for (const message of steeringMessages) {
    const text = theme.fg("dim", `Steering: ${message}`);
    this.pendingMessagesContainer.addChild(new TruncatedText(text, 1, 0));
}
for (const message of followUpMessages) {
    const text = theme.fg("dim", `Follow-up: ${message}`);
    this.pendingMessagesContainer.addChild(new TruncatedText(text, 1, 0));
}
```

**Queue modes** (settings):
- `steeringMode: "all" | "one-at-a-time"` — deliver all steering messages at once, or one per turn
- `followUpMode: "all" | "one-at-a-time"` — same for follow-ups

### rollouts (`rollouts/frontends`)

**Current implementation** (`tui_frontend.py`, `components/pending_messages.py`):
- Single `_pending_messages` list (no steering/follow-up distinction)
- `Enter` while busy → queues message (no interrupt semantics)
- `Up arrow` when editor empty → restores queued messages
- No `Alt+Enter` or `Alt+Up` keybindings

**Missing**:
- No agent-level `steer()` / `follow_up()` methods
- No queue type distinction in UI
- No keybinding system (all keys hardcoded)

---

## Design

### Option A: Full pi-mono parity

Port the complete steering/follow-up model with agent-level integration.

**Pros**: Feature parity, enables interrupt semantics
**Cons**: Requires changes to agent/runner, more complex

### Option B: UI-only distinction

Track steering vs follow-up in the frontend only, but deliver all messages the same way.

**Pros**: Simpler, isolated to frontend
**Cons**: No actual interrupt behavior, just visual distinction

### Recommendation: Option A (phased)

Phase 1: Frontend keybindings and UI
Phase 2: Agent-level queue integration

---

## Phase 1: Frontend Changes

### 1.1 Add keybinding detection

pytui (`pytui/pytui/terminal.py`) already has `alt+up` mapped in `KNOWN_SEQUENCES`:
```python
"\x1b[1;3A": "alt+up",
```

Need to add `alt+enter`:
```python
# In KNOWN_SEQUENCES dict, add:
"\x1b\r": "alt+enter",   # ESC + CR
"\x1b\n": "alt+enter",   # ESC + LF (some terminals)
```

**Note**: pytui's escape sequence disambiguation (`_pending_esc_deadline`, 30ms timeout) correctly handles ESC followed by another key. When ESC is followed by `\r` within the timeout, it reads as `\x1b\r`.

**File**: `rollouts/frontends/tui_frontend.py`

In `_run_input_loop()`, add handling for the new sequences:

```python
# After existing Escape handling, before routing to component:
if isinstance(msg, KeyPress):
    if msg.key == "\x1b\r" or msg.key == "\x1b\n":  # Alt+Enter
        self._handle_follow_up_submit()
        continue
    if msg.key == "\x1b[1;3A":  # Alt+Up
        self._handle_dequeue()
        continue
```

### 1.2 Split PendingMessages queue

**File**: `rollouts/frontends/tui/components/pending_messages.py`

Change from single list to two lists:

```python
class PendingMessages(Component):
    def __init__(self, ...):
        self._steering: list[str] = []
        self._follow_up: list[str] = []

    def add_steering(self, text: str) -> None:
        self._steering.append(text)

    def add_follow_up(self, text: str) -> None:
        self._follow_up.append(text)

    def get_all(self) -> tuple[list[str], list[str]]:
        return (list(self._steering), list(self._follow_up))

    def render(self, width: int) -> list[str]:
        # Show "Steering: {msg}" for steering messages
        # Show "Follow-up: {msg}" for follow-up messages
        ...
```

### 1.3 Update TUIFrontend to use new queues

**File**: `rollouts/frontends/tui_frontend.py`

```python
def _handle_input_submit(self, text: str) -> None:
    # If agent is busy:
    #   Enter → add_steering(text)
    #   Alt+Enter → add_follow_up(text)
    # If agent is idle:
    #   Send immediately
    ...

def _handle_alt_up(self) -> None:
    # Restore all queued messages to editor (both steering and follow-up)
    ...
```

### 1.4 Add hint for Alt+Enter

Update the restore hint to show both keybindings:

```python
restore_hint="↑ restore, Alt+Enter to queue follow-up"
```

---

## Phase 2: Agent Integration

### 2.1 Add queue methods to runner

**File**: `rollouts/frontends/runner.py`

```python
class InteractiveRunner:
    def __init__(self, ...):
        self._steering_queue: list[str] = []
        self._follow_up_queue: list[str] = []

    async def steer(self, text: str) -> None:
        """Queue a steering message (interrupts at tool boundary)."""
        self._steering_queue.append(text)
        # Signal agent to check queue after current tool

    async def follow_up(self, text: str) -> None:
        """Queue a follow-up message (waits for turn to complete)."""
        self._follow_up_queue.append(text)
```

### 2.2 Integrate with agent loop

The agent loop needs to:
1. After each tool execution, check for steering messages
2. If steering message exists, skip remaining tools and deliver it
3. After turn completes, deliver follow-up messages

This requires changes to the core agent loop in the driver.

---

## Files to Modify

### Phase 1 (Frontend only)

| File | Changes |
|------|---------|
| `pytui/pytui/terminal.py` | Add `"\x1b\r": "alt+enter"` to KNOWN_SEQUENCES |
| `rollouts/frontends/tui/components/pending_messages.py` | Split into steering/follow-up queues, update render |
| `rollouts/frontends/tui_frontend.py` | Add Alt+Enter/Alt+Up handling, wire up new queue methods |

### Phase 2 (Agent integration)

| File | Changes |
|------|---------|
| `rollouts/frontends/runner.py` | Add steer()/follow_up() methods |
| `rollouts/frontends/protocol.py` | Add SteeringMessage/FollowUpMessage types |
| `rollouts/drivers/*.py` | Integrate queue checking into agent loop |

---

## Key Differences to Handle

### Terminal escape sequence detection

pi-mono uses Kitty keyboard protocol which sends unambiguous sequences. rollouts/pytui uses standard ANSI with a 30ms ESC timeout for disambiguation.

pytui's `Terminal.read_input()` already handles this correctly:
- If ESC is followed by another byte within 30ms, they're read together
- `\x1b\r` (Alt+Enter) will be returned as a single string
- `\x1b[1;3A` (Alt+Up) is already in `KNOWN_SEQUENCES`

Just need to add `"\x1b\r": "alt+enter"` to `KNOWN_SEQUENCES` in `pytui/pytui/terminal.py`.

### Queue delivery semantics

pi-mono's steering actually interrupts — the agent skips remaining tools. rollouts would need driver changes to support this. For Phase 1, we can queue the messages but deliver them all after the turn completes (no actual interrupt).

---

## Testing

1. Start agent, submit a prompt that triggers multiple tool calls
2. While tools are running:
   - Press `Enter` with text → should show "Steering: {text}"
   - Press `Alt+Enter` with text → should show "Follow-up: {text}"
3. Press `Alt+Up` → both messages should restore to editor
4. Submit empty editor → messages should be sent (Phase 1: all at once; Phase 2: with proper semantics)

---

## References

- pi-mono keybindings: `/tmp/pi-mono/packages/coding-agent/src/core/keybindings.ts`
- pi-mono agent queue: `/tmp/pi-mono/packages/coding-agent/src/core/agent-session.ts:916-979`
- pi-mono UI display: `/tmp/pi-mono/packages/coding-agent/src/modes/interactive/interactive-mode.ts:2860-2876`
- rollouts pending messages: `rollouts/frontends/tui/components/pending_messages.py`
- rollouts frontend: `rollouts/frontends/tui_frontend.py`
