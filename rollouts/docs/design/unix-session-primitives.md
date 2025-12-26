# Unix Session Primitives: Design Doc

## Status: Draft

A proposal for Unix-style CLI primitives that enable:
1. Spawning and managing sub-agents from within agent sessions
2. Composable thread/context manipulation (slice, summarize, inject)
3. Better observability and control of running agents

---

## Existing Primitives (What We Already Have)

### Session Management
| Command | Description |
|---------|-------------|
| `-s <id>` | Resume specific session |
| `-s` (no arg) | Interactive session picker |
| `-c` | Continue most recent session |
| `--no-session` | Don't persist session |

### Input/Output
| Command | Description |
|---------|-------------|
| `-p <query>` | Non-interactive, run query and exit |
| `-p` (stdin) | Read query from stdin |
| `--stream-json` | NDJSON output per event |
| `-q` / `--quiet` | Only print final response |
| `< file.txt` | Pipe input to interactive mode |

### Export
| Command | Description |
|---------|-------------|
| `--export-md` | Export session to markdown |
| `--export-html` | Export session to HTML |
| `--handoff GOAL` | Goal-directed context extraction to stdout |

### Session Repair
| Command | Description |
|---------|-------------|
| `--doctor` | Show session diagnostics |
| `--trim N` | Remove last N messages (creates new session) |
| `--fix` | Auto-fix detected issues |

### Composable Patterns That Already Work

```bash
# 1. Handoff → new session (the canonical pattern)
rollouts --handoff "fix the tests" -s abc123 | rollouts --env coding

# 2. Non-interactive query, get answer
rollouts -p "what is 2+2" --env calculator -q

# 3. Stream JSON for scripting
rollouts -p "task" --stream-json 2>/dev/null | jq '.message.content[]?'

# 4. Stdin piping to interactive
echo "explain this codebase" | rollouts --env coding

# 5. Resume + non-interactive follow-up
rollouts -s abc123 -p "what was the last thing we discussed?"

# 6. Export for external processing  
rollouts --export-md -s abc123 > session.md

# 7. Trim and continue (poor man's slice-from-end)
rollouts --trim 10 -s abc123  # creates new session with last 10 removed
rollouts -s <new_id>          # continue from trimmed state
```

### Getting Results Programmatically (Already Possible)

```bash
# Get last assistant message via stream-json + jq:
rollouts -p "task" --env coding --stream-json 2>/dev/null \
    | jq -r 'select(.type=="assistant") | .message.content[0].text' \
    | tail -1
```

---

## What's Missing

| Capability | Current Workaround | Gap |
|------------|-------------------|-----|
| Spawn background agent | Can't (blocking only) | Need `--spawn` |
| Check running status | Can't (must wait) | Need `--status` |
| Wait for completion | Implicit (everything blocks) | Need `--wait` |
| Inject into running session | Can't | Need `--send` |
| Slice by range | Only `--trim` (from end) | Need `--slice` |
| Summarize arbitrary range | `--handoff` (whole session) | Need ranged summarize |

**The key missing piece is background/async execution.** Everything else is close.

---

## Motivation

### Current State (REPL Environment)

The REPL environment has `agent()` for spawning sub-agents:
```python
result = agent("Find bugs in auth code", context[5000:10000])
```

This is **synchronous and in-process**:
- Parent blocks until child completes
- Child session not persisted
- Can't observe/interrupt child
- No parallelism

### Desired State

Sub-agents should be **first-class sessions**:
```bash
# Spawn returns immediately with session ID
sid=$(rollouts --spawn -p "analyze auth code" < context.txt)

# Check status, send messages, wait, get results
rollouts --status $sid      # running | completed | error
rollouts --send $sid        # inject user message
rollouts --wait $sid        # block until done  
rollouts --result $sid      # get final answer
```

This enables:
- True async/parallel exploration
- Observable, interruptible work
- Persistent sessions (resumable, browseable)
- Tree visualization of exploration

---

## Part 1: Session Lifecycle Primitives

### New CLI Commands

| Command | Description | Output |
|---------|-------------|--------|
| `--spawn` | Start session in background, return ID | Session ID |
| `--status <id>` | Get session status | `running`, `completed`, `error`, `pending` |
| `--wait <id>` | Block until session completes | Exit code |
| `--result <id>` | Get final answer/last assistant message | Text |
| `--send <id>` | Inject user message into session | None |
| `--kill <id>` | Abort a running session | None |

### Usage Examples

```bash
# 1. Spawn and poll
sid=$(rollouts --spawn --env coding -p "refactor the auth module")
while [ "$(rollouts --status $sid)" = "running" ]; do
    sleep 5
done
rollouts --result $sid

# 2. Parallel exploration
sid1=$(rollouts --spawn -p "approach A" < context.txt)
sid2=$(rollouts --spawn -p "approach B" < context.txt)
rollouts --wait $sid1 $sid2  # wait for both
echo "A: $(rollouts --result $sid1)"
echo "B: $(rollouts --result $sid2)"

# 3. Interactive sub-agent
sid=$(rollouts --spawn --env coding -p "investigate the bug")
# ... later, inject guidance ...
echo "focus on the database layer" | rollouts --send $sid
rollouts --wait $sid

# 4. From within an agent session (via bash tool)
# Agent can spawn sub-agents:
bash("sid=$(rollouts --spawn --env coding -p 'fix tests' < /tmp/context.txt); echo $sid")
# ... continue other work ...
bash("rollouts --wait $sid && rollouts --result $sid")
```

### Implementation Notes

**--spawn**
- Creates session via `SessionStore.create()`
- Forks process (or uses background job)
- Returns session ID immediately
- Session runs with `--no-session` semantics initially, then saves on completion

**--status**
- Reads `session.json` for status field
- Also checks if process is still running (for `running` status)
- Could add: turn count, last activity time

**--wait**
- Polls status or uses file watcher
- Supports multiple session IDs
- Returns combined exit code (0 if all succeeded)

**--result**
- For completed sessions: return `environment._final_answer` if available
- Fallback: last assistant message content
- For REPL: the `final_answer` tool result
- Could add: `--result --format json` for structured output

**--send**
- Appends user message to `messages.jsonl`
- Sets a flag file (`.interrupt`) that the running agent checks
- Agent picks up new message on next turn start

---

## Part 2: Thread Manipulation Primitives

### The Problem

User wants to do:
> "Keep messages 0-3, summarize 4-17, keep 18-19, continue from 19 with an injected user message"

Currently available:
- `--trim N` — remove last N messages (creates new session)
- `--handoff GOAL` — LLM-generated summary for new session
- `--doctor --fix` — remove duplicate tool results

Not available:
- Slice by range
- Summarize arbitrary ranges
- Inject messages at arbitrary points
- Compose these operations

### Proposed: `--slice` Command

```bash
# Keep only messages 0-3, create new session
rollouts --slice 0:3 -s abc123

# Keep 0-3, then summarize 4-17, then keep 18-19
rollouts --slice "0:3,summarize:4:17,18:19" -s abc123

# Inject a user message at position 4
rollouts --slice "0:3,inject:'focus on tests',4:" -s abc123

# Full example: slice, summarize, inject, continue
rollouts --slice "0:3,summarize:4:17,18:19" -s abc123 \
    | rollouts --env coding
```

### Slice Spec Grammar

```
slice_spec := segment ("," segment)*
segment    := range | summarize | inject
range      := start ":" end?     # Python slice notation
summarize  := "summarize:" start ":" end
inject     := "inject:" quoted_string
```

Examples:
- `0:3` — messages 0, 1, 2
- `0:3,10:` — messages 0-2, then 10 to end
- `summarize:4:17` — replace messages 4-16 with LLM summary
- `inject:'check the tests'` — insert user message at current position

### Alternative: Composable Subcommands

Instead of one complex `--slice`, use composable subcommands:

```bash
# Get messages as JSON
rollouts messages -s abc123 --range 0:3 --format json > /tmp/msgs.json

# Summarize a range (outputs a single message)
rollouts summarize -s abc123 --range 4:17 >> /tmp/msgs.json

# Add more messages
rollouts messages -s abc123 --range 18:19 --format json >> /tmp/msgs.json

# Create new session from messages
rollouts create --from-messages /tmp/msgs.json

# Or pipe directly
rollouts messages -s abc123 --range 0:3 \
    | rollouts summarize -s abc123 --range 4:17 --append \
    | rollouts messages -s abc123 --range 18:19 --append \
    | rollouts create --from-stdin
```

This is more Unix-y but more verbose. Trade-off between power and ergonomics.

### What's Easy vs. Hard

| Operation | Current Support | Difficulty |
|-----------|----------------|------------|
| Trim last N | ✅ `--trim N` | Done |
| Handoff summary | ✅ `--handoff GOAL` | Done |
| Slice by range | ❌ | Easy - just array slice |
| Keep multiple ranges | ❌ | Easy - concat slices |
| Summarize range | ❌ | Medium - need LLM call |
| Inject message | ❌ | Easy - insert in array |
| Compose operations | ❌ | Medium - parsing/ordering |
| Continue from modified | ❌ | Easy - create child session |

### Implementation Approach

```python
# In cli.py, add:

@dataclass
class SliceSegment:
    """One segment of a slice operation."""
    type: str  # "range", "summarize", "inject"
    start: int | None = None
    end: int | None = None
    content: str | None = None  # For inject

def parse_slice_spec(spec: str) -> list[SliceSegment]:
    """Parse slice spec into segments."""
    ...

async def apply_slice(
    session: AgentSession, 
    segments: list[SliceSegment],
    endpoint: Endpoint | None = None,  # For summarize
) -> list[Message]:
    """Apply slice operations, return new message list."""
    result = []
    for seg in segments:
        if seg.type == "range":
            result.extend(session.messages[seg.start:seg.end])
        elif seg.type == "summarize":
            summary = await summarize_messages(
                session.messages[seg.start:seg.end], 
                endpoint
            )
            result.append(Message(role="user", content=f"[Summary of messages {seg.start}-{seg.end}]\n{summary}"))
        elif seg.type == "inject":
            result.append(Message(role="user", content=seg.content))
    return result
```

---

## Part 3: UI/TUI Integration

### Session Tree View

The TUI should show the session DAG, not just a list:

```
┌─ Session Tree ────────────────────────────────────────┐
│ ○ 20251225_140000 (current) - coding, 12 msgs        │
│ │                                                     │
│ ○ 20251225_135000 - haiku, branched at msg 8         │
│ │                                                     │
│ │ ○ 20251225_133000 - completed, 15 msgs             │
│ ├─╯                                                   │
│ ○ 20251225_120000 (root) - 8 msgs                    │
└───────────────────────────────────────────────────────┘
```

**Keys:**
- `↑/↓` — navigate tree
- `Enter` — resume session
- `d` — show diff from parent
- `s` — slice/edit this session
- `h` — handoff from this session

### Slice Editor (TUI)

Interactive slice editor for the current example:

```
┌─ Slice Editor ─────────────────────────────────────────┐
│ Source: 20251225_140000 (20 messages)                  │
│                                                        │
│ Operations:                                            │
│   [x] Keep 0-3 (system + initial exchange)            │
│   [x] Summarize 4-17 (14 messages → 1 summary)        │
│   [x] Keep 18-19 (recent context)                     │
│   [ ] Inject: ___________________________             │
│                                                        │
│ Preview:                                               │
│   0: [system] You are a coding assistant...           │
│   1: [user] Help me refactor auth...                  │
│   2: [assistant] I'll help with that...               │
│   3: [tool] read auth.py → 234 lines                  │
│   4: [user] [Summary] Explored auth module...         │ ← generated
│   5: [user] Now focus on the tests                    │
│   6: [assistant] ...                                  │
│                                                        │
│ [Enter] Create new session  [Esc] Cancel              │
└────────────────────────────────────────────────────────┘
```

### Running Sessions Panel

Show spawned sub-agents:

```
┌─ Sub-agents ──────────────────────────────────────────┐
│ ● abc123 - running (turn 5) - "analyze auth"         │
│ ● def456 - running (turn 2) - "check tests"          │
│ ✓ ghi789 - completed - "find TODOs"                  │
│                                                       │
│ [Enter] View  [k] Kill  [s] Send message             │
└───────────────────────────────────────────────────────┘
```

---

## Part 4: Agent Tools for Session Management

For the `coding` environment, add tools that wrap CLI commands:

```python
# In environments/coding.py, add these tools:

Tool(
    name="spawn_agent",
    description="Start a sub-agent session for a complex task. Returns session ID.",
    parameters={
        "task": "What the sub-agent should do",
        "context": "Context to provide (will be written to temp file)",
        "env": "Environment for sub-agent (default: coding)",
    }
)

Tool(
    name="check_agent",
    description="Check status of a spawned agent session.",
    parameters={
        "session_id": "Session ID from spawn_agent",
    }
)

Tool(
    name="get_agent_result",
    description="Get the result from a completed agent session.",
    parameters={
        "session_id": "Session ID from spawn_agent",
    }
)

Tool(
    name="send_to_agent",
    description="Send a message to a running agent session.",
    parameters={
        "session_id": "Session ID",
        "message": "Message to send",
    }
)
```

Implementation:
```python
async def exec_spawn_agent(self, tool_call: ToolCall, ...) -> ToolResult:
    task = tool_call.args["task"]
    context = tool_call.args.get("context", "")
    env = tool_call.args.get("env", "coding")
    
    # Write context to temp file
    context_file = Path(tempfile.mktemp(suffix=".txt"))
    context_file.write_text(context)
    
    # Spawn via subprocess
    result = await trio.run_process([
        "python", "-m", "rollouts",
        "--spawn", "--env", env,
        "--context-file", str(context_file),
        "-p", task,
    ], capture_stdout=True)
    
    session_id = result.stdout.decode().strip()
    return ToolResult(content=f"Spawned session: {session_id}")
```

---

## Part 5: Context Flow / Flame Graph Model

### Conceptual Model

Each spawned agent creates a new "stack frame":

```
[Main Session: abc123]
    context: "user's original request"
    messages: 0-10
    │
    ├── [Sub-agent: def456] (spawned at msg 5)
    │       context: subset of abc123's state
    │       messages: 0-8
    │       result: "found 3 bugs in auth.py"
    │
    └── [Sub-agent: ghi789] (spawned at msg 7)  
            context: different subset
            messages: 0-5
            │
            └── [Sub-sub-agent: jkl012] (spawned at msg 3)
                    context: even smaller subset
                    result: "confirmed: race condition"
```

### Metadata Tracking

```python
@dataclass
class AgentSession:
    # ... existing fields ...
    
    # New: spawning metadata
    spawned_by: str | None = None       # Parent session that spawned this
    spawn_context: str | None = None    # What context was passed
    spawn_task: str | None = None       # What task was requested
    spawn_result: str | None = None     # Result returned to parent
```

### Aggregation

When parent needs results from children:

```python
# Get all child sessions
children = await session_store.list(filter_tags={"spawned_by": parent_id})

# Wait for all to complete
for child in children:
    await wait_session(child.session_id)

# Aggregate results
results = [
    {"task": child.spawn_task, "result": child.spawn_result}
    for child in children
]
```

---

## Implementation Phases

### Phase 1: Core CLI Primitives (1-2 days)
- [ ] `--spawn` — start background session
- [ ] `--status` — get session status
- [ ] `--wait` — block until complete
- [ ] `--result` — get final answer

### Phase 2: Message Injection (1 day)
- [ ] `--send` — inject message into running session
- [ ] Agent checks for new messages at turn boundaries

### Phase 3: Slice Operations (2-3 days)
- [ ] `--slice` with range notation
- [ ] `summarize` segment type
- [ ] `inject` segment type
- [ ] Create child session from sliced messages

### Phase 4: Agent Tools (1-2 days)
- [ ] `spawn_agent` tool
- [ ] `check_agent` tool
- [ ] `get_agent_result` tool
- [ ] `send_to_agent` tool

### Phase 5: TUI Integration (2-3 days)
- [ ] Session tree view
- [ ] Running sub-agents panel
- [ ] Interactive slice editor

### Phase 6: Documentation & Polish (1 day)
- [ ] Update README
- [ ] Add examples
- [ ] Integration tests

---

## Open Questions

1. **Process management**: Fork vs. background job vs. subprocess?
   - Fork: Complex, zombie process risk
   - Background job: Shell-dependent
   - Subprocess: Clean but need PID tracking

2. **Status polling vs. events**: How does `--wait` work?
   - Polling: Simple, portable, slight latency
   - File watcher: Efficient, OS-dependent
   - IPC: Complex but real-time

3. **Context serialization**: How to pass context to sub-agents?
   - Temp files: Simple, works for large contexts
   - Stdin: Elegant but size-limited
   - Shared state: Complex, needs careful design

4. **Abort semantics**: What happens when you `--kill` a session?
   - Send SIGTERM to process?
   - Set status to ABORTED?
   - What about sub-sub-agents?

5. **Slice conflict handling**: What if summarize range overlaps with keep range?
   - Error out?
   - Last operation wins?
   - Merge intelligently?

---

## See Also

- `session-branching-design.md` — Tree structure for sessions
- `SESSION_DESIGN.md` — Current session format
- `repl.py` — Current in-process agent() implementation

---

## Appendix: Concrete Example Analysis

> "keep message 0-3, summarize 4-17, keep 18-19, continue from 19 with an injected user message"

### Current State

| Step | Possible Now? | How |
|------|---------------|-----|
| Keep 0-3 | ❌ No direct way | Would need to manually create new session |
| Summarize 4-17 | 🟡 Partial | `--handoff` does goal-directed summary, but includes ALL messages |
| Keep 18-19 | ❌ No | Can only trim from end, not select ranges |
| Inject message | ❌ No | No way to inject at arbitrary point |
| Continue | ✅ Yes | Resume child session with `-s` |

### What Would Need to Change

**Easy (array operations):**
- Slice by index range → `messages[0:4]`
- Multiple ranges → `messages[0:4] + messages[18:20]`
- Inject message → `messages.insert(pos, new_msg)`

**Medium (need LLM call):**
- Summarize range → call LLM with messages 4-17, get summary, create user message

**Already exists but not exposed:**
- `SessionStore.save()` can write a complete session with arbitrary messages
- `branch_point` metadata tracks where fork occurred
- Child sessions duplicate messages up to branch point

### Minimum Viable Implementation

```python
# This could work TODAY with ~50 lines of code:

async def slice_session(
    session: AgentSession,
    keep_ranges: list[tuple[int, int]],  # [(0, 4), (18, 20)]
    summarize_ranges: list[tuple[int, int]],  # [(4, 18)]
    inject_at: dict[int, str],  # {4: "focus on tests"}
    endpoint: Endpoint,
    session_store: SessionStore,
) -> AgentSession:
    """Create new session with sliced/summarized/injected messages."""
    
    # 1. Build new message list
    new_messages = []
    
    # Keep ranges
    for start, end in sorted(keep_ranges):
        new_messages.extend(session.messages[start:end])
    
    # Summarize ranges (insert as user message)
    for start, end in summarize_ranges:
        msgs_to_summarize = session.messages[start:end]
        summary = await summarize_with_llm(msgs_to_summarize, endpoint)
        new_messages.append(Message(
            role="user", 
            content=f"[Summary of previous {end-start} messages]\n{summary}"
        ))
    
    # Sort by original position (roughly)
    # ... ordering logic ...
    
    # Inject messages
    for pos, content in inject_at.items():
        new_messages.insert(pos, Message(role="user", content=content))
    
    # 2. Create child session
    child = await session_store.create(
        endpoint=session.endpoint,
        environment=session.environment,
        parent_id=session.session_id,
        branch_point=len(new_messages),
        tags={"sliced": "true"},
    )
    
    # 3. Save messages
    for msg in new_messages:
        await session_store.append_message(child.session_id, msg)
    
    return child
```

### CLI Syntax Proposal

```bash
# The full example as one command:
rollouts --slice "0:4, summarize:4:18, 18:20, inject:'now focus on the tests'" \
    -s abc123 \
    | rollouts --env coding

# Breakdown:
# 0:4           → keep messages 0,1,2,3
# summarize:4:18 → summarize messages 4-17, insert as one user message  
# 18:20         → keep messages 18,19
# inject:'...'  → add user message at end

# Output: creates new session, prints session ID
# Then pipes to new interactive session that starts with that context
```

### Alternative: Message-Level Commands

More Unix-y, more composable:

```bash
# Export specific messages as JSON
rollouts messages -s abc123 --range 0:4 > /tmp/slice.jsonl

# Summarize a range, append to file  
rollouts summarize -s abc123 --range 4:18 >> /tmp/slice.jsonl

# Add more messages
rollouts messages -s abc123 --range 18:20 >> /tmp/slice.jsonl

# Inject custom message
echo '{"role":"user","content":"now focus on tests"}' >> /tmp/slice.jsonl

# Create session from messages
rollouts create --from-messages /tmp/slice.jsonl --parent abc123

# Or all as a pipeline:
{
  rollouts messages -s abc123 --range 0:4
  rollouts summarize -s abc123 --range 4:18
  rollouts messages -s abc123 --range 18:20  
  echo '{"role":"user","content":"focus on tests"}'
} | rollouts create --from-stdin --parent abc123 | xargs rollouts -s
```

This follows the Unix philosophy better but is more verbose. The `--slice` syntax is a convenience wrapper.

---

## Implementation Status (Updated)

### ✅ Implemented: `--slice`

```bash
# Basic usage
rollouts --slice "0:4, 10:" -s abc123

# With summarize (uses LLM)
rollouts --slice "0:4, summarize:4:18, 18:" -s abc123

# With per-summarize goal
rollouts --slice "0:4, summarize:4:18:'security review', 18:" -s abc123

# With compact (shrinks tool results)
rollouts --slice "0:4, compact:4:10, 10:" -s abc123

# With inject
rollouts --slice "0:4, inject:'now focus on tests'" -s abc123

# Full example (your original use case!)
rollouts --slice "0:4, summarize:4:18:'security', 18:20, inject:'focus on tests'" -s abc123 \
    | xargs rollouts -s
```

### Slice Spec Grammar

```
segment := range | summarize | compact | inject

range     := START:END?                   # "0:4", "10:", ":5"
summarize := summarize:START:END(:GOAL)?  # "summarize:4:18", "summarize:4:18:'goal'"
compact   := compact:START:END            # "compact:5:15"
inject    := inject:QUOTED                # "inject:'message'"
```

### Files

- `rollouts/slice.py` — parsing, apply, compact, summarize
- `rollouts/cli.py` — `--slice`, `--slice-goal` args + `cmd_slice()`
- `rollouts/tests/test_slice.py` — 26 passing tests

### Compact Behavior

| Tool | Original | Compacted |
|------|----------|-----------|
| `read` | Full file (5000 chars) | `📄 [read: 234 lines, 5000 chars] first line...` |
| `write` | Confirmation | `✏️ [wrote 156 lines]` |
| `edit` | Full diff | `🔧 [edit: +12/-5 lines]` |
| `bash` | Full stdout | First 3 lines + `... (47 more lines)` |
| Other | Full content | `🔧 [tool: N lines, M chars] preview...` |
