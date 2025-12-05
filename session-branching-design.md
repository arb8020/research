# Session Branching: Loom-Style Conversation Trees

## Current State: Linear Sessions

Config changes are logged **within the same session file**:

```jsonl
{"type": "session", "id": "abc-123", "provider": "anthropic", "model": "claude-sonnet-4"}
{"type": "message", "role": "user", "content": "hello"}
{"type": "message", "role": "assistant", "content": "hi there"}
{"type": "config_change", "endpoint": {...}, "environment_type": "coding"}
{"type": "message", "role": "user", "content": "test"}
{"type": "message", "role": "assistant", "content": "testing with new config"}
```

**Problem**: Single linear timeline. Config changes mutate session state inline.

## Proposed: Tree Structure (Loom-Style)

Each config change or message edit creates a **new session file** that branches from parent:

```
session_abc-123.jsonl (root)
├─ "hello" → assistant reply
├─ /model haiku (branch point)
│   └─ session_def-456.jsonl (child: haiku branch)
│       └─ "test with haiku"
└─ (edit message #1 to "hi")
    └─ session_ghi-789.jsonl (child: edited branch)
        └─ "hi (edited)" → different trajectory
```

### Session Graph Example

```
         [abc-123]
         (sonnet-4)
        /           \
   [def-456]     [ghi-789]
   (haiku-4)     (sonnet-4, edited msg)
       |
   [jkl-012]
   (opus-4)
```

## Design

### 1. Extended SessionHeader

```python
@dataclass(frozen=True)
class SessionHeader:
    type: str  # Always "session"
    id: str
    timestamp: str
    cwd: str
    provider: str
    model: str

    # NEW: Branching metadata
    parent_id: str | None  # Parent session ID (None for root)
    branch_point: int | None  # Message index where branch occurred
    branch_reason: str | None  # "config_change" | "message_edit" | "retry"
    branch_metadata: dict | None  # Extra context (e.g., which config changed)
```

### 2. Branching on Config Change

When `/model`, `/thinking`, or `/tools` is used:

```python
# In interactive_agent.py restart loop
if run_states and run_states[-1].stop == StopReason.CONFIG_CHANGE:
    last_state = run_states[-1]

    # Create NEW session branching from current
    new_session = branch_session_for_config_change(
        source=self.session,
        branch_after_idx=len(last_state.actor.trajectory.messages) - 1,
        working_dir=Path.cwd(),
        new_endpoint=self.endpoint,
        new_environment=self.environment,
        branch_metadata={
            "old_model": f"{old_endpoint.provider}/{old_endpoint.model}",
            "new_model": f"{self.endpoint.provider}/{self.endpoint.model}",
        }
    )

    # Switch to new session
    self.session = new_session
    self.config_changed = False
```

### 3. Branching on Message Edit

Future feature: Edit a previous message and replay:

```python
def branch_session_for_edit(
    source: Session,
    edit_message_idx: int,
    new_content: str,
    working_dir: Path,
    provider: str,
    model: str,
) -> Session:
    """Create new session with edited message and replay from that point."""

    # Load messages up to edit point
    messages = load_messages(source)

    # Create new session
    new_session = create_session(
        working_dir=working_dir,
        provider=provider,
        model=model,
        parent_id=source.session_id,
        branch_point=edit_message_idx,
        branch_reason="message_edit",
    )

    # Copy messages before edit point
    for i, msg in enumerate(messages[:edit_message_idx]):
        append_message(new_session, msg)

    # Add edited message
    edited_msg = Message(role=messages[edit_message_idx].role, content=new_content)
    append_message(new_session, edited_msg)

    return new_session
```

### 4. Session Graph Utilities

```python
def get_session_children(session: Session, working_dir: Path) -> list[Session]:
    """Find all sessions that branch from this one."""
    all_sessions = list_sessions(working_dir)
    children = []

    for s in all_sessions:
        header = load_header(s)
        if header.parent_id == session.session_id:
            children.append(s)

    return children


def get_session_root(session: Session, working_dir: Path) -> Session:
    """Walk parent links to find root session."""
    current = session

    while True:
        header = load_header(current)
        if header.parent_id is None:
            return current

        # Find parent
        parent = next(
            (s for s in list_sessions(working_dir)
             if s.session_id == header.parent_id),
            None
        )

        if parent is None:
            # Parent not found (deleted?), treat current as root
            return current

        current = parent


def build_session_tree(root: Session, working_dir: Path) -> dict:
    """Build tree structure from root session.

    Returns:
        {
            "session": root,
            "header": SessionHeader(...),
            "children": [
                {
                    "session": child1,
                    "header": SessionHeader(...),
                    "children": [...]
                },
                ...
            ]
        }
    """
    header = load_header(root)
    children = get_session_children(root, working_dir)

    return {
        "session": root,
        "header": header,
        "children": [build_session_tree(child, working_dir) for child in children]
    }
```

## Use Cases

### 1. Config Changes Create Branches

```bash
agent --model anthropic/claude-sonnet-4
> hello
> /model anthropic/claude-haiku-4  # Creates new session, branches here
> test with haiku
```

Session tree:
```
session_abc.jsonl (sonnet)
└─ session_def.jsonl (haiku, parent=abc, branch_point=1)
```

### 2. Message Editing Creates Branches

```bash
agent --continue session_abc
> /edit 2 "different question"  # Edit message #2, replay from there
```

Session tree:
```
session_abc.jsonl (original)
└─ session_xyz.jsonl (edited, parent=abc, branch_point=2)
```

### 3. Multiple Branches (Exploration)

```
session_root.jsonl
├─ session_haiku.jsonl (/model haiku)
│   └─ session_haiku_no_tools.jsonl (/tools readonly)
└─ session_opus.jsonl (/model opus)
```

### 4. Resume Any Branch

```bash
# List all sessions (shows tree)
agent --list-sessions

# Resume specific branch
agent --resume session_haiku_no_tools
```

## Benefits

### For Users

1. **Exploration without fear**: Try different models/configs without losing original trajectory
2. **Message editing**: Fix typos or rephrase and continue from that point
3. **Compare trajectories**: See how different configs affect agent behavior
4. **Git-like workflow**: Sessions are like commits, branches are config changes

### For Training Data

1. **Clean trajectories**: Each session file = one complete trajectory with consistent config
2. **No mid-trajectory config mutations**: Each file has immutable execution context
3. **Branching metadata**: Training can use `branch_reason` to filter/weight data
4. **Easy replay**: `--resume session_id` loads exact branch state

### For Research

1. **A/B testing**: Compare agent performance across model/config branches
2. **Counterfactuals**: "What if I had used haiku instead of sonnet here?"
3. **Tree search**: Explore conversation tree like MCTS for optimal paths

## Implementation Phases

### Phase 1: Config Change Branching (Immediate)

- [x] Add `CONFIG_CHANGE` stop reason
- [x] Agent restart loop
- [ ] Add `parent_id`, `branch_point`, `branch_reason` to `SessionHeader`
- [ ] Modify restart loop to call `branch_session()` instead of continuing same session
- [ ] Update session list/resume to show parent-child relationships

### Phase 2: Message Editing (Future)

- [ ] Add `/edit <index> <new_content>` command
- [ ] Implement `branch_session_for_edit()`
- [ ] Replay trajectory from edit point with new message

### Phase 3: Visualization (Future)

- [ ] TUI command to show session tree (like `git log --graph`)
- [ ] Web UI to visualize conversation tree (Loom-style)
- [ ] Interactive branch switching in TUI

### Phase 4: Advanced Features (Future)

- [ ] Branch merging (combine trajectories)
- [ ] Differential view (compare two branches)
- [ ] Export branch to markdown/html
- [ ] Training data export with branch filtering

## Session File Format After Branching

Root session:
```jsonl
{"type": "session", "id": "abc-123", "parent_id": null, "branch_point": null, "branch_reason": null, ...}
{"type": "message", "role": "user", "content": "hello"}
{"type": "message", "role": "assistant", "content": "hi there"}
```

Child session (after `/model haiku`):
```jsonl
{"type": "session", "id": "def-456", "parent_id": "abc-123", "branch_point": 1, "branch_reason": "config_change", "branch_metadata": {"old_model": "anthropic/claude-sonnet-4", "new_model": "anthropic/claude-haiku-4"}, ...}
{"type": "message", "role": "user", "content": "hello"}
{"type": "message", "role": "assistant", "content": "hi there"}
{"type": "message", "role": "user", "content": "test with haiku"}
{"type": "message", "role": "assistant", "content": "testing..."}
```

Note: Child session **duplicates** messages from parent up to branch point. This makes each session file **self-contained** and easy to replay.

## Compatibility

### Backward Compatibility

Old sessions (without `parent_id`) work fine:
- `parent_id = None` → treated as root
- Existing code continues to work

### Migration

No migration needed! Old sessions are automatically roots in the new tree structure.

## Open Questions

1. **Disk space**: Duplicating messages across branches costs storage. Accept tradeoff for simplicity?
   - **Answer**: Yes. Storage is cheap, simplicity is valuable.

2. **Garbage collection**: Should we auto-delete old branches?
   - **Answer**: Manual deletion only (like git). Add `agent --gc` command later.

3. **Branch naming**: Should branches have human-readable names?
   - **Answer**: Phase 2. Use session_id for now, add optional `branch_name` field later.

4. **Config changes mid-tool-execution**: What if user runs `/model` while tools are running?
   - **Answer**: Config change only triggers on user input (in `handle_no_tool_interactive`). Tool execution blocks config changes naturally.

## Comparison to Other Systems

### Git
- **Similar**: Sessions are commits, branches are divergence points
- **Different**: No merging (yet), automatic branching on config changes

### Loom (Google's conversational AI tool)
- **Similar**: Tree structure for conversations, visual branching UI
- **Different**: We have explicit config change branches, not just user edits

### Claude Artifacts / ChatGPT Branches
- **Similar**: Branching on message edits
- **Different**: We also branch on config changes, richer metadata

## Summary

Session branching makes config changes and message edits **first-class citizens** in the trajectory model. Instead of mutating a linear timeline, we create a **tree of immutable trajectories**, each with consistent execution context.

This aligns perfectly with:
- Immutable dataclass architecture
- Clean training data
- User exploration workflows
- Research/analysis needs

Implementation is straightforward using existing `branch_session()` primitive. Phase 1 (config change branching) can be done immediately.
