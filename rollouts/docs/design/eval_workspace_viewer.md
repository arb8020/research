# Eval Workspace Viewer

**Claude session**: `b5ac8810-f7ec-4e2e-bbe0-b6d6b49233af` (slug: logical-booping-treasure)
**Branch**: `charisma-external-eval-observability`
**Date**: 2026-03-18

---

## Goal

Make the eval viewer show not just the conversation trajectory but also the workspace
state at each turn — filesystem contents, bash history, and line-level attribution of
edits to conversation turns. Think minimal IDE embedded in the trajectory viewer.

---

## Layout

```
┌──────────────────────────────────────────────────────────────────┐
│ conversation (left ~55%)    │ workspace panel (right ~45%)       │
│                             │                                    │
│ [message list, scrolls]     │ ┌─────────────┬──────────────────┐│
│                             │ │ file tree   │ file viewer      ││
│ clicking a message          │ │ (left ~30%) │ (syntax hl'd)    ││
│ selects that turn ──────────┼─┤             │                  ││
│                             │ │ folder/file │ click line →     ││
│                             │ │ list        │ blame popover    ││
│                             │ ├─────────────┴──────────────────┤│
│                             │ │ bash history (bottom)          ││
│                             │ │ cmd | exit | stdout preview    ││
│                             │ └────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

When no workspace snapshots exist (e.g. simple API evals), the viewer falls back to
the current full-width conversation layout with no workspace panel.

---

## Event Schema

One new event type written to `events.jsonl` per sample:

```json
{
  "type": "workspace_snapshot",
  "turn": 3,
  "sample_id": "sample_0000",
  "timestamp": "2026-03-18T00:30:00Z",
  "snapshot": {
    "files": {
      "perf_takehome.py": "<full file contents>",
      "tests/submission_tests.py": "<full file contents>"
    },
    "cwd": "/workspace",
    "bash_history": [
      {
        "turn": 1,
        "cmd": "python tests/submission_tests.py",
        "stdout": "Testing forest_height=10...\nCYCLES: 3809\n",
        "stderr": "",
        "exit_code": 0
      }
    ]
  }
}
```

Alongside each snapshot, a `line_history` index is computed and served by the API
(not stored in the event log — derived at read time from the full snapshot sequence):

```json
{
  "perf_takehome.py": {
    "42": [
      {
        "turn": 3,
        "type": "edit",
        "diff": "- old line\n+ new line",
        "message_index": 47
      },
      {
        "turn": 8,
        "type": "bash",
        "cmd": "sed -i 's/old/new/' perf_takehome.py",
        "message_index": 103
      }
    ]
  }
}
```

---

## Emission Strategy

### Native rollouts agents (CodingEnvironment)

Add `emit_workspace_snapshot(turn, event_logger)` method to `CodingEnvironment`.
Called after each `exec_tool` returns. Takes a directory listing of `working_dir`,
reads all non-binary files under a size threshold, appends bash_history entry for
the most recent command.

The `Environment` protocol gets an optional hook:

```python
class Environment(Protocol):
    # Optional — environments that want workspace snapshots implement this
    async def get_workspace_snapshot(self, turn: int) -> dict[str, Any] | None: ...
```

The eval loop in `native.py` calls this after each tool execution:

```python
# In run_agent loop, after exec_tool:
snap_fn = getattr(environment, "get_workspace_snapshot", None)
if snap_fn is not None:
    snapshot = await snap_fn(turn=current_turn)
    if snapshot is not None:
        _event_logger.info(
            "workspace_snapshot",
            extra={"sample_id": sample_id, "turn": current_turn, **snapshot}
        )
```

### External drivers (Claude Code, Codex)

No live hooks available — reconstruct post-hoc from trajectory.

The server (`server.py`) reconstructs snapshots when loading a sample that has no
`workspace_snapshot` events but has a trajectory with tool calls. It walks the
trajectory, applies tool call effects to a virtual filesystem, and synthesizes
the snapshot sequence.

```python
def reconstruct_workspace_snapshots(messages: list[dict]) -> list[dict]:
    """Walk trajectory tool calls and build workspace snapshots per turn.

    Returns list of snapshot dicts in the workspace_snapshot event format.
    """
```

#### Filesystem mutation detection

Track these tool calls for file state:
- `write` / `create` → set `files[path] = content`
- `edit` → apply patch to `files[path]`, record diff in line_history
- `read` → no mutation, but confirms file exists at this turn

Track these bash patterns for side effects:
- `mv <src> <dst>` → rename key in files dict
- `cp <src> <dst>` → copy file contents
- `mkdir <dir>` → record dir exists (no file content)
- `rm <path>` → delete from files dict
- `rmdir <dir>` → delete directory
- `touch <path>` → create empty file if not exists
- `> <path>` / `>> <path>` (shell redirections) → overwrite/append to file
- `sed -i 's/old/new/' <path>` → apply sed substitution to file contents, record in line_history
- `sed -i 's/old/new/g' <path>` → same, global flag
- `tee <path>` → capture stdin to file

Mark any unrecognized bash command that touches a file as `"uncertain"` in the
snapshot — the viewer renders these with a visual indicator.

#### Turn boundary detection

For external drivers, a turn boundary = one complete assistant message in the
trajectory. Emit a snapshot after each assistant message that contained at least
one tool call affecting the filesystem.

---

## Driver ↔ Environment Coupling (Future / TODO)

**Current state**: Driver and Environment are composed externally in `attempt_executor`.
The driver (Claude Code, Codex) runs the agent against the environment's workspace
directory, but the environment has no visibility into individual turns as they happen.
Snapshots must be reconstructed post-hoc from the trajectory.

**Desired state**: The environment mediates the driver's interaction with the LLM, so
it observes each turn as it happens and can emit live snapshots.

The right model (inspired by prime-rl's `CliAgentEnv`) is an HTTP interception proxy:

```
Agent CLI ──HTTP──▶ InterceptionProxy ──HTTP──▶ LLM API
                          │
                          ▼
                   turn boundary hook
                          │
                          ▼
                   env.emit_snapshot(turn)
```

When the agent makes an LLM API call, the proxy intercepts it, triggers the
environment's snapshot hook, then forwards the request to the real API. This gives
exact per-turn state without any change to the agent CLI.

**Protocol sketch** (not yet implemented):

```python
class TurnAwareEnvironment(Protocol):
    """Extension of Environment for environments that support live turn hooks."""

    async def on_turn_start(self, turn: int, messages: list[Message]) -> None:
        """Called before each LLM API call. Emit snapshot here for live capture."""
        ...

    async def on_turn_end(self, turn: int, response: Message) -> None:
        """Called after each LLM API response."""
        ...

class InterceptionProxy:
    """HTTP proxy that intercepts agent LLM calls and triggers turn hooks.

    Run as a local server. Set ANTHROPIC_BASE_URL / OPENAI_BASE_URL to point
    the agent CLI at this proxy.

    On each intercepted request:
      1. Call env.on_turn_start(turn, messages)
      2. Forward request to real API
      3. Call env.on_turn_end(turn, response)
      4. Return response to agent
    """
```

This is the path to live workspace snapshots for external drivers without
post-hoc reconstruction. Implement after the post-hoc path is working.

---

## API Changes

New endpoint on `server.py`:

```
GET /api/trace/{run_id}/sample/{sample_id}/workspace
```

Response:

```json
{
  "snapshots": [
    {
      "turn": 0,
      "timestamp": "...",
      "files": {"perf_takehome.py": "..."},
      "cwd": "/workspace",
      "bash_history": [...]
    },
    ...
  ],
  "line_history": {
    "perf_takehome.py": {
      "42": [{"turn": 3, "type": "edit", "diff": "...", "message_index": 47}]
    }
  },
  "source": "live" | "reconstructed"
}
```

`source` tells the frontend whether these are live-captured or reconstructed,
so it can show an indicator ("⚠ reconstructed from trajectory").

---

## React Components

### `WorkspacePanel`

Top-level panel. Takes `snapshots`, `lineHistory`, `selectedTurn`, `onJumpToMessage`.

```tsx
<WorkspacePanel
  snapshots={workspaceData.snapshots}
  lineHistory={workspaceData.line_history}
  selectedTurn={selectedTurn}       // controlled by clicking conversation messages
  onJumpToMessage={(messageIndex) => scrollConversationTo(messageIndex)}
/>
```

Internal layout: `FileTree` (left) + `FileViewer` (center/right) + `BashHistory` (bottom).

### `FileTree`

Simple `<ul>` tree of file paths from the current snapshot. Clicking selects a file.
Folders inferred from path separators. Uses `lucide-react` folder/file icons (already
in deps). Highlights files that have been modified since turn 0.

### `FileViewer`

`react-syntax-highlighter` (already in deps) with:
- Language detection from file extension
- Line numbers
- Per-line click handler → opens `LineBlamePopover`
- Lines touched in the current turn highlighted (diff overlay)

```tsx
<SyntaxHighlighter
  language={detectLanguage(filename)}
  showLineNumbers
  wrapLines
  lineProps={(lineNumber) => ({
    onClick: () => onLineClick(lineNumber),
    style: {
      cursor: lineHistory[filename]?.[lineNumber] ? 'pointer' : 'default',
      background: lineHistory[filename]?.[lineNumber] ? 'rgba(59,130,246,0.1)' : undefined,
    }
  })}
>
  {fileContents}
</SyntaxHighlighter>
```

### `LineBlamePopover`

Small overlay showing all edits that touched the clicked line:
- Turn number, type (edit/bash/sed), diff snippet
- "Jump to turn" button → calls `onJumpToMessage(messageIndex)`
- Sorted most-recent first

### `BashHistory`

Scrollable list of `{cmd, exit_code, stdout preview}` entries from the current
snapshot's `bash_history`. Monospace, compact. Exit code colored green/red.
Click expands full stdout. Optionally click to jump to the conversation turn
that produced this command.

---

## Implementation Order

1. **`workspace_snapshot.py`** (new file in `rollouts/eval/`) — reconstruction logic:
   `reconstruct_workspace_snapshots(messages) -> list[dict]`
   plus `build_line_history(snapshots) -> dict`

2. **`server.py`** — add `/workspace` endpoint, call reconstruction when no live
   snapshots exist

3. **`native.py`** + **`dtypes.py`** — `get_workspace_snapshot` optional protocol
   method, call it in the eval loop

4. **`CodingEnvironment`** — implement `get_workspace_snapshot`

5. **React**: `WorkspacePanel`, `FileTree`, `FileViewer`, `LineBlamePopover`,
   `BashHistory` — wire into `RunViewer`

6. **`RunViewer`** — split layout when workspace data present, conversation message
   click sets `selectedTurn`

---

## Open Questions / Future Work

- **Snapshot size**: Full file contents per turn is fine for small workspaces (perf
  challenge, KernelBench). For SWELancer with large repos, switch to diff-only mode:
  only store changed files per snapshot, reconstruct full state by replaying from
  initial snapshot. Add `snapshot_mode: "full" | "diff"` to the workspace endpoint.

- **Binary files**: Skip binary files in snapshots. Detect via null bytes in first
  512 bytes of content.

- **Large files**: Cap individual file size at ~100KB in snapshots. Truncate with
  indicator in viewer.

- **InterceptionProxy**: See "Driver ↔ Environment Coupling" section above. Implement
  once post-hoc reconstruction is working and validated.

- **sed parsing**: `sed -i 's/pattern/replacement/g' file` needs regex application
  to reconstruct the file mutation. Use Python `re.sub` to simulate. Handle flags:
  `g` (global), line-address prefixes (`42s/old/new/`), multiple expressions (`-e`).
