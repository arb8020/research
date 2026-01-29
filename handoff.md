# RL Training Monitor - Handoff

## CURRENT TASK

Deploy logging and timeout have been added. Next run should show exactly where
time is spent (or where it hangs). Run and check the logs.

Previous issue: deploy step hung for 5+ min with no visibility.
Latest run log: `results/rl/run_20260129-202241/run.jsonl`

### What was added (deploy observability)
- `deploy_code()` now has a 5-minute timeout (`DEPLOY_TIMEOUT_SECONDS`) — raises `TimeoutError` with the step name
- Every sub-step logs timing: workspace-exists check, bundle create, SFTP upload, remote clone/fetch+reset, rev-parse verify
- SFTP upload logs bundle size, progress (every ~10%), transfer rate in MB/s
- `client.py:push()` logs SSH connection time and total deploy_code time
- Timeout checker is threaded through `_create_workspace` and `_update_workspace` — checked after each step

## OPEN THREADS (not yet finished)

### 2. Scroll: remove auto_scroll dual-mode, match bubbles
- Our `auto_scroll` bool + `scroll` int dual-mode keeps causing regressions
  (transition logic between modes is where bugs appear)
- Bubbles has NO `auto_scroll` — just `YOffset` clamped to `[0, maxYOffset()]`
- Caller calls `GotoBottom()` when new content arrives if user was already at bottom
- Refactor: remove `auto_scroll` from Model, add `was_at_bottom` check in content handlers
- Added asserts to `_scroll_down/up()` to catch directional violations
- Files: `rollouts/rollouts/tui/rlmon.py`

### 3. Viewport still uses approximate height
- `_scroll_down/up()` hardcode `viewport_height=20`
- Should derive from actual terminal height, but update() doesn't know it
- Could store last terminal height in Model, or compute from view params
- Bubbles stores `Height` on the viewport Model and uses it everywhere
- Files: `rollouts/rollouts/tui/rlmon.py`

### 4. Cache longestLineWidth
- `visible_width()` now has LRU cache (1024 entries), which helps
- But we still iterate all visible lines per render for width checks
- Bubbles caches `longestLineWidth` once per `SetContent()` call
- Files: `pytui/pytui/text.py`, `rollouts/rollouts/tui/rlmon.py`

### 5. Wrap/truncate toggle
- User wants `w` key to toggle between wrap and truncate modes for log lines
- Not started
- Files: `rollouts/rollouts/tui/rlmon.py`

### 6. Monitor debug logging gaps
- update() message handling not logged (we log `dispatch` with msg_type but not what happened)
- Experiment type detection not logged
- Sync thread errors could be more detailed
- Files: `rollouts/rollouts/tui/rlmon.py`, `rollouts/rollouts/tui/monitor_cli.py`

## WHAT WAS DONE THIS SESSION

### Fixes
- Removed stderr warning spam (renderer already truncates)
- Removed redundant sync/cleanup in run.py (monitor handles internally)
- file_tail reads existing content before tailing new lines
- Added `if __name__ == "__main__"` to config script
- Increased HF download timeout to 5 minutes
- Fixed scroll bugs from bubbles viewport comparison

### Observability
- `run.jsonl` in run directory with provisioning/deploy/bootstrap/submit events
- `monitor.jsonl` with subscription, file_tail, dispatch events
- Run directory created immediately on start

### pytui improvements (matching bubbletea)
- Pre-compiled ANSI regex, LRU cache on `visible_width()`
- Buffered 256-byte input reads (was 1 byte at a time)
- Adaptive escape sequence timeout with known sequence map (~100 sequences from bubbletea)
- Bracketed paste handling (`PasteEvent`) — prevents pasted "q" from quitting
- Focus/blur events (`FocusEvent`)
- Mouse support (`MouseEvent`, off by default)

## GIT STATE

- Branch: `main`
- 143 commits ahead of origin (not pushed)
- Clean working tree (no staged changes)
- Uncommitted files (not ours): `broker/broker/providers/modal.py`, `rollouts-tmux-sessions`

### Recent commits (this session)
```
371bb9d2 feat(pytui): expand escape sequence map from bubbletea
66cc59e1 feat(pytui): add paste, focus events and sequence map
36796b73 perf(pytui): buffered input reads like bubbletea
4d9403d8 perf(pytui): optimize text width and input handling
f532c82f feat(pytui): add mouse support (off by default)
d25fcbeb fix(rlmon): fix scroll bugs found in bubbles viewport comparison
0c2dc40e fix(training): increase HF download timeout to 5 minutes
765fcdec fix(examples): add __main__ block to config script
ae6ac6fc feat(run): create local run dir immediately and log all steps
53329f15 feat(monitor): add structured debug logging for observability
5fb60d3c fix(pytui): file_tail reads existing content before tailing
ab5f7c88 fix(monitor): remove redundant warning and sync
```

## KEY FILES TO LOAD

### For deploy debugging
```
bifrost/bifrost/client.py:255-319       # push() method
bifrost/bifrost/git_sync.py             # deploy_code() — where it hangs
rollouts/rollouts/run.py:55-120         # _deploy_and_submit() with logging
```

### For TUI work
```
rollouts/rollouts/tui/rlmon.py          # Main monitor (Model, update, view, scroll)
rollouts/rollouts/tui/monitor_cli.py    # CLI, attach mode, SSH tunnel, sync loop
pytui/pytui/app.py                      # Elm architecture, input parsing, paste/focus
pytui/pytui/terminal.py                 # Raw mode, buffered input, sequence map
pytui/pytui/text.py                     # visible_width (cached), slice_ansi, truncate
pytui/pytui/renderer.py                 # Differential rendering
```

### Reference
```
/tmp/bubbletea/                         # Cloned bubbletea source (if still there)
/tmp/bubbles/                           # Cloned bubbles source (viewport impl)
~/research/docs/code_style/             # Code style docs (errors, testing, tiger style)
```

## HOW TO TEST
```bash
cd ~/research/rollouts
uv run run.py --config examples/rl/calculator/grpo_01_01.py --provision

# Monitor debug logs in another terminal:
tail -f results/rl/run_*/run.jsonl | jq .
tail -f results/rl/run_*/monitor.jsonl | jq .
```

## DESIGN NOTES

### pytui Elm Architecture
- Model: immutable dataclass with all state
- update(model, msg) -> (new_model, cmd): pure state transitions
- view(model, width, height) -> list[str]: pure render function
- subscriptions(model) -> Sub: file tails, timers

### Scroll state (from bubbles comparison)
- `scroll` is line offset, only used when `auto_scroll=False`
- When `auto_scroll=True`, view takes `lines[-content_h:]` directly
- Transitioning from auto to manual: initialize to current bottom position
- Clamp scroll in update(), not in view()

### Input handling (from bubbletea comparison)
- 256-byte buffered reads with input buffer between calls
- Known sequence map for O(1) lookup (longest prefix match)
- Heuristic fallback for unknown sequences
- Bracketed paste state machine in App._parse_input()
