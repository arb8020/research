# RL Training Monitor - Handoff Document

## GOAL
Fix the RL training monitor TUI for remote GPU training runs.

## CURRENT STATE
The monitor is working well now. This session fixed scroll bugs and added observability.

## WHAT WAS FIXED THIS SESSION

### 1. Scroll Bugs (from bubbles viewport comparison)
- **Problem**: j/k navigation had stale scroll state, boxes got messed up
- **Root cause**: Auto-scroll was setting scroll to `len-1` instead of `max_scroll`, and scroll wasn't clamped in update()
- **Fix**:
  - Removed broken auto-scroll formula (view already handles it)
  - Added `_clamp_scroll()`, `_scroll_down/up()` helpers
  - Scroll now properly transitions from auto-scroll to manual mode
- **Commit**: `d25fcbeb`

### 2. Stderr Warning Spam
- **Problem**: WARNING messages polluted terminal on j/k navigation
- **Fix**: Removed redundant warning - renderer already truncates silently
- **Commit**: `ab5f7c88`

### 3. Duplicate Sync After Monitor
- **Problem**: run.py tried to sync/cleanup after monitor exited, but instance was already terminated
- **Fix**: Removed redundant `_sync_and_cleanup()` call - monitor handles this internally
- **Commit**: `ab5f7c88`

### 4. File Tail Not Showing Existing Content
- **Problem**: `Sub.file_tail()` seeked to end, only showing new lines
- **Fix**: Read from beginning so existing log content is visible on attach
- **Commit**: `5fb60d3c`

### 5. Observability - Debug Logging
- **Added**: `run.jsonl` in run directory with provisioning/deploy/bootstrap events
- **Added**: `monitor.jsonl` with subscription, file_tail, dispatch events
- **Commit**: `53329f15`, `ae6ac6fc`

### 6. Training Script Not Running
- **Problem**: Config scripts were executed directly but had no `if __name__ == "__main__"` block
- **Fix**: Added `if __name__ == "__main__": train(config)` to example config
- **Commit**: `765fcdec`

### 7. HuggingFace Download Timeout
- **Problem**: Default 10s timeout caused model downloads to fail
- **Fix**: Set `HF_HUB_DOWNLOAD_TIMEOUT=300` (5 min) for SGLang and vLLM
- **Commit**: `0c2dc40e`

### 8. Mouse Support in pytui
- **Added**: `MouseEvent` message type, SGR mouse parsing
- **Added**: `mouse=True` option in App (off by default)
- **Commit**: `f532c82f`

## KEY FILES

### Monitor TUI
```
rollouts/rollouts/tui/rlmon.py          # Main monitor app (Model, update, view)
rollouts/rollouts/tui/monitor_cli.py    # CLI entry point, attach mode, SSH tunnel
```

### pytui (TUI framework)
```
pytui/pytui/text.py                     # visible_width, slice_ansi, truncate_to_width
pytui/pytui/app.py                      # Elm architecture (App, Cmd, Sub, MouseEvent)
pytui/pytui/terminal.py                 # Raw mode, mouse tracking
pytui/pytui/renderer.py                 # Differential rendering
```

### Runner
```
rollouts/run.py                         # Entry point wrapper
rollouts/rollouts/run.py                # Unified runner implementation
```

## HOW TO TEST

### Run training with TUI
```bash
cd ~/research/rollouts
uv run run.py --config examples/rl/calculator/grpo_01_01.py --provision
```

### Monitor debug logs
```bash
# While running:
tail -f results/rl/run_*/run.jsonl | jq .
tail -f results/rl/run_*/monitor.jsonl | jq .

# After run:
cat results/rl/run_YYYYMMDD-HHMMSS/run.jsonl | jq .
```

### Attach to existing run
```bash
uv run rollouts monitor --attach run_YYYYMMDD-HHMMSS
uv run rollouts monitor --latest
```

## DESIGN NOTES

### pytui Elm Architecture
- **Model**: immutable dataclass with all state
- **update(model, msg) -> (new_model, cmd)**: pure state transitions
- **view(model, width, height) -> list[str]**: pure render function
- **subscriptions(model) -> Sub**: declares what to watch (file tails, timers)

### Scroll State Management (from bubbles comparison)
- `scroll` is a line offset, only used when `auto_scroll=False`
- When `auto_scroll=True`, view takes `lines[-content_h:]` directly
- Transitioning from auto to manual: initialize scroll to current bottom position
- Always clamp scroll in update(), not just in view()

### Debug Logging
- `run.jsonl`: provisioning, deploy, bootstrap, submit events
- `monitor.jsonl`: subscriptions, file_tail, dispatch events
- Both written to run directory automatically

## POTENTIAL FUTURE WORK
- Cache `longestLineWidth` to avoid O(N) `visible_width()` calls per render
- Add wrap/truncate toggle (`w` key)
- Enable mouse wheel scrolling (infrastructure is in place)
