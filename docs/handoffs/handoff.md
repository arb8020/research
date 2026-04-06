# RL Training Monitor - Handoff

## CURRENT TASK

Reward stays stuck at 0 in the monitor. Need to investigate:
- Is the training loop actually computing rewards?
- Is `rollouts.jsonl` being written with reward data?
- Is the monitor parsing it correctly (`_parse_rollout`)?
- Is the metrics panel displaying reward_history?

Latest run: `results/rl/run_20260129-205205/`

## OPEN THREADS (not yet finished)

### 1. Reward stuck at 0 (BLOCKING)
- Monitor shows reward = 0 throughout training
- Could be training bug (reward not computed), logging bug (not written to rollouts.jsonl), or TUI bug (not parsed/displayed)
- Check `rollouts.jsonl` for actual reward values
- Check `_parse_rollout()` in rlmon.py
- Check metrics panel rendering of `reward_history`
- Files: `rollouts/rollouts/tui/rlmon.py` (_parse_rollout, view), `rollouts/rollouts/training/grpo.py`

### 2. Viewport uses approximate height
- `_VIEWPORT_HEIGHT = 20` hardcoded in scroll helpers and `_at_bottom()`
- Should derive from actual terminal height via `Resize` messages
- Bubbles stores `Height` on the viewport Model and uses it everywhere
- rlmon doesn't handle `Resize` messages yet
- Files: `rollouts/rollouts/tui/rlmon.py`

### 3. Cache longestLineWidth
- `visible_width()` has LRU cache (1024 entries), which helps
- But we still iterate all visible lines per render for width checks
- Bubbles caches `longestLineWidth` once per `SetContent()` call
- Files: `pytui/pytui/text.py`, `rollouts/rollouts/tui/rlmon.py`

### 4. Wrap/truncate toggle
- `w` key to toggle between wrap and truncate modes for log lines
- Not started
- Files: `rollouts/rollouts/tui/rlmon.py`

### 5. Monitor debug logging gaps
- update() message handling not logged (we log `dispatch` with msg_type but not what happened)
- Experiment type detection not logged
- Sync thread errors could be more detailed
- Files: `rollouts/rollouts/tui/rlmon.py`, `rollouts/rollouts/tui/monitor_cli.py`

### 6. ty type errors
- Worktree at `/Users/chiraagbalu/research-ty-errors` on branch `fix/ty-errors`
- 5284 diagnostics total, almost all in submodules
- Our code: missing `Callable` import in `rollouts/rollouts/run.py` (fixed on main, not on branch yet)
- Pre-existing errors in broker, bifrost callers

## WHAT WAS DONE (across sessions)

### This session
- Deploy logging + 5-min timeout in `git_sync.py` (every step timed, SFTP progress logged)
- Drain all buffered input per frame (fixes j/k lag on hold)
- Log box titles show `5,000+` when at cap
- Fixed `_scroll_down` regression (was scrolling wrong direction from auto-scroll)
- Refactored scroll: removed `auto_scroll` bool, now single-offset like bubbles
  - `_at_bottom()`, `_goto_bottom()` derived from scroll position
  - Content handlers check at_bottom before append, goto_bottom after if was following
  - Scroll offset adjusted when lines trimmed at 5000 cap (no viewport drift)
- `Callable` import fix in `run.py`

### Previous session
- Removed stderr warning spam, redundant sync/cleanup
- file_tail reads existing content before tailing
- `if __name__ == "__main__"` in config script
- HF download timeout increased to 5 minutes
- Scroll bugs fixed from bubbles viewport comparison
- `run.jsonl` + `monitor.jsonl` structured logging
- pytui: pre-compiled ANSI regex, LRU cache, buffered input, paste/focus events, mouse support, ~100 escape sequences from bubbletea

## GIT STATE

- Branch: `main`
- 147 commits ahead of origin (not pushed)
- Uncommitted files (not ours): `broker/broker/providers/modal.py`, `rollouts-tmux-sessions`
- Worktree: `fix/ty-errors` at `/Users/chiraagbalu/research-ty-errors`

### Recent commits
```
86eff168 fix(rlmon): adjust scroll offset when log lines trimmed at cap
c5803c0c refactor(rlmon): remove auto_scroll, match bubbles single-offset model
b7cd8aa1 fix(tui): drain all buffered input per frame and show 5000+ line count
3733a462 feat(deploy): add step-level timing logs and 5-min timeout to deploy_code
371bb9d2 feat(pytui): expand escape sequence map from bubbletea
66cc59e1 feat(pytui): add paste, focus events and sequence map
36796b73 perf(pytui): buffered input reads like bubbletea
4d9403d8 perf(pytui): optimize text width and input handling
f532c82f feat(pytui): add mouse support (off by default)
d25fcbeb fix(rlmon): fix scroll bugs found in bubbles viewport comparison
```

## KEY FILES TO LOAD

### For reward debugging
```
rollouts/rollouts/tui/rlmon.py:344-362  # _parse_rollout
rollouts/rollouts/tui/rlmon.py:148-175  # Model (reward_history, metrics)
rollouts/rollouts/tui/rlmon.py:780-800  # view metrics rendering
rollouts/rollouts/training/grpo.py      # GRPO training loop, reward computation
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

### For deploy debugging
```
bifrost/bifrost/client.py:255-319       # push() method (now with timing logs)
bifrost/bifrost/git_sync.py             # deploy_code() with timeout + step logging
rollouts/rollouts/run.py:55-120         # _deploy_and_submit() with JSONL logging
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

# Check if rollouts.jsonl has reward data:
cat results/rl/run_*/rollouts.jsonl | jq '.reward' | head -20
```

## DESIGN NOTES

### pytui Elm Architecture
- Model: immutable dataclass with all state
- update(model, msg) -> (new_model, cmd): pure state transitions
- view(model, width, height) -> list[str]: pure render function
- subscriptions(model) -> Sub: file tails, timers

### Scroll state (matching bubbles)
- `scroll` is line offset (like bubbles `YOffset`), always clamped to `[0, max_scroll]`
- No `auto_scroll` bool — `_at_bottom(model)` derived from `scroll >= max_scroll`
- Content handlers: check `_at_bottom()` before append, `_goto_bottom()` after if was following
- `_append_log()` returns `(new_lines, trimmed_count)` — scroll adjusted by trimmed
- `_scroll_down/up()` are pure arithmetic + clamp with directional asserts
- `_VIEWPORT_HEIGHT = 20` approximation until Resize is wired into Model

### Input handling (from bubbletea comparison)
- 256-byte buffered reads with input buffer between calls
- All available input drained per frame (not one-per-frame)
- Known sequence map for O(1) lookup (longest prefix match)
- Heuristic fallback for unknown sequences
- Bracketed paste state machine in App._parse_input()
