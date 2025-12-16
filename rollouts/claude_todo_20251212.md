# Rollouts Harness Improvements

## Session UX
- [x] `--session` should inherit env/preset/model from session config (flags as overrides only)
  - Added `get_config_sync()` and `get_latest_id_sync()` to FileSessionStore
  - CLI now loads session config before creating endpoint/environment
  - Also works with `--continue` / `-c`

## Agent Exit Survey
- [x] On agent pause/exit, prompt agent for feedback on harness + task success assessment
- [x] Triggers: explicit stop/abort, natural completion, yield for user input
- [x] Write feedback to `~/.rollouts/feedback/all.jsonl` (append)
  - Created `rollouts/feedback.py` with `run_exit_survey()`, `check_should_survey()`, `collect_exit_survey()`
  - Light check first (haiku), then full survey if warranted
  - Added to `interactive_agent.py` finally block (exit) and yield point

## Crash Handling
- [x] Stop dumping full (sanitized) request on 400 errors
- [x] Write crash info to `~/.rollouts/crashes/` instead (error + traceback, not full messages)
  - Added `log_crash()` to store.py
  - Updated anthropic.py, openai_completions.py, openai_responses.py
