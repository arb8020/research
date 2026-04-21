# agent-blame

git blame for coding agents. Given a repo, reconstruct which Claude Code
(later: Codex, OpenCode) session wrote each line, by reading agent
transcripts on disk and replaying their edits.

## Status

v0: Claude Code + Codex adapters. CLI prints coverage stats, top sessions,
and per-file line-range attribution. No UI yet.

Codex edits are parsed from `apply_patch` tool calls (both `custom_tool_call`
and `function_call` shapes). Shell-based edits (heredocs, `sed`, Python
one-liners) are deliberately not parsed in v0 — they'd need a separate
`shell_edits` adapter with per-pattern confidence.

## Run it

```bash
cd /Users/chiraagbalu/research/dev/agent-blame
/Users/chiraagbalu/research/.venv/bin/python -m agent_blame.cli /path/to/repo

# Show per-line runs for a specific file
... --sample-file path/to/file.py
```

Expects Claude Code transcripts at `~/.claude/projects/<encoded-cwd>/<uuid>.jsonl`.

## How it works

1. **Adapter** (`agent_blame.adapters.claude_code`): parses JSONL -> stream of
   `FileEdit` records (source, session_id, message_uuid, tool_call_id, ts,
   path, op, new_content, old_content). Only Write / Edit / MultiEdit emit
   edits; Read / Bash / Glob are ignored.

2. **Fold** (`agent_blame.fold`): replays edits in timestamp order against a
   per-file virtual state. Each line in the virtual state is tagged with
   the `FileEdit` that introduced it. Unchanged lines preserve their prior
   attribution across subsequent edits.

3. **Reconcile** (`agent_blame.reconcile`): joins virtual state against
   current repo files by line-text match. Same-path match always wins;
   cross-file match only allowed for lines ≥20 non-whitespace chars to
   avoid `}` / blank-line noise. Unmatched lines get `edit=None` ("unknown").

## Design notes

- **No commit tagging.** We do not require sessions to be stamped into
  commit messages. We match retroactively via content, so this works on
  any repo where Claude Code transcripts exist on disk.
- **Honest failure channels.** Stale edits (agent's `old_string` doesn't
  appear in our virtual state) are recorded, not fabricated. Ambiguous
  matches are flagged, not silently resolved.
- **Fold invariants are tested.** See `tests/test_fold.py`. Unchanged
  lines across multiple sessions keep their original attribution.

## Limits / known gaps

- **Staleness dominates.** On a real repo (rollouts, ~43 CC + 42 Codex
  sessions), 72% of agent edits fail to match their `old_string` in our
  virtual state. Cause: cross-session drift — humans and uncaptured
  sessions modified files between the agent sessions we do have. Fixing
  this properly needs timestamped git integration (seed each session's
  initial file state from `git show <sha-at-timestamp>:path`) rather than
  a single current-HEAD seed. That's the next correctness improvement.
- **Low absolute coverage** (~2% on rollouts). Expected for a repo that
  long predates the indexed sessions. Coverage will climb as more
  sessions are persisted going forward.
- **Shell-based edits are opaque.** `cat > file <<EOF`, `sed -i`, Python
  one-liners. A separate `shell_edits` adapter is planned: whitelist
  common heredoc / sed patterns, emit FileEdits with a confidence field.
- **OpenCode adapter not written.** Would mirror `codex.py` structure.
- **No UI.** Next: Monaco blame gutter + per-line transcript panel.
- The 20-char cross-file threshold is empirical. A smarter approach
  would be TF-IDF scoring of line distinctiveness.
