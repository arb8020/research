# rollouts

CLI agent for coding tasks with session persistence.

## Install

```bash
uv pip install -e .
```

## Quick Start

```bash
# Interactive coding agent
rollouts --env coding

# Ask a question (non-interactive)
rollouts -p "explain what this repo does"

# Continue previous session
rollouts -c --env coding
```

## Model Configuration

Models use `provider/model` format:

```bash
# Anthropic (default)
rollouts --model anthropic/claude-sonnet-4-5-20250929

# OpenAI
rollouts --model openai/gpt-4o

# With API key
rollouts --model anthropic/claude-sonnet-4-5-20250929 --api-key sk-...

# Or use environment variables
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
```

### Claude Pro/Max (OAuth)

```bash
# Login once
rollouts --login-claude

# Then use without API key (no invoices!)
rollouts --env coding

# Verify OAuth is being used
python check_auth.py
```

**Important:** When using OAuth, you should see `🔐 Using OAuth authentication (Claude Pro/Max)` in the output. If you don't see this message, you're using an API key and will get invoiced. See [docs/OAUTH.md](docs/OAUTH.md) for details.

## Environments

```bash
--env none        # No tools (default)
--env coding      # File read/write/edit + bash
--env git         # Like coding, but with isolated git history for undo
--env calculator  # Math tools (for testing)
```

The coding environment provides: `read`, `write`, `edit`, `bash`

```bash
# Specify working directory
rollouts --env coding --cwd /path/to/project

# Restrict tools (useful for read-only exploration or sub-agents)
rollouts --env coding --tools readonly    # Just read
rollouts --env coding --tools no-write    # read, edit, bash (no write)
rollouts --env coding --tools full        # All tools (default)
```

## Session Management

Sessions persist to `~/.rollouts/sessions/<session_id>/`

```bash
# Continue most recent session
rollouts -c

# Pick from recent sessions
rollouts -s

# Resume specific session
rollouts -s 20241210_143052_a1b2c3

# Don't persist session
rollouts --no-session
```

### Session Config Inheritance

When resuming a session (`-s` or `-c`), the original session's configuration is inherited automatically. CLI flags act as overrides:

```bash
# Original session used haiku + coding env
rollouts --model anthropic/claude-3-haiku-20240307 --env coding

# Resume inherits config - no need to re-specify flags
rollouts -s 20241210_143052_a1b2c3

# Override to fork with different model
rollouts -s 20241210_143052_a1b2c3 --model anthropic/claude-sonnet-4-5-20250929
# Prints: "Forking from session: 20241210_143052_a1b2c3"
```

Inherited settings: model, environment, thinking mode, confirm_tools.

### Diagnostics

Crash info and agent feedback are stored in `~/.rollouts/`:

```bash
# Crash logs (400 errors, etc)
ls ~/.rollouts/crashes/

# Agent exit surveys (task success, harness feedback)
cat ~/.rollouts/feedback/all.jsonl | jq .
```

The exit survey runs automatically when the agent pauses or exits, collecting self-reported task progress and tooling feedback.

## Stdin Input

Piped input works in both interactive and non-interactive modes:

```bash
# Interactive TUI with initial prompt from stdin
echo "explain this codebase" | rollouts --env coding

# Non-interactive (print mode)
rollouts -p "what does main.py do"
echo "explain this" | rollouts -p

# Combine with session
rollouts -c -p "what was the last thing we did"
```

## Stream JSON Output

For scripting and pipelines, use `--stream-json` to get NDJSON output:

```bash
rollouts -p "calculate 5+3" --env calculator --stream-json
```

Each line is a JSON object:

```json
{"type":"system","subtype":"init","session_id":"","tools":["add","subtract",...]}
{"type":"assistant","message":{"content":[{"type":"text","text":"I'll help..."},{"type":"tool_use","id":"...","name":"add","input":{"value":5}}]}}
{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"...","content":"Added 5...","is_error":false}]}}
{"type":"result","subtype":"success","session_id":"","num_turns":2,"duration_ms":1234}
```

Process with jq:

```bash
rollouts -p "task" --stream-json 2>/dev/null | jq '.message.content[]?'
```

## Handoff (Context Transfer)

Extract goal-directed context from a session to start fresh:

```bash
# Extract context for a specific goal
rollouts --handoff "implement the API endpoints" -s abc123 > handoff.md

# Review/edit the handoff
vim handoff.md

# Start new interactive session with that context
rollouts --env coding < handoff.md

# Or non-interactive
rollouts -p < handoff.md
```

One-liner - pipe handoff directly to new interactive session:

```bash
rollouts --handoff "fix the failing tests" -s abc123 | rollouts --env coding
```

Handoff is goal-directed: it extracts only context relevant to your next task,
not a generic summary. This keeps sessions focused.

## Slice (Session Surgery)

Slice, summarize, compact, and inject messages to create a focused child session:

```bash
# Keep only first 4 messages
rollouts --slice "0:4" -s abc123

# Keep beginning and end, summarize the middle
rollouts --slice "0:4, summarize:4:18, 18:" -s abc123

# Same, but focus the summary on a specific goal
rollouts --slice "0:4, summarize:4:18:'security review', 18:" -s abc123

# Compact tool results (shrink verbose output, keep structure)
rollouts --slice "0:4, compact:4:15, 15:" -s abc123

# Inject a user message
rollouts --slice "0:10, inject:'now focus on tests'" -s abc123

# Full example: slice, summarize, inject, then continue
rollouts --slice "0:4, summarize:4:18, 18:20, inject:'focus on tests'" -s abc123 \
    | xargs rollouts -s --env coding
```

### Percentage-Based Slicing

Use percentages when you don't know the exact message count:

```bash
# Keep last 20% of messages
rollouts --slice "80%:" -s abc123

# Keep system prompt, summarize first 80%, keep last 20%
rollouts --slice "0:2, summarize:2:80%, 80%:" -s abc123

# Compact first half, keep second half
rollouts --slice "compact:0%:50%, 50%:" -s abc123

# Self-compaction pattern (agent approaching context limits)
rollouts --slice "0:2, summarize:2:80%:'key progress', 80%:" -s abc123
```

### Slice Spec Format

Grammar:
```
slice_spec := segment ("," segment)*
segment    := range | summarize | compact | inject
range      := start ":" end?
summarize  := "summarize:" start ":" end (":" quoted_string)?
compact    := "compact:" start ":" end
inject     := "inject:" quoted_string
```

| Segment | Syntax | Description |
|---------|--------|-------------|
| Range | `0:4`, `10:`, `-5:`, `80%:` | Keep messages (Python slice notation) |
| Summarize | `summarize:4:18`, `summarize:2:80%` | Collapse range into single user message via LLM |
| Summarize+Goal | `summarize:4:18:'goal'` | Focused summary |
| Compact | `compact:0:`, `compact:0%:50%` | Shrink tool results, keep message structure |
| Inject | `inject:'message'` | Insert user message |

- **Percentages** work in any position: `80%:`, `0:50%`, `summarize:2:80%`
- **Negative indices** work for ranges: `-5:` keeps last 5 messages
- **Compact vs Summarize**: Compact preserves each message but shrinks tool output. Summarize collapses the entire range into one summary message.

### Output

```
Slicing: abc123 (25 messages, ~12,000 tokens)
Spec: 0:4, summarize:4:18:'security', 18:
Created: def456 (8 messages, ~3,200 tokens)
Reduction: 73% fewer tokens
def456
```

- **stderr**: Progress and stats
- **stdout**: New session ID (for piping)

### Compact Behavior

Compact preserves conversation structure but shrinks verbose tool outputs:

| Tool | Before | After |
|------|--------|-------|
| `read` | Full file (5000 chars) | `📄 [read: 234 lines] first line...` |
| `bash` | Full stdout (1000 lines) | First 3 lines + `... (997 more)` |
| `edit` | Full diff | `🔧 [edit: +12/-5 lines]` |
| `write` | Confirmation | `✏️ [wrote 156 lines]` |

## Context Management for Agents

When an agent's context window fills up, it can manage its own context using bash:

```bash
# Check current session size
rollouts --doctor -s $SESSION_ID

# Compact tool results to save tokens (keeps structure)
rollouts --slice "compact:0:$(rollouts --doctor -s $SESSION_ID | grep Messages | cut -d: -f2)" -s $SESSION_ID

# Summarize old work, keep recent context
rollouts --slice "0:2, summarize:2:100:'key decisions', 100:" -s $SESSION_ID | xargs rollouts -s

# Spawn a sub-agent for isolated exploration (doesn't pollute main context)
rollouts --spawn --env coding -p "explore the auth module" --no-session
```

### When to Use What

| Situation | Solution |
|-----------|----------|
| Tool results too verbose | `compact:START:END` |
| Old context no longer relevant | `summarize:START:END:'goal'` |
| Need isolated exploration | Spawn sub-agent via bash |
| Context window nearly full | Summarize first 80%, keep recent 20% |
| Starting fresh with learnings | `--handoff "goal" \| rollouts` |

### Self-Compaction Pattern

An agent approaching context limits can compact itself:

```bash
# Get current session ID from environment or parse from rollouts output
# Then create a compacted child session and continue there
NEW_SESSION=$(rollouts --slice "0:2, summarize:2:80, compact:80:150, 150:" -s $CURRENT_SESSION)
echo "Continuing in compacted session: $NEW_SESSION"
```

The original session is preserved—compaction creates a child session, so you can always trace back.

## Export

```bash
# Export session to markdown
rollouts --export-md -s abc123 > session.md

# Export to HTML
rollouts --export-html -s abc123 > session.html

# Export most recent session
rollouts -c --export-md
```

## Extended Thinking

Anthropic models use extended thinking by default:

```bash
# Disable thinking
rollouts --thinking disabled

# Enabled by default
rollouts --thinking enabled
```

## Presets

```bash
# List available presets
rollouts --list-presets

# Use a preset
rollouts --preset fast_coder
```

## All Options

```
--model MODEL           Model in "provider/model" format
--api-base URL          API base URL
--api-key KEY           API key
--system-prompt TEXT    Custom system prompt
--env ENV               Environment: none, coding, git, calculator
--tools PRESET          Tool preset: full, readonly, no-write (coding env only)
--cwd PATH              Working directory
--max-turns N           Max agent turns (default: 50)
--confirm-tools         Require confirmation before tool execution

-c, --continue          Continue most recent session
-s, --session [ID]      Resume session (picker if no ID)
--no-session            Don't persist session

-p, --print [QUERY]     Non-interactive mode (stdin if no QUERY)
--stream-json           Output NDJSON per turn (for -p mode)

--handoff GOAL          Extract context for goal to stdout
--slice SPEC            Slice/summarize/compact messages (see Slice section)
--slice-goal GOAL       Default goal for summarize segments

--export-md [FILE]      Export to markdown
--export-html [FILE]    Export to HTML

--doctor                Show session diagnostics
--trim N                Remove last N messages
--fix                   Auto-fix session issues

--thinking {enabled,disabled}  Extended thinking (Anthropic)
--preset NAME           Use agent preset
--login-claude          OAuth login for Claude Pro/Max
--logout-claude         Revoke OAuth tokens

--theme THEME           TUI theme: dark, rounded, minimal
--debug                 Debug logging
```

## See Also

- `docs/SESSION_DESIGN.md` - Session persistence and handoff design
- `rollouts/agent_presets/README.md` - Creating custom presets
