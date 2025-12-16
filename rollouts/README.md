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

--export-md [FILE]      Export to markdown
--export-html [FILE]    Export to HTML

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
