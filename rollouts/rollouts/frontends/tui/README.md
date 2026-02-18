# TUI Interactive Agent

Interactive terminal UI for running agents with streaming responses, tool execution, and session persistence.

## Quick Start

```bash
# Interactive coding agent (default)
rollouts --env coding

# Continue most recent session
rollouts -c

# Pick from previous sessions
rollouts -s

# Non-interactive query
rollouts -p "explain this error"
```

## CLI Options

### Model Configuration
```bash
--provider {openai,anthropic,google}  # API provider
--model MODEL                          # Model name
--system-prompt TEXT                   # Custom system prompt
```

### Environment
```bash
--env {coding,calculator,none}  # Tool environment (default: coding)
--cwd PATH                      # Working directory for coding env
```

### Session Management
```bash
-c, --continue      # Resume most recent session
-s, --session       # Interactive session picker
--no-session        # Don't persist to disk
```

### Execution Mode
```bash
-p "query"          # Non-interactive: run query, print result, exit
--max-turns N       # Maximum agent turns (default: 50)
```

### Frontend Selection
```bash
--frontend {tui,none,minimal}  # Frontend type (default: tui)
```

## Session Persistence

Sessions are stored in `~/.rollouts/sessions/<session_id>/` as JSONL files.

```bash
# Start new session (auto-created)
rollouts --env coding

# Continue where you left off
rollouts -c

# Pick from previous sessions
rollouts -s
```

## Coding Environment Tools

The `--env coding` flag provides:
- **read**: Read file contents with offset/limit
- **write**: Write files (auto-creates directories)
- **edit**: Replace exact text in files
- **bash**: Execute shell commands

## Keyboard Shortcuts

- **Enter**: Submit message
- **Ctrl+C**: Cancel and exit
- **Arrow keys**: Navigate input text

## Architecture

```
tui.py                 # Differential rendering engine
agent_renderer.py      # StreamEvent → TUI components
terminal.py            # Raw mode, cursor, escape sequences
theme.py               # Color definitions
slash_commands.py      # Slash command handling
control_flow_types.py  # Control flow type definitions
components/
  input.py             # Text editor
  assistant_message.py # Streaming text display
  user_message.py      # User message display
  markdown.py          # Markdown rendering
  tool_execution.py    # Tool call display
  status_line.py       # Status bar
```

## See Also

- `docs/SESSION_DESIGN.md` - Session persistence design doc
