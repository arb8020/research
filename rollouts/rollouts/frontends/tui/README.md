# TUI Interactive Agent

Interactive terminal UI for running agents with streaming responses, tool execution, and session persistence.

## Quick Start

```bash
# Basic chat (no tools)
python -m rollouts.frontends.tui.cli --provider anthropic --model claude-sonnet-4-5

# Coding agent with file/shell tools
python -m rollouts.frontends.tui.cli --env coding --provider anthropic --model claude-sonnet-4-5

# Non-interactive query
python -m rollouts.frontends.tui.cli -p "explain this error" --provider anthropic --model claude-sonnet-4-5
```

## CLI Options

### Model Configuration
```bash
--provider {openai,anthropic}  # API provider (default: openai)
--model MODEL                   # Model name (default: gpt-4o-mini)
--api-key KEY                   # API key (or use env var)
--system-prompt TEXT            # Custom system prompt
```

### Environment
```bash
--env {none,calculator,coding}  # Tool environment (default: none)
--cwd PATH                      # Working directory for coding env
```

### Session Management
```bash
-c, --continue      # Resume most recent session
-s, --session       # Interactive session picker
-s PATH             # Resume specific session file
--no-session        # Don't persist to disk
```

### Execution Mode
```bash
-p "query"          # Non-interactive: run query, print result, exit
--max-turns N       # Maximum agent turns (default: 50)
```

## Session Persistence

Sessions are stored in `~/.rollouts/sessions/--encoded-cwd--/` as JSONL files.

```bash
# Start new session (auto-created)
python -m rollouts.frontends.tui.cli --env coding

# Continue where you left off
python -m rollouts.frontends.tui.cli --env coding -c

# Pick from previous sessions
python -m rollouts.frontends.tui.cli --env coding -s
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
cli.py                 # Entry point, arg parsing
interactive_agent.py   # Agent loop coordinator
agent_renderer.py      # StreamEvent → TUI components
sessions.py            # Session persistence (functional)
terminal.py            # Raw mode, cursor, escape sequences
tui.py                 # Differential rendering engine
theme.py               # Color definitions
components/
  input.py             # Text editor
  assistant_message.py # Streaming text display
  user_message.py      # User message display
  markdown.py          # Markdown rendering
```

## See Also

- `TUI_TODO.md` - Roadmap and missing features
- `docs/SESSION_DESIGN.md` - Session persistence design doc
