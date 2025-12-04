# TUI Interactive Agent

Interactive terminal UI for running agents with streaming responses, tool execution, and user input.

## Usage

### Basic Usage

```bash
# Run with default settings (GPT-4o-mini)
python -m rollouts.frontends.tui.cli

# Specify model
python -m rollouts.frontends.tui.cli --model gpt-4o

# Use Anthropic
python -m rollouts.frontends.tui.cli --model claude-sonnet-4-5 --provider anthropic

# Custom system prompt
python -m rollouts.frontends.tui.cli --system-prompt "You are a helpful coding assistant."
```

### Programmatic Usage

```python
from rollouts.frontends.tui.interactive_agent import run_interactive_agent
from rollouts.rollouts.dtypes import Endpoint, Message, Trajectory
import trio

endpoint = Endpoint(provider="openai", model="gpt-4o-mini")
trajectory = Trajectory(messages=[Message(role="system", content="You are helpful.")])

states = trio.run(run_interactive_agent, trajectory, endpoint)
```

## Features

- ✅ Streaming text responses with markdown rendering
- ✅ Thinking/reasoning blocks (muted styling)
- ✅ Tool execution display with status colors
- ✅ Multi-line text input with cursor
- ✅ Ctrl+C cancellation support
- ✅ Message queuing while agent streams
- ✅ Differential rendering (only updates changed lines)

## Keyboard Shortcuts

- **Enter**: Submit message
- **Ctrl+C**: Cancel agent and exit
- **Ctrl+K**: Delete to end of line (in input)
- **Ctrl+U**: Delete to start of line (in input)
- **Ctrl+W**: Delete word backwards (in input)
- **Arrow keys**: Navigate text (in input)

## Architecture

- `AgentRenderer`: Connects `StreamEvent` types to TUI components
- `InteractiveAgentRunner`: Coordinates agent loop with TUI
- `Input`: Multi-line text editor component
- `AssistantMessage`: Streaming text/thinking display
- `ToolExecution`: Tool call display with results

## Missing Features (Future Work)

- [ ] Tool result event handling (currently tool results are added as messages)
- [ ] Interactive tool confirmation (currently auto-confirms)
- [ ] Input history (up/down arrows for previous messages)
- [ ] Scrollback for long conversations
- [ ] Environment-specific tool result formatting

