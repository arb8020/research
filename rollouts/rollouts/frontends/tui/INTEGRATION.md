# TUI Integration with Rollouts Agent

## Overview

The TUI rendering engine is complete and tested. It needs to be connected to the agent's `StreamEvent` system.

## What's Done

```
frontends/tui/
├── terminal.py      # Terminal I/O, raw mode, resize handling
├── tui.py           # Component, Container, TUI (differential rendering)
├── utils.py         # visible_width, wrap_text_with_ansi, truncate_to_width
└── components/
    ├── text.py      # Text with word wrapping
    ├── spacer.py    # Vertical spacing
    └── markdown.py  # Markdown rendering with ANSI styling
```

All tests pass: `python3 -m frontends.tui.test_rendering`

## What's Needed

### 1. Create `AgentRenderer` class

Connect TUI to `rollouts/agents.py` by handling `StreamEvent` types:

```python
# frontends/tui/agent_renderer.py

from rollouts.dtypes import (
    StreamEvent, TextDelta, ThinkingDelta,
    ToolCallStart, ToolCallEnd, ToolCallError,
    StreamStart, StreamDone, StreamError
)

class AgentRenderer:
    """Renders agent StreamEvents to TUI."""

    def __init__(self, tui: TUI):
        self.tui = tui
        self.chat_container = Container()
        self.current_message = None  # Text component for streaming
        self.current_tool = None     # Tool component being built

    async def handle_event(self, event: StreamEvent) -> None:
        """Route StreamEvent to appropriate handler."""
        match event:
            case TextDelta(delta=text):
                # Append to current streaming text
                self.current_message.append(text)

            case ThinkingDelta(delta=text):
                # Append to thinking block (styled differently)
                self.current_thinking.append(text)

            case ToolCallStart(name=name, id=id):
                # Create new tool execution component
                self.current_tool = ToolComponent(name)

            case ToolCallEnd(tool_call=tc):
                # Finalize tool call display
                self.current_tool.set_args(tc.args)

            case StreamDone():
                # Finalize current message
                pass

        self.tui.request_render()
```

### 2. Wire into RunConfig

The agent uses `run_config.on_chunk` callback:

```python
# In your CLI entry point

async def main():
    terminal = ProcessTerminal()
    tui = TUI(terminal)
    renderer = AgentRenderer(tui)

    run_config = RunConfig(
        on_chunk=renderer.handle_event,  # Connect here
        # ... other config
    )

    tui.start()
    try:
        await run_agent(state, run_config)
    finally:
        tui.stop()
```

### 3. Components to Add

Based on pi-mono's TUI, you'll want:

- `AssistantMessageComponent` - streaming text with thinking blocks
- `ToolExecutionComponent` - shows tool name, args, result, spinner while running
- `UserMessageComponent` - user input display
- `InputComponent` - text input with cursor, history (port from pi-mono)
- `LoaderComponent` - spinning animation for "Working..."

### 4. Input Handling

For interactive mode, need to:
1. Read user input in raw mode
2. Handle Ctrl+C (abort), Escape (cancel), etc.
3. Queue messages while agent is streaming (pi-mono pattern)

See `pi-mono/packages/tui/src/components/editor.ts` for multi-line input.

## Key Files to Reference

| Feature | Pi-mono file |
|---------|--------------|
| Event handling | `coding-agent/src/tui/tui-renderer.ts:621` (`handleEvent`) |
| Tool component | `coding-agent/src/tui/tool-execution.ts` |
| Assistant message | `coding-agent/src/tui/assistant-message.ts` |
| Input/editor | `tui/src/components/editor.ts` |
| Streaming text | `tui/src/components/text.ts` (already ported) |

## StreamEvent → TUI Mapping

| StreamEvent | TUI Action |
|-------------|------------|
| `StreamStart` | Show loader, create new message container |
| `TextDelta` | Append to current text component |
| `ThinkingDelta` | Append to thinking block (magenta/dim) |
| `ToolCallStart` | Create tool component with spinner |
| `ToolCallDelta` | Update tool args as they stream |
| `ToolCallEnd` | Finalize tool display |
| `ToolCallError` | Show error in tool component |
| `StreamDone` | Hide loader, finalize message |
| `StreamError` | Show error message |

## Testing

Run the demo to see basic rendering:
```bash
python3 -m frontends.tui.demo
```

Run unit tests (no terminal needed):
```bash
python3 -m frontends.tui.test_rendering
```
