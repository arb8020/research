# Frontend Abstraction

**DRI:** Chiraag
**Claude:** [this conversation]

## Context
Decouple the agent loop from TUI rendering so users can choose their preferred frontend: our Python TUI, bubbletea (Go), opentui (TypeScript), or no TUI at all.

## Out of Scope
- Building a bubbletea/opentui frontend (just define the interface)
- Web frontend / browser rendering
- Remote/networked frontends (WebSocket, etc.)

## Solution
**Input:** `StreamEvent` dataclasses emitted by agent loop
**Output:** Frontend-agnostic protocol that any renderer can implement

## Usage
```bash
# Current Python TUI (default)
rollouts --frontend=tui

# Go TUI via IPC
rollouts --frontend=bubbletea

# TypeScript TUI via IPC
rollouts --frontend=opentui

# No TUI - plain stdout
rollouts --frontend=none
```

```python
# Programmatic usage
from rollouts.frontends import TUIFrontend, NoneFrontend

frontend = TUIFrontend(theme="dark")
await run_interactive(trajectory, endpoint, frontend=frontend)
```

---

## Details

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Agent Loop                              │
│  run_agent() yields StreamEvent on each chunk               │
└──────────────────────┬──────────────────────────────────────┘
                       │ StreamEvent (JSON-serializable)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Frontend Protocol                         │
│  - handle_event(event: StreamEvent)                         │
│  - get_input() -> str                                       │
│  - show_loader(text: str)                                   │
│  - hide_loader()                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┬───────────────┐
       ▼               ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ TUIFrontend │ │ Bubbletea   │ │ OpenTUI     │ │ NoneFrontend│
│ (Python)    │ │ (Go/IPC)    │ │ (TS/IPC)    │ │ (stdout)    │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

### Frontend Protocol

```python
from typing import Protocol
from rollouts.dtypes import StreamEvent

class Frontend(Protocol):
    """Protocol for agent frontends."""

    async def start(self) -> None:
        """Initialize frontend (enter raw mode, spawn subprocess, etc.)."""
        ...

    async def stop(self) -> None:
        """Cleanup frontend (restore terminal, kill subprocess, etc.)."""
        ...

    async def handle_event(self, event: StreamEvent) -> None:
        """Handle a streaming event from the agent.

        Events include: TextDelta, ToolCallStart, ToolResultReceived, etc.
        """
        ...

    async def get_input(self, prompt: str = "") -> str:
        """Get user input. Blocks until user submits."""
        ...

    async def confirm_tool(self, tool_name: str, args: dict) -> bool:
        """Prompt user to confirm tool execution."""
        ...

    def show_loader(self, text: str) -> None:
        """Show loading indicator with text."""
        ...

    def hide_loader(self) -> None:
        """Hide loading indicator."""
        ...
```

### StreamEvent as Contract

`StreamEvent` is already the right abstraction - it's JSON-serializable and contains all info needed for rendering:

```python
# Events emitted by agent loop (dtypes.py)
StreamEvent = (
    LLMCallStart      # About to call LLM
    | StreamStart     # First token received
    | TextStart       # Text block started
    | TextDelta       # Text token received
    | TextEnd         # Text block complete
    | ThinkingStart   # Thinking block started
    | ThinkingDelta   # Thinking token received
    | ThinkingEnd     # Thinking block complete
    | ToolCallStart   # Tool call started
    | ToolCallDelta   # Tool args streaming
    | ToolCallEnd     # Tool call complete
    | ToolCallError   # Tool call failed to parse
    | ToolResultReceived  # Tool execution result
    | StreamDone      # Stream complete
    | StreamError     # Stream error
)
```

### IPC Protocol (for Go/TS frontends)

External frontends communicate over stdin/stdout with JSON lines:

```
# Python → Frontend (events)
{"type": "text_delta", "content_index": 0, "delta": "Hello", "timestamp": 1734567890.123}
{"type": "toolcall_start", "content_index": 1, "tool_call_id": "abc", "tool_name": "bash"}

# Frontend → Python (input/responses)
{"type": "input", "text": "user message here"}
{"type": "confirm", "approved": true}
```

### Frontend Implementations

#### TUIFrontend (Python, current)

```python
class TUIFrontend:
    """Current Python TUI implementation."""

    def __init__(self, theme: str = "dark", debug: bool = False):
        self.terminal = ProcessTerminal()
        self.tui = TUI(self.terminal, theme=get_theme(theme))
        self.renderer = AgentRenderer(self.tui)
        # ... existing implementation
```

#### NoneFrontend (stdout only)

```python
class NoneFrontend:
    """No TUI - just print to stdout."""

    async def handle_event(self, event: StreamEvent) -> None:
        match event:
            case TextDelta(delta=text):
                print(text, end="", flush=True)
            case ToolCallStart(tool_name=name):
                print(f"\n[Tool: {name}]", flush=True)
            case ToolResultReceived(content=content):
                print(f"[Result: {content[:100]}...]", flush=True)

    async def get_input(self, prompt: str = "") -> str:
        return input(prompt)
```

#### IPCFrontend (base for Go/TS)

```python
class IPCFrontend:
    """Base class for subprocess-based frontends."""

    def __init__(self, command: list[str]):
        self.command = command
        self.process: trio.Process | None = None

    async def start(self) -> None:
        self.process = await trio.lowlevel.open_process(
            self.command,
            stdin=trio.PIPE,
            stdout=trio.PIPE,
        )

    async def handle_event(self, event: StreamEvent) -> None:
        line = event.to_json() + "\n"
        await self.process.stdin.send_all(line.encode())

    async def get_input(self, prompt: str = "") -> str:
        # Read JSON line from subprocess stdout
        line = await self.process.stdout.receive_some(4096)
        msg = json.loads(line)
        return msg["text"]

class BubbleteaFrontend(IPCFrontend):
    def __init__(self):
        super().__init__(["rollouts-tui-go"])

class OpenTUIFrontend(IPCFrontend):
    def __init__(self):
        super().__init__(["npx", "rollouts-tui"])
```

### Refactoring InteractiveAgentRunner

Current `InteractiveAgentRunner` does too much. Split into:

1. **`AgentRunner`** - Pure agent loop, no UI knowledge
2. **`InteractiveRunner`** - Wires agent to frontend

```python
class InteractiveRunner:
    """Runs agent with a frontend."""

    def __init__(
        self,
        frontend: Frontend,
        endpoint: Endpoint,
        environment: Environment | None = None,
    ):
        self.frontend = frontend
        self.endpoint = endpoint
        self.environment = environment

    async def run(self, initial_trajectory: Trajectory) -> list[AgentState]:
        await self.frontend.start()
        try:
            # Wire up callbacks
            run_config = RunConfig(
                on_chunk=self.frontend.handle_event,
                on_input=self.frontend.get_input,
                confirm_tool=self._make_confirm_handler(),
            )

            # Run agent loop
            return await run_agent(initial_state, run_config)
        finally:
            await self.frontend.stop()
```

### Flow

1. User runs `rollouts --frontend=bubbletea`
2. CLI parses flag, instantiates `BubbleteaFrontend`
3. `InteractiveRunner` starts frontend (spawns Go process)
4. Agent loop emits `StreamEvent` → frontend.handle_event() → JSON to Go process
5. Go process renders TUI, sends input back as JSON
6. On exit, frontend.stop() kills subprocess

### Comparison of Approaches

| Aspect | Python TUI | Bubbletea (Go) | OpenTUI (TS) |
|--------|------------|----------------|--------------|
| Startup | Instant | ~50ms (binary) | ~200ms (Node) |
| Rendering | Differential, inline | Alt-screen | Alt-screen |
| Scrollback | Preserved | Lost | Lost |
| Charts | Need to build | ntcharts (braille) | Yoga layout |
| Dependencies | None (pure Python) | Single binary | Node/Bun |

### Open Questions

- [ ] Should IPC use JSON lines or msgpack for performance?
- [ ] How to handle terminal resize events across IPC boundary?
- [ ] Should frontends handle their own input loop or should Python poll?
- [ ] How to pass theme/config to external frontends?

### Files

**Read:**
- `rollouts/frontends/tui/interactive_agent.py` - Current coupled implementation
- `rollouts/frontends/tui/agent_renderer.py` - StreamEvent → Component mapping
- `rollouts/dtypes.py` - StreamEvent definitions

**Create:**
- `rollouts/frontends/protocol.py` - Frontend protocol definition
- `rollouts/frontends/none.py` - NoneFrontend (stdout)
- `rollouts/frontends/ipc.py` - IPCFrontend base class
- `rollouts/runner.py` - InteractiveRunner (decoupled from TUI)

**Modify:**
- `rollouts/cli.py` - Add --frontend flag
- `rollouts/frontends/tui/__init__.py` - Export TUIFrontend

### References

- [sst/opentui](https://github.com/sst/opentui) - TypeScript TUI with Yoga layout
- [charmbracelet/bubbletea](https://github.com/charmbracelet/bubbletea) - Go TUI framework
- [wandb LEET](https://github.com/wandb/wandb/tree/main/core/internal/leet) - Go TUI over IPC example
