# Multi-Frontend Architecture for Rollouts

## Vision

Build **multiple frontends** (web, CLI, React Ink) that all consume the **same backend interface**, enabling:
- **Kernel-gen coworkers** to use rollouts synchronously for agent development
- **Researchers** to use rollouts in batch mode for training
- **Different UX preferences** without duplicating backend logic

---

## Current State Analysis

### **What Rollouts Already Has (Perfect Foundation!)**

#### **1. Backend Interface: Callback-Based Streaming**

Rollouts already has a **clean backend interface** via `RunConfig`:

```python
# rollouts/dtypes.py:471-492
@dataclass(frozen=True)
class RunConfig:
    # 🔑 This is your backend interface!
    on_chunk: Callable[[StreamChunk], Awaitable[None]]
    emit_event: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None

    # Other callbacks
    on_input: Callable[[str], Awaitable[str]] = ...
    confirm_tool: Callable[[ToolCall, AgentState, RunConfig], ...] = ...
    handle_tool_error: Callable[[ToolResult, AgentState], ...] = ...
    on_step_start: Callable[[AgentState], ...] = ...
    handle_stop: Callable[[AgentState], ...] = ...
    handle_no_tool: Callable[[AgentState, RunConfig], ...] = ...
```

**StreamChunk types** (rollouts/dtypes.py:63-66):
```python
@dataclass(frozen=True)
class StreamChunk:
    kind: str  # "token", "tool_call_complete", "tool_result", "thinking"
    data: Mapping[str, Any]
```

**This is already a perfect backend interface!** Frontends just implement different `on_chunk` handlers.

#### **2. Existing Web Frontend**

You already have `rollouts/frontend/` with:
- `server.py` - HTTP server with API endpoints
- `index.html` - Web UI
- Live streaming via polling (subprocess monitoring)

**Current architecture:**
```
Web Browser → HTTP Server → subprocess → rollouts agent
                ↓
          Polls output logs
```

---

## Proposed Architecture: Backend-Frontend Separation

### **Core Principle: Pure Functions + Callbacks**

Rollouts' functional design is **perfect** for multi-frontend support:

```python
# Backend (pure function)
async def agent_loop(
    state: AgentState,
    run_config: RunConfig  # 🔑 Frontend injected here
) -> AgentState:
    # Pure logic - no knowledge of frontend
    next_actor = await rollout(state.actor, run_config.on_chunk)  # Frontend callback!
    return updated_state

# Frontend 1: Web (already exists)
async def web_on_chunk(chunk: StreamChunk):
    await websocket.send(chunk.to_json())

# Frontend 2: Rich CLI (proposed)
async def rich_on_chunk(chunk: StreamChunk):
    if chunk.kind == "token":
        console.print(chunk.data["text"], end="")
    elif chunk.kind == "tool_call_complete":
        console.print(f"[cyan]🔧 {chunk.data['name']}[/cyan]")

# Frontend 3: React Ink (proposed)
async def ink_on_chunk(chunk: StreamChunk):
    update_react_state(chunk)
```

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Web Frontend │  │ Rich CLI     │  │ React Ink    │      │
│  │              │  │              │  │              │      │
│  │ - Browser UI │  │ - Terminal   │  │ - Terminal   │      │
│  │ - WebSockets │  │ - Rich lib   │  │ - React      │      │
│  │ - Live stream│  │ - Colors     │  │ - Components │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          │    Implement on_chunk callbacks     │
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼─────────────┐
│         ▼                  ▼                  ▼             │
│              PRESENTATION ADAPTER LAYER                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ WebAdapter   │  │ RichAdapter  │  │ InkAdapter   │      │
│  │              │  │              │  │              │      │
│  │ - Session    │  │ - Live       │  │ - State      │      │
│  │   mgmt       │  │   console    │  │   updates    │      │
│  │ - WebSocket  │  │ - Progress   │  │ - Components │      │
│  │ - Event log  │  │   bars       │  │   rendering  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          │    All use same backend interface    │
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼─────────────┐
│         ▼                  ▼                  ▼             │
│                    BACKEND LAYER                            │
│                                                             │
│  Pure Functions (No UI Knowledge):                         │
│  - agent_loop(state, run_config) -> state                 │
│  - rollout(actor, on_chunk) -> actor                      │
│  - execute_tool(tool_call, state) -> result               │
│                                                             │
│  Data Structures:                                          │
│  - AgentState, Trajectory, Message                        │
│  - RunConfig (callbacks injected by frontend)             │
│                                                             │
│  Persistence:                                              │
│  - Checkpointing (FileCheckpointStore)                    │
│  - Trajectory storage (JSONL)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### **Backend Interface (Already Exists!)**

The backend exposes **callback-based interface** via `RunConfig`:

```python
# rollouts/dtypes.py (already exists)
@dataclass(frozen=True)
class RunConfig:
    """Backend interface - frontends implement these callbacks."""

    # Primary streaming interface
    on_chunk: Callable[[StreamChunk], Awaitable[None]]

    # Event emission (optional)
    emit_event: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None

    # User input (for interactive frontends)
    on_input: Callable[[str], Awaitable[str]] = default_stdin_handler

    # Tool confirmation (for interactive frontends)
    confirm_tool: Callable[[ToolCall, AgentState, RunConfig], ...] = default_confirm_tool

# Backend emits chunks (already exists)
async def rollout_openai(actor: Actor, on_chunk: Callable) -> Actor:
    # Stream tokens
    await on_chunk(StreamChunk("token", {"text": "Hello"}))

    # Stream tool calls
    await on_chunk(StreamChunk("tool_call_complete", {
        "name": "bash",
        "args": {"command": "ls"}
    }))

    # Stream tool results
    await on_chunk(StreamChunk("tool_result", {
        "ok": True,
        "content": "file1.py\nfile2.py"
    }))

    return updated_actor
```

**No changes needed to backend!** It's already perfect.

---

### **Frontend 1: Web (Already Exists)**

**Current implementation:**
- `rollouts/frontend/server.py` - HTTP server with SSE/polling
- `rollouts/frontend/index.html` - Web UI

**Architecture:**
```python
# server.py (simplified)
class DevLoopServer:
    def do_POST(self):
        if path == "/api/launch":
            # Launch agent in subprocess
            process = subprocess.Popen([
                "python", "-m", "rollouts.run_eval",
                "--config", config_name
            ])

            # Store process for polling
            _active_runs[run_id] = {
                "process": process,
                "output_file": output_path,
            }

    def do_GET(self):
        if path.startswith("/api/run-status/"):
            # Poll subprocess output
            with open(output_file) as f:
                lines = f.readlines()
            return {"status": "running", "output": lines}
```

**Flow:**
```
Browser → POST /api/launch → Spawn subprocess
   ↓
   └─→ Poll GET /api/run-status → Read output file → Send to browser
```

**Limitations:**
- ❌ No real-time streaming (polls file)
- ❌ Subprocess isolation (can't access agent state)
- ❌ No WebSocket support

---

### **Frontend 2: Rich CLI (Proposed)**

**Goal:** Interactive terminal UI for synchronous agent development

**Architecture:**

```python
# rollouts/cli/rich_frontend.py (new file)

import asyncio
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown

from rollouts.dtypes import StreamChunk, RunConfig, AgentState
from rollouts.agents import agent_loop

class RichAdapter:
    """Adapter that renders StreamChunks to Rich terminal."""

    def __init__(self):
        self.console = Console()
        self.current_message = []
        self.tool_calls = []
        self.turn_count = 0

    async def on_chunk(self, chunk: StreamChunk) -> None:
        """Handle StreamChunk and render to terminal."""

        if chunk.kind == "token":
            # Stream tokens to terminal
            text = chunk.data["text"]
            self.console.print(text, end="", style="green")

        elif chunk.kind == "thinking":
            # Show thinking in different style
            text = chunk.data["text"]
            self.console.print(text, end="", style="dim cyan")

        elif chunk.kind == "tool_call_complete":
            # Show tool call
            name = chunk.data["name"]
            args = chunk.data["args"]
            self.console.print(f"\n🔧 [cyan]{name}[/cyan]({args})")

        elif chunk.kind == "tool_result":
            # Show tool result
            ok = chunk.data["ok"]
            content = chunk.data["content"]
            status = "✓" if ok else "✗"
            style = "green" if ok else "red"
            self.console.print(f"{status} [{'style}]{content[:100]}...[/]")

    async def on_input(self, prompt: str) -> str:
        """Get user input."""
        return self.console.input(f"[yellow]{prompt}[/yellow] ")

    async def confirm_tool(
        self, tc: ToolCall, state: AgentState, run_config: RunConfig
    ) -> Tuple[AgentState, ToolConfirmResult]:
        """Confirm tool execution."""
        self.console.print(f"\n▶️  Execute [cyan]{tc.name}({tc.args})[/cyan]?")
        self.console.print("  [green][y][/green] Yes, execute")
        self.console.print("  [red][n][/red] No, provide feedback")

        response = await self.on_input("Choice")

        if response.lower() == "y":
            return state, ToolConfirmResult(proceed=True)
        else:
            feedback = await self.on_input("Feedback")
            return state, ToolConfirmResult(proceed=False, feedback=feedback)

    def get_run_config(self) -> RunConfig:
        """Create RunConfig with Rich callbacks."""
        return RunConfig(
            on_chunk=self.on_chunk,
            on_input=self.on_input,
            confirm_tool=self.confirm_tool,
        )


# CLI entry point
async def run_interactive(
    environment_name: str,
    model: str = "gpt-4o",
    max_turns: int = 10
):
    """Run interactive agent session with Rich UI."""

    # Create adapter
    adapter = RichAdapter()
    run_config = adapter.get_run_config()

    # Initialize environment
    from rollouts.environments.calculator import CalculatorEnvironment
    env = CalculatorEnvironment()

    # Create initial state
    from rollouts.dtypes import AgentState, Actor, Endpoint
    endpoint = Endpoint(
        base_url="https://api.openai.com/v1",
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Get initial prompt from user
    adapter.console.print("[bold cyan]Rollouts Interactive Agent[/bold cyan]")
    user_prompt = await adapter.on_input("Prompt")

    messages = [
        Message(role="system", content=env.get_system_message()),
        Message(role="user", content=user_prompt),
    ]

    actor = Actor(endpoint=endpoint, messages=messages, tools=env.get_tools())
    state = AgentState(
        actor=actor,
        environment=env,
        turn_idx=0,
        max_turns=max_turns,
    )

    # Run agent loop (backend)
    final_state = await agent_loop(state, run_config)

    # Show final result
    adapter.console.print("\n[bold green]✓ Agent completed[/bold green]")
    adapter.console.print(f"Stop reason: {final_state.stop}")


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="calculator")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(run_interactive(args.env, args.model, args.max_turns))
```

**Usage:**
```bash
$ python -m rollouts.cli.rich_frontend --env calculator

Rollouts Interactive Agent
Prompt: Calculate 2^10 + 5^3

🔧 calculator({"expr": "2**10 + 5**3"})
✓ 1149

The answer is 1149.

✓ Agent completed
Stop reason: SUCCESS
```

**Benefits:**
- ✅ Real-time streaming (not polling)
- ✅ Direct access to backend (no subprocess)
- ✅ Rich formatting (colors, progress bars, panels)
- ✅ Interactive (tool confirmation, user feedback)

---

### **Frontend 3: React Ink (Proposed)**

**Goal:** Terminal UI with React component model

**Architecture:**

```typescript
// rollouts/cli/ink_frontend.tsx (new file)

import React, { useState, useEffect } from 'react';
import { render, Box, Text, Newline } from 'ink';
import Spinner from 'ink-spinner';

interface Message {
  role: string;
  content: string;
  tool_calls?: any[];
}

interface StreamChunk {
  kind: string;
  data: any;
}

const AgentRunner: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [currentToken, setCurrentToken] = useState('');

  useEffect(() => {
    // Connect to backend via Python bridge
    const eventSource = new EventSource('/api/stream');

    eventSource.onmessage = (event) => {
      const chunk: StreamChunk = JSON.parse(event.data);

      if (chunk.kind === 'token') {
        setCurrentToken(prev => prev + chunk.data.text);
      } else if (chunk.kind === 'tool_call_complete') {
        // Add tool call to messages
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: '',
          tool_calls: [chunk.data]
        }]);
      } else if (chunk.kind === 'tool_result') {
        setMessages(prev => [...prev, {
          role: 'tool',
          content: chunk.data.content
        }]);
      }
    };

    return () => eventSource.close();
  }, []);

  return (
    <Box flexDirection="column">
      <Box borderStyle="round" borderColor="cyan" padding={1}>
        <Text bold color="cyan">Rollouts Agent</Text>
      </Box>

      <Box flexDirection="column" marginTop={1}>
        {messages.map((msg, idx) => (
          <MessageComponent key={idx} message={msg} />
        ))}

        {streaming && (
          <Box>
            <Text color="green">
              <Spinner type="dots" />
              {' '}
              {currentToken}
            </Text>
          </Box>
        )}
      </Box>
    </Box>
  );
};

const MessageComponent: React.FC<{ message: Message }> = ({ message }) => {
  if (message.role === 'user') {
    return (
      <Box borderStyle="single" borderColor="blue" padding={1} marginY={1}>
        <Text bold color="blue">User:</Text>
        <Newline />
        <Text>{message.content}</Text>
      </Box>
    );
  }

  if (message.role === 'assistant') {
    return (
      <Box borderStyle="single" borderColor="green" padding={1} marginY={1}>
        <Text bold color="green">Assistant:</Text>
        <Newline />
        <Text>{message.content}</Text>

        {message.tool_calls && message.tool_calls.map((tc, idx) => (
          <Box key={idx} marginTop={1}>
            <Text color="cyan">🔧 {tc.name}({JSON.stringify(tc.args)})</Text>
          </Box>
        ))}
      </Box>
    );
  }

  return null;
};

render(<AgentRunner />);
```

**Python bridge:**
```python
# rollouts/cli/ink_bridge.py

import asyncio
import json
from aiohttp import web

from rollouts.dtypes import StreamChunk, RunConfig
from rollouts.agents import agent_loop

class InkBridge:
    """Bridge between Python backend and React Ink frontend."""

    def __init__(self):
        self.app = web.Application()
        self.app.router.add_get('/api/stream', self.stream_handler)
        self.subscribers = []

    async def stream_handler(self, request):
        """SSE endpoint for streaming chunks to Ink frontend."""
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        await response.prepare(request)

        # Add to subscribers
        queue = asyncio.Queue()
        self.subscribers.append(queue)

        try:
            while True:
                chunk = await queue.get()
                data = json.dumps(chunk.to_dict())
                await response.write(f"data: {data}\n\n".encode())
        finally:
            self.subscribers.remove(queue)

        return response

    async def on_chunk(self, chunk: StreamChunk) -> None:
        """Forward chunks to all subscribers."""
        for queue in self.subscribers:
            await queue.put(chunk)

    def get_run_config(self) -> RunConfig:
        return RunConfig(on_chunk=self.on_chunk)
```

---

### **Comparison: Three Frontends**

| Feature | Web Frontend | Rich CLI | React Ink |
|---------|-------------|----------|-----------|
| **Rendering** | Browser HTML/CSS | Terminal ANSI | Terminal React |
| **Streaming** | WebSocket/SSE | Direct callbacks | SSE bridge |
| **Process** | Subprocess (isolated) | In-process | In-process |
| **Latency** | ~100ms (network) | <1ms (direct) | ~10ms (bridge) |
| **Interactivity** | Form inputs | readline | React state |
| **Complexity** | High (HTTP server) | Low (direct) | Medium (bridge) |
| **Multi-session** | ✅ Yes | ❌ No | ❌ No |
| **Remote access** | ✅ Yes | ❌ No | ❌ No |
| **Styling** | CSS | ANSI colors | Ink components |
| **State mgmt** | Server-side | Adapter class | React hooks |

---

## Session Management Architecture

### **Agent-Runner's Approach: SessionManager**

Agent-runner has a **stateful SessionManager** for incremental message saving:

```python
# agent-runner/src/agentrunner/core/session.py
class SessionManager:
    """Manages CLI sessions with incremental message saving."""

    def __init__(self, workspace: Workspace):
        self.sessions_dir = Path("~/.agentrunner/sessions").expanduser()

    async def save(
        self,
        session_id: str,
        messages: list[Message],
        config: AgentConfig,
        meta: dict[str, Any],
    ):
        """Save session (can be called incrementally)."""
        # Atomic write to session directory
        # ~/.agentrunner/sessions/{session_id}/messages.jsonl
        pass

    async def load(self, session_id: str):
        """Load session and resume."""
        pass
```

**Storage:**
```
~/.agentrunner/sessions/
├── session_abc123/
│   ├── messages.jsonl  # Incrementally updated
│   ├── config.json
│   └── meta.json
```

### **Rollouts' Approach: Checkpointing + Trajectory**

Rollouts has **two mechanisms**:

1. **Checkpointing** - Mid-execution state snapshots
2. **Trajectory storage** - Final episode data

**For interactive CLI, you need a third mechanism:**

```python
# rollouts/session.py (new file)

from pathlib import Path
from typing import List, Dict, Any
import json
import time

@dataclass(frozen=True)
class InteractiveSession:
    """Interactive session state for CLI frontends."""
    session_id: str
    messages: Tuple[Message, ...]
    environment_name: str
    model: str
    created_at: float
    updated_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class SessionStore:
    """Store for interactive sessions (like agent-runner's SessionManager)."""

    def __init__(self, directory: str = "~/.rollouts/sessions"):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)

    async def save(self, session: InteractiveSession) -> None:
        """Save session (can be called incrementally)."""
        session_dir = self.directory / session.session_id
        session_dir.mkdir(exist_ok=True)

        # Save messages incrementally
        messages_path = session_dir / "messages.jsonl"
        with open(messages_path, "w") as f:
            for msg in session.messages:
                f.write(msg.to_json() + "\n")

        # Save metadata
        meta_path = session_dir / "meta.json"
        meta = {
            "session_id": session.session_id,
            "environment_name": session.environment_name,
            "model": session.model,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            **session.metadata,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    async def load(self, session_id: str) -> InteractiveSession:
        """Load session from disk."""
        session_dir = self.directory / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        # Load messages
        messages_path = session_dir / "messages.jsonl"
        messages = []
        for line in messages_path.read_text().splitlines():
            if line:
                messages.append(Message.from_json(line))

        # Load metadata
        meta_path = session_dir / "meta.json"
        meta = json.loads(meta_path.read_text())

        return InteractiveSession(
            session_id=session_id,
            messages=tuple(messages),
            environment_name=meta["environment_name"],
            model=meta["model"],
            created_at=meta["created_at"],
            updated_at=meta["updated_at"],
            metadata={k: v for k, v in meta.items() if k not in [
                "session_id", "environment_name", "model",
                "created_at", "updated_at"
            ]},
        )

    async def list(self) -> List[Dict[str, Any]]:
        """List all sessions with metadata."""
        sessions = []
        for session_dir in self.directory.iterdir():
            if not session_dir.is_dir():
                continue

            meta_path = session_dir / "meta.json"
            if not meta_path.exists():
                continue

            try:
                meta = json.loads(meta_path.read_text())
                sessions.append(meta)
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by updated_at
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions
```

**Usage in Rich CLI:**

```python
# rollouts/cli/rich_frontend.py

class RichAdapter:
    def __init__(self, session_id: str | None = None):
        self.console = Console()
        self.session_store = SessionStore()
        self.session_id = session_id or f"cli_{int(time.time())}"
        self.messages = []

        # Load existing session if resuming
        if session_id:
            session = await self.session_store.load(session_id)
            self.messages = list(session.messages)
            self.console.print(f"[green]Resumed session: {session_id}[/green]")

    async def save_session(
        self,
        environment_name: str,
        model: str,
        metadata: Dict[str, Any]
    ):
        """Save session after each turn."""
        session = InteractiveSession(
            session_id=self.session_id,
            messages=tuple(self.messages),
            environment_name=environment_name,
            model=model,
            created_at=metadata.get("created_at", time.time()),
            updated_at=time.time(),
            metadata=metadata,
        )
        await self.session_store.save(session)

# Usage
async def run_interactive(session_id: str | None = None):
    adapter = RichAdapter(session_id=session_id)  # Resume if provided

    # ... run agent ...

    # Save after each turn
    await adapter.save_session(
        environment_name="calculator",
        model="gpt-4o",
        metadata={"total_tokens": 1542},
    )
```

---

## Recommended Architecture

### **Use Rollouts' Existing Patterns + Add SessionStore**

```python
# Rollouts already has:
1. ✅ Backend interface (RunConfig with callbacks)
2. ✅ Checkpointing (FileCheckpointStore)
3. ✅ Trajectory storage (Trajectory.save_jsonl)

# Add for interactive CLI:
4. 🆕 SessionStore (like agent-runner's SessionManager)
```

**Why this is better than agent-runner's approach:**

| Feature | Agent-Runner | Rollouts (Proposed) |
|---------|-------------|---------------------|
| **Backend interface** | Stateful Agent class | Pure functions + callbacks ✅ |
| **Session storage** | SessionManager ✅ | SessionStore 🆕 |
| **Multi-frontend** | ❌ Coupled to CLI | ✅ Decoupled via callbacks |
| **Streaming** | EventBus (global state) | Callbacks (injected) ✅ |
| **Checkpointing** | ❌ No mid-execution | ✅ FileCheckpointStore |
| **Training data** | ❌ Not designed for this | ✅ Trajectory storage |

**Rollouts' approach is superior because:**
- ✅ Backend is pure functions (no UI coupling)
- ✅ Frontend injects callbacks (clean separation)
- ✅ Can add multiple frontends without changing backend
- ✅ Already has checkpointing + trajectory storage for training
- ✅ Just need to add SessionStore for interactive CLI

---

## Implementation Plan

### **Phase 1: Add SessionStore (Week 1)**

**Day 1-2: Implement SessionStore**

```python
# rollouts/session.py (new file)

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List
import json
import time

from rollouts.dtypes import Message

@dataclass(frozen=True)
class InteractiveSession:
    session_id: str
    messages: Tuple[Message, ...]
    environment_name: str
    model: str
    created_at: float
    updated_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class SessionStore:
    """Storage for interactive CLI sessions."""

    def __init__(self, directory: str = "~/.rollouts/sessions"):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)

    async def save(self, session: InteractiveSession) -> None:
        # Implementation as shown above
        pass

    async def load(self, session_id: str) -> InteractiveSession:
        # Implementation as shown above
        pass

    async def list(self) -> List[Dict[str, Any]]:
        # Implementation as shown above
        pass

    async def delete(self, session_id: str) -> None:
        session_dir = self.directory / session_id
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
```

**Day 3: Tests**

```python
# tests/test_session.py

import pytest
from rollouts.session import SessionStore, InteractiveSession
from rollouts.dtypes import Message

@pytest.mark.asyncio
async def test_session_save_load(tmp_path):
    store = SessionStore(directory=str(tmp_path))

    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
    ]

    session = InteractiveSession(
        session_id="test123",
        messages=tuple(messages),
        environment_name="calculator",
        model="gpt-4o",
        created_at=time.time(),
        updated_at=time.time(),
    )

    await store.save(session)
    loaded = await store.load("test123")

    assert loaded.session_id == "test123"
    assert len(loaded.messages) == 2
    assert loaded.messages[0].content == "Hello"

@pytest.mark.asyncio
async def test_session_list(tmp_path):
    store = SessionStore(directory=str(tmp_path))

    # Save multiple sessions
    for i in range(3):
        session = InteractiveSession(
            session_id=f"session_{i}",
            messages=tuple([]),
            environment_name="calc",
            model="gpt-4o",
            created_at=time.time(),
            updated_at=time.time() + i,
        )
        await store.save(session)

    sessions = await store.list()
    assert len(sessions) == 3
    # Should be sorted by updated_at descending
    assert sessions[0]["session_id"] == "session_2"
```

### **Phase 2: Rich CLI Frontend (Week 2)**

**Day 1-3: Implement RichAdapter**

```python
# rollouts/cli/rich_frontend.py (implementation shown above)
```

**Day 4-5: CLI commands**

```python
# rollouts/cli/__main__.py

import click

@click.group()
def cli():
    """Rollouts CLI"""
    pass

@cli.command()
@click.option("--env", default="calculator", help="Environment name")
@click.option("--model", default="gpt-4o", help="Model to use")
@click.option("--session", help="Session ID to resume")
def chat(env, model, session):
    """Interactive chat with agent."""
    asyncio.run(run_interactive(env, model, session))

@cli.command()
def sessions():
    """List all interactive sessions."""
    store = SessionStore()
    sessions = await store.list()

    console = Console()
    table = Table(title="Interactive Sessions")
    table.add_column("Session ID")
    table.add_column("Environment")
    table.add_column("Model")
    table.add_column("Updated")

    for s in sessions:
        table.add_row(
            s["session_id"],
            s["environment_name"],
            s["model"],
            s["updated_at"],
        )

    console.print(table)

if __name__ == "__main__":
    cli()
```

**Usage:**
```bash
# Start new interactive session
$ python -m rollouts.cli chat --env calculator

# Resume session
$ python -m rollouts.cli chat --session cli_1234567

# List sessions
$ python -m rollouts.cli sessions
```

### **Phase 3: React Ink Frontend (Optional - Week 3)**

Only implement if you need more sophisticated terminal UI.

---

## Summary

### **Key Architectural Principles**

1. **Backend is pure functions** - No UI knowledge
2. **Frontend injects callbacks** - Clean separation via RunConfig
3. **Multiple frontends share same interface** - StreamChunk callbacks
4. **Session storage is frontend-specific** - Different frontends have different needs

### **What Rollouts Already Has (Perfect!)**

- ✅ Backend interface (RunConfig with callbacks)
- ✅ Checkpointing (FileCheckpointStore)
- ✅ Trajectory storage (for training)
- ✅ Web frontend (rollouts/frontend/)

### **What to Add**

- 🆕 SessionStore (for interactive CLI)
- 🆕 Rich CLI frontend
- 🆕 (Optional) React Ink frontend

### **Is SessionManager Better Than What You Have?**

**No!** Rollouts' architecture is **superior** to agent-runner's:

| Aspect | Agent-Runner | Rollouts |
|--------|-------------|----------|
| Backend coupling | ❌ Stateful Agent class | ✅ Pure functions |
| Frontend flexibility | ❌ Hardcoded for CLI | ✅ Callback-based |
| Multi-frontend | ❌ Difficult | ✅ Easy (just implement callbacks) |
| Training support | ❌ Not designed for it | ✅ Built-in (Trajectory) |
| Session storage | ✅ Has it | ❌ Need to add |

**Recommendation:**
- Keep rollouts' functional backend (it's perfect!)
- Add SessionStore (inspired by agent-runner's SessionManager)
- Implement Rich CLI frontend using callbacks
- Web frontend already exists

**Your architecture is already right!** Just add the session storage layer for interactive CLI.

Would you like me to implement the SessionStore + Rich CLI frontend?