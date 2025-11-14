# Agent Dev Loop Tool

Visual config builder and trace viewer for rollouts-based agents.

## Features

- **Config Builder**: Visually configure agents (model, prompts, tools, eval settings)
- **Trace Viewer**: Browse evaluation results and inspect conversation trajectories
- **General Purpose**: Works with any rollouts-based agent evaluation

## Usage

### Quick Start

From your agent project directory:

```bash
cd ~/wafer_stuff/kernels-gpumode-agent
~/research/rollouts/rollouts/frontend/dev.sh
```

This will:
1. Start server on http://localhost:8080
2. Open browser automatically
3. Scan `configs/` and `results/` directories
4. Log to `logs/devloop_TIMESTAMP.log`
5. Prefix all output with `[SERVER]`

Press ENTER or Ctrl+C to stop.

### Options

```bash
# Custom project directory
~/research/rollouts/rollouts/frontend/dev.sh ~/my-agent

# Custom port
~/research/rollouts/rollouts/frontend/dev.sh ~/my-agent 9000

# Or use Python directly for more control
python3 ~/research/rollouts/rollouts/frontend/run.py --project ~/my-agent --port 9000 --no-browser
```

## Workflow

### 1. Build Config

Use the left panel to configure your agent:
- Select model (GPT-4, Claude, etc.)
- Write system prompt
- Adjust temperature, max turns, num samples
- Enable tools (filesystem, web search, etc.)

Click "Generate Config" to download a Python config file.

### 2. Run Evaluation

```bash
python entrypoint.py configs/config_20251114T143022.py
```

### 3. View Traces

The right panel shows evaluation results:
- Click a trace to view details
- See conversation messages
- Inspect rewards and metrics

## Architecture

```
rollouts/frontend/
├── __init__.py       # Package init
├── __main__.py       # Module entry point
├── server.py         # HTTP server + API (200 lines)
├── index.html        # Single-page UI (400 lines)
├── run.py            # Convenience script
└── README.md         # This file
```

**Design Principles:**
- No frameworks (vanilla HTML/CSS/JS)
- No build step (just Python stdlib)
- Coupled to rollouts abstractions, not specific environments
- Compression-oriented (simple first, extend later)

## API Endpoints

- `GET /` - Serve frontend
- `GET /api/configs` - List config files in `configs/`
- `GET /api/traces` - List evaluation results in `results/`
- `GET /api/trace/:id` - Load specific trace with trajectories
- `POST /api/generate` - Generate config from JSON

## Extending

### Adding Custom Tools

The generated config includes a `get_tools()` method:

```python
def get_tools(self) -> List[Tool]:
    """Return tools available to agent."""
    return [
        Tool(
            type="function",
            function=ToolFunction(
                name="my_custom_tool",
                description="Does something custom",
                parameters=ToolFunctionParameter(
                    type="object",
                    properties={
                        "arg": {"type": "string"}
                    }
                )
            )
        )
    ]
```

### Custom Environment Logic

Implement `on_assistant_message()` to handle agent outputs:

```python
async def on_assistant_message(self, message: Message, state):
    """React to agent messages."""
    # Parse message, extract actions, update state
    return updated_state
```

## Future Extensions

- [ ] Tool management UI (add/edit/remove tools visually)
- [ ] Dataset preview
- [ ] Trace comparison (diff two runs)
- [ ] Export trajectory as markdown
- [ ] Custom reward function editor
