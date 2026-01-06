# Progress Display Redesign

## Problem

Current `MultiProgress` has two issues:

1. **Logs interfere with rendering** - `logger.info()` calls print to stdout while MultiProgress is trying to control terminal state, causing glitchy/twitchy display

2. **Embedded in eval code** - `evaluate()` creates MultiProgress internally and passes it through call stack, coupling UI to business logic

## Goal

Clean progress display like `uv` or `cargo`:
- Single updating view showing what's happening
- No log noise interfering  
- Satisfying to watch
- When done, back to normal terminal (no jarring context switch)

## Three Display Modes

| Mode | Flag | Behavior |
|------|------|----------|
| **Raw/Verbose** | `--verbose` | No special display. Logs print normally. Good for debugging, piping to files, CI/CD. |
| **Clean Progress** | (default) | Alternate screen with passive progress display. No keyboard input. Exits when eval completes. |
| **Full TUI** | `--tui` | Multi-pane interactive explorer. Keyboard navigation, expandable samples, live tokens. |

## Core Architecture

### Single Source of Truth: JSONL File

The JSONL file is the **only source of state**. The renderer is stateless - it derives current view by reading from the file.

```
┌─────────────────────────────────────────────────────────────┐
│  evaluate() / GEPA / RL training                            │
│                                                             │
│  logger.info("Starting", extra={"type": "eval_start", ...}) │
│  logger.info("Sample", extra={"type": "sample_start", ...}) │
│  logger.info("Done", extra={"type": "sample_end", ...})     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ Python logging (root logger)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  RotatingFileHandler                                        │
│  └── Writes all logs to events.jsonl with JSONFormatter     │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ File on disk (single source of truth)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ProgressRenderer (stateless)                               │
│                                                             │
│  - Polls events.jsonl for new lines                         │
│  - Derives current state from events (no internal state)    │
│  - Renders to terminal (alternate screen)                   │
│  - Can attach/detach at any time                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why File-Based?

1. **Single source of truth** - No in-memory state to get out of sync
2. **Reattachable** - Can start/stop renderer without losing state
3. **Debuggable** - Can inspect events.jsonl directly
4. **Replayable** - Can replay an old eval through the renderer
5. **Decoupled** - Writer doesn't know/care if anyone is watching

### Key Design Decisions

1. **ProgressRenderer is stateless** - Derives view from file on each render cycle
2. **Polling, not filesystem events** - Simple 100ms poll loop (per sys design: "simple components")
3. **Two execution modes supported**:
   - **Integrated**: Eval in background thread, renderer in main thread (single command)
   - **Detached**: Eval runs normally, renderer in separate process (two terminals)

## Event Types

Events are logged with `extra={"type": ...}` dict:

```python
# Eval lifecycle
logger.info("Starting", extra={"type": "eval_start", "name": "gsm8k", "total": 50})
logger.info("Done", extra={"type": "eval_end", "total": 50, "accuracy": 0.85})

# Sample lifecycle (specific sample IDs for debugging/correlation)
logger.info("Started", extra={"type": "sample_start", "id": "sample_0001", "name": "question_1"})
logger.info("Turn", extra={"type": "turn", "id": "sample_0001", "turn": 1})
logger.info("Done", extra={"type": "sample_end", "id": "sample_0001", "score": 0.85})

# Modal/GPU progress
logger.info("Phase", extra={"type": "modal_progress", "id": "sample_0001", "phase": "compiling"})

# GEPA (header only in minimal version)
logger.info("Iter", extra={"type": "gepa_iteration", "iter": 3, "total": 10, "best": 0.42})

# Retry indicator for failed samples
logger.info("Retry", extra={"type": "sample_retry", "id": "sample_0001", "attempt": 2})

# Generic logs (no "type" - written to file but not special-cased in rendering)
logger.info("Some debug info")
logger.debug("Detailed stuff")
```

### Event Schema Validation

**Strict validation** - if `extra["type"]` is present but malformed, crash with clear error. Programmer error should fail fast (per code philosophy: "parse at boundary, assert internally").

## Clean Progress Display

Alternate screen, passive (no keyboard input), stateless render from file:

```
┌────────────────────────────────────────────────────────────────┐
│ gsm8k_eval: 12/50 (24%) ████████░░░░░░░░░░░░░░░░░░░░░░ ETA 3m  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   sample_0012  T:1  compiling...                               │
│   sample_0013  T:0  streaming...                               │
│   sample_0014  T:2  ✓ score=0.85                               │
│   sample_0015  T:1  checking...                                │
│   sample_0010  T:3  ✗ retry (attempt 2)                        │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│ correct: [████████░░] 0.8  |  fast_1: [██████░░░░] 0.6         │
│          ▂▃▅▆▇█▇▅▃▂        |          ▁▂▃▄▅▆▇█▇▆               │
└────────────────────────────────────────────────────────────────┘
```

### Display Elements

1. **Header**: Eval name, progress bar, completed/total, ETA
2. **Sample list**: In-flight samples with:
   - Sample ID (for debugging correlation)
   - Turn count (T:N)
   - Status: `streaming...`, `compiling...`, `checking...`, `benchmarking...`
   - Completion: `✓ score=X.XX` or `✗ retry (attempt N)`
3. **Histograms**: Per-metric score distributions (j/k to switch, like RL monitor)

### GEPA Display (Minimal)

Just header, samples show flat:

```
┌────────────────────────────────────────────────────────────────┐
│ GEPA iter 3/10 │ best: 42%                                     │
├────────────────────────────────────────────────────────────────┤
│   sample_0012  T:1  compiling...                               │
│   sample_0013  T:0  streaming...                               │
├────────────────────────────────────────────────────────────────┤
│ correct: [████████░░] 0.8                                      │
└────────────────────────────────────────────────────────────────┘
```

### State Derivation from File

On each render cycle, renderer reads events.jsonl and derives:

```python
def derive_state(events: list[dict]) -> RenderState:
    """Derive current display state from event stream. Stateless."""
    state = RenderState()
    
    for event in events:
        event_type = event.get("type")
        
        if event_type == "eval_start":
            state.eval_name = event["name"]
            state.total = event["total"]
        
        elif event_type == "sample_start":
            state.samples[event["id"]] = Sample(
                id=event["id"],
                name=event["name"],
                status="started",
            )
        
        elif event_type == "modal_progress":
            if event["id"] in state.samples:
                state.samples[event["id"]].phase = event["phase"]
        
        elif event_type == "sample_end":
            if event["id"] in state.samples:
                state.samples[event["id"]].score = event["score"]
                state.samples[event["id"]].status = "complete"
                state.completed += 1
        
        elif event_type == "gepa_iteration":
            state.gepa_iter = event["iter"]
            state.gepa_total = event["total"]
            state.gepa_best = event["best"]
    
    return state
```

## Full TUI Mode (`--tui`)

Multi-pane like current `TrainingMonitor`:

- **Eval pane**: Logs routed by logger name
- **Modal pane**: Modal/GPU logs
- **Traces pane**: Rollout traces

Additional features over clean progress:
- **Keyboard navigation**: j/k scroll, 1-4 switch panes, q quit
- **On-demand sample expansion**: Modal fullscreen with live tokens
- **Histogram switching**: j/k to cycle through metrics

## File Logging

### JSONL Format

All logs (including DEBUG) written to `{output_dir}/events.jsonl`:

```json
{"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "logger": "rollouts.eval", "message": "Starting", "type": "eval_start", "name": "gsm8k", "total": 50}
{"timestamp": "2024-01-15T10:30:01Z", "level": "DEBUG", "logger": "httpx", "message": "HTTP Request: POST https://api.openai.com/..."}
```

### Rotation

- Rotate at **10MB**
- **Keep all rotations** indefinitely: `events.jsonl`, `events.jsonl.1`, `events.jsonl.2`, ...

### Default Behavior

- File logging is **on by default** when `output_dir` is set
- Use `--no-log-file` flag to disable

## Terminal Handling

### Alternate Screen

- Enter alternate screen on start
- Exit alternate screen on completion
- Content doesn't pollute terminal scrollback

### Cleanup

Use existing `frontends/tui/terminal.py` which provides:
- `atexit` handler for normal exit
- `stty sane` fallback for crash recovery
- SIGWINCH handling for resize

### Ctrl+C Behavior

1. Catch SIGINT
2. Exit alternate screen cleanly
3. Print brief summary to normal terminal: `"Interrupted. 12/50 samples completed. Results in {output_dir}"`
4. Exit with appropriate code

### Successful Completion

1. Receive `eval_end` event
2. Exit alternate screen immediately
3. Return to normal terminal (no pause/wait)

### Crash/Exception

1. Error banner appears in alternate screen: `"Error: {exception}"`
2. Wait for keypress
3. Exit alternate screen
4. Print traceback to normal terminal

## Rendering

### Polling Loop

Single-threaded polling, like current TrainingMonitor:

```python
def render_loop(events_file: Path, terminal: Terminal):
    """Main render loop. Polls file, derives state, renders."""
    file_pos = 0
    
    while True:
        # Read new events from file
        with open(events_file) as f:
            f.seek(file_pos)
            new_lines = f.readlines()
            file_pos = f.tell()
        
        if new_lines:
            events = [json.loads(line) for line in new_lines]
            state = derive_state(events)
            render(terminal, state)
        
        time.sleep(0.1)  # 100ms poll interval
```

### Rate Limiting

Start without rate limiting. If flickering occurs at high event rates, add 30fps cap later.

## API

### Usage - Integrated Mode (Single Command)

```python
# Eval runs in background thread, renderer in main thread
with progress_display(desc="gsm8k_eval", output_dir=output_dir):
    await evaluate(dataset, config)
```

### Usage - Detached Mode (Two Terminals)

```bash
# Terminal 1: Run eval (writes to events.jsonl)
python eval_script.py --output-dir ./results

# Terminal 2: Watch progress (reads events.jsonl)
python -m rollouts.progress_watch ./results/events.jsonl
```

### Implementation

```python
# rollouts/progress_display.py

@contextmanager
def progress_display(desc: str = "", output_dir: Path | None = None):
    """Context manager for clean progress display.
    
    Sets up file logging, runs eval in background thread, renders in main thread.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    
    events_file = output_dir / "events.jsonl"
    
    # Setup file logging
    file_handler = RotatingFileHandler(
        events_file,
        maxBytes=10_000_000,  # 10MB
        backupCount=999,      # Keep all
    )
    file_handler.setFormatter(JSONFormatter())
    logging.getLogger().addHandler(file_handler)
    
    # Setup terminal
    terminal = Terminal(use_alternate_screen=True)
    terminal.start()
    
    # Run renderer in main thread
    renderer_stop = threading.Event()
    
    def run_renderer():
        render_loop(events_file, terminal, stop_event=renderer_stop)
    
    # Eval will run in the with block, renderer runs alongside
    renderer_thread = threading.Thread(target=run_renderer, daemon=True)
    renderer_thread.start()
    
    try:
        yield  # Eval runs here
    except KeyboardInterrupt:
        terminal.stop()
        print(f"Interrupted. Results in {output_dir}")
        raise
    finally:
        renderer_stop.set()
        terminal.stop()
        logging.getLogger().removeHandler(file_handler)
```

## File Structure

```
rollouts/
├── progress_display.py      # NEW: progress_display() context manager + render loop
├── progress_watch.py        # NEW: CLI for detached viewing (python -m rollouts.progress_watch)
├── progress.py              # EXISTING: Keep MultiProgress for now (deprecate later)
├── evaluation.py            # MODIFY: Add logger.info(..., extra={"type": ...}) calls
├── _logging/
│   └── json_formatter.py    # EXISTING: JSON formatter for JSONL output
├── frontends/tui/
│   └── terminal.py          # EXISTING: Use this for terminal control
└── tui/
    └── monitor.py           # EXISTING: Full TUI mode
```

## Implementation Phases

### Phase 0: Minimal Viable

1. Basic `progress_display.py` with:
   - File handler setup
   - Simple render loop (progress bar + sample list only)
   - No histograms, no GEPA header
2. Add logging calls to `evaluate()` for basic events
3. Test with one eval

### Phase 1: Core Features

1. Add histogram rendering
2. Add GEPA header support
3. Add `progress_watch.py` for detached mode

### Phase 2: Polish

1. Error handling (crash banner, Ctrl+C)
2. Log rotation
3. Update all examples

### Phase 3: Migration

1. Deprecate `show_progress` config option
2. Remove MultiProgress from evaluate() internals

## Open Questions (Resolved)

| Question | Decision |
|----------|----------|
| State model? | Stateless - derive from JSONL file on each render |
| Polling vs filesystem events? | Polling (simple, 100ms interval) |
| Threading model? | Support both: integrated (background eval) and detached (separate process) |
| Sample IDs vs aggregate counts? | Show specific sample IDs |
| GEPA display? | Header only (minimal), samples flat |
| Event validation? | Strict - crash on malformed events |
| Log rotation? | 10MB, keep all rotations |
| File logging default? | On by default |
| Completion behavior? | Exit immediately |
| Ctrl+C behavior? | Exit cleanly, print summary |

## References

- Current `MultiProgress`: `rollouts/progress.py`
- Current `TrainingMonitor`: `rollouts/tui/monitor.py`  
- Terminal abstraction: `rollouts/frontends/tui/terminal.py`
- JSON logging: `rollouts/_logging/json_formatter.py`
- mcoding logging guide: `docs/code_style/mcoding_logging.md`
- Code philosophy: `docs/code_style/code_philosophy.md`
- Sys design guide: `docs/code_style/sys_design_sean_goedecke.md`
