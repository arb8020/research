# Unified TUI Design

## Goal

Single pattern for GEPA, RL training, and normal eval:
1. Process emits JSONL events to a file
2. TUI tails the file and renders progress
3. Completely decoupled - process doesn't know if anyone is watching

## Current State

| Mode | Event emission | TUI |
|------|---------------|-----|
| RL Training | `remote_runner.py` wraps process + tails logs → stdout | `monitor.py` reads stdin |
| Eval | `MultiProgress` embedded in `evaluate()` | N/A (MultiProgress IS the UI) |
| GEPA | Nested `MultiProgress` instances fight | Broken |

## Target State

All modes:
1. Write JSONL events to `{output_dir}/events.jsonl`
2. TUI tails that file (or stdin if piped)
3. `MultiProgress` becomes one possible renderer of events

```
┌─────────────────────────────────────────────────────────────┐
│  Any Process (eval, GEPA, RL training)                      │
│                                                             │
│  emit_event({"type": "sample_start", "id": "001", ...})    │
│  emit_event({"type": "turn", "id": "001", "turn": 1})      │
│  emit_event({"type": "modal_progress", "phase": "compile"})│
│  emit_event({"type": "sample_end", "id": "001", ...})      │
│                                                             │
│            ↓ writes to                                      │
│  {output_dir}/events.jsonl                                  │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ tail -f (or pipe)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  TUI (rollouts.tui.watch)                                   │
│                                                             │
│  Renders based on event type:                               │
│  - sample_start/end → progress bar                          │
│  - turn → update turn count                                 │
│  - modal_progress → show phase (compiling, checking, etc)   │
│  - gepa_iteration → show GEPA progress                      │
│  - metrics → show charts/sparklines                         │
│  - log → route to appropriate pane                          │
└─────────────────────────────────────────────────────────────┘
```

## Event Types

```python
# Core eval events (from evaluation.py)
{"type": "sample_start", "id": "001", "name": "Square_matmul", "total": 10}
{"type": "turn", "id": "001", "turn": 1, "status": "streaming"}
{"type": "modal_progress", "id": "001", "phase": "compiling"}  # or correctness, benchmarking
{"type": "sample_end", "id": "001", "score": 0.85, "time_sec": 45.2}

# GEPA events (from prompt_optimization/engine.py)
{"type": "gepa_iteration", "iteration": 3, "evals_used": 12, "evals_budget": 50, "best_score": 0.42}
{"type": "gepa_accepted", "old_score": 0.40, "new_score": 0.42}
{"type": "gepa_rejected", "old_score": 0.42, "new_score": 0.38}

# RL training events (from training/grpo.py)
{"type": "rl_step", "step": 10, "reward_mean": 0.65, "loss": 0.023}
{"type": "rl_checkpoint", "step": 100, "path": "/checkpoints/step_100"}

# Generic log events (from any logger)
{"type": "log", "logger": "kernelbench", "level": "INFO", "message": "..."}
```

## Implementation Plan

### Phase 1: Event emitter (no UI changes)

1. Add `EventEmitter` class that writes JSONL to file
2. Wire into `evaluate()` - emit events alongside existing `on_chunk` callbacks
3. Wire into GEPA engine - emit `gepa_iteration`, `gepa_accepted`, etc.
4. Wire into RL training - emit `rl_step`, etc.

```python
# rollouts/events.py
class EventEmitter:
    """Writes structured events to JSONL file."""
    
    def __init__(self, output_dir: Path):
        self.file = open(output_dir / "events.jsonl", "a")
    
    def emit(self, event: dict) -> None:
        event["timestamp"] = datetime.now().isoformat()
        self.file.write(json.dumps(event) + "\n")
        self.file.flush()
```

### Phase 2: TUI consumer

1. Create `rollouts.tui.watch` that tails events.jsonl
2. Renders progress based on event types
3. Can show multiple concurrent samples (like current MultiProgress)
4. Can show GEPA iteration progress in header
5. Can show RL training metrics/charts

```bash
# Usage
python -m rollouts.tui.watch /path/to/output/events.jsonl

# Or with auto-discovery
python -m rollouts.tui.watch --latest  # finds most recent run
```

### Phase 3: Remove embedded UI

1. Remove `MultiProgress` from `evaluate()`
2. Remove nested progress displays from GEPA
3. All progress viewing goes through TUI

## File Structure

```
rollouts/
├── events.py           # EventEmitter - writes JSONL
├── evaluation.py       # Uses EventEmitter (no MultiProgress)
├── tui/
│   ├── watch.py        # Main TUI entry point (tails events.jsonl)
│   ├── monitor.py      # TrainingMonitor (rename to EventRenderer?)
│   ├── progress.py     # Progress bar rendering (extracted from MultiProgress)
│   └── terminal.py     # Terminal abstraction
```

## Migration Path

1. Add EventEmitter alongside existing code (non-breaking)
2. Add `rollouts.tui.watch` (new capability)
3. Deprecate `show_progress` flag
4. Remove `MultiProgress` from evaluation internals
5. Update docs to recommend TUI approach

## Open Questions

1. Should events go to stdout (like RL training) or file (like our GEPA sketch)?
   - File is simpler for local runs
   - Stdout works better for remote/piped scenarios
   - Could support both via `--events-to stdout` flag

2. How to handle existing `on_chunk` callbacks?
   - Keep for backwards compat, but events.jsonl is the primary output
   - Eventually deprecate on_chunk in favor of event file

3. How to integrate with existing `TrainingMonitor` panes?
   - Events with `type: "log"` get routed to panes by logger name
   - Other event types rendered in dedicated progress section
