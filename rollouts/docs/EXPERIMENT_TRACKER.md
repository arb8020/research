# Experiment Tracker ("wandb at home")

**DRI:** Chiraag
**Claude:** [this conversation]

## Context
Replace wandb dependency with a local-first experiment tracker that uses TUI for real-time visualization - no SaaS, no accounts, just files.

## Out of Scope
- Cloud sync / team collaboration
- Web dashboard
- Artifact versioning/registry
- Hyperparameter sweeps
- Alerts/notifications

## Solution
**Input:** Python `log()` calls writing metrics to local `.jsonl` files
**Output:** TUI that reads these files and renders live charts + run comparisons

## Usage
```python
# Logging (library side)
from rollouts.tracker import Tracker

tracker = Tracker("my-experiment")
for step in range(1000):
    loss = train_step()
    tracker.log({"loss": loss, "step": step})
tracker.finish()

# Viewing (CLI)
$ rollouts track              # Watch latest run
$ rollouts track ./runs/      # Compare all runs in directory
$ rollouts track run-abc123   # Watch specific run
```

---

## Details

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      TUI (Python)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ RunSelector │  │ MetricsGrid │  │   ChartPanel    │ │
│  │  (sidebar)  │  │  (overview) │  │ (braille lines) │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                            │ reads
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    File Watcher                         │
│              (inotify/kqueue on .jsonl)                 │
└───────────────────────────┬─────────────────────────────┘
                            │ watches
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Storage Format                        │
│  runs/                                                  │
│  ├── run-20241215-abc123/                              │
│  │   ├── meta.json      # run config, start time       │
│  │   ├── metrics.jsonl  # {"step": 0, "loss": 1.2}     │
│  │   └── system.jsonl   # GPU util, memory (optional)  │
│  └── run-20241215-def456/                              │
│       └── ...                                           │
└─────────────────────────────────────────────────────────┘
```

### Components

#### 1. Tracker (logging library)
```python
class Tracker:
    def __init__(self, name: str, config: dict = None, dir: str = "./runs"):
        # Create run directory with timestamp + random suffix
        # Write meta.json with config, git hash, start time

    def log(self, data: dict, step: int = None):
        # Append to metrics.jsonl
        # Auto-increment step if not provided

    def log_system(self):
        # Optional: GPU util, memory, CPU (separate file)

    def finish(self):
        # Write end time to meta.json
```

#### 2. File Format
```json
// meta.json
{
  "name": "my-experiment",
  "run_id": "run-20241215-abc123",
  "config": {"lr": 0.001, "batch_size": 32},
  "git_hash": "abc123",
  "started_at": "2024-12-15T10:00:00Z",
  "finished_at": null  // set on finish()
}

// metrics.jsonl (append-only)
{"_step": 0, "_timestamp": 1734567890.123, "loss": 2.3, "accuracy": 0.1}
{"_step": 1, "_timestamp": 1734567891.456, "loss": 2.1, "accuracy": 0.15}
```

#### 3. TUI Components

| Component | Purpose | Keybindings |
|-----------|---------|-------------|
| RunSelector | List runs, filter, select for comparison | `j/k` navigate, `space` toggle, `/` filter |
| MetricsGrid | Show latest values for all metrics | auto-updates |
| ChartPanel | Braille line charts for selected metrics | `←/→` pan, `+/-` zoom, `Tab` cycle metrics |
| HelpOverlay | Keybinding reference | `?` toggle |

#### 4. Braille Charts

Use Unicode braille characters for high-resolution terminal charts:
```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣠⣤⣤⣶
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣴⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣀⣀⣤⣤⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
loss ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Each braille char is 2x4 dots = 8 possible points per character cell.
Gives ~4x vertical resolution vs ASCII art.

### Flow

1. User starts training with `Tracker("exp-name")`
2. Each `log()` appends JSON line to `metrics.jsonl`
3. TUI watches file with `watchfiles` (cross-platform inotify/kqueue)
4. On change, TUI reads new lines, updates internal data store
5. Charts re-render with new data points

### Dependencies

```
# New (minimal)
watchfiles  # async file watching

# Already have
trio        # async runtime (already in rollouts)
```

No new TUI framework needed - extend existing `rollouts/frontends/tui/`.

### Open Questions
- [ ] How to handle runs with different metric sets? (union vs intersection for comparison)
- [ ] System metrics: poll in separate thread or rely on nvidia-smi wrapper?
- [ ] Max data points before downsampling for chart performance?
- [ ] Config format: flat dict or nested? (wandb uses flat with `/` for nesting)

### Files
**Read:**
- `rollouts/frontends/tui/` - existing TUI infrastructure
- `rollouts/frontends/tui/tui.py` - Component/Container base classes

**Create:**
- `rollouts/tracker/__init__.py` - Tracker class
- `rollouts/tracker/storage.py` - File format read/write
- `rollouts/frontends/tui/components/chart.py` - Braille chart component
- `rollouts/frontends/tui/components/run_selector.py` - Run list sidebar
- `rollouts/cli.py` - Add `track` subcommand

### References
- W&B LEET uses [ntcharts](https://github.com/NimbleMarkets/ntcharts) for Go braille charts
- [plotext](https://github.com/piccolomo/plotext) - Python terminal plotting (could vendor/simplify)
- [asciichartpy](https://github.com/kroitor/asciichartpy) - Simpler ASCII charts
