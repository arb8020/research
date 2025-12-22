"""Training TUI for monitoring RL training runs.

Components:
- remote_runner: Wraps training + log sources into unified JSONL stream
- monitor: Local TUI that consumes stream and renders panes
- terminal: Terminal abstraction (raw mode, input handling)
"""

from rollouts.tui.monitor import (
    TrainingMonitor,
    PaneConfig,
    RL_TRAINING_PANES,
    EVAL_PANES,
    PANE_PRESETS,
)

__all__ = [
    "TrainingMonitor",
    "PaneConfig",
    "RL_TRAINING_PANES",
    "EVAL_PANES",
    "PANE_PRESETS",
]
