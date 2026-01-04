"""Training TUI for monitoring RL training runs.

Components:
- remote_runner: Wraps training + log sources into unified JSONL stream
- monitor: Local TUI that consumes stream and renders panes
- terminal: Terminal abstraction (raw mode, input handling)
"""

from ..tui.monitor import (
    EVAL_PANES,
    PANE_PRESETS,
    RL_TRAINING_PANES,
    PaneConfig,
    TrainingMonitor,
)

__all__ = [
    "TrainingMonitor",
    "PaneConfig",
    "RL_TRAINING_PANES",
    "EVAL_PANES",
    "PANE_PRESETS",
]
