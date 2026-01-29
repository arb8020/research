# Console: Unified Output Manager

## Problem

The pytui Spinner and Python's logging module both write to stderr but don't coordinate. When log messages emit while a spinner is animating, output interleaves and corrupts:

```
⠹ Deploying code...⚠️  Found 1 untracked file(s) that will NOT be deployed:
```

The spinner writes `\r\033[K` (carriage return + clear line) to animate in place. Log messages write complete lines with `\n`. They collide because neither knows the other exists.

## Root Cause

We had three separate output systems hitting the same terminal:
1. `pytui.Spinner` — standalone, knows nothing about logging
2. `logging` module — standalone, knows nothing about spinners
3. `print()` — standalone (mostly migrated away now)

The print→logging migration unified (3) into (2), but didn't address the spinner/logging collision.

## Solution: Console

A single "progress reporter" abstraction that owns stderr and coordinates all output:

```python
from pytui import Console
import logging

console = Console()  # Owns stderr
console.install_logging_handler(logging.getLogger())  # Explicit wiring

with console.spinner("Provisioning..."):
    result = await acquire_node()  # Any logger.warning() inside is coordinated
# ✓ Provisioning... (1m7s)
```

When a log message emits while a spinner is active:
1. Clear the spinner line (`\r\033[K`)
2. Print the log message on its own line
3. Redraw the spinner below

## Design Principles (from code_style docs)

**Context manager, not start/stop** — Explicit scope, guaranteed cleanup, no forgotten `stop()` calls. Spinner is active only inside the `with` block.

**Explicit handler installation** — Don't secretly hook into logging. Caller does `console.install_logging_handler(logger)` so the wiring is visible.

**Three levels of granularity** (Casey Muratori's rule — no API holes):
- Low-level: `pause_spinner()`, `resume_spinner()`, `write()`
- Mid-level: `status(text)` for one-off messages
- High-level: `with console.spinner(text):` context manager

Each level uses the lower level, so callers can escape to primitives when needed.

**Immediate mode** — No retained state to track. Spinner exists only in its `with` block.

**Console owns spinner state, caller owns logger** — Clear responsibility boundaries.

## API

```python
class Console:
    def __init__(self, output: TextIO = sys.stderr, spinner_interval: float = 0.08):
        """Create console. Defaults to stderr (matches logging convention)."""

    # Low-level
    def write(self, text: str) -> None: ...
    def write_line(self, text: str) -> None: ...
    def pause_spinner(self) -> None: ...
    def resume_spinner(self) -> None: ...

    # Mid-level
    def status(self, text: str, success: bool = True) -> None:
        """Print ✓/✗ status line."""

    # High-level
    @contextmanager
    def spinner(self, message: str) -> Iterator[SpinnerHandle]:
        """Spinner active only in this block. Logs are coordinated."""

    # Logging integration
    def install_logging_handler(self, logger: Logger, level: int = DEBUG, ...) -> Handler:
        """Explicitly wire logging to this console."""

    def remove_logging_handlers(self) -> None:
        """Clean up installed handlers."""


class SpinnerHandle:
    def update(self, message: str) -> None:
        """Change spinner text mid-flight."""


class ConsoleHandler(logging.Handler):
    """Logging handler that pauses spinner, emits message, resumes spinner."""
```

## Features

- **Live elapsed time**: Shows `(45s)`, `(1m7s)` while spinning and on completion
- **Spinner/logging coordination**: Log messages appear cleanly above the spinner
- **Context manager cleanup**: Guaranteed stop on exit or exception

## Bifrost Integration

Added `on_bootstrap_step` callback to `bifrost.push()` for observability without inverting control:

```python
with console.spinner("Deploying code...") as spinner:
    def on_step(cmd: str, index: int, total: int) -> None:
        spinner.update(f"{labels[index]}...")

    workspace = bifrost.push(
        workspace_path,
        bootstrap_cmd=commands,
        on_bootstrap_step=on_step,
    )
```

This shows labeled progress during deploy:
```
⠹ Installing system deps... (5s)
⠹ Installing uv... (12s)
⠹ Syncing Python deps... (45s)
⠹ Installing ML packages... (1m30s)
✓ Installing ML packages... (2m15s)
```

## Branch

Implementation is on `debug-frames` branch (worktree at `../research-debug-frames`).

Commits:
- `b37effbf` — feat(pytui): add Console for unified spinner/logging coordination
- `299d3496` — feat(bifrost): add on_bootstrap_step callback to push()

## Related

- `handoff.md` — print→logging migration context
- `docs/code_style/mcoding_logging_dense.md` — logging best practices
- `docs/code_style/casey_granularity.md` — API granularity principles

## TODO

- [ ] Monitor UI cleanup — currently shows raw JSON logs, layout is broken
- [ ] Parse JSON logs into metrics (loss, reward, step)
- [ ] btop-style layout with sparklines and clean log formatting
