"""Clean progress display using alternate screen and JSONL event stream.

Replaces MultiProgress with a cleaner approach:
- Alternate screen (no scrollback pollution)
- Stateless renderer (derives state from events.jsonl file)
- Works with existing EventEmitter infrastructure

Usage:
    with progress_display(output_dir=output_dir):
        await evaluate(dataset, config)

Or detached mode (two terminals):
    # Terminal 1: Run eval (writes to events.jsonl)
    python eval_script.py --output-dir ./results

    # Terminal 2: Watch progress
    python -m rollouts.progress_watch ./results/events.jsonl

Note: Events are written by EventEmitter (rollouts/events.py), not Python logging.
The progress display just reads the events.jsonl file that evaluate() already produces.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import TextIO

# ANSI escape codes
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
CLEAR_LINE = "\x1b[2K"
CURSOR_UP = "\x1b[{n}A"
# Synchronized output (reduces flicker in supported terminals)
SYNC_START = "\x1b[?2026h"
SYNC_END = "\x1b[?2026l"

# Colors
DIM = "\x1b[90m"
RESET = "\x1b[0m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"
BOLD = "\x1b[1m"

# Known event types for validation
KNOWN_EVENT_TYPES = frozenset({
    "eval_start",
    "eval_end",
    "sample_start",
    "sample_end",
    "turn",
    "modal_progress",
    "gepa_iteration",
    "gepa_accepted",
    "gepa_rejected",
    "sample_retry",
})


def _format_time(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    h = int(seconds) // 3600
    m = int(seconds) % 3600 // 60
    s = int(seconds) % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _format_bar(progress: float, width: int) -> str:
    """Render a progress bar with unicode blocks."""
    if width <= 0:
        return ""
    filled = progress * width
    full_blocks = int(filled)
    partial_idx = int(8 * filled) % 8
    partial = " ▏▎▍▌▋▊▉"[partial_idx] if partial_idx else ""
    bar = "█" * full_blocks + partial
    return bar.ljust(width)


@dataclass
class SampleState:
    """State of a single sample."""

    id: str
    name: str = ""
    turn: int = 0
    phase: str = ""  # streaming, compiling, checking, etc.
    score: float | None = None
    status: str = "started"  # started, complete, retry
    retry_attempt: int = 0
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)  # For sorting by recency


@dataclass
class RenderState:
    """Display state derived from events."""

    eval_name: str = ""
    total: int = 0
    completed: int = 0
    samples: dict[str, SampleState] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    # GEPA state
    gepa_iter: int | None = None
    gepa_total: int | None = None
    gepa_best: float | None = None

    # Histogram data (scores per metric)
    scores: list[float] = field(default_factory=list)


def derive_state(events: list[dict]) -> RenderState:
    """Derive current display state from event stream. Stateless."""
    state = RenderState()

    for event in events:
        event_type = event.get("type")
        if event_type is None:
            continue

        # Strict validation: crash on unknown event types
        if event_type not in KNOWN_EVENT_TYPES:
            raise ValueError(f"Unknown event type: {event_type}. Known types: {KNOWN_EVENT_TYPES}")

        if event_type == "eval_start":
            state.eval_name = event.get("name", "eval")
            state.total = event.get("total", 0)
            # Use event timestamp if available, otherwise fall back to now
            if ts := event.get("timestamp"):
                from datetime import datetime

                state.start_time = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            else:
                state.start_time = time.time()

        elif event_type == "sample_start":
            sample_id = event["id"]
            now = time.time()
            state.samples[sample_id] = SampleState(
                id=sample_id,
                name=event.get("name", sample_id),
                last_update=now,
                phase="starting",  # Default phase so samples show immediately
            )

        elif event_type == "turn":
            sample_id = event["id"]
            if sample_id in state.samples:
                # Only update turn if explicitly provided (status updates don't include turn)
                if "turn" in event:
                    state.samples[sample_id].turn = event["turn"]
                state.samples[sample_id].last_update = time.time()
                # Set phase from turn status if provided, or default to "running"
                # This ensures samples show up even without modal_progress events
                status = event.get("status", "running")
                state.samples[sample_id].phase = status

        elif event_type == "modal_progress":
            sample_id = event["id"]
            if sample_id in state.samples:
                state.samples[sample_id].phase = event.get("phase", "")
                state.samples[sample_id].last_update = time.time()

        elif event_type == "sample_end":
            sample_id = event["id"]
            if sample_id in state.samples:
                state.samples[sample_id].status = "complete"
                score = event.get("score")
                if score is not None:
                    state.samples[sample_id].score = score
                    state.scores.append(score)
                state.completed += 1

        elif event_type == "sample_retry":
            sample_id = event["id"]
            if sample_id in state.samples:
                state.samples[sample_id].status = "retry"
                state.samples[sample_id].retry_attempt = event.get("attempt", 1)

        elif event_type == "gepa_iteration":
            state.gepa_iter = event.get("iter")
            state.gepa_total = event.get("total")
            state.gepa_best = event.get("best")

        elif event_type == "eval_end":
            pass  # Will trigger exit in render loop

    return state


def render(state: RenderState, width: int, height: int) -> list[str]:
    """Render state to list of lines. Pure function."""
    lines = []

    # Header with GEPA or eval progress
    if state.gepa_iter is not None:
        header = (
            f"GEPA iter {state.gepa_iter}/{state.gepa_total or '?'} │ best: {state.gepa_best:.0%}"
            if state.gepa_best
            else f"GEPA iter {state.gepa_iter}/{state.gepa_total or '?'}"
        )
    else:
        elapsed = time.time() - state.start_time
        if state.total > 0:
            progress = state.completed / state.total
            bar_width = min(30, width - 50)
            bar = _format_bar(progress, bar_width)
            eta = ""
            if state.completed > 0 and progress < 1.0:
                remaining = (elapsed / progress) - elapsed
                eta = f" <{_format_time(remaining)}"
            header = (
                f"{state.eval_name}: {BOLD}{state.completed}/{state.total}{RESET} "
                f"|{bar}| {100 * progress:3.0f}% "
                f"[{_format_time(elapsed)}{eta}]"
            )
        else:
            header = f"{state.eval_name}: {state.completed} samples [{_format_time(elapsed)}]"

    lines.append(header[:width])

    # Sample list - only show active (have a phase), sorted by most recently updated
    # This matches uv's behavior: most active items float to the top
    active = [s for s in state.samples.values() if s.status != "complete" and s.phase]
    active.sort(key=lambda s: s.last_update, reverse=True)

    # Reserve lines: 1 header + 1 for "... and X more" + 1 for score summary
    max_samples = max(1, height - 4)

    for sample in active[:max_samples]:
        line = _format_sample_row(sample, width)
        lines.append(line)

    # Show how many more are in flight (including non-active)
    total_in_flight = sum(1 for s in state.samples.values() if s.status != "complete")
    hidden = total_in_flight - len(active[:max_samples])
    if hidden > 0:
        lines.append(f"{DIM}  ... and {hidden} more in flight{RESET}")

    # Score summary (compact)
    if state.scores:
        mean_score = sum(state.scores) / len(state.scores)
        lines.append(f"{DIM}score: {mean_score:.1%}{RESET}")

    return lines


def _format_sample_row(sample: SampleState, width: int) -> str:
    """Format a single sample row."""
    name = sample.name[:25].ljust(25)
    turn_info = f"T:{sample.turn}"

    if sample.status == "complete":
        if sample.score is not None:
            icon = f"{GREEN}✓{RESET}" if sample.score > 0.5 else f"{RED}✗{RESET}"
            return f"  {name} {turn_info:>4}  {icon} score={sample.score:.2f}"
        else:
            return f"  {name} {turn_info:>4}  {GREEN}✓{RESET} done"
    elif sample.status == "retry":
        return f"  {name} {turn_info:>4}  {YELLOW}⟳{RESET} retry (attempt {sample.retry_attempt})"
    else:
        phase = sample.phase or "running"
        return f"  {name} {turn_info:>4}  {CYAN}{phase}...{RESET}"


class ProgressDisplay:
    """In-place progress display that reads from JSONL file.

    Renders progress by overwriting lines in place (no alternate screen).
    Similar to how MultiProgress and the rollouts chat CLI work.
    """

    def __init__(
        self,
        events_file: Path,
        poll_interval: float = 0.2,
        output_stream: TextIO | None = None,
    ) -> None:
        self.events_file = events_file
        self.poll_interval = poll_interval
        self._output: TextIO = output_stream or sys.stdout
        self._stop_event = threading.Event()
        self._file_pos = 0
        self._events: list[dict] = []
        self._lines_rendered = 0
        self._old_sigwinch = None

    def start(self) -> None:
        """Start rendering (hide cursor)."""
        self._output.write(HIDE_CURSOR)
        self._output.flush()

        # Install resize handler
        self._old_sigwinch = signal.signal(signal.SIGWINCH, self._handle_resize)

        # Register atexit for cleanup
        atexit.register(self._cleanup)

    def stop(self) -> None:
        """Stop rendering and restore terminal."""
        self._stop_event.set()
        self._cleanup()

    def _cleanup(self) -> None:
        """Restore terminal state."""
        # Show cursor
        self._output.write(SHOW_CURSOR)
        self._output.flush()

        # Restore signal handlers
        if self._old_sigwinch is not None:
            signal.signal(signal.SIGWINCH, self._old_sigwinch)

        try:
            atexit.unregister(self._cleanup)
        except Exception:
            pass

    def _handle_resize(self, signum: int, frame: FrameType | None) -> None:
        """Handle terminal resize."""
        self._render()

    def run(self) -> None:
        """Main render loop. Polls file, derives state, renders."""
        while not self._stop_event.is_set():
            self._poll_events()
            self._render()

            # Check for eval_end event
            for event in self._events:
                if event.get("type") == "eval_end":
                    # Final render then exit
                    self._render()
                    return

            time.sleep(self.poll_interval)

    def _poll_events(self) -> None:
        """Read new events from file."""
        if not self.events_file.exists():
            return

        try:
            with open(self.events_file) as f:
                f.seek(self._file_pos)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            self._events.append(event)
                        except json.JSONDecodeError:
                            pass  # Skip malformed lines
                self._file_pos = f.tell()
        except OSError:
            pass  # File might be locked

    def _render(self) -> None:
        """Render current state by overwriting previous lines."""
        try:
            size = os.get_terminal_size()
            width, height = size.columns, size.lines
        except OSError:
            width, height = 80, 24

        state = derive_state(self._events)
        lines = render(state, width, height)

        # Fixed display height - always use same number of lines to avoid jumpiness
        display_height = min(12, height - 2)  # Cap at 12 lines

        # Pad or truncate to fixed height
        while len(lines) < display_height:
            lines.append("")
        lines = lines[:display_height]

        # Build complete output buffer with synchronized output
        buf = [SYNC_START]  # Begin synchronized update

        # Move cursor up to overwrite previous render
        if self._lines_rendered > 0:
            buf.append(CURSOR_UP.format(n=self._lines_rendered))

        # Write all lines (fixed count)
        for line in lines:
            buf.append(CLEAR_LINE + line + "\n")

        buf.append(SYNC_END)  # End synchronized update

        # Single write + flush
        self._output.write("".join(buf))
        self._output.flush()
        self._lines_rendered = len(lines)


@contextmanager
def progress_display(
    output_dir: Path | str,
    disable: bool = False,
    suppress_output: bool = True,
) -> Generator[Path, None, None]:
    """Context manager for clean progress display.

    Reads events.jsonl written by EventEmitter (from evaluate()) and
    renders a progress display in alternate screen.

    Args:
        output_dir: Directory containing events.jsonl (required - evaluate() writes here)
        disable: If True, skip progress display (verbose mode)
        suppress_output: If True, redirect stdout/stderr to log file to prevent display glitches

    Usage:
        # The output_dir must match what you pass to EvalConfig
        with progress_display(output_dir=config.output_dir):
            await evaluate(dataset, config)
    """
    if disable:
        yield Path(output_dir)
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_file = output_dir / "events.jsonl"

    # Save original stdout/stderr for display rendering
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = None

    if suppress_output:
        # Redirect stdout/stderr to a log file to prevent display glitches
        # The progress display will write directly to original_stdout
        log_file = open(output_dir / "output.log", "w")
        sys.stdout = log_file
        sys.stderr = log_file

    # Create and start display (uses original_stdout for rendering)
    display = ProgressDisplay(events_file, output_stream=original_stdout)
    display.start()

    # Run display in background thread
    display_thread = threading.Thread(target=display.run, daemon=True)
    display_thread.start()

    try:
        yield output_dir
    except KeyboardInterrupt:
        display.stop()
        raise
    except Exception:
        display.stop()
        raise
    finally:
        display.stop()
        # Restore stdout/stderr
        if suppress_output:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if log_file:
                log_file.close()
