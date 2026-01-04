"""Multi-task progress display with ANSI terminal control.

Replaces the simple tqdm-style single bar with a multi-row display
showing all concurrent tasks, similar to uv's download progress.

Usage:
    # As a drop-in tqdm replacement (single task)
    with tqdm(total=100, desc="Processing") as pbar:
        for i in range(100):
            pbar.update(1)

    # Multi-task mode for concurrent evaluations
    async with MultiProgress(total=100, desc="Evaluating") as mp:
        mp.add_task("sample_0001", name="level1/matmul", total=10)
        mp.update_task("sample_0001", completed=3, status="generating...")
        mp.complete_task("sample_0001", success=True, message="✓ 2.3x")
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")  # Only used for legacy tqdm Generic

# ANSI escape codes
CURSOR_UP = "\x1b[{n}A"
CURSOR_DOWN = "\x1b[{n}B"
CLEAR_LINE = "\x1b[2K"
CURSOR_HOME = "\r"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"

# Colors
DIM = "\x1b[90m"
RESET = "\x1b[0m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"
BOLD = "\x1b[1m"


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
class TaskState:
    """State of a single task in the progress display."""

    task_id: str
    name: str
    turn: int = 0  # Current turn number
    status: str = ""
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    success: bool | None = None  # None = in progress, True/False = done
    message: str = ""  # Final message on completion

    @property
    def is_done(self) -> bool:
        return self.success is not None

    @property
    def elapsed(self) -> float:
        end = self.end_time or time.perf_counter()
        return end - self.start_time


class MultiProgress:
    """Multi-task progress display for concurrent operations.

    Shows a header with overall progress and one row per active task.
    Completed tasks show their final status before being removed or kept.

    Thread-safe for updates from multiple async tasks.
    """

    def __init__(
        self,
        total: int | None = None,
        desc: str = "",
        unit: str = "sample",
        disable: bool = False,
        keep_completed: bool = True,
        max_visible: int | None = None,  # None = show all
        verbose: bool = False,
    ) -> None:
        """Initialize multi-progress display.

        Args:
            total: Total number of items (for header progress)
            desc: Description prefix
            unit: Unit name (e.g., "sample", "task")
            disable: If True, don't render anything
            keep_completed: If True, keep completed tasks visible
            max_visible: Maximum number of task rows to show (None = show all)
            verbose: If True, show INFO/DEBUG logs; if False, only WARNING+
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.keep_completed = keep_completed
        self.max_visible = max_visible

        self.tasks: dict[str, TaskState] = {}
        self.task_order: list[str] = []  # Insertion order
        self.completed_count = 0
        self.start_time = time.perf_counter()

        self._lock = Lock()
        self._lines_rendered = 0
        self._last_render = 0.0
        self._render_interval = 0.05  # 20 FPS max

        # Recent log messages to display
        self._log_messages: list[str] = []
        self._max_log_messages = 3

        # Log level control - only show WARNING+ by default
        self._saved_log_level: int | None = None
        self._verbose = verbose

    def __enter__(self) -> MultiProgress:
        if not self.disable:
            # Hide cursor during rendering
            sys.stderr.write(HIDE_CURSOR)
            sys.stderr.flush()
            # Set log level: WARNING+ by default, INFO+ if verbose
            self._saved_log_level = logging.root.level
            if not self._verbose:
                logging.root.setLevel(logging.WARNING)
        return self

    def __exit__(self, *_: object) -> None:
        if not self.disable:
            # Final render and show cursor
            self._render(force=True, final=True)
            sys.stderr.write(SHOW_CURSOR)
            sys.stderr.flush()
            # Restore log level
            if self._saved_log_level is not None:
                logging.root.setLevel(self._saved_log_level)

    async def __aenter__(self) -> MultiProgress:
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        self.__exit__(*args)

    def add_task(
        self,
        task_id: str,
        name: str = "",
    ) -> None:
        """Add a new task to track.

        Args:
            task_id: Unique identifier for the task
            name: Display name
        """
        with self._lock:
            display_name = name or task_id
            self.tasks[task_id] = TaskState(
                task_id=task_id,
                name=display_name,
            )
            self.task_order.append(task_id)
        self._render()

    def update_task(
        self,
        task_id: str,
        turn: int | None = None,
        status: str | None = None,
    ) -> None:
        """Update a task's progress.

        Args:
            task_id: Task to update
            turn: Current turn number
            status: Status message (e.g., "generating...", "evaluating...")
        """
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            if turn is not None:
                task.turn = turn
            if status is not None:
                task.status = status
        self._render()

    def complete_task(
        self,
        task_id: str,
        success: bool = True,
        message: str = "",
    ) -> None:
        """Mark a task as complete.

        Args:
            task_id: Task to complete
            success: Whether the task succeeded
            message: Final status message (e.g., "✓ 2.3x speedup")
        """
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            task.success = success
            task.message = message
            task.end_time = time.perf_counter()
            self.completed_count += 1
        self._render()

    def log(self, message: str) -> None:
        """Add a log message to display below tasks.

        Use this for important status updates that should be visible
        in the progress display (e.g., "Retrying 3 failed samples...").

        Args:
            message: Message to display
        """
        with self._lock:
            self._log_messages.append(message)
            # Keep only recent messages
            if len(self._log_messages) > self._max_log_messages:
                self._log_messages.pop(0)
        self._render()

    def _render(self, force: bool = False, final: bool = False) -> None:
        """Render the progress display."""
        if self.disable:
            return

        # Rate limit rendering
        now = time.perf_counter()
        if not force and (now - self._last_render) < self._render_interval:
            return
        self._last_render = now

        with self._lock:
            lines = self._build_display(final)

        # Move cursor up to overwrite previous render
        if self._lines_rendered > 0:
            sys.stderr.write(CURSOR_UP.format(n=self._lines_rendered))

        # Write new lines
        for line in lines:
            sys.stderr.write(CLEAR_LINE + line + "\n")

        # Clear any extra lines from previous render
        extra_lines = self._lines_rendered - len(lines)
        for _ in range(extra_lines):
            sys.stderr.write(CLEAR_LINE + "\n")
        if extra_lines > 0:
            sys.stderr.write(CURSOR_UP.format(n=extra_lines))

        sys.stderr.flush()
        self._lines_rendered = len(lines)

    def _build_display(self, final: bool = False) -> list[str]:
        """Build the lines to display."""
        lines = []
        width = shutil.get_terminal_size().columns

        # Header: overall progress
        elapsed = time.perf_counter() - self.start_time
        if self.total:
            progress = self.completed_count / self.total
            bar_width = min(30, width - 50)
            bar = _format_bar(progress, bar_width)
            eta = ""
            if self.completed_count > 0 and progress < 1.0:
                remaining = (elapsed / progress) - elapsed
                eta = f"<{_format_time(remaining)}"
            header = (
                f"{self.desc}: {BOLD}{self.completed_count}/{self.total}{RESET} "
                f"|{bar}| {100 * progress:3.0f}% "
                f"[{_format_time(elapsed)}{eta}]"
            )
        else:
            header = f"{self.desc}: {self.completed_count} {self.unit}s [{_format_time(elapsed)}]"
        lines.append(header[:width])

        # Separator
        lines.append(DIM + "─" * min(60, width) + RESET)

        # Task rows
        visible_tasks = self._get_visible_tasks()
        name_width = max((len(t.name) for t in visible_tasks), default=20)
        name_width = min(name_width, 30)  # Cap at 30 chars

        for task in visible_tasks:
            line = self._format_task_row(task, name_width, width)
            lines.append(line)

        # Show count of hidden tasks (only if max_visible is set)
        if self.max_visible is not None:
            hidden_count = len(self.task_order) - len(visible_tasks)
            if hidden_count > 0:
                lines.append(f"{DIM}  ... and {hidden_count} more{RESET}")

        # Show recent log messages
        if self._log_messages:
            lines.append(DIM + "─" * min(60, width) + RESET)
            for msg in self._log_messages:
                lines.append(f"{DIM}  {msg}{RESET}"[:width])

        return lines

    def _get_visible_tasks(self) -> list[TaskState]:
        """Get tasks to display, respecting max_visible."""
        # Prioritize in-progress tasks, then recent completions
        in_progress = []
        completed = []

        for task_id in self.task_order:
            task = self.tasks.get(task_id)
            if not task:
                continue
            if task.is_done:
                if self.keep_completed:
                    completed.append(task)
            else:
                in_progress.append(task)

        # Show in-progress first, then completed (most recent first)
        visible = in_progress + list(reversed(completed))
        if self.max_visible is not None:
            return visible[: self.max_visible]
        return visible

    def _format_task_row(self, task: TaskState, name_width: int, total_width: int) -> str:
        """Format a single task row."""
        name = task.name[:name_width].ljust(name_width)

        if task.is_done:
            # Completed task
            if task.success:
                icon = f"{GREEN}✓{RESET}"
                msg = task.message or "done"
            else:
                icon = f"{RED}✗{RESET}"
                msg = task.message or "failed"
            turn_info = f"T:{task.turn}" if task.turn > 0 else ""
            elapsed = _format_time(task.elapsed)
            return f"  {name}  {icon} {turn_info} {msg} {DIM}({elapsed}){RESET}"
        else:
            # In-progress task
            turn_info = f"T:{task.turn}" if task.turn > 0 else "T:0"
            status = task.status or "running..."
            return f"  {name}  {turn_info:>4}  {CYAN}{status}{RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# Legacy tqdm compatibility wrapper
# ─────────────────────────────────────────────────────────────────────────────


class tqdm(Generic[T]):
    """Minimal tqdm-compatible progress bar.

    This is a thin wrapper for backwards compatibility. For concurrent
    task tracking, use MultiProgress directly.

    Supports:
    - ETA calculation
    - set_postfix() for custom metrics
    - Context manager and manual update
    """

    def __init__(
        self,
        total: int | None = None,
        desc: str = "",
        disable: bool = False,
        unit: str = "it",
        unit_scale: bool = False,
        rate: int = 100,
        bar_format: str | None = None,
    ) -> None:
        self.disable = disable
        self.unit = unit
        self.unit_scale = unit_scale
        self.rate = rate
        self.t = total
        self.bar_format = bar_format
        self.use_inverse_rate = (
            bar_format and "{rate_inv_fmt}" in bar_format if bar_format else False
        )

        # Timing and counters
        self.st = time.perf_counter()
        self.i = -1
        self.n = 0
        self.skip = 1

        # Postfix for custom metrics
        self.postfix_dict: dict[str, str] = {}

        self.set_description(desc)
        self.update(0)

    def __enter__(self) -> tqdm[T]:
        return self

    def __exit__(self, *_: object) -> None:
        self.update(close=True)

    def set_description(self, desc: str) -> None:
        """Set the description prefix."""
        self.desc = f"{desc}: " if desc else ""

    def set_postfix(self, postfix_dict: dict[str, str]) -> None:
        """Set custom metrics to display."""
        self.postfix_dict = postfix_dict

    def update(self, n: int = 0, close: bool = False) -> None:
        """Update progress by n items."""
        self.n += n
        self.i += 1

        if self.disable or (not close and self.i % self.skip != 0):
            return

        prog = self.n / self.t if self.t else 0
        elapsed = time.perf_counter() - self.st
        ncols = shutil.get_terminal_size().columns

        # Adaptive refresh rate
        if elapsed and self.i / elapsed > self.rate and self.i:
            self.skip = max(int(self.i / elapsed) // self.rate, 1)

        # Build progress text
        prog_text = f"{self.n}/{self.t}" if self.t else str(self.n)

        # ETA
        if self.t and self.n:
            remaining = (elapsed / prog) - elapsed
            eta = f"<{_format_time(remaining)}"
        else:
            eta = ""

        # Rate
        if self.n:
            if self.use_inverse_rate:
                rate_text = f"{elapsed / self.n:5.2f}s/{self.unit}"
            else:
                rate_text = f"{self.n / elapsed:5.2f}{self.unit}/s"
        else:
            rate_text = f"?{self.unit}/s"

        # Postfix
        postfix_str = ""
        if self.postfix_dict:
            postfix_str = ", " + ", ".join(f"{k}={v}" for k, v in self.postfix_dict.items())

        # Build suffix
        suf = f"{prog_text} [{_format_time(elapsed)}{eta}, {rate_text}{postfix_str}]"

        # Build bar
        bar_width = max(ncols - len(self.desc) - len(suf) - 10, 1)
        if self.t:
            bar = _format_bar(prog, bar_width)
            line = f"\r{self.desc}{100 * prog:3.0f}%|{bar}| {suf}"
        else:
            line = f"\r{self.desc}{suf}"

        print(line[:ncols], flush=True, end="\n" if close else "", file=sys.stderr)

    def close(self) -> None:
        """Finalize the progress bar."""
        self.update(close=True)
