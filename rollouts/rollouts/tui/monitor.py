"""Training monitor TUI - multi-pane log viewer.

Reads unified JSONL stream from stdin, routes to panes by logger name.
Vim-style keybindings for navigation.

Usage:
    # Pipe from bifrost
    bifrost exec 'python -m rollouts.tui.remote_runner ...' | python -m rollouts.tui.monitor

    # Or with a local JSONL file
    tail -f training.jsonl | python -m rollouts.tui.monitor

Keybindings:
    1/2/3   - Switch to pane (training/sglang/metrics)
    j/k     - Scroll down/up (or switch charts on metrics pane)
    g/G     - Go to top/bottom
    q       - Quit

TODO: Support viewing old runs via results directory
    - Store experiment outputs in results/{experiment}_{timestamp}/ dirs
      (like wafer_stuff/clicker: "07_sglang_smoke_08_20251108-040540")
    - Each dir contains: config.json, metrics.jsonl, training.log, sglang.log
    - Add CLI flag: python -m rollouts.tui.monitor --replay results/grpo_01_01_20251216-143022/
    - TUI reads from log files instead of stdin, allows scrubbing through history
    - Could add timeline scrubber at bottom to jump to specific step
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field

from .terminal import Terminal

# Colors (ANSI)
DIM = "\x1b[90m"
WHITE = "\x1b[37m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"
BG_HEADER = "\x1b[48;2;40;44;52m"  # Dark background for header/footer

# Sparkline characters (8 levels)
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 20) -> str:
    """Render a sparkline from values.

    Args:
        values: List of numeric values
        width: Max width of sparkline

    Returns:
        String of unicode block characters
    """
    if not values:
        return ""

    # Take last `width` values
    values = values[-width:]

    # Normalize to 0-7 range
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if range_val == 0:
        # All same value - show middle
        return SPARK_CHARS[4] * len(values)

    result = []
    for v in values:
        normalized = (v - min_val) / range_val  # 0.0 to 1.0
        idx = min(7, int(normalized * 8))  # 0 to 7
        result.append(SPARK_CHARS[idx])

    return "".join(result)


@dataclass
class LogLine:
    """A parsed log line."""

    logger: str
    message: str
    level: str = "INFO"
    extra: dict = field(default_factory=dict)


@dataclass
class Pane:
    """A scrollable log pane."""

    name: str
    lines: deque[LogLine] = field(default_factory=lambda: deque(maxlen=10000))
    scroll: int = 0
    auto_scroll: bool = True  # Follow tail

    def add_line(self, line: LogLine) -> None:
        self.lines.append(line)
        if self.auto_scroll:
            # Keep scroll at bottom
            self.scroll = max(0, len(self.lines) - 1)

    def scroll_up(self, n: int = 1) -> None:
        self.scroll = max(0, self.scroll - n)
        self.auto_scroll = False

    def scroll_down(self, n: int, visible_height: int) -> None:
        max_scroll = max(0, len(self.lines) - visible_height)
        self.scroll = min(max_scroll, self.scroll + n)
        # Re-enable auto scroll if at bottom
        if self.scroll >= max_scroll:
            self.auto_scroll = True

    def scroll_to_top(self) -> None:
        self.scroll = 0
        self.auto_scroll = False

    def scroll_to_bottom(self, visible_height: int) -> None:
        self.scroll = max(0, len(self.lines) - visible_height)
        self.auto_scroll = True


class TrainingMonitor:
    """Multi-pane TUI for monitoring training logs."""

    def __init__(self) -> None:
        self.terminal = Terminal(use_alternate_screen=True)
        self.panes = {
            "training": Pane(name="Training"),
            "sglang": Pane(name="SGLang"),
            "metrics": Pane(name="Metrics"),
        }
        self.pane_order = ["training", "sglang", "metrics"]
        self.active_pane = "training"
        self._running = False
        self._needs_redraw = True
        self._stdin_buffer = ""

        # Metrics tracking (for sparklines)
        self._metrics: dict[str, deque[float]] = {}
        self._current_step = 0
        self._selected_metric = 0  # Index of currently viewed metric chart

    def route_log_line(self, line: LogLine) -> str:
        """Route log line to appropriate pane based on logger name."""
        logger = line.logger.lower()

        if "sglang" in logger:
            return "sglang"
        elif "metrics" in logger:
            return "metrics"
        else:
            return "training"

    def feed_line(self, raw: str) -> None:
        """Feed a single line to the monitor (for use with kerbal callback).

        Casey: Continuous granularity - can be called from external streaming.

        Args:
            raw: Raw log line (JSONL or plain text)
        """
        if not raw.strip():
            return

        log_line = self.parse_jsonl_line(raw.strip())
        if log_line:
            pane_name = self.route_log_line(log_line)
            self.panes[pane_name].add_line(log_line)
            self._needs_redraw = True

    def parse_jsonl_line(self, raw: str) -> LogLine | None:
        """Parse a JSONL line into LogLine."""
        try:
            data = json.loads(raw)

            # Check if this is a metrics entry (has step + numeric values)
            if "step" in data and any(
                isinstance(v, (int, float)) and k not in ("step", "timestamp")
                for k, v in data.items()
            ):
                # Extract and store metrics for sparklines
                step = data.get("step", 0)
                self._current_step = max(self._current_step, step)

                for key, value in data.items():
                    if key in ("step", "timestamp", "logger", "message", "level"):
                        continue
                    if isinstance(value, (int, float)):
                        if key not in self._metrics:
                            self._metrics[key] = deque(maxlen=100)
                        self._metrics[key].append(value)

                # Create a message summarizing the metrics
                metric_parts = []
                for key, value in data.items():
                    if key in ("step", "timestamp"):
                        continue
                    if isinstance(value, (int, float)):
                        metric_parts.append(f"{key}={value:.4f}")

                return LogLine(
                    logger="metrics",
                    message=f"[step {step}] " + "  ".join(metric_parts),
                    level="INFO",
                    extra=data,
                )

            return LogLine(
                logger=data.get("logger", "unknown"),
                message=data.get("message", raw),
                level=data.get("level", "INFO"),
                extra={k: v for k, v in data.items() if k not in ("logger", "message", "level")},
            )
        except json.JSONDecodeError:
            # Not JSON, treat as raw message
            return LogLine(logger="raw", message=raw)

    def read_stdin_nonblocking(self) -> list[str]:
        """Read available lines from stdin without blocking."""
        import select

        lines = []

        # Check if stdin has data
        while select.select([sys.stdin], [], [], 0)[0]:
            try:
                chunk = sys.stdin.read(4096)
                if not chunk:
                    break
                self._stdin_buffer += chunk
            except (OSError, BlockingIOError):
                break

        # Extract complete lines
        while "\n" in self._stdin_buffer:
            line, self._stdin_buffer = self._stdin_buffer.split("\n", 1)
            if line.strip():
                lines.append(line.strip())

        return lines

    def run(self) -> None:
        """Main TUI loop."""
        self._running = True
        self.terminal.start(on_input=lambda x: None, on_resize=self._on_resize)

        # Make stdin non-blocking
        import fcntl
        import os

        flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)

        try:
            self._main_loop()
        finally:
            # Restore stdin flags
            fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags)
            self.terminal.stop()

    def _on_resize(self) -> None:
        self._needs_redraw = True

    def _main_loop(self) -> None:
        while self._running:
            # Read and process incoming log lines
            for raw_line in self.read_stdin_nonblocking():
                log_line = self.parse_jsonl_line(raw_line)
                if log_line:
                    pane_name = self.route_log_line(log_line)
                    self.panes[pane_name].add_line(log_line)
                    self._needs_redraw = True

            # Handle keyboard input
            data = self.terminal.read_input()
            if data:
                self._handle_input(data)

            # Render if needed
            if self._needs_redraw:
                self._render()
                self._needs_redraw = False

            time.sleep(0.05)  # 20fps

    def _handle_input(self, data: str) -> None:
        pane = self.panes[self.active_pane]
        content_height = self.terminal.rows - 3  # header + footer + tab bar

        # Quit
        if data == "q":
            self._running = False
            return

        # Pane switching
        if data == "1":
            self.active_pane = "training"
            self._needs_redraw = True
        elif data == "2":
            self.active_pane = "sglang"
            self._needs_redraw = True
        elif data == "3":
            self.active_pane = "metrics"
            self._needs_redraw = True

        # Scrolling - metrics pane scrolls through charts, others scroll logs
        elif data in ("j", "\x1b[B"):  # Down
            if self.active_pane == "metrics" and self._metrics:
                # Scroll to next metric chart
                self._selected_metric = min(
                    self._selected_metric + 1, len(self._metrics) - 1
                )
            else:
                pane.scroll_down(1, content_height)
            self._needs_redraw = True
        elif data in ("k", "\x1b[A"):  # Up
            if self.active_pane == "metrics" and self._metrics:
                # Scroll to previous metric chart
                self._selected_metric = max(self._selected_metric - 1, 0)
            else:
                pane.scroll_up(1)
            self._needs_redraw = True
        elif data == "\x04":  # Ctrl+D - half page down
            pane.scroll_down(content_height // 2, content_height)
            self._needs_redraw = True
        elif data == "\x15":  # Ctrl+U - half page up
            pane.scroll_up(content_height // 2)
            self._needs_redraw = True
        elif data == "g":
            # gg = top
            next_char = self._wait_for_char()
            if next_char == "g":
                pane.scroll_to_top()
                self._needs_redraw = True
        elif data == "G":
            pane.scroll_to_bottom(content_height)
            self._needs_redraw = True

    def _wait_for_char(self, timeout: float = 0.5) -> str | None:
        start = time.time()
        while time.time() - start < timeout:
            data = self.terminal.read_input()
            if data:
                return data
            time.sleep(0.01)
        return None

    def _render(self) -> None:
        width = self.terminal.columns
        height = self.terminal.rows
        content_height = height - 3  # tab bar + header + footer

        self.terminal.clear_screen()

        output = []

        # Tab bar
        output.append(self._render_tab_bar(width))

        # Content - special handling for metrics pane
        if self.active_pane == "metrics" and self._metrics:
            # Use plotext chart (takes most of the space)
            chart_height = min(content_height - 5, 15)  # Leave room for log lines
            chart_lines = self._render_plotext_chart(width, chart_height)
            output.extend(chart_lines)

            # Show recent log lines below chart
            pane = self.panes[self.active_pane]
            remaining_height = content_height - len(chart_lines)
            visible_lines = list(pane.lines)[-remaining_height:]  # Show most recent

            for log_line in visible_lines:
                output.append(self._render_log_line(log_line, width))
        else:
            # Normal pane rendering
            pane = self.panes[self.active_pane]
            visible_lines = list(pane.lines)[pane.scroll : pane.scroll + content_height]

            for log_line in visible_lines:
                output.append(self._render_log_line(log_line, width))

        # Pad with empty lines
        while len(output) < height - 1:
            output.append(" " * width)

        # Footer
        output.append(self._render_footer(width))

        # Write to terminal
        for i, line in enumerate(output):
            self.terminal.write(f"\x1b[{i + 1};1H{line}")

    def _render_sparklines(self, width: int) -> list[str]:
        """Render sparkline header for metrics pane."""
        lines = []

        # Header
        lines.append(
            f"{BG_HEADER}{BOLD} Metrics (step {self._current_step}){RESET}{BG_HEADER}{' ' * (width - 20)}{RESET}"
        )

        # Sparkline width (leave room for label and current value)
        spark_width = min(40, width - 30)

        for name, values in sorted(self._metrics.items()):
            if not values:
                continue

            # Get current value and sparkline
            current = values[-1]
            spark = sparkline(list(values), width=spark_width)

            # Format: "  loss: 0.1234 ▁▂▃▄▅▆▇█"
            label = f"{name}:"
            value_str = f"{current:.4f}"

            line = f"  {CYAN}{label:>12}{RESET} {GREEN}{value_str:>10}{RESET} {spark}"
            lines.append(line[:width])

        # Add separator
        lines.append(f"{DIM}{'─' * width}{RESET}")

        return lines

    def _render_plotext_chart(self, width: int, height: int) -> list[str]:
        """Render single metric chart using plotext braille charts.

        Shows one metric at a time. Use j/k to scroll through metrics.
        """
        try:
            import plotext as plt
        except ImportError:
            return [f"{DIM}(plotext not installed){RESET}"]

        if not self._metrics:
            return []

        # Get sorted metric names for consistent ordering
        metric_names = sorted(self._metrics.keys())
        total_metrics = len(metric_names)

        # Clamp selected index
        self._selected_metric = max(0, min(self._selected_metric, total_metrics - 1))

        # Get the selected metric
        metric_name = metric_names[self._selected_metric]
        values = list(self._metrics[metric_name])

        if not values:
            return []

        # Clear previous plot
        plt.clf()

        # Plot the selected metric
        plt.plot(values, marker="braille")

        # Title shows metric name and navigation hint
        current_val = values[-1] if values else 0
        plt.title(f"{metric_name}: {current_val:.4f}  ({self._selected_metric + 1}/{total_metrics})")
        plt.xlabel(f"Step (latest: {self._current_step})")
        plt.plotsize(width - 2, height)
        plt.theme("dark")

        # Build and split into lines
        chart_str = plt.build()
        return chart_str.split("\n")

    def _render_tab_bar(self, width: int) -> str:
        """Render tab bar showing all panes."""
        tabs = []
        for i, name in enumerate(self.pane_order):
            pane = self.panes[name]
            num = str(i + 1)
            count = len(pane.lines)

            if name == self.active_pane:
                tabs.append(f"{BOLD}{WHITE}[{num}] {pane.name} ({count}){RESET}")
            else:
                tabs.append(f"{DIM}[{num}] {pane.name} ({count}){RESET}")

        tab_str = "  ".join(tabs)
        padding = width - len("[1] Training (0)  [2] SGLang (0)  [3] Metrics (0)")
        return f"{BG_HEADER} {tab_str}{' ' * max(0, padding)}{RESET}"

    def _render_log_line(self, line: LogLine, width: int) -> str:
        """Render a single log line."""
        # Color by level
        level_color = {
            "DEBUG": DIM,
            "INFO": WHITE,
            "WARNING": YELLOW,
            "ERROR": RED,
            "CRITICAL": RED + BOLD,
        }.get(line.level, WHITE)

        # Truncate message to fit
        msg = line.message
        if len(msg) > width - 1:
            msg = msg[: width - 4] + "..."

        return f"{level_color}{msg}{RESET}"

    def _render_footer(self, width: int) -> str:
        """Render footer with keybindings and scroll position."""
        pane = self.panes[self.active_pane]
        content_height = self.terminal.rows - 3

        # Different hints for metrics pane (j/k scrolls charts, not logs)
        if self.active_pane == "metrics" and self._metrics:
            hints = "1/2/3:pane  j/k:chart  q:quit"
        else:
            hints = "1/2/3:pane  j/k:scroll  gg/G:top/end  q:quit"

        # Scroll position (or metric position for metrics pane)
        if self.active_pane == "metrics" and self._metrics:
            metric_names = sorted(self._metrics.keys())
            total_metrics = len(metric_names)
            pos = f"chart {self._selected_metric + 1}/{total_metrics}"
        elif len(pane.lines) > 0:
            total = len(pane.lines)
            pos = f"{pane.scroll + 1}-{min(pane.scroll + content_height, total)}/{total}"
            if pane.auto_scroll:
                pos += " [FOLLOW]"
        else:
            pos = "0/0"

        visible_hints = len(hints)
        visible_pos = len(pos) + 2
        padding = width - visible_hints - visible_pos - 2

        return (
            f"{BG_HEADER} {DIM}{hints}{RESET}{BG_HEADER}{' ' * max(0, padding)}{WHITE}{pos} {RESET}"
        )


def main() -> None:
    """Entry point for CLI."""
    monitor = TrainingMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
