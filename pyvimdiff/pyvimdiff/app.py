"""
Diff viewer TUI - simple, correct rendering.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

from .terminal import Terminal
from .utils import pad_line, truncate

# Colors (hex)
GREEN = "#98c379"  # Added lines
RED = "#e06c75"  # Removed lines
DIM = "#5c6370"  # Context/line numbers
CYAN = "#56b6c2"  # Hunk headers
WHITE = "#abb2bf"  # Normal text
BG_HEADER = "#282c34"  # Header/footer background


def hex_to_fg(hex_color: str) -> str:
    """Convert hex to ANSI foreground."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"\x1b[38;2;{r};{g};{b}m"


def hex_to_bg(hex_color: str) -> str:
    """Convert hex to ANSI background."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"\x1b[48;2;{r};{g};{b}m"


RESET = "\x1b[0m"
BOLD = "\x1b[1m"


@dataclass
class DiffLine:
    """A single line in the diff."""

    type: str  # "add", "remove", "context", "hunk", "header"
    content: str
    old_num: int | None = None
    new_num: int | None = None


def parse_unified_diff(diff_text: str) -> list[DiffLine]:
    """Parse unified diff into structured lines."""
    lines = []
    old_num = 0
    new_num = 0

    for raw_line in diff_text.split("\n"):
        if raw_line.startswith("@@"):
            # Parse hunk header: @@ -old,count +new,count @@
            lines.append(DiffLine(type="hunk", content=raw_line))
            # Extract starting line numbers
            try:
                parts = raw_line.split()
                old_part = parts[1]  # -X,Y
                new_part = parts[2]  # +X,Y
                old_num = int(old_part.split(",")[0][1:])
                new_num = int(new_part.split(",")[0][1:])
            except (IndexError, ValueError):
                pass
        elif raw_line.startswith("---") or raw_line.startswith("+++"):
            lines.append(DiffLine(type="header", content=raw_line))
        elif raw_line.startswith("-"):
            lines.append(DiffLine(type="remove", content=raw_line[1:], old_num=old_num))
            old_num += 1
        elif raw_line.startswith("+"):
            lines.append(DiffLine(type="add", content=raw_line[1:], new_num=new_num))
            new_num += 1
        elif raw_line.startswith(" "):
            lines.append(
                DiffLine(type="context", content=raw_line[1:], old_num=old_num, new_num=new_num)
            )
            old_num += 1
            new_num += 1
        elif raw_line == "":
            # Empty line in diff (context)
            if old_num > 0:  # Only if we're inside a hunk
                lines.append(DiffLine(type="context", content="", old_num=old_num, new_num=new_num))
                old_num += 1
                new_num += 1

    return lines


class DiffViewer:
    """Single-file diff viewer."""

    def __init__(self, diff_text: str, filename: str = "") -> None:
        self.diff_text = diff_text
        self.filename = filename
        self.lines = parse_unified_diff(diff_text)
        self.scroll = 0
        self.terminal = Terminal(use_alternate_screen=True)
        self._running = False
        self._needs_redraw = True

    def run(self) -> None:
        if not self.lines:
            print("No diff to display.")
            return

        self._running = True
        self.terminal.start(on_input=lambda x: None, on_resize=self._on_resize)

        try:
            self._main_loop()
        finally:
            self.terminal.stop()

    def _on_resize(self) -> None:
        self._needs_redraw = True

    def _main_loop(self) -> None:
        while self._running:
            if self._needs_redraw:
                self._render()
                self._needs_redraw = False

            data = self.terminal.read_input()
            if data:
                self._handle_input(data)

            time.sleep(0.01)

    def _handle_input(self, data: str) -> None:
        # Quit
        if data == "q":
            self._running = False
            return

        height = self.terminal.rows
        content_height = height - 2  # header + footer
        max_scroll = max(0, len(self.lines) - content_height)

        # Scroll
        if data in ("j", "\x1b[B"):  # Down
            self.scroll = min(max_scroll, self.scroll + 1)
            self._needs_redraw = True
        elif data in ("k", "\x1b[A"):  # Up
            self.scroll = max(0, self.scroll - 1)
            self._needs_redraw = True
        elif data == "\x04":  # Ctrl+D
            self.scroll = min(max_scroll, self.scroll + content_height // 2)
            self._needs_redraw = True
        elif data == "\x15":  # Ctrl+U
            self.scroll = max(0, self.scroll - content_height // 2)
            self._needs_redraw = True
        elif data == "g":
            # gg = top
            next_char = self._wait_for_char()
            if next_char == "g":
                self.scroll = 0
                self._needs_redraw = True
        elif data == "G":
            self.scroll = max_scroll
            self._needs_redraw = True

        # Hunk navigation
        elif data == "]":
            next_char = self._wait_for_char()
            if next_char == "c":
                self._next_hunk()
        elif data == "[":
            next_char = self._wait_for_char()
            if next_char == "c":
                self._prev_hunk()

    def _wait_for_char(self, timeout: float = 0.5) -> str | None:
        start = time.time()
        while time.time() - start < timeout:
            data = self.terminal.read_input()
            if data:
                return data
            time.sleep(0.01)
        return None

    def _next_hunk(self) -> None:
        for i, line in enumerate(self.lines):
            if i > self.scroll and line.type == "hunk":
                self.scroll = i
                self._needs_redraw = True
                return

    def _prev_hunk(self) -> None:
        for i in range(self.scroll - 1, -1, -1):
            if self.lines[i].type == "hunk":
                self.scroll = i
                self._needs_redraw = True
                return

    def _render(self) -> None:
        width = self.terminal.columns
        height = self.terminal.rows
        content_height = height - 2

        output = []

        # Header
        output.append(self._render_header(width))

        # Content
        visible = self.lines[self.scroll : self.scroll + content_height]
        for line in visible:
            output.append(self._render_line(line, width))

        # Pad if needed
        while len(output) < height - 1:
            output.append(" " * width)

        # Footer
        output.append(self._render_footer(width))

        self.terminal.move_cursor(1, 1)
        self.terminal.write("\n".join(output))

    def _render_header(self, width: int) -> str:
        """Render header bar."""
        # Count stats
        adds = sum(1 for line in self.lines if line.type == "add")
        dels = sum(1 for line in self.lines if line.type == "remove")

        left = f" {self.filename}" if self.filename else " [diff]"

        fg_green = hex_to_fg(GREEN)
        fg_red = hex_to_fg(RED)
        bg = hex_to_bg(BG_HEADER)

        # Build with colors
        colored_stats = f" {fg_green}+{adds}{RESET}{bg} {fg_red}-{dels}{RESET}{bg}"

        # Calculate padding
        visible_left = len(left)
        visible_stats = len(f" +{adds} -{dels}")
        padding = width - visible_left - visible_stats

        return f"{bg}{BOLD}{left}{RESET}{bg}{' ' * max(0, padding)}{colored_stats} {RESET}"

    def _render_line(self, line: DiffLine, width: int) -> str:
        """Render a single diff line."""
        fg_green = hex_to_fg(GREEN)
        fg_red = hex_to_fg(RED)
        fg_dim = hex_to_fg(DIM)
        fg_cyan = hex_to_fg(CYAN)

        if line.type == "hunk":
            # Hunk header in cyan
            text = truncate(line.content, width)
            return f"{fg_cyan}{pad_line(text, width)}{RESET}"

        elif line.type == "header":
            # File header (--- / +++)
            text = truncate(line.content, width)
            return f"{fg_dim}{pad_line(text, width)}{RESET}"

        elif line.type == "add":
            # Added line: green
            ln = f"{line.new_num:>4}" if line.new_num else "    "
            marker = "+"
            content = line.content
            full = f"{ln} {marker} {content}"
            text = truncate(full, width)
            return f"{fg_green}{pad_line(text, width)}{RESET}"

        elif line.type == "remove":
            # Removed line: red
            ln = f"{line.old_num:>4}" if line.old_num else "    "
            marker = "-"
            content = line.content
            full = f"{ln} {marker} {content}"
            text = truncate(full, width)
            return f"{fg_red}{pad_line(text, width)}{RESET}"

        else:  # context
            # Context: dim
            ln_old = f"{line.old_num:>4}" if line.old_num else "    "
            marker = " "
            content = line.content
            full = f"{ln_old} {marker} {content}"
            text = truncate(full, width)
            return f"{fg_dim}{pad_line(text, width)}{RESET}"

    def _render_footer(self, width: int) -> str:
        """Render footer bar."""
        bg = hex_to_bg(BG_HEADER)
        fg = hex_to_fg(WHITE)
        fg_dim = hex_to_fg(DIM)

        hints = "j/k:scroll  gg/G:top/end  ]c/[c:hunk  q:quit"

        # Scroll position
        total = len(self.lines)
        if total > 0:
            pct = min(100, int((self.scroll / max(1, total)) * 100))
            pos = f"{pct}%"
        else:
            pos = "0%"

        visible_hints = len(hints)
        visible_pos = len(pos) + 2
        padding = width - visible_hints - visible_pos

        return f"{bg}{fg} {fg_dim}{hints}{RESET}{bg}{' ' * max(0, padding)}{fg}{pos} {RESET}"


def diff_files(local_path: str, remote_path: str) -> str:
    """Generate unified diff between two files."""
    result = subprocess.run(
        ["diff", "-u", local_path, remote_path],
        capture_output=True,
        text=True,
    )
    return result.stdout
