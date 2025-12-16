"""
Main application - ties together terminal, git, and rendering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from .git import FileDiff, get_diff, diff_files, Hunk, HunkLine
from .terminal import Terminal
from .theme import DEFAULT_THEME, RESET, Theme


class Mode(Enum):
    NORMAL = "normal"
    SEARCH = "search"


@dataclass
class AppState:
    """Application state."""

    files: list[FileDiff] = field(default_factory=list)
    file_index: int = 0
    scroll_offset: int = 0
    mode: Mode = Mode.NORMAL
    search_query: str = ""
    search_matches: list[int] = field(default_factory=list)
    search_match_index: int = 0
    message: str = ""


class App:
    """Main diff viewer application."""

    def __init__(
        self,
        ref: str | None = None,
        staged: bool = False,
        cwd: str | None = None,
        theme: Theme | None = None,
    ) -> None:
        self.ref = ref
        self.staged = staged
        self.cwd = cwd
        self.theme = theme or DEFAULT_THEME
        self.terminal = Terminal(use_alternate_screen=True)
        self.state = AppState()
        self._running = False
        self._needs_redraw = True
        self._cached_diff_lines: list[str] = []

    def run(self) -> None:
        """Run the application."""
        self.state.files = get_diff(ref=self.ref, staged=self.staged, cwd=self.cwd)

        if not self.state.files:
            print("No changes to display.")
            return

        self._running = True
        self.terminal.start(on_input=lambda x: None, on_resize=self._on_resize)

        try:
            self._main_loop()
        finally:
            self.terminal.stop()

    def _on_resize(self) -> None:
        self._cached_diff_lines = []  # Invalidate cache
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
        """Handle keyboard input."""
        state = self.state

        if state.mode == Mode.SEARCH:
            self._handle_search_input(data)
            return

        # Quit
        if data == "q":
            self._running = False
            return

        # Navigation
        if data in ("j", "\x1b[B"):  # j or Down
            self._scroll(1)
        elif data in ("k", "\x1b[A"):  # k or Up
            self._scroll(-1)
        elif data == "\x04":  # Ctrl+D
            self._scroll(self.terminal.rows // 2)
        elif data == "\x15":  # Ctrl+U
            self._scroll(-self.terminal.rows // 2)
        elif data == "g":
            next_char = self._wait_for_char()
            if next_char == "g":
                state.scroll_offset = 0
                self._needs_redraw = True
        elif data == "G":
            state.scroll_offset = max(0, self._get_total_lines() - self.terminal.rows + 2)
            self._needs_redraw = True

        # File navigation
        elif data in ("h", "\x1b[D"):  # h or Left
            self._prev_file()
        elif data in ("l", "\x1b[C"):  # l or Right
            self._next_file()
        elif data == "H":
            state.file_index = 0
            state.scroll_offset = 0
            self._cached_diff_lines = []
            self._needs_redraw = True
        elif data == "L":
            state.file_index = len(state.files) - 1
            state.scroll_offset = 0
            self._cached_diff_lines = []
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

        # Search
        elif data == "/":
            state.mode = Mode.SEARCH
            state.search_query = ""
            self._needs_redraw = True
        elif data == "n":
            self._next_search_match()
        elif data == "N":
            self._prev_search_match()

        # Escape clears message
        elif data == "\x1b":
            state.message = ""
            self._needs_redraw = True

    def _wait_for_char(self, timeout: float = 0.5) -> str | None:
        start = time.time()
        while time.time() - start < timeout:
            data = self.terminal.read_input()
            if data:
                return data
            time.sleep(0.01)
        return None

    def _handle_search_input(self, data: str) -> None:
        state = self.state

        if data == "\x1b":  # Escape
            state.mode = Mode.NORMAL
            state.search_query = ""
            self._needs_redraw = True
        elif data == "\r":  # Enter
            state.mode = Mode.NORMAL
            self._do_search()
            self._needs_redraw = True
        elif data in ("\x7f", "\x08"):  # Backspace
            state.search_query = state.search_query[:-1]
            self._needs_redraw = True
        elif len(data) == 1 and ord(data) >= 32:
            state.search_query += data
            self._needs_redraw = True

    def _scroll(self, delta: int) -> None:
        state = self.state
        max_scroll = max(0, self._get_total_lines() - self.terminal.rows + 2)
        state.scroll_offset = max(0, min(max_scroll, state.scroll_offset + delta))
        self._needs_redraw = True

    def _prev_file(self) -> None:
        if self.state.file_index > 0:
            self.state.file_index -= 1
            self.state.scroll_offset = 0
            self._cached_diff_lines = []
            self._needs_redraw = True

    def _next_file(self) -> None:
        if self.state.file_index < len(self.state.files) - 1:
            self.state.file_index += 1
            self.state.scroll_offset = 0
            self._cached_diff_lines = []
            self._needs_redraw = True

    def _get_total_lines(self) -> int:
        if not self.state.files:
            return 0
        file_diff = self.state.files[self.state.file_index]
        total = 0
        for hunk in file_diff.hunks:
            total += len(hunk.lines)
        return total

    def _next_hunk(self) -> None:
        file_diff = self.state.files[self.state.file_index]
        current_line = 0

        for hunk in file_diff.hunks:
            hunk_start = current_line
            if hunk_start > self.state.scroll_offset:
                self.state.scroll_offset = hunk_start
                self._needs_redraw = True
                return
            current_line += len(hunk.lines)

    def _prev_hunk(self) -> None:
        file_diff = self.state.files[self.state.file_index]
        hunk_positions = []
        current_line = 0

        for hunk in file_diff.hunks:
            hunk_positions.append(current_line)
            current_line += len(hunk.lines)

        for pos in reversed(hunk_positions):
            if pos < self.state.scroll_offset:
                self.state.scroll_offset = pos
                self._needs_redraw = True
                return

    def _do_search(self) -> None:
        query = self.state.search_query.lower()
        if not query:
            return

        file_diff = self.state.files[self.state.file_index]
        matches = []
        line_num = 0

        for hunk in file_diff.hunks:
            for hunk_line in hunk.lines:
                if query in hunk_line.content.lower():
                    matches.append(line_num)
                line_num += 1

        self.state.search_matches = matches
        self.state.search_match_index = 0

        if matches:
            self.state.scroll_offset = matches[0]
            self.state.message = f"/{query}: {len(matches)} matches"
        else:
            self.state.message = f"/{query}: no matches"

    def _next_search_match(self) -> None:
        if not self.state.search_matches:
            return
        self.state.search_match_index = (self.state.search_match_index + 1) % len(self.state.search_matches)
        self.state.scroll_offset = self.state.search_matches[self.state.search_match_index]
        self._needs_redraw = True

    def _prev_search_match(self) -> None:
        if not self.state.search_matches:
            return
        self.state.search_match_index = (self.state.search_match_index - 1) % len(self.state.search_matches)
        self.state.scroll_offset = self.state.search_matches[self.state.search_match_index]
        self._needs_redraw = True

    def _render(self) -> None:
        """Render the full screen."""
        width = self.terminal.columns
        height = self.terminal.rows
        state = self.state

        output = []

        # Header (1 line)
        output.append(self._render_header(width))

        # Diff content (height - 2 for header and footer)
        content_height = height - 2

        if not self._cached_diff_lines:
            self._cached_diff_lines = self._render_diff_lines(width)

        visible_lines = self._cached_diff_lines[state.scroll_offset : state.scroll_offset + content_height]

        # Pad with empty lines
        empty_line = self._style_line("~", " " * (width - 1), "context", width)
        while len(visible_lines) < content_height:
            visible_lines.append(empty_line)

        output.extend(visible_lines)

        # Footer (1 line)
        output.append(self._render_footer(width))

        self.terminal.move_cursor(1, 1)
        self.terminal.write("\n".join(output))

    def _render_header(self, width: int) -> str:
        """Render header with filename and stats."""
        state = self.state
        file_diff = state.files[state.file_index] if state.files else None

        if not file_diff:
            bg = "\x1b[48;2;40;44;52m"
            reset = "\x1b[0m"
            return f"{bg}{' ' * width}{reset}"

        # Count additions/deletions
        additions = 0
        deletions = 0
        for hunk in file_diff.hunks:
            for line in hunk.lines:
                if line.type == "add":
                    additions += 1
                elif line.type == "remove":
                    deletions += 1

        # Build visible parts first (no ANSI)
        filename = file_diff.path
        nav = f"[{state.file_index + 1}/{len(state.files)}]"
        if len(state.files) > 1:
            nav += " h/l"

        visible_left = f" {filename}"
        visible_middle = f" +{additions} -{deletions}"
        visible_right = f" {nav} "

        total_visible = len(visible_left) + len(visible_middle) + len(visible_right)
        padding = max(0, width - total_visible)

        # Now build with ANSI colors
        green = "\x1b[38;2;152;195;121m"
        red = "\x1b[38;2;224;108;117m"
        dim = "\x1b[38;5;245m"
        bold = "\x1b[1m"
        reset = "\x1b[0m"
        bg = "\x1b[48;2;40;44;52m"

        styled_left = f" {bold}{filename}{reset}{bg}"
        styled_middle = f" {green}+{additions}{reset}{bg} {red}-{deletions}{reset}{bg}"
        styled_right = f" {dim}{nav}{reset}{bg} "

        return f"{bg}{styled_left}{styled_middle}{' ' * padding}{styled_right}{reset}"

    def _render_diff_lines(self, width: int) -> list[str]:
        """Render diff content lines."""
        if not self.state.files:
            return []

        file_diff = self.state.files[self.state.file_index]
        lines = []

        # Calculate line number width based on max line number
        max_ln = 0
        for hunk in file_diff.hunks:
            max_ln = max(max_ln, hunk.old_start + hunk.old_count, hunk.new_start + hunk.new_count)
        ln_width = max(4, len(str(max_ln)))

        for hunk in file_diff.hunks:
            old_ln = hunk.old_start
            new_ln = hunk.new_start

            for hl in hunk.lines:
                if hl.type == "context":
                    ln_str = f"{old_ln:>{ln_width}}  {new_ln:>{ln_width}}"
                    old_ln += 1
                    new_ln += 1
                elif hl.type == "add":
                    ln_str = f"{' ':>{ln_width}}  {new_ln:>{ln_width}}"
                    new_ln += 1
                else:  # remove
                    ln_str = f"{old_ln:>{ln_width}}  {' ':>{ln_width}}"
                    old_ln += 1

                lines.append(self._style_line(ln_str, hl.content, hl.type, width))

        return lines

    def _style_line(self, ln_str: str, content: str, line_type: str, width: int) -> str:
        """Style a single diff line with background colors like Critique."""
        reset = "\x1b[0m"

        # Line number styling
        ln_bg = "\x1b[48;2;5;5;5m"        # Very dark for line numbers
        ln_fg = "\x1b[38;5;243m"          # Dim gray text

        if line_type == "add":
            ln_bg = "\x1b[48;2;0;50;0m"   # Dark green for added line numbers
            content_bg = "\x1b[48;2;100;250;120;12m".replace(";12m", "m")  # Can't do alpha, use solid
            content_bg = "\x1b[48;2;30;50;30m"  # Dark green tint
            content_fg = "\x1b[38;2;152;195;121m"  # Bright green text
            prefix = "+"
        elif line_type == "remove":
            ln_bg = "\x1b[48;2;60;0;0m"   # Dark red for removed line numbers
            content_bg = "\x1b[48;2;50;30;30m"  # Dark red tint
            content_fg = "\x1b[38;2;224;108;117m"  # Bright red text
            prefix = "-"
        else:  # context
            content_bg = "\x1b[48;2;15;15;15m"  # Very dark gray
            content_fg = "\x1b[38;5;250m"  # Light gray text
            prefix = " "

        # Build the line
        ln_part = f"{ln_bg}{ln_fg} {ln_str} {reset}"

        # Content part - pad to fill width
        content_with_prefix = f"{prefix} {content}"
        ln_visible_width = len(ln_str) + 2  # +2 for spaces around it
        content_width = width - ln_visible_width - 1  # -1 for the separator

        if len(content_with_prefix) > content_width:
            content_with_prefix = content_with_prefix[:content_width - 1] + "~"

        padded_content = content_with_prefix + " " * max(0, content_width - len(content_with_prefix))
        content_part = f"{content_bg}{content_fg}{padded_content}{reset}"

        return f"{ln_part}{content_part}"

    def _render_footer(self, width: int) -> str:
        """Render footer with keybind hints."""
        state = self.state

        bg = "\x1b[48;2;40;44;52m"
        fg = "\x1b[38;5;250m"
        dim = "\x1b[38;5;243m"
        reset = "\x1b[0m"

        if state.mode == Mode.SEARCH:
            left = f"/{state.search_query}_"
        elif state.message:
            left = f" {state.message}"
        else:
            left = ""

        # Keybind hints
        hints = "j/k:scroll  ]c/[c:hunk  /:search  q:quit"

        # Scroll position
        total = self._get_total_lines()
        if total > 0:
            pct = min(100, int((state.scroll_offset / max(1, total)) * 100))
            right = f" {pct}% "
        else:
            right = " Top "

        padding = width - len(left) - len(hints) - len(right)
        if padding < 0:
            hints = ""
            padding = width - len(left) - len(right)

        line = f"{left}{' ' * max(0, padding)}{dim}{hints}{reset}{right}"
        padded = line + " " * max(0, width - len(left) - max(0, padding) - len(hints) - len(right))

        return f"{bg}{fg}{padded}{reset}"


class AppFilePair(App):
    """Diff viewer for two files (used by git difftool)."""

    def __init__(
        self,
        local_path: str,
        remote_path: str,
        name: str | None = None,
        theme: Theme | None = None,
    ) -> None:
        self.local_path = local_path
        self.remote_path = remote_path
        self.name = name or remote_path
        self.theme = theme or DEFAULT_THEME
        self.terminal = Terminal(use_alternate_screen=True)
        self.state = AppState()
        self._running = False
        self._needs_redraw = True
        self._cached_diff_lines: list[str] = []

    def run(self) -> None:
        """Run the application."""
        file_diff = diff_files(self.local_path, self.remote_path, self.name)

        if not file_diff.hunks:
            print("Files are identical.")
            return

        self.state.files = [file_diff]
        self._running = True
        self.terminal.start(on_input=lambda x: None, on_resize=self._on_resize)

        try:
            self._main_loop()
        finally:
            self.terminal.stop()
