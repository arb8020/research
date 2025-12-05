"""
Markdown component - renders markdown with ANSI styling.

Uses mistune for parsing and converts to ANSI-styled terminal output.
"""

from __future__ import annotations

from typing import Callable, Optional, List, Protocol
import re

from ..tui import Component
from ..theme import Theme, DARK_THEME, hex_to_fg, RESET
from ..utils import wrap_text_with_ansi
from .text import _render_line_with_margins, _render_empty_lines


class MarkdownTheme(Protocol):
    """Theme functions for markdown elements."""

    def heading(self, text: str) -> str: ...
    def link(self, text: str) -> str: ...
    def link_url(self, text: str) -> str: ...
    def code(self, text: str) -> str: ...
    def code_block(self, text: str) -> str: ...
    def code_block_border(self, text: str) -> str: ...
    def quote(self, text: str) -> str: ...
    def quote_border(self, text: str) -> str: ...
    def hr(self, text: str) -> str: ...
    def list_bullet(self, text: str) -> str: ...
    def bold(self, text: str) -> str: ...
    def italic(self, text: str) -> str: ...
    def strikethrough(self, text: str) -> str: ...
    def underline(self, text: str) -> str: ...


class DefaultMarkdownTheme:
    """Default markdown theme using colors from Theme."""

    def __init__(self, theme: Optional[Theme] = None) -> None:
        self._theme = theme or DARK_THEME

    def heading(self, text: str) -> str:
        # Golden heading (pi-mono style)
        return f"\x1b[1m{hex_to_fg(self._theme.md_heading)}{text}{RESET}"

    def link(self, text: str) -> str:
        return f"\x1b[4m{hex_to_fg(self._theme.md_link)}{text}{RESET}"

    def link_url(self, text: str) -> str:
        return f"{hex_to_fg(self._theme.md_link_url)}{text}{RESET}"

    def code(self, text: str) -> str:
        return f"{hex_to_fg(self._theme.md_code)}{text}{RESET}"

    def code_block(self, text: str) -> str:
        return f"{hex_to_fg(self._theme.md_code_block)}{text}{RESET}"

    def code_block_border(self, text: str) -> str:
        return f"{hex_to_fg(self._theme.md_code_border)}{text}{RESET}"

    def quote(self, text: str) -> str:
        return f"\x1b[3m{hex_to_fg(self._theme.md_quote)}{text}{RESET}"

    def quote_border(self, text: str) -> str:
        return f"{hex_to_fg(self._theme.md_quote_border)}{text}{RESET}"

    def hr(self, text: str) -> str:
        return f"{hex_to_fg(self._theme.md_hr)}{text}{RESET}"

    def list_bullet(self, text: str) -> str:
        return f"{hex_to_fg(self._theme.md_list_bullet)}{text}{RESET}"

    def bold(self, text: str) -> str:
        return f"\x1b[1m{text}\x1b[0m"

    def italic(self, text: str) -> str:
        return f"\x1b[3m{text}\x1b[0m"

    def strikethrough(self, text: str) -> str:
        return f"\x1b[9m{text}\x1b[0m"

    def underline(self, text: str) -> str:
        return f"\x1b[4m{text}\x1b[0m"


class Markdown(Component):
    """Component that renders markdown with ANSI styling."""

    def __init__(
        self,
        text: str,
        padding_x: int = 1,
        padding_y: int = 0,
        theme: Optional[MarkdownTheme] = None,
        bg_fn: Optional[Callable[[str], str]] = None,
        gutter_prefix: Optional[str] = None,
    ) -> None:
        self._text = text
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._theme = theme or DefaultMarkdownTheme()
        self._bg_fn = bg_fn
        self._gutter_prefix = gutter_prefix

        # Extract TUI theme if available (for use_compact_padding setting)
        self._tui_theme = getattr(self._theme, '_theme', None)

        # Cache
        self._cached_text: Optional[str] = None
        self._cached_width: Optional[int] = None
        self._cached_lines: Optional[List[str]] = None
        self._cached_gutter_prefix: Optional[str] = None

    def set_text(self, text: str) -> None:
        """Update the markdown text."""
        self._text = text
        self.invalidate()

    def invalidate(self) -> None:
        """Clear cached rendering."""
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None
        self._cached_gutter_prefix = None

    def render(self, width: int) -> List[str]:
        """Render markdown to styled lines."""
        # Check cache
        if (
            self._cached_lines is not None
            and self._cached_text == self._text
            and self._cached_width == width
            and self._cached_gutter_prefix == self._gutter_prefix
        ):
            return self._cached_lines

        # Empty text - return empty list
        if not self._text or self._text.strip() == "":
            self._cached_text = self._text
            self._cached_width = width
            self._cached_lines = []
            return []

        # Normalize and render markdown
        normalized_text = self._text.replace("\t", "   ")
        # Account for gutter prefix width if present
        gutter_width = 0
        if self._gutter_prefix:
            from ..utils import visible_width
            gutter_width = visible_width(self._gutter_prefix) + 1  # +1 for space after prefix
        content_width = max(1, width - self._padding_x * 2 - gutter_width)
        rendered_lines = self._render_markdown(normalized_text, content_width)

        # Wrap lines
        wrapped_lines: List[str] = []
        for line in rendered_lines:
            wrapped_lines.extend(wrap_text_with_ansi(line, content_width))

        # Render content lines with margins and background
        # If we have a gutter, render to reduced width so final line fits after adding gutter
        render_width = width - gutter_width if gutter_width > 0 else width
        content_lines = [
            _render_line_with_margins(line, render_width, self._padding_x, self._bg_fn)
            for line in wrapped_lines
        ]

        # Add vertical padding (use render_width if we have a gutter)
        use_compact = self._tui_theme and getattr(self._tui_theme, "use_compact_padding", False)
        top_lines = _render_empty_lines(self._padding_y, render_width, self._padding_x, self._bg_fn, use_compact, is_top=True)
        bottom_lines = _render_empty_lines(self._padding_y, render_width, self._padding_x, self._bg_fn, use_compact, is_top=False)

        result = [*top_lines, *content_lines, *bottom_lines]

        # Add gutter prefix if specified
        if self._gutter_prefix:
            from ..utils import visible_width
            gutter_len = visible_width(self._gutter_prefix)
            # Find the position where visible characters start (skip ANSI codes)
            def skip_ansi_and_get_pos(line, visible_chars_to_skip):
                """Find byte position after skipping N visible characters, preserving ANSI."""
                pos = 0
                visible_count = 0
                while pos < len(line) and visible_count < visible_chars_to_skip:
                    if line[pos:pos+2] == '\x1b[':
                        # Skip ANSI escape sequence
                        end = line.find('m', pos)
                        if end != -1:
                            pos = end + 1
                        else:
                            pos += 1
                    else:
                        # Regular visible character
                        visible_count += 1
                        pos += 1
                return pos

            # Calculate which lines are padding vs content
            num_top_padding = len(top_lines)
            num_bottom_padding = len(bottom_lines)
            num_content = len(content_lines)

            new_result = []
            for i, line in enumerate(result):
                # Padding lines get spaces instead of gutter prefix
                if i < num_top_padding or i >= (num_top_padding + num_content):
                    # This is a padding line - add spaces instead of gutter
                    pos = skip_ansi_and_get_pos(line, self._padding_x)
                    new_result.append(" " * (gutter_len + 1) + line[pos:])
                elif i == num_top_padding:
                    # First content line: add gutter prefix
                    pos = skip_ansi_and_get_pos(line, self._padding_x)
                    new_result.append(self._gutter_prefix + " " + line[pos:])
                else:
                    # Other content lines: add spacing
                    pos = skip_ansi_and_get_pos(line, self._padding_x)
                    new_result.append(" " * (gutter_len + 1) + line[pos:])
            result = new_result

        # Update cache
        self._cached_text = self._text
        self._cached_width = width
        self._cached_gutter_prefix = self._gutter_prefix
        self._cached_lines = result

        return result

    def _render_markdown(self, text: str, width: int) -> List[str]:
        """Simple markdown renderer.

        Handles basic markdown without external dependencies.
        For full markdown support, use mistune library.
        """
        lines: List[str] = []
        in_code_block = False
        code_lang = ""

        for line in text.split("\n"):
            # Code blocks
            if line.startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    code_lang = line[3:].strip()
                    lines.append(self._theme.code_block_border("```" + code_lang))
                else:
                    in_code_block = False
                    code_lang = ""
                    lines.append(self._theme.code_block_border("```"))
                continue

            if in_code_block:
                lines.append("  " + self._theme.code_block(line))
                continue

            # Headings
            if line.startswith("### "):
                heading = line[4:]
                lines.append(self._theme.heading(self._theme.bold("### " + self._render_inline(heading))))
                lines.append("")
                continue
            if line.startswith("## "):
                heading = line[3:]
                lines.append(self._theme.heading(self._theme.bold(self._render_inline(heading))))
                lines.append("")
                continue
            if line.startswith("# "):
                heading = line[2:]
                lines.append(self._theme.heading(self._theme.bold(self._theme.underline(self._render_inline(heading)))))
                lines.append("")
                continue

            # Horizontal rule
            if re.match(r"^[-*_]{3,}$", line.strip()):
                lines.append(self._theme.hr("─" * min(width, 80)))
                lines.append("")
                continue

            # Blockquote
            if line.startswith("> "):
                quote_text = line[2:]
                lines.append(
                    self._theme.quote_border("│ ")
                    + self._theme.quote(self._theme.italic(self._render_inline(quote_text)))
                )
                continue

            # Unordered list
            if re.match(r"^[-*+] ", line):
                bullet = line[0]
                content = line[2:]
                lines.append(self._theme.list_bullet("- ") + self._render_inline(content))
                continue

            # Ordered list
            match = re.match(r"^(\d+)\. ", line)
            if match:
                num = match.group(1)
                content = line[len(match.group(0)):]
                lines.append(self._theme.list_bullet(f"{num}. ") + self._render_inline(content))
                continue

            # Empty line
            if not line.strip():
                lines.append("")
                continue

            # Regular paragraph
            lines.append(self._render_inline(line))

        return lines

    def _render_inline(self, text: str) -> str:
        """Render inline markdown elements."""
        result = text

        # Inline code: `text` - MUST process FIRST to protect from other formatters
        result = re.sub(
            r"`([^`]+)`",
            lambda m: self._theme.code(m.group(1)),
            result,
        )

        # Bold: **text** or __text__
        result = re.sub(
            r"\*\*(.+?)\*\*|__(.+?)__",
            lambda m: self._theme.bold(m.group(1) or m.group(2)),
            result,
        )

        # Italic: *text* or _text_
        result = re.sub(
            r"\*(.+?)\*|_(.+?)_",
            lambda m: self._theme.italic(m.group(1) or m.group(2)),
            result,
        )

        # Strikethrough: ~~text~~
        result = re.sub(
            r"~~(.+?)~~",
            lambda m: self._theme.strikethrough(m.group(1)),
            result,
        )

        # Links: [text](url)
        result = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            lambda m: (
                self._theme.link(self._theme.underline(m.group(1)))
                if m.group(1) == m.group(2)
                else self._theme.link(self._theme.underline(m.group(1)))
                + self._theme.link_url(f" ({m.group(2)})")
            ),
            result,
        )

        return result
