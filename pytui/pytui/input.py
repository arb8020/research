"""Input message types and parsing helpers.

This module is intentionally small and reusable:
- `Terminal` produces raw key strings (single chars or escape sequences).
- `InputParser` groups higher-level constructs like bracketed paste into messages.

This keeps bracketed-paste parsing out of individual UI components.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeyPress:
    """Keyboard input. `key` is the raw string (e.g. "j", "\\x1b[A")."""

    key: str


@dataclass(frozen=True)
class PasteEvent:
    """Bracketed paste content.

    When bracketed paste mode is enabled, pasted text is wrapped in escape
    sequences so it can be distinguished from typed input. This prevents
    pasted text from triggering keybindings.
    """

    text: str


@dataclass(frozen=True)
class FocusEvent:
    """Terminal focus change.

    Sent when the terminal window gains or loses focus.
    Requires focus reporting to be enabled.
    """

    focused: bool  # True = gained focus, False = lost focus


@dataclass(frozen=True)
class MouseEvent:
    """Mouse event (button press, release, wheel scroll).

    Button values:
        0 = left click
        1 = middle click
        2 = right click
        64 = wheel up
        65 = wheel down
        66 = wheel left
        67 = wheel right

    Action values:
        "press" = button pressed
        "release" = button released
        "motion" = mouse moved while button held
    """

    button: int
    x: int  # 1-indexed column
    y: int  # 1-indexed row
    action: str  # "press", "release", or "motion"

    @property
    def is_wheel_up(self) -> bool:
        return self.button == 64

    @property
    def is_wheel_down(self) -> bool:
        return self.button == 65


# Bracketed paste markers
PASTE_START = "\x1b[200~"
PASTE_END = "\x1b[201~"


def parse_mouse_sgr(seq: str) -> MouseEvent | None:
    """Parse SGR mouse sequence: ESC [ < Btn ; X ; Y M/m."""
    import re

    match = re.match(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])", seq)
    if not match:
        return None

    btn = int(match.group(1))
    x = int(match.group(2))
    y = int(match.group(3))
    release = match.group(4) == "m"

    action = "release" if release else "press"
    if btn & 32:
        action = "motion"

    return MouseEvent(button=btn & ~32, x=x, y=y, action=action)


def parse_focus(seq: str) -> FocusEvent | None:
    """Parse focus event: ESC [ I (focus) or ESC [ O (blur)."""
    if seq == "\x1b[I":
        return FocusEvent(focused=True)
    if seq == "\x1b[O":
        return FocusEvent(focused=False)
    return None


class InputParser:
    """Turns raw key strings into higher-level messages (paste, mouse, focus, keys)."""

    def __init__(self) -> None:
        self._paste_buffer: str | None = None

    def parse(self, key: str) -> object | None:
        """Parse a raw input key string into a message, or None if buffering."""
        if key == PASTE_START:
            self._paste_buffer = ""
            return None

        if self._paste_buffer is not None:
            if key == PASTE_END:
                text = self._paste_buffer
                self._paste_buffer = None
                return PasteEvent(text=text)
            self._paste_buffer += key
            return None

        mouse = parse_mouse_sgr(key)
        if mouse is not None:
            return mouse

        focus = parse_focus(key)
        if focus is not None:
            return focus

        return KeyPress(key=key)
