"""PendingMessages component - shows queued user messages near the editor."""

from __future__ import annotations

from collections.abc import Callable

from ..theme import DARK_THEME, Theme
from ..tui import Component
from ..utils import truncate_to_width, visible_width


class PendingMessages(Component):
    """Shows queued messages and a hint to restore them into the editor."""

    def __init__(
        self,
        *,
        theme: Theme | None = None,
        max_preview: int = 3,
        get_blocked_reason: Callable[[], str | None] | None = None,
        restore_hint: str = "↑ restore to edit",
    ) -> None:
        self._theme = theme or DARK_THEME
        self._messages: list[str] = []
        self._max_preview = max_preview
        self._get_blocked_reason = get_blocked_reason
        self._restore_hint = restore_hint

    def count(self) -> int:
        return len(self._messages)

    def get_all(self) -> list[str]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def add(self, text: str) -> None:
        self._messages.append(text)

    def pop_left(self) -> str | None:
        if not self._messages:
            return None
        return self._messages.pop(0)

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        if not self._messages:
            return []

        gray = "\x1b[38;5;245m"
        reset = "\x1b[0m"

        reason = self._get_blocked_reason() if self._get_blocked_reason else None
        if reason:
            header = f"{len(self._messages)} queued. Blocked: {reason}. {self._restore_hint}"
        else:
            header = f"{len(self._messages)} queued. {self._restore_hint}"

        header = truncate_to_width(header, width, ellipsis="…")
        pad = " " * max(0, width - visible_width(header))
        lines = [f"{gray}{header}{pad}{reset}"]

        for i, msg in enumerate(self._messages[: self._max_preview]):
            prefix = f"{i + 1}. "
            available = max(0, width - len(prefix))
            preview = msg.replace("\n", " ")
            if visible_width(preview) > available:
                preview = truncate_to_width(preview, available, ellipsis="…")
            line = prefix + preview
            line = truncate_to_width(line, width, ellipsis="…")
            pad = " " * max(0, width - visible_width(line))
            lines.append(f"{gray}{line}{pad}{reset}")

        if len(self._messages) > self._max_preview:
            more = f"… +{len(self._messages) - self._max_preview} more"
            more = truncate_to_width(more, width, ellipsis="…")
            pad = " " * max(0, width - visible_width(more))
            lines.append(f"{gray}{more}{pad}{reset}")

        return lines
