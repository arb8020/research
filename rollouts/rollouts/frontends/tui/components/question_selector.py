"""
QuestionSelectorComponent - Interactive multiple-choice question selector.

Displays options with arrow key navigation for the ask_user_question tool.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import trio

from ..tui import Component, Container
from .spacer import Spacer
from .text import Text

if TYPE_CHECKING:
    from ..theme import Theme


class QuestionSelectorComponent(Container):
    """Interactive selector for a single question with multiple options.

    Displays options with arrow key navigation:
    - Up/Down or j/k to navigate
    - Enter to select
    - Escape to cancel (selects nothing)
    """

    def __init__(
        self,
        question: dict[str, Any],
        theme: Theme | None = None,
        on_select: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._question = question
        self._theme = theme
        self._on_select = on_select
        self._on_cancel = on_cancel

        self._question_text = question.get("question", "")
        self._header = question.get("header", "Question")
        self._options = question.get("options", [])
        self._multi_select = question.get("multiSelect", False)

        self._selected_index = 0
        self._selected_indices: set[int] = set()  # For multi-select

        # Add "Other" option
        self._all_options = list(self._options) + [
            {"label": "Other", "description": "Type a custom answer"}
        ]

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI components."""
        self.clear()

        # Header and question
        header_text = f"[{self._header}] {self._question_text}"
        if self._theme:
            header_styled = self._theme.accent_fg(header_text)
        else:
            header_styled = header_text
        self.add_child(Text(header_styled, padding_x=2, padding_y=0, theme=self._theme))
        self.add_child(Spacer(1))

        # Options
        for i, opt in enumerate(self._all_options):
            label = opt.get("label", f"Option {i + 1}")
            desc = opt.get("description", "")
            is_selected = i == self._selected_index
            is_checked = i in self._selected_indices  # For multi-select

            # Build option text
            if self._multi_select:
                checkbox = "[x]" if is_checked else "[ ]"
                prefix = f"  {checkbox} "
            else:
                prefix = "  "

            if is_selected:
                # Highlighted option
                arrow = "→ " if not self._multi_select else ""
                if self._theme:
                    option_text = self._theme.accent_fg(f"{arrow}{label}")
                    if desc:
                        option_text += self._theme.muted_fg(f": {desc}")
                else:
                    option_text = f"{arrow}{label}"
                    if desc:
                        option_text += f": {desc}"
            else:
                if self._theme:
                    option_text = self._theme.fg(label)
                    if desc:
                        option_text += self._theme.muted_fg(f": {desc}")
                else:
                    option_text = label
                    if desc:
                        option_text += f": {desc}"

            full_text = prefix + option_text
            self.add_child(Text(full_text, padding_x=2, padding_y=0, theme=self._theme))

        self.add_child(Spacer(1))

        # Instructions
        if self._multi_select:
            hint = "↑↓ navigate  space toggle  enter confirm  esc cancel"
        else:
            hint = "↑↓ navigate  enter select  esc cancel"
        if self._theme:
            hint_styled = self._theme.muted_fg(hint)
        else:
            hint_styled = hint
        self.add_child(Text(hint_styled, padding_x=2, padding_y=0, theme=self._theme))

    def handle_input(self, data: str) -> None:
        """Handle keyboard input."""
        # Up arrow or k
        if data == "\x1b[A" or data == "k":
            self._selected_index = max(0, self._selected_index - 1)
            self._build_ui()
            return

        # Down arrow or j
        if data == "\x1b[B" or data == "j":
            self._selected_index = min(len(self._all_options) - 1, self._selected_index + 1)
            self._build_ui()
            return

        # Space - toggle selection (multi-select only)
        if data == " " and self._multi_select:
            if self._selected_index in self._selected_indices:
                self._selected_indices.remove(self._selected_index)
            else:
                self._selected_indices.add(self._selected_index)
            self._build_ui()
            return

        # Enter - confirm selection
        if len(data) == 1 and ord(data[0]) == 13:
            if self._multi_select:
                # Return comma-separated labels
                selected_labels = [
                    self._all_options[i].get("label", "") for i in sorted(self._selected_indices)
                ]
                result = ", ".join(selected_labels) if selected_labels else ""
            else:
                result = self._all_options[self._selected_index].get("label", "")

            if self._on_select:
                self._on_select(result)
            return

        # Escape - cancel
        if data == "\x1b":
            if self._on_cancel:
                self._on_cancel()
            return

    def render(self, width: int) -> list[str]:
        """Render all children."""
        lines: list[str] = []
        for child in self.children:
            lines.extend(child.render(width))
        return lines


class MultiQuestionSelector:
    """Manages asking multiple questions sequentially using QuestionSelectorComponent.

    This class handles the flow of:
    1. Displaying a question
    2. Waiting for user selection
    3. Moving to the next question
    4. Returning all answers when complete
    """

    def __init__(
        self,
        questions: list[dict[str, Any]],
        tui: Any,  # TUI instance
        theme: Theme | None = None,
    ) -> None:
        self._questions = questions
        self._tui = tui
        self._theme = theme
        self._answers: dict[str, str] = {}
        self._current_index = 0

        # Channel for receiving selections
        self._send: trio.MemorySendChannel[str | None] | None = None
        self._receive: trio.MemoryReceiveChannel[str | None] | None = None

        # Current selector component
        self._selector: QuestionSelectorComponent | None = None

        # Original focused component to restore
        self._original_focus: Component | None = None

    async def ask_all(self) -> dict[str, str]:
        """Ask all questions and return answers.

        Returns:
            Dictionary mapping question text to selected answer.
        """
        self._send, self._receive = trio.open_memory_channel[str | None](1)
        self._original_focus = self._tui._focused_component

        try:
            for i, question in enumerate(self._questions):
                self._current_index = i
                answer = await self._ask_single_question(question)

                question_text = question.get("question", f"Question {i + 1}")

                if answer is None:
                    # Cancelled - use empty string
                    self._answers[question_text] = ""
                elif answer == "Other":
                    # Need to get custom input - restore focus to input component
                    # For now, just use "Other" as the answer
                    # TODO: Implement custom input flow
                    self._answers[question_text] = "Other"
                else:
                    self._answers[question_text] = answer

            return self._answers

        finally:
            # Restore original focus
            if self._original_focus and self._tui:
                self._tui.set_focus(self._original_focus)
            self._tui.request_render()

    async def _ask_single_question(self, question: dict[str, Any]) -> str | None:
        """Ask a single question and wait for response."""
        assert self._send is not None
        assert self._receive is not None

        def on_select(answer: str) -> None:
            assert self._send is not None
            try:
                self._send.send_nowait(answer)
            except trio.WouldBlock:
                pass

        def on_cancel() -> None:
            assert self._send is not None
            try:
                self._send.send_nowait(None)
            except trio.WouldBlock:
                pass

        # Create selector component
        self._selector = QuestionSelectorComponent(
            question=question,
            theme=self._theme,
            on_select=on_select,
            on_cancel=on_cancel,
        )

        # Add to TUI and set focus
        # We need to temporarily add it to the TUI's children
        self._tui.add_child(self._selector)
        self._tui.set_focus(self._selector)
        self._tui.request_render()

        try:
            # Wait for selection
            answer = await self._receive.receive()
            return answer
        finally:
            # Remove selector from TUI
            if self._selector in self._tui.children:
                self._tui.children.remove(self._selector)
            self._selector = None
