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
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        super().__init__()
        self._question = question
        self._theme = theme
        self._on_select = on_select
        self._on_cancel = on_cancel
        self._current = current
        self._total = total

        self._question_text = question.get("question", "")
        self._header = question.get("header", "Question")
        self._options = question.get("options", [])
        self._multi_select = question.get("multiSelect", False)

        self._selected_index = 0
        self._selected_indices: set[int] = set()  # For multi-select

        # Add "Other" option, and "Done" for multi-select
        self._all_options = list(self._options) + [
            {"label": "Other", "description": "Type a custom answer"}
        ]
        self._other_index = len(self._all_options) - 1

        if self._multi_select:
            self._all_options.append({"label": "Done", "description": "Confirm your selections"})
            self._done_index = len(self._all_options) - 1
        else:
            self._done_index = -1

        # Custom text input state (for "Other" option)
        self._custom_text = ""
        self._is_typing_custom = False

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI components."""
        self.clear()

        # Progress indicator (if multi-question)
        if self._current is not None and self._total is not None and self._total > 1:
            progress_text = f"Question {self._current} of {self._total}"
            if self._theme:
                progress_styled = self._theme.muted_fg(progress_text)
            else:
                progress_styled = progress_text
            self.add_child(Text(progress_styled, padding_x=2, padding_y=0, theme=self._theme))

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
            is_other = i == self._other_index
            is_done = self._multi_select and i == self._done_index

            # For "Other" option, show custom text if present
            if is_other and self._custom_text:
                if self._is_typing_custom and is_selected:
                    # Actively typing: show cursor
                    display_label = (
                        self._custom_text + "\x1b[7m \x1b[0m"
                    )  # Highlighted cursor block
                else:
                    # Custom text confirmed: show with quotes to indicate it's a value
                    display_label = f'"{self._custom_text}"'
            elif is_other and self._is_typing_custom and is_selected:
                # Just started typing, no text yet - show cursor
                display_label = "\x1b[7m \x1b[0m"
            else:
                display_label = label

            # Build option text
            if self._multi_select and not is_done:
                checkbox = "[x]" if is_checked else "[ ]"
                prefix = f"  {checkbox} "
            elif is_done:
                prefix = "  ──── "  # Visual separator for Done
            else:
                prefix = "  "

            if is_selected:
                # Highlighted option
                arrow = "→ " if not self._multi_select else ""
                if self._theme:
                    option_text = self._theme.accent_fg(f"{arrow}{display_label}")
                    # Only show description if not typing custom text
                    if desc and not (is_other and self._custom_text):
                        option_text += self._theme.muted_fg(f": {desc}")
                else:
                    option_text = f"{arrow}{display_label}"
                    if desc and not (is_other and self._custom_text):
                        option_text += f": {desc}"
            else:
                if self._theme:
                    # Use text color for unselected options
                    option_text = self._theme.fg(self._theme.text)(display_label)
                    if desc and not (is_other and self._custom_text):
                        option_text += self._theme.muted_fg(f": {desc}")
                else:
                    option_text = display_label
                    if desc and not (is_other and self._custom_text):
                        option_text += f": {desc}"

            full_text = prefix + option_text
            self.add_child(Text(full_text, padding_x=2, padding_y=0, theme=self._theme))

        self.add_child(Spacer(1))

        # Instructions - context-aware hints
        is_on_other = self._selected_index == self._other_index
        if self._is_typing_custom:
            hint = "type to enter text  enter confirm  esc clear"
        elif self._multi_select:
            if is_on_other:
                hint = "type for custom answer  ↑↓ navigate  [Done] to submit  esc cancel"
            else:
                hint = "↑↓ navigate  space/enter toggle  [Done] to submit  esc cancel"
        else:
            hint = "↑↓ navigate  enter select  esc cancel"
        if self._theme:
            hint_styled = self._theme.muted_fg(hint)
        else:
            hint_styled = hint
        self.add_child(Text(hint_styled, padding_x=2, padding_y=0, theme=self._theme))

    def handle_input(self, data: str) -> None:
        """Handle keyboard input."""
        # Strip bracketed paste escape sequences (ignore paste content in selector)
        if "\x1b[200~" in data or "\x1b[201~" in data:
            data = data.replace("\x1b[200~", "").replace("\x1b[201~", "")
            if not data:
                return

        is_on_other = self._selected_index == self._other_index
        is_on_done = self._multi_select and self._selected_index == self._done_index

        # Ctrl+C - treat as cancel (let it propagate for TUI to handle)
        if len(data) > 0 and ord(data[0]) == 3:
            if self._on_cancel:
                self._on_cancel()
            return

        # Up arrow - navigate (but not if actively typing custom text)
        if data == "\x1b[A":
            if not self._is_typing_custom:
                self._selected_index = max(0, self._selected_index - 1)
                self._build_ui()
            return

        # Down arrow - navigate (but not if actively typing custom text)
        if data == "\x1b[B":
            if not self._is_typing_custom:
                self._selected_index = min(len(self._all_options) - 1, self._selected_index + 1)
                self._build_ui()
            return

        # If on "Other" option, handle text input
        if is_on_other:
            # Backspace
            if len(data) > 0 and ord(data[0]) in (127, 8):
                if self._custom_text:
                    self._custom_text = self._custom_text[:-1]
                    self._build_ui()
                return

            # Enter - confirm custom text
            if len(data) == 1 and ord(data[0]) == 13:
                if self._multi_select:
                    # In multi-select mode, Enter on Other confirms the custom text
                    # and adds it to selections, then user can continue selecting
                    if self._custom_text:
                        # Mark Other as selected (the custom text is stored in _custom_text)
                        self._selected_indices.add(self._other_index)
                        self._is_typing_custom = False
                        self._build_ui()
                    # If no custom text, just toggle Other off if it was selected
                    elif self._other_index in self._selected_indices:
                        self._selected_indices.remove(self._other_index)
                        self._build_ui()
                else:
                    # Single-select: Enter submits immediately with the custom text
                    custom_answer = self._custom_text if self._custom_text else "Other"
                    if self._on_select:
                        self._on_select(custom_answer)
                return

            # Escape - cancel (or clear custom text if typing)
            if data == "\x1b":
                if self._custom_text and self._is_typing_custom:
                    # Clear custom text and exit typing mode
                    self._custom_text = ""
                    self._is_typing_custom = False
                    # Also remove from selected if it was selected
                    if self._other_index in self._selected_indices:
                        self._selected_indices.remove(self._other_index)
                    self._build_ui()
                elif self._on_cancel:
                    self._on_cancel()
                return

            # Regular printable characters - add to custom text
            if len(data) > 0 and ord(data[0]) >= 32:
                self._custom_text += data
                self._is_typing_custom = True
                self._build_ui()
                return

        # j/k navigation (only when not actively typing custom text)
        if data == "k" and not self._is_typing_custom:
            self._selected_index = max(0, self._selected_index - 1)
            self._build_ui()
            return

        if data == "j" and not self._is_typing_custom:
            self._selected_index = min(len(self._all_options) - 1, self._selected_index + 1)
            self._build_ui()
            return

        # Space or Enter - toggle selection (multi-select only, not on Other or Done)
        if (
            (data == " " or (len(data) == 1 and ord(data[0]) == 13))
            and self._multi_select
            and not is_on_other
            and not is_on_done
        ):
            if self._selected_index in self._selected_indices:
                self._selected_indices.remove(self._selected_index)
            else:
                self._selected_indices.add(self._selected_index)
            self._build_ui()
            return

        # Enter on Done - confirm multi-select
        if len(data) == 1 and ord(data[0]) == 13 and is_on_done:
            selected_labels = [
                self._all_options[i].get("label", "")
                for i in sorted(self._selected_indices)
                if i != self._other_index and i != self._done_index
            ]
            # If "Other" was selected, include the custom text
            if self._other_index in self._selected_indices and self._custom_text:
                selected_labels.append(self._custom_text)
            result = ", ".join(selected_labels) if selected_labels else ""
            if self._on_select:
                self._on_select(result)
            return

        # Enter - confirm selection (for non-Other options, single-select only)
        if len(data) == 1 and ord(data[0]) == 13 and not self._multi_select:
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


class AnswerReviewComponent(Container):
    """Shows a summary of answers with confirm/cancel options.

    Displays all Q&A pairs and allows user to confirm or cancel (redo).
    """

    def __init__(
        self,
        answers: dict[str, str],
        theme: Theme | None = None,
        on_confirm: Callable[[], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._answers = answers
        self._theme = theme
        self._on_confirm = on_confirm
        self._on_cancel = on_cancel
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the review UI."""
        self.clear()

        # Title
        title = "Review your answers"
        if self._theme:
            title_styled = self._theme.accent_fg(title)
        else:
            title_styled = title
        self.add_child(Text(title_styled, padding_x=2, padding_y=0, theme=self._theme))
        self.add_child(Spacer(1))

        # Show each Q&A
        for question, answer in self._answers.items():
            # Question (truncate if too long)
            q_display = question if len(question) <= 60 else question[:57] + "..."
            if self._theme:
                q_styled = self._theme.muted_fg(f"  {q_display}")
            else:
                q_styled = f"  {q_display}"
            self.add_child(Text(q_styled, padding_x=2, padding_y=0, theme=self._theme))

            # Answer
            a_display = answer if answer else "(no answer)"
            if self._theme:
                a_styled = self._theme.accent_fg(f"    → {a_display}")
            else:
                a_styled = f"    → {a_display}"
            self.add_child(Text(a_styled, padding_x=2, padding_y=0, theme=self._theme))

        self.add_child(Spacer(1))

        # Instructions
        hint = "enter confirm  esc redo"
        if self._theme:
            hint_styled = self._theme.muted_fg(hint)
        else:
            hint_styled = hint
        self.add_child(Text(hint_styled, padding_x=2, padding_y=0, theme=self._theme))

    def handle_input(self, data: str) -> None:
        """Handle keyboard input."""
        # Enter - confirm
        if len(data) == 1 and ord(data[0]) == 13:
            if self._on_confirm:
                self._on_confirm()
            return

        # Escape - cancel/redo
        if data == "\x1b":
            if self._on_cancel:
                self._on_cancel()
            return

        # Ctrl+C - treat as cancel
        if len(data) > 0 and ord(data[0]) == 3:
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
    4. Showing review with confirm/cancel option
    5. Returning all answers when confirmed
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

        # Review component
        self._review: AnswerReviewComponent | None = None

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
            while True:
                # Reset answers for this attempt
                self._answers = {}

                # Ask all questions
                for i, question in enumerate(self._questions):
                    self._current_index = i
                    answer = await self._ask_single_question(question)

                    question_text = question.get("question", f"Question {i + 1}")

                    if answer is None:
                        # Cancelled mid-question - use empty string
                        self._answers[question_text] = ""
                    else:
                        self._answers[question_text] = answer

                # Show review if multiple questions
                if len(self._questions) > 1:
                    confirmed = await self._show_review()
                    if confirmed:
                        return self._answers
                    # User cancelled - loop back and redo all questions
                else:
                    # Single question - no review needed
                    return self._answers

        finally:
            # Restore original focus
            if self._original_focus and self._tui:
                self._tui.set_focus(self._original_focus)
            self._tui.request_render()

    async def _show_review(self) -> bool:
        """Show review component and return True if confirmed, False if cancelled."""
        assert self._send is not None
        assert self._receive is not None

        confirmed = False

        def on_confirm() -> None:
            nonlocal confirmed
            confirmed = True
            assert self._send is not None
            try:
                self._send.send_nowait("confirm")
            except trio.WouldBlock:
                pass

        def on_cancel() -> None:
            assert self._send is not None
            try:
                self._send.send_nowait(None)
            except trio.WouldBlock:
                pass

        self._review = AnswerReviewComponent(
            answers=self._answers,
            theme=self._theme,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
        )

        if len(self._tui.children) >= 2:
            self._tui.children.insert(-1, self._review)
        else:
            self._tui.children.append(self._review)

        self._tui.set_focus(self._review)
        self._tui.request_render()

        try:
            await self._receive.receive()
            return confirmed
        finally:
            if self._review in self._tui.children:
                self._tui.children.remove(self._review)
            self._review = None

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
            current=self._current_index + 1,
            total=len(self._questions),
        )

        # Insert before the last spacer (after status line) for tighter layout
        # The TUI children are: [..., status_line, spacer(5)]
        # We want to insert before the spacer
        if len(self._tui.children) >= 2:
            self._tui.children.insert(-1, self._selector)
        else:
            self._tui.children.append(self._selector)

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
