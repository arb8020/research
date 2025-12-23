"""Hierarchical trace viewer for training rollouts.

Navigable tree structure:
    Step → Group (prompt) → Sample (rollout) → Trace (messages)

Based on pyvimdiff pattern: each level is a picker that opens the next level.

Usage:
    from rollouts.tui.traces import StepPicker, TraceData

    # Load from rollouts.jsonl
    data = TraceData.from_jsonl("results/rl/run_xxx/rollouts.jsonl")
    picker = StepPicker(data)
    picker.run()
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .terminal import Terminal

# Colors (ANSI)
DIM = "\x1b[90m"
WHITE = "\x1b[37m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
MAGENTA = "\x1b[35m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"
BG_HEADER = "\x1b[48;2;40;44;52m"


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Message:
    """A single message in a trace."""

    role: str  # "user", "assistant", "tool", "system"
    content: str


@dataclass
class Rollout:
    """A single rollout (sample)."""

    step: int
    group_id: int  # Which prompt group
    sample_id: int  # Which sample within group
    prompt: str
    response: str
    reward: float
    messages: list[Message] = field(default_factory=list)
    is_streaming: bool = False  # True if still receiving updates
    sample_id_str: str = ""  # Original string sample ID for matching updates


@dataclass
class Group:
    """A group of rollouts for the same prompt."""

    step: int
    group_id: int
    prompt: str
    rollouts: list[Rollout] = field(default_factory=list)

    @property
    def avg_reward(self) -> float:
        if not self.rollouts:
            return 0.0
        return sum(r.reward for r in self.rollouts) / len(self.rollouts)

    @property
    def max_reward(self) -> float:
        if not self.rollouts:
            return 0.0
        return max(r.reward for r in self.rollouts)

    @property
    def min_reward(self) -> float:
        if not self.rollouts:
            return 0.0
        return min(r.reward for r in self.rollouts)


@dataclass
class Step:
    """A training step containing multiple groups."""

    step: int
    groups: list[Group] = field(default_factory=list)

    @property
    def avg_reward(self) -> float:
        all_rewards = [r.reward for g in self.groups for r in g.rollouts]
        if not all_rewards:
            return 0.0
        return sum(all_rewards) / len(all_rewards)

    @property
    def num_rollouts(self) -> int:
        return sum(len(g.rollouts) for g in self.groups)


@dataclass
class TraceData:
    """All trace data from a training run."""

    steps: list[Step] = field(default_factory=list)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> TraceData:
        """Load trace data from rollouts.jsonl file."""
        path = Path(path)
        if not path.exists():
            return cls()

        # Group by step, then by group_index (or hash of prompt as fallback)
        step_groups: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))

        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    step = record.get("step", 0)
                    # Use group_index if available (new format), else hash prompt
                    group_id = record.get("group_index")
                    if group_id is None:
                        prompt = record.get("prompt", "")
                        group_id = hash(str(prompt)) % 10000
                    step_groups[step][group_id].append(record)
                except json.JSONDecodeError:
                    continue

        # Build structured data
        steps = []
        for step_num in sorted(step_groups.keys()):
            groups = []
            for group_id, records in step_groups[step_num].items():
                if not records:
                    continue

                prompt = records[0].get("prompt", "")
                rollouts = []

                for i, record in enumerate(records):
                    # Extract messages - check top-level first (new format), then metadata (old format)
                    messages = []
                    raw_messages = record.get("messages") or record.get("metadata", {}).get(
                        "messages", []
                    )
                    for msg in raw_messages:
                        messages.append(
                            Message(
                                role=msg.get("role", "unknown"),
                                content=msg.get("content", ""),
                            )
                        )

                    rollouts.append(
                        Rollout(
                            step=step_num,
                            group_id=group_id,
                            sample_id=i,
                            prompt=prompt,
                            response=record.get("response", ""),
                            reward=record.get("reward", 0.0),
                            messages=messages,
                        )
                    )

                groups.append(
                    Group(
                        step=step_num,
                        group_id=group_id,
                        prompt=prompt,
                        rollouts=rollouts,
                    )
                )

            steps.append(Step(step=step_num, groups=groups))

        return cls(steps=steps)

    def add_record(self, record: dict) -> None:
        """Add or update a rollout record (from log stream).

        For streaming rollouts, updates existing rollout if sample_id matches.
        """
        sample_id_str = str(record.get("step", ""))  # The string ID like "sample_0003"
        status = record.get("status", "")
        is_streaming = status == "streaming"

        # Use prompt hash as group_id
        group_id = record.get("group_index")
        if group_id is None:
            prompt = record.get("prompt", "")
            group_id = hash(str(prompt)) % 10000

        # Extract messages
        messages = []
        raw_messages = record.get("messages") or record.get("metadata", {}).get("messages", [])
        for msg in raw_messages:
            messages.append(
                Message(
                    role=msg.get("role", "unknown"),
                    content=msg.get("content", ""),
                )
            )

        # Check if we already have this rollout (for streaming updates)
        existing_rollout = self.find_rollout_by_sample_id(sample_id_str)
        if existing_rollout:
            # Update existing rollout
            existing_rollout.response = record.get("response", "")
            existing_rollout.reward = record.get("reward", 0.0)
            existing_rollout.messages = messages
            existing_rollout.is_streaming = is_streaming
            return

        # Create new rollout - need to find/create step and group first
        # Use 0 as step_num since we're using sample_id_str for identification
        step_num = 0

        # Find or create step
        step = None
        for s in self.steps:
            if s.step == step_num:
                step = s
                break
        if step is None:
            step = Step(step=step_num, groups=[])
            self.steps.append(step)
            self.steps.sort(key=lambda s: s.step)

        # Find or create group
        group = None
        for g in step.groups:
            if g.group_id == group_id:
                group = g
                break
        if group is None:
            group = Group(
                step=step_num,
                group_id=group_id,
                prompt=record.get("prompt", ""),
                rollouts=[],
            )
            step.groups.append(group)

        # Add new rollout
        rollout = Rollout(
            step=step_num,
            group_id=group_id,
            sample_id=len(group.rollouts),
            prompt=record.get("prompt", ""),
            response=record.get("response", ""),
            reward=record.get("reward", 0.0),
            messages=messages,
            is_streaming=is_streaming,
            sample_id_str=sample_id_str,
        )
        group.rollouts.append(rollout)

    def find_rollout_by_sample_id(self, sample_id_str: str) -> Rollout | None:
        """Find a rollout by its string sample ID."""
        for step in self.steps:
            for group in step.groups:
                for rollout in group.rollouts:
                    if rollout.sample_id_str == sample_id_str:
                        return rollout
        return None

    def handle_stream_event(self, sample_id: str, event_type: str, event_data: dict) -> None:
        """Handle a StreamEvent for live streaming display.

        Uses AgentRenderer-style event handling - builds messages from
        TextEnd, ThinkingEnd, ToolCallEnd events (not manual accumulation).
        """
        # Find or create rollout for this sample
        rollout = self.find_rollout_by_sample_id(sample_id)
        if rollout is None:
            # Create a placeholder rollout for streaming
            step_num = 0
            group_id = hash(sample_id) % 10000

            # Find or create step
            step = None
            for s in self.steps:
                if s.step == step_num:
                    step = s
                    break
            if step is None:
                step = Step(step=step_num, groups=[])
                self.steps.append(step)

            # Find or create group
            group = None
            for g in step.groups:
                if g.group_id == group_id:
                    group = g
                    break
            if group is None:
                group = Group(step=step_num, group_id=group_id, prompt="", rollouts=[])
                step.groups.append(group)

            # Create new rollout
            rollout = Rollout(
                step=step_num,
                group_id=group_id,
                sample_id=len(group.rollouts),
                prompt="",
                response="",
                reward=0.0,
                messages=[],
                is_streaming=True,
                sample_id_str=sample_id,
            )
            group.rollouts.append(rollout)

        # Handle different event types
        # sample_start contains initial messages including user prompt
        if event_type == "sample_start":
            messages = event_data.get("messages", [])
            for msg in messages:
                rollout.messages.append(
                    Message(role=msg.get("role", "user"), content=msg.get("content", ""))
                )
            # Also set prompt for fallback display
            if messages:
                rollout.prompt = messages[0].get("content", "")

        # Use "End" events which contain complete content (like AgentRenderer)
        elif event_type == "TextEnd":
            content = event_data.get("content", "")
            # Add or update assistant message with text
            self._update_assistant_text(rollout, content)

        elif event_type == "ThinkingEnd":
            content = event_data.get("content", "")
            # Add thinking to assistant message
            self._update_assistant_thinking(rollout, content)

        elif event_type == "ToolCallEnd":
            tool_call = event_data.get("tool_call", {})
            # Add tool call to assistant message
            self._add_tool_call(rollout, tool_call)

        elif event_type == "ToolResultReceived":
            tool_call_id = event_data.get("tool_call_id", "")
            content = event_data.get("content", "")
            is_error = event_data.get("is_error", False)
            # Add tool result message
            self._add_tool_result(rollout, tool_call_id, content, is_error)

        elif event_type == "StreamDone":
            # Mark streaming as complete
            rollout.is_streaming = False

        # For delta events, accumulate text for live preview
        elif event_type == "TextDelta":
            delta = event_data.get("delta", "")
            rollout.response += delta

        elif event_type == "ThinkingDelta":
            # Could accumulate for live thinking preview if needed
            pass

    def _update_assistant_text(self, rollout: Rollout, content: str) -> None:
        """Update or create assistant message with text content."""
        # Find last assistant message or create one
        if rollout.messages and rollout.messages[-1].role == "assistant":
            rollout.messages[-1] = Message(role="assistant", content=content)
        else:
            rollout.messages.append(Message(role="assistant", content=content))
        rollout.response = content

    def _update_assistant_thinking(self, rollout: Rollout, content: str) -> None:
        """Add thinking content to assistant message."""
        # For now, prepend thinking to content with a marker
        if rollout.messages and rollout.messages[-1].role == "assistant":
            existing = rollout.messages[-1].content
            rollout.messages[-1] = Message(
                role="assistant", content=f"[thinking]\n{content}\n[/thinking]\n\n{existing}"
            )

    def _add_tool_call(self, rollout: Rollout, tool_call: dict) -> None:
        """Add tool call info to display."""
        name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {})
        # Format as text for now
        tool_text = f"[tool_call: {name}]\n{args}"
        if rollout.messages and rollout.messages[-1].role == "assistant":
            existing = rollout.messages[-1].content
            rollout.messages[-1] = Message(role="assistant", content=f"{existing}\n\n{tool_text}")

    def _add_tool_result(
        self, rollout: Rollout, tool_call_id: str, content: str, is_error: bool
    ) -> None:
        """Add tool result message."""
        prefix = "[tool_error]" if is_error else "[tool_result]"
        rollout.messages.append(Message(role="tool", content=f"{prefix}\n{content}"))


# ─────────────────────────────────────────────────────────────────────────────
# Trace Viewer (deepest level - shows message trace)
# ─────────────────────────────────────────────────────────────────────────────


class TraceViewer:
    """View a single rollout's message trace with nvim-style folding and cursor."""

    def __init__(self, rollout: Rollout) -> None:
        self.rollout = rollout
        self.scroll = 0
        self.h_scroll = 0  # Horizontal scroll
        self.wrap = True  # Wrap mode (False = truncate + h-scroll)
        self.terminal = Terminal(use_alternate_screen=True)
        self._running = False
        self._needs_redraw = True
        self._rendered_lines: list[str] = []

        # Cursor: which message is currently selected
        self.cursor = 0  # Index into messages list

        # Fold state: track which messages are folded (by index)
        # Start with all messages folded except user messages
        self._folds: dict[int, bool] = {}  # msg_index -> is_folded
        self._init_folds()

        # Message-to-line mapping for scrolling to cursor
        self._msg_to_line: dict[int, int] = {}  # msg_index -> first rendered line

    def _init_folds(self) -> None:
        """Initialize fold state - fold all non-user messages by default."""
        messages = self._get_all_messages()
        for i, msg in enumerate(messages):
            # Fold everything except user messages by default
            self._folds[i] = msg.role != "user"

    def _get_all_messages(self) -> list[Message]:
        """Get all messages including fallback assistant response."""
        messages = list(self.rollout.messages)
        # Add fallback response as assistant if not in messages
        has_assistant = any(m.role == "assistant" for m in messages)
        if not has_assistant and self.rollout.response:
            messages.append(Message(role="assistant", content=self.rollout.response))
        return messages

    def run(self) -> None:
        self._running = True
        self._render_messages()  # Pre-render message lines
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
        if data == "q":
            self._running = False
            return

        messages = self._get_all_messages()
        num_messages = len(messages)
        height = self.terminal.rows
        content_height = height - 2

        # j/k move cursor between messages
        if data in ("j", "\x1b[B"):
            if self.cursor < num_messages - 1:
                self.cursor += 1
                self._scroll_to_cursor(content_height)
            self._needs_redraw = True
        elif data in ("k", "\x1b[A"):
            if self.cursor > 0:
                self.cursor -= 1
                self._scroll_to_cursor(content_height)
            self._needs_redraw = True
        elif data == "\x04":  # Ctrl+D - move cursor down by half page worth of messages
            self.cursor = min(num_messages - 1, self.cursor + 5)
            self._scroll_to_cursor(content_height)
            self._needs_redraw = True
        elif data == "\x15":  # Ctrl+U - move cursor up by half page worth of messages
            self.cursor = max(0, self.cursor - 5)
            self._scroll_to_cursor(content_height)
            self._needs_redraw = True
        elif data == "g":
            next_char = self._wait_for_char()
            if next_char == "g":
                self.cursor = 0
                self._scroll_to_cursor(content_height)
                self._needs_redraw = True
        elif data == "G":
            self.cursor = num_messages - 1
            self._scroll_to_cursor(content_height)
            self._needs_redraw = True
        # Horizontal scrolling
        elif data in ("h", "\x1b[D"):  # Left
            self.h_scroll = max(0, self.h_scroll - 20)
            self.wrap = False
            if self.h_scroll == 0:
                self.wrap = True
            self._needs_redraw = True
        elif data in ("l", "\x1b[C"):  # Right
            self.h_scroll += 20
            self.wrap = False
            self._needs_redraw = True
        elif data == "0":  # Beginning
            self.h_scroll = 0
            self._needs_redraw = True
        elif data == "w":  # Toggle wrap
            self.wrap = not self.wrap
            if self.wrap:
                self.h_scroll = 0
            self._needs_redraw = True
        # Fold commands (nvim-style)
        elif data == "z":
            self._handle_fold_command()
        # Tab/Space/Enter to toggle fold at cursor
        elif data in ("\t", " ", "\r", "\n"):
            self._toggle_fold_at_cursor()
            self._needs_redraw = True

    def _scroll_to_cursor(self, content_height: int) -> None:
        """Ensure cursor is visible by adjusting scroll."""
        if self.cursor not in self._msg_to_line:
            return
        cursor_line = self._msg_to_line[self.cursor]
        # If cursor is above visible area, scroll up
        if cursor_line < self.scroll:
            self.scroll = cursor_line
        # If cursor is below visible area, scroll down
        elif cursor_line >= self.scroll + content_height:
            self.scroll = cursor_line - content_height + 1

    def _handle_fold_command(self) -> None:
        """Handle nvim-style fold commands (za, zo, zc, zM, zR)."""
        next_char = self._wait_for_char()
        if next_char is None:
            return

        if next_char == "a":  # za - toggle fold at cursor
            self._toggle_fold_at_cursor()
        elif next_char == "o":  # zo - open fold at cursor
            self._set_fold_at_cursor(False)
        elif next_char == "c":  # zc - close fold at cursor
            self._set_fold_at_cursor(True)
        elif next_char == "M":  # zM - close all folds
            for i in self._folds:
                self._folds[i] = True
            self._render_messages()
        elif next_char == "R":  # zR - open all folds
            for i in self._folds:
                self._folds[i] = False
            self._render_messages()

        self._needs_redraw = True

    def _toggle_fold_at_cursor(self) -> None:
        """Toggle fold for the message at cursor."""
        self._folds[self.cursor] = not self._folds.get(self.cursor, False)
        self._render_messages()

    def _set_fold_at_cursor(self, folded: bool) -> None:
        """Set fold state for the message at cursor."""
        self._folds[self.cursor] = folded
        self._render_messages()

    def _open_in_nvim(self) -> None:
        """Open trace content in neovim for full vim navigation."""
        import os
        import subprocess
        import tempfile

        content = self._get_plain_text()

        fd, path = tempfile.mkstemp(suffix=".md")
        os.write(fd, content.encode())
        os.close(fd)

        self.terminal.stop()

        try:
            init_cmd = " | ".join([
                "set filetype=markdown",
                "set number",
                "set cursorline",
                "set nomodifiable",
                "nnoremap <silent> q :qa!<CR>",
                "set laststatus=2",
                "set statusline=%f\\ [TRACE]\\ %l/%L",
            ])
            subprocess.run(["nvim", "-R", f"+{init_cmd}", path])
        finally:
            os.unlink(path)
            self.terminal.start(on_input=lambda x: None, on_resize=self._on_resize)

    def _get_plain_text(self) -> str:
        """Get trace content as plain text."""
        lines = []
        for msg in self.rollout.messages:
            lines.append(f"# [{msg.role}]")
            lines.append("")
            for line in msg.content.split("\n"):
                lines.append(line)
            lines.append("")
        return "\n".join(lines)

    def _wait_for_char(self, timeout: float = 0.5) -> str | None:
        start = time.time()
        while time.time() - start < timeout:
            data = self.terminal.read_input()
            if data:
                return data
            time.sleep(0.01)
        return None

    def _render_messages(self) -> None:
        """Pre-render all message lines with fold support and cursor highlight."""
        self._rendered_lines = []
        self._msg_to_line = {}

        role_colors = {
            "user": CYAN,
            "assistant": GREEN,
            "tool": YELLOW,
            "system": MAGENTA,
        }

        # Cursor styling - bright yellow background for clear visibility
        BG_CURSOR = "\x1b[48;2;80;80;40m"  # Muted yellow-brown background
        CURSOR_ARROW = "\x1b[33;1m▸\x1b[0m"  # Bright yellow arrow

        messages = self._get_all_messages()

        for msg_idx, msg in enumerate(messages):
            color = role_colors.get(msg.role, WHITE)
            is_folded = self._folds.get(msg_idx, False)
            is_cursor = msg_idx == self.cursor
            content_lines = msg.content.split("\n")
            num_lines = len(content_lines)

            # Track which line this message starts at
            line_idx = len(self._rendered_lines)
            self._msg_to_line[msg_idx] = line_idx

            # Cursor indicator - bright arrow when selected, space otherwise
            cursor_marker = CURSOR_ARROW if is_cursor else " "
            bg = BG_CURSOR if is_cursor else ""
            bg_reset = RESET if is_cursor else ""

            if is_folded:
                # Show folded summary: ▸ ▶ [role] --- N lines ---
                fold_marker = "▶"
                preview = content_lines[0][:40] if content_lines else ""
                if len(content_lines[0]) > 40 if content_lines else False:
                    preview += "..."
                summary = f"{bg}{cursor_marker} {fold_marker} {color}{BOLD}[{msg.role}]{RESET}{bg} {DIM}--- {num_lines} lines: {preview}{RESET}{bg_reset}"
                self._rendered_lines.append(summary)
            else:
                # Show expanded: ▸ ▼ [role] followed by content
                fold_marker = "▼"
                header = (
                    f"{bg}{cursor_marker} {fold_marker} {color}{BOLD}[{msg.role}]{RESET}{bg_reset}"
                )
                self._rendered_lines.append(header)
                for line in content_lines:
                    self._rendered_lines.append(f"    {line}")

            self._rendered_lines.append("")

    def _render(self) -> None:
        width = self.terminal.columns
        height = self.terminal.rows
        content_height = height - 2

        self.terminal.write("\x1b[2J\x1b[H")

        output = []

        # Header
        output.append(self._render_header(width))

        # Content - either wrap or truncate based on mode
        visible = self._rendered_lines[self.scroll : self.scroll + content_height]
        for ln in visible:
            if self.wrap:
                # Wrap long lines
                while ln and len(output) < content_height + 1:
                    chunk = ln[: width - 1]
                    ln = ln[width - 1 :]
                    output.append(chunk + " " * max(0, width - len(chunk)))
                if not ln:
                    continue
            else:
                # Truncate with h-scroll
                display_ln = ln
                if self.h_scroll > 0:
                    display_ln = (
                        display_ln[self.h_scroll :] if self.h_scroll < len(display_ln) else ""
                    )
                if len(display_ln) > width - 1:
                    display_ln = display_ln[: width - 4] + "..."
                output.append(display_ln + " " * max(0, width - len(display_ln)))

        # Pad
        while len(output) < height - 1:
            output.append(" " * width)

        # Footer
        output.append(self._render_footer(width))

        for i, line in enumerate(output):
            self.terminal.write(f"\x1b[{i + 1};1H{line}")

    def _render_header(self, width: int) -> str:
        r = self.rollout
        reward_color = GREEN if r.reward > 0 else RED if r.reward < 0 else WHITE
        left = f" Step {r.step} / Group {r.group_id} / Sample {r.sample_id}"
        right = f"reward: {reward_color}{r.reward:.3f}{RESET}{BG_HEADER} "

        padding = width - len(left) - len(f"reward: {r.reward:.3f} ")
        return f"{BG_HEADER}{BOLD}{left}{RESET}{BG_HEADER}{' ' * max(0, padding)}{right}{RESET}"

    def _render_footer(self, width: int) -> str:
        wrap_hint = "w:truncate" if self.wrap else "h/l:scroll  w:wrap"
        hints = f"j/k:move  Space:fold  zR/zM:all  {wrap_hint}  q:back"
        messages = self._get_all_messages()
        num_messages = len(messages)
        pos = f"msg {self.cursor + 1}/{num_messages}"

        padding = width - len(hints) - len(pos) - 4
        return (
            f"{BG_HEADER} {DIM}{hints}{RESET}{BG_HEADER}{' ' * max(0, padding)}{WHITE}{pos} {RESET}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Trace Viewer (live updating)
# ─────────────────────────────────────────────────────────────────────────────


class TraceStreamingViewer:
    """View a streaming rollout with live token updates, nvim-style folding and cursor.

    Similar to TraceViewer but polls the Rollout object for updates and
    auto-scrolls as new content arrives.
    """

    def __init__(
        self,
        rollout: Rollout,
    ) -> None:
        """Initialize streaming viewer.

        Args:
            rollout: The Rollout object (gets updated by handle_stream_event)
        """
        self.rollout = rollout
        self.sample_id = rollout.sample_id_str or str(rollout.sample_id)

        self.scroll = 0
        self.h_scroll = 0
        self.wrap = True
        self.auto_scroll = True  # Follow tail (also moves cursor to last message)
        self.terminal = Terminal(use_alternate_screen=True)
        self._running = False
        self._needs_redraw = True
        self._last_response_len = 0  # Track changes
        self._rendered_lines: list[str] = []

        # Cursor: which message is currently selected
        self.cursor = 0  # Index into messages list

        # Fold state: track which messages are folded (by index)
        # Start with all messages folded except user messages
        self._folds: dict[int, bool] = {}  # msg_index -> is_folded
        self._msg_to_line: dict[int, int] = {}  # msg_index -> first rendered line
        self._last_msg_count = 0  # Track when new messages arrive

    def _init_folds_for_new_messages(self, messages: list) -> None:
        """Initialize fold state for any new messages."""
        for i, msg in enumerate(messages):
            if i not in self._folds:
                # Fold everything except user messages by default
                role = msg.role if hasattr(msg, "role") else msg.get("role", "")
                self._folds[i] = role != "user"

    def run(self) -> None:
        self._running = True
        self._render_content()
        self.terminal.start(on_input=lambda x: None, on_resize=self._on_resize)

        try:
            self._main_loop()
        finally:
            self.terminal.stop()

    def _on_resize(self) -> None:
        self._needs_redraw = True

    def _main_loop(self) -> None:
        while self._running:
            # Check if rollout.response has changed
            current_len = len(self.rollout.response)
            if current_len != self._last_response_len:
                self._render_content()
                self._last_response_len = current_len
                self._needs_redraw = True

                # Auto-scroll: move cursor to last message and scroll to it
                if self.auto_scroll:
                    messages = self._get_all_messages()
                    if messages:
                        self.cursor = len(messages) - 1
                    height = self.terminal.rows
                    content_height = height - 2
                    self._scroll_to_cursor(content_height)

            if self._needs_redraw:
                self._render()
                self._needs_redraw = False

            data = self.terminal.read_input()
            if data:
                self._handle_input(data)

            time.sleep(0.02)  # 50fps for smooth streaming

    def _get_all_messages(self) -> list[Message]:
        """Get all messages including prompt fallback and streaming response."""
        messages: list[Message] = []

        # Show prompt as user message if no messages yet
        has_user_message = any(msg.role == "user" for msg in self.rollout.messages)
        if not has_user_message and self.rollout.prompt:
            messages.append(Message(role="user", content=self.rollout.prompt))

        # Add messages from rollout
        messages.extend(self.rollout.messages)

        # Add streaming response as a separate message if present
        if self.rollout.response:
            messages.append(Message(role="assistant", content=self.rollout.response))

        return messages

    def _scroll_to_cursor(self, content_height: int) -> None:
        """Ensure cursor is visible by adjusting scroll."""
        if self.cursor not in self._msg_to_line:
            return
        cursor_line = self._msg_to_line[self.cursor]
        # If cursor is above visible area, scroll up
        if cursor_line < self.scroll:
            self.scroll = cursor_line
        # If cursor is below visible area, scroll down
        elif cursor_line >= self.scroll + content_height:
            self.scroll = cursor_line - content_height + 1

    def _handle_input(self, data: str) -> None:
        if data == "q":
            self._running = False
            return

        messages = self._get_all_messages()
        num_messages = len(messages)
        height = self.terminal.rows
        content_height = height - 2

        # j/k move cursor between messages
        if data in ("j", "\x1b[B"):
            if self.cursor < num_messages - 1:
                self.cursor += 1
                self._scroll_to_cursor(content_height)
                self.auto_scroll = self.cursor >= num_messages - 1
            self._needs_redraw = True
        elif data in ("k", "\x1b[A"):
            if self.cursor > 0:
                self.cursor -= 1
                self._scroll_to_cursor(content_height)
                self.auto_scroll = False
            self._needs_redraw = True
        elif data == "\x04":  # Ctrl+D - move cursor down by half page worth of messages
            self.cursor = min(num_messages - 1, self.cursor + 5)
            self._scroll_to_cursor(content_height)
            self.auto_scroll = self.cursor >= num_messages - 1
            self._needs_redraw = True
        elif data == "\x15":  # Ctrl+U - move cursor up by half page worth of messages
            self.cursor = max(0, self.cursor - 5)
            self._scroll_to_cursor(content_height)
            self.auto_scroll = False
            self._needs_redraw = True
        elif data == "g":
            next_char = self._wait_for_char()
            if next_char == "g":
                self.cursor = 0
                self._scroll_to_cursor(content_height)
                self.auto_scroll = False
                self._needs_redraw = True
        elif data == "G":
            self.cursor = num_messages - 1
            self._scroll_to_cursor(content_height)
            self.auto_scroll = True
            self._needs_redraw = True
        elif data in ("h", "\x1b[D"):
            self.h_scroll = max(0, self.h_scroll - 20)
            self.wrap = self.h_scroll == 0
            self._needs_redraw = True
        elif data in ("l", "\x1b[C"):
            self.h_scroll += 20
            self.wrap = False
            self._needs_redraw = True
        elif data == "0":
            self.h_scroll = 0
            self._needs_redraw = True
        elif data == "w":
            self.wrap = not self.wrap
            if self.wrap:
                self.h_scroll = 0
            self._needs_redraw = True
        # Fold commands (nvim-style)
        elif data == "z":
            self._handle_fold_command()
        # Tab/Space/Enter to toggle fold at cursor
        elif data in ("\t", " ", "\r", "\n"):
            self._toggle_fold_at_cursor()
            self._needs_redraw = True

    def _handle_fold_command(self) -> None:
        """Handle nvim-style fold commands (za, zo, zc, zM, zR)."""
        next_char = self._wait_for_char()
        if next_char is None:
            return

        if next_char == "a":  # za - toggle fold at cursor
            self._toggle_fold_at_cursor()
        elif next_char == "o":  # zo - open fold at cursor
            self._set_fold_at_cursor(False)
        elif next_char == "c":  # zc - close fold at cursor
            self._set_fold_at_cursor(True)
        elif next_char == "M":  # zM - close all folds
            for i in self._folds:
                self._folds[i] = True
            self._render_content()
        elif next_char == "R":  # zR - open all folds
            for i in self._folds:
                self._folds[i] = False
            self._render_content()

        self._needs_redraw = True

    def _toggle_fold_at_cursor(self) -> None:
        """Toggle fold for the message at cursor."""
        self._folds[self.cursor] = not self._folds.get(self.cursor, False)
        self._render_content()

    def _set_fold_at_cursor(self, folded: bool) -> None:
        """Set fold state for the message at cursor."""
        self._folds[self.cursor] = folded
        self._render_content()

    def _get_plain_text(self) -> str:
        """Get trace content as plain text for nvim."""
        lines = []
        for msg in self._get_messages():
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"[{role}]")
            lines.append("")
            for line in content.split("\n"):
                lines.append(line)
            lines.append("")
        return "\n".join(lines)

    def _open_in_nvim(self) -> None:
        """Open streaming trace in neovim with auto-reload."""
        import os
        import subprocess
        import tempfile
        import threading

        content = self._get_plain_text()

        fd, path = tempfile.mkstemp(suffix=".md")
        os.write(fd, content.encode())
        os.close(fd)

        self.terminal.stop()

        stop_streaming = threading.Event()
        last_hash = [hash(content)]  # Use list to allow mutation in closure

        def stream_updates() -> None:
            while not stop_streaming.is_set() and self._is_streaming:
                new_content = self._get_plain_text()
                new_hash = hash(new_content)
                if new_hash != last_hash[0]:
                    with open(path, "w") as f:
                        f.write(new_content)
                    last_hash[0] = new_hash
                time.sleep(0.1)

        thread = threading.Thread(target=stream_updates, daemon=True)
        thread.start()

        try:
            init_cmd = " | ".join([
                "set filetype=markdown",
                "set number",
                "set cursorline",
                "set nomodifiable",
                "set autoread",
                # Timer-based auto-reload (every 200ms, even while scrolling)
                "let g:stream_timer = timer_start(200, {-> execute('silent! checktime')}, {'repeat': -1})",
                "autocmd FileChangedShellPost * set modifiable | silent! %d | silent! read | 1d | set nomodifiable | normal! G",
                "nnoremap <silent> q :call timer_stop(g:stream_timer) \\| qa!<CR>",
                "set laststatus=2",
                "set statusline=%f\\ [STREAMING]\\ %l/%L",
                "normal! G",
            ])
            subprocess.run(["nvim", "-R", f"+{init_cmd}", path])
        finally:
            stop_streaming.set()
            os.unlink(path)
            self.terminal.start(on_input=lambda x: None, on_resize=self._on_resize)

    def _wait_for_char(self, timeout: float = 0.5) -> str | None:
        start = time.time()
        while time.time() - start < timeout:
            data = self.terminal.read_input()
            if data:
                return data
            time.sleep(0.01)
        return None

    def _render_content(self) -> None:
        """Render content lines from rollout with fold support and cursor highlight."""
        self._rendered_lines = []
        self._msg_to_line = {}

        role_colors = {
            "user": CYAN,
            "assistant": GREEN,
            "tool": YELLOW,
            "system": MAGENTA,
        }

        # Cursor styling - bright yellow background for clear visibility
        BG_CURSOR = "\x1b[48;2;80;80;40m"  # Muted yellow-brown background
        CURSOR_ARROW = "\x1b[33;1m▸\x1b[0m"  # Bright yellow arrow

        # Get all messages
        messages = self._get_all_messages()

        # Initialize folds for any new messages
        self._init_folds_for_new_messages(messages)

        # Render each message with fold support and cursor
        for msg_idx, msg in enumerate(messages):
            color = role_colors.get(msg.role, WHITE)
            is_folded = self._folds.get(msg_idx, False)
            is_cursor = msg_idx == self.cursor
            content_lines = msg.content.split("\n")
            num_lines = len(content_lines)

            # Track which line this message starts at
            line_idx = len(self._rendered_lines)
            self._msg_to_line[msg_idx] = line_idx

            # Cursor indicator - bright arrow when selected, space otherwise
            cursor_marker = CURSOR_ARROW if is_cursor else " "
            bg = BG_CURSOR if is_cursor else ""
            bg_reset = RESET if is_cursor else ""

            if is_folded:
                # Show folded summary: ▸ ▶ [role] --- N lines ---
                fold_marker = "▶"
                preview = content_lines[0][:40] if content_lines else ""
                if len(content_lines[0]) > 40 if content_lines else False:
                    preview += "..."
                summary = f"{bg}{cursor_marker} {fold_marker} {color}{BOLD}[{msg.role}]{RESET}{bg} {DIM}--- {num_lines} lines: {preview}{RESET}{bg_reset}"
                self._rendered_lines.append(summary)
            else:
                # Show expanded: ▸ ▼ [role] followed by content
                fold_marker = "▼"
                header = (
                    f"{bg}{cursor_marker} {fold_marker} {color}{BOLD}[{msg.role}]{RESET}{bg_reset}"
                )
                self._rendered_lines.append(header)
                for line in content_lines:
                    self._rendered_lines.append(f"    {line}")

            self._rendered_lines.append("")

    def _render(self) -> None:
        width = self.terminal.columns
        height = self.terminal.rows
        content_height = height - 2

        self.terminal.write("\x1b[2J\x1b[H")

        output = []
        output.append(self._render_header(width))

        visible = self._rendered_lines[self.scroll : self.scroll + content_height]
        for ln in visible:
            if self.wrap:
                while ln and len(output) < content_height + 1:
                    chunk = ln[: width - 1]
                    ln = ln[width - 1 :]
                    output.append(chunk + " " * max(0, width - len(chunk)))
                if not ln:
                    continue
            else:
                display_ln = ln
                if self.h_scroll > 0:
                    display_ln = (
                        display_ln[self.h_scroll :] if self.h_scroll < len(display_ln) else ""
                    )
                if len(display_ln) > width - 1:
                    display_ln = display_ln[: width - 4] + "..."
                output.append(display_ln + " " * max(0, width - len(display_ln)))

        while len(output) < height - 1:
            output.append(" " * width)

        output.append(self._render_footer(width))

        for i, line in enumerate(output):
            self.terminal.write(f"\x1b[{i + 1};1H{line}")

    def _render_header(self, width: int) -> str:
        sample_id = self.sample_id
        char_count = len(self.rollout.response)

        if self.rollout.is_streaming:
            status = f"{YELLOW}● STREAMING{RESET}{BG_HEADER}"
        else:
            reward = self.rollout.reward or 0.0
            reward_color = GREEN if reward > 0 else RED if reward < 0 else WHITE
            status = f"{reward_color}reward: {reward:.3f}{RESET}{BG_HEADER}"

        left = f" [{sample_id}] {status} ({char_count} chars)"
        padding = width - len(f" [{sample_id}]  STREAMING ({char_count} chars)")
        return f"{BG_HEADER}{BOLD}{left}{RESET}{BG_HEADER}{' ' * max(0, padding)}{RESET}"

    def _render_footer(self, width: int) -> str:
        wrap_hint = "w:truncate" if self.wrap else "h/l:scroll  w:wrap"
        hints = f"j/k:move  Space:fold  zR/zM:all  {wrap_hint}  q:back"
        messages = self._get_all_messages()
        num_messages = len(messages)
        pos = f"msg {self.cursor + 1}/{num_messages}"
        if self.auto_scroll and self.rollout.is_streaming:
            pos += " [FOLLOW]"

        padding = width - len(hints) - len(pos) - 6
        return (
            f"{BG_HEADER} {DIM}{hints}{RESET}{BG_HEADER}{' ' * max(0, padding)}{WHITE}{pos} {RESET}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Picker (samples within a group)
# ─────────────────────────────────────────────────────────────────────────────


class RolloutPicker:
    """Pick a rollout from a group."""

    def __init__(self, group: Group, line_queue: list[str] | None = None) -> None:
        self.group = group
        self.line_queue = line_queue
        self.cursor = 0
        self.scroll = 0
        self.terminal = Terminal(use_alternate_screen=True)
        self._running = False
        self._needs_redraw = True

    def run(self) -> None:
        if not self.group.rollouts:
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
        if data == "q":
            self._running = False
            return

        height = self.terminal.rows
        content_height = height - 2

        if data in ("j", "\x1b[B"):
            if self.cursor < len(self.group.rollouts) - 1:
                self.cursor += 1
                if self.cursor >= self.scroll + content_height:
                    self.scroll = self.cursor - content_height + 1
                self._needs_redraw = True
        elif data in ("k", "\x1b[A"):
            if self.cursor > 0:
                self.cursor -= 1
                if self.cursor < self.scroll:
                    self.scroll = self.cursor
                self._needs_redraw = True
        elif data == "g":
            next_char = self._wait_for_char()
            if next_char == "g":
                self.cursor = 0
                self.scroll = 0
                self._needs_redraw = True
        elif data == "G":
            self.cursor = len(self.group.rollouts) - 1
            self.scroll = max(0, self.cursor - content_height + 1)
            self._needs_redraw = True
        elif data in ("\r", "\n"):
            self._open_trace()

    def _wait_for_char(self, timeout: float = 0.5) -> str | None:
        start = time.time()
        while time.time() - start < timeout:
            data = self.terminal.read_input()
            if data:
                return data
            time.sleep(0.01)
        return None

    def _open_trace(self) -> None:
        rollout = self.group.rollouts[self.cursor]
        self.terminal.stop()

        # Use streaming viewer for streaming rollouts
        if rollout.is_streaming:
            viewer = TraceStreamingViewer(rollout)
        else:
            viewer = TraceViewer(rollout)

        viewer.run()

        self.terminal.start(on_input=lambda x: None, on_resize=self._on_resize)
        self._needs_redraw = True

    def _render(self) -> None:
        width = self.terminal.columns
        height = self.terminal.rows
        content_height = height - 2

        self.terminal.write("\x1b[2J\x1b[H")

        output = []
        output.append(self._render_header(width))

        visible = self.group.rollouts[self.scroll : self.scroll + content_height]
        for i, rollout in enumerate(visible):
            idx = self.scroll + i
            output.append(self._render_rollout_line(rollout, idx == self.cursor, width))

        while len(output) < height - 1:
            output.append(" " * width)

        output.append(self._render_footer(width))

        for i, line in enumerate(output):
            self.terminal.write(f"\x1b[{i + 1};1H{line}")

    def _render_header(self, width: int) -> str:
        g = self.group
        # Truncate prompt for header
        prompt_preview = g.prompt[:50] + "..." if len(g.prompt) > 50 else g.prompt
        prompt_preview = prompt_preview.replace("\n", " ")
        left = f" Step {g.step} / Group {g.group_id}: {prompt_preview}"

        if len(left) > width - 20:
            left = left[: width - 23] + "..."

        padding = width - len(left)
        return f"{BG_HEADER}{BOLD}{left}{RESET}{BG_HEADER}{' ' * max(0, padding)}{RESET}"

    def _render_rollout_line(self, r: Rollout, selected: bool, width: int) -> str:
        cursor = "> " if selected else "  "

        # Show streaming indicator or reward
        if r.is_streaming:
            char_count = len(r.response)
            status_str = f"{YELLOW}● streaming ({char_count} chars){RESET}"
        else:
            reward_color = GREEN if r.reward > 0.5 else YELLOW if r.reward > 0 else RED
            status_str = f"reward={reward_color}{r.reward:.3f}{RESET}"

        # Response preview
        response_preview = r.response[:50].replace("\n", " ")
        if len(r.response) > 50:
            response_preview += "..."

        # Use sample_id_str if available, otherwise sample_id
        sample_label = r.sample_id_str if r.sample_id_str else f"Sample {r.sample_id}"

        if selected:
            line = f"{BOLD}{cursor}{sample_label}  {status_str}  {response_preview}{RESET}"
        else:
            line = (
                f"{DIM}{cursor}{sample_label}  {RESET}{status_str}{DIM}  {response_preview}{RESET}"
            )

        return line

    def _render_footer(self, width: int) -> str:
        hints = "j/k:move  Enter:view  q:back"
        pos = f"{self.cursor + 1}/{len(self.group.rollouts)}"

        padding = width - len(hints) - len(pos) - 4
        return (
            f"{BG_HEADER} {DIM}{hints}{RESET}{BG_HEADER}{' ' * max(0, padding)}{WHITE}{pos} {RESET}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Group Picker (prompt groups within a step)
# ─────────────────────────────────────────────────────────────────────────────


class GroupPicker:
    """Pick a group from a step."""

    def __init__(self, step: Step, line_queue: list[str] | None = None) -> None:
        self.step = step
        self.line_queue = line_queue
        self.cursor = 0
        self.scroll = 0
        self.terminal = Terminal(use_alternate_screen=True)
        self._running = False
        self._needs_redraw = True

    def run(self) -> None:
        if not self.step.groups:
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
        if data == "q":
            self._running = False
            return

        height = self.terminal.rows
        content_height = height - 2

        if data in ("j", "\x1b[B"):
            if self.cursor < len(self.step.groups) - 1:
                self.cursor += 1
                if self.cursor >= self.scroll + content_height:
                    self.scroll = self.cursor - content_height + 1
                self._needs_redraw = True
        elif data in ("k", "\x1b[A"):
            if self.cursor > 0:
                self.cursor -= 1
                if self.cursor < self.scroll:
                    self.scroll = self.cursor
                self._needs_redraw = True
        elif data == "g":
            next_char = self._wait_for_char()
            if next_char == "g":
                self.cursor = 0
                self.scroll = 0
                self._needs_redraw = True
        elif data == "G":
            self.cursor = len(self.step.groups) - 1
            self.scroll = max(0, self.cursor - content_height + 1)
            self._needs_redraw = True
        elif data in ("\r", "\n"):
            self._open_group()

    def _wait_for_char(self, timeout: float = 0.5) -> str | None:
        start = time.time()
        while time.time() - start < timeout:
            data = self.terminal.read_input()
            if data:
                return data
            time.sleep(0.01)
        return None

    def _open_group(self) -> None:
        group = self.step.groups[self.cursor]
        self.terminal.stop()

        picker = RolloutPicker(group, line_queue=self.line_queue)
        picker.run()

        self.terminal.start(on_input=lambda x: None, on_resize=self._on_resize)
        self._needs_redraw = True

    def _render(self) -> None:
        width = self.terminal.columns
        height = self.terminal.rows
        content_height = height - 2

        self.terminal.write("\x1b[2J\x1b[H")

        output = []
        output.append(self._render_header(width))

        visible = self.step.groups[self.scroll : self.scroll + content_height]
        for i, group in enumerate(visible):
            idx = self.scroll + i
            output.append(self._render_group_line(group, idx == self.cursor, width))

        while len(output) < height - 1:
            output.append(" " * width)

        output.append(self._render_footer(width))

        for i, line in enumerate(output):
            self.terminal.write(f"\x1b[{i + 1};1H{line}")

    def _render_header(self, width: int) -> str:
        s = self.step
        left = f" Step {s.step} - {len(s.groups)} groups, {s.num_rollouts} rollouts"
        reward_color = GREEN if s.avg_reward > 0.5 else YELLOW if s.avg_reward > 0 else WHITE
        right = f"avg reward: {reward_color}{s.avg_reward:.3f}{RESET}{BG_HEADER} "

        padding = width - len(left) - len(f"avg reward: {s.avg_reward:.3f} ")
        return f"{BG_HEADER}{BOLD}{left}{RESET}{BG_HEADER}{' ' * max(0, padding)}{right}{RESET}"

    def _render_group_line(self, g: Group, selected: bool, width: int) -> str:
        cursor = "> " if selected else "  "

        # Reward range
        reward_color = GREEN if g.avg_reward > 0.5 else YELLOW if g.avg_reward > 0 else RED
        reward_str = f"{reward_color}{g.avg_reward:.3f}{RESET}"

        # Prompt preview
        prompt_preview = g.prompt[:50].replace("\n", " ")
        if len(g.prompt) > 50:
            prompt_preview += "..."

        n_samples = len(g.rollouts)

        if selected:
            line = f"{BOLD}{cursor}Group {g.group_id}  ({n_samples} samples)  avg={reward_str}  {prompt_preview}{RESET}"
        else:
            line = f"{DIM}{cursor}Group {g.group_id}  ({n_samples} samples)  avg={RESET}{reward_str}{DIM}  {prompt_preview}{RESET}"

        return line

    def _render_footer(self, width: int) -> str:
        hints = "j/k:move  Enter:view  q:back"
        pos = f"{self.cursor + 1}/{len(self.step.groups)}"

        padding = width - len(hints) - len(pos) - 4
        return (
            f"{BG_HEADER} {DIM}{hints}{RESET}{BG_HEADER}{' ' * max(0, padding)}{WHITE}{pos} {RESET}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step Picker (top level)
# ─────────────────────────────────────────────────────────────────────────────


class StepPicker:
    """Pick a training step."""

    def __init__(self, data: TraceData, line_queue: list[str] | None = None) -> None:
        self.data = data
        self.line_queue = line_queue
        self.cursor = 0
        self.scroll = 0
        self.terminal = Terminal(use_alternate_screen=True)
        self._running = False
        self._needs_redraw = True

    def run(self) -> None:
        if not self.data.steps:
            print("No trace data.")
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
        if data == "q":
            self._running = False
            return

        height = self.terminal.rows
        content_height = height - 2

        if data in ("j", "\x1b[B"):
            if self.cursor < len(self.data.steps) - 1:
                self.cursor += 1
                if self.cursor >= self.scroll + content_height:
                    self.scroll = self.cursor - content_height + 1
                self._needs_redraw = True
        elif data in ("k", "\x1b[A"):
            if self.cursor > 0:
                self.cursor -= 1
                if self.cursor < self.scroll:
                    self.scroll = self.cursor
                self._needs_redraw = True
        elif data == "g":
            next_char = self._wait_for_char()
            if next_char == "g":
                self.cursor = 0
                self.scroll = 0
                self._needs_redraw = True
        elif data == "G":
            self.cursor = len(self.data.steps) - 1
            self.scroll = max(0, self.cursor - content_height + 1)
            self._needs_redraw = True
        elif data in ("\r", "\n"):
            self._open_step()

    def _wait_for_char(self, timeout: float = 0.5) -> str | None:
        start = time.time()
        while time.time() - start < timeout:
            data = self.terminal.read_input()
            if data:
                return data
            time.sleep(0.01)
        return None

    def _open_step(self) -> None:
        step = self.data.steps[self.cursor]
        self.terminal.stop()

        picker = GroupPicker(step, line_queue=self.line_queue)
        picker.run()

        self.terminal.start(on_input=lambda x: None, on_resize=self._on_resize)
        self._needs_redraw = True

    def _render(self) -> None:
        width = self.terminal.columns
        height = self.terminal.rows
        content_height = height - 2

        self.terminal.write("\x1b[2J\x1b[H")

        output = []
        output.append(self._render_header(width))

        visible = self.data.steps[self.scroll : self.scroll + content_height]
        for i, step in enumerate(visible):
            idx = self.scroll + i
            output.append(self._render_step_line(step, idx == self.cursor, width))

        while len(output) < height - 1:
            output.append(" " * width)

        output.append(self._render_footer(width))

        for i, line in enumerate(output):
            self.terminal.write(f"\x1b[{i + 1};1H{line}")

    def _render_header(self, width: int) -> str:
        total_rollouts = sum(s.num_rollouts for s in self.data.steps)
        left = f" Training Steps ({len(self.data.steps)} steps, {total_rollouts} rollouts)"
        padding = width - len(left)
        return f"{BG_HEADER}{BOLD}{left}{RESET}{BG_HEADER}{' ' * max(0, padding)}{RESET}"

    def _render_step_line(self, s: Step, selected: bool, width: int) -> str:
        cursor = "> " if selected else "  "

        # Reward with color
        reward_color = GREEN if s.avg_reward > 0.5 else YELLOW if s.avg_reward > 0 else RED
        reward_str = f"{reward_color}{s.avg_reward:.3f}{RESET}"

        # Stats
        stats = f"{len(s.groups)} groups, {s.num_rollouts} rollouts"

        if selected:
            line = f"{BOLD}{cursor}Step {s.step:>3}  avg_reward={reward_str}  {stats}{RESET}"
        else:
            line = f"{DIM}{cursor}Step {s.step:>3}  avg_reward={RESET}{reward_str}{DIM}  {stats}{RESET}"

        return line

    def _render_footer(self, width: int) -> str:
        hints = "j/k:move  Enter:view  q:quit"
        pos = f"{self.cursor + 1}/{len(self.data.steps)}"

        padding = width - len(hints) - len(pos) - 4
        return (
            f"{BG_HEADER} {DIM}{hints}{RESET}{BG_HEADER}{' ' * max(0, padding)}{WHITE}{pos} {RESET}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m rollouts.tui.traces <rollouts.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    data = TraceData.from_jsonl(path)

    if not data.steps:
        print(f"No traces found in {path}")
        sys.exit(1)

    picker = StepPicker(data)
    picker.run()


if __name__ == "__main__":
    main()
