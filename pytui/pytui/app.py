"""Elm-style TUI application runtime.

Provides the App class that runs the Model-Update-View loop,
plus Cmd and Sub types for side effects and subscriptions.

Usage:
    from dataclasses import dataclass, replace
    from pytui import App, Cmd, Sub, KeyPress

    @dataclass(frozen=True)
    class Model:
        count: int = 0

    def update(model: Model, msg: object) -> tuple[Model, Cmd]:
        match msg:
            case KeyPress(key="q"):
                return model, Cmd.quit()
            case KeyPress(key="j"):
                return replace(model, count=model.count + 1), Cmd.none()
        return model, Cmd.none()

    def view(model: Model, width: int, height: int) -> list[str]:
        return [f"Count: {model.count}", "", "j: increment  q: quit"]

    App(init=(Model(), Cmd.none()), update=update, view=view).run()
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .renderer import RenderState, diff_render
from .terminal import Terminal


# ---------------------------------------------------------------------------
# Debug logging (file-based, not stderr)
# ---------------------------------------------------------------------------

_DEBUG_LOG: Path | None = None


def _log(event: str, **data: Any) -> None:
    """Append a debug event to the log file (if enabled)."""
    if _DEBUG_LOG is None:
        return
    entry = {
        "ts": datetime.now().isoformat(),
        "event": event,
        **data,
    }
    with open(_DEBUG_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ---------------------------------------------------------------------------
# Built-in messages (sent by runtime)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyPress:
    """Keyboard input. key is the raw string (e.g. "j", "\x1b[A")."""

    key: str


@dataclass(frozen=True)
class Resize:
    """Terminal was resized."""

    width: int
    height: int


# ---------------------------------------------------------------------------
# Cmd: side effect descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cmd:
    """Side effect descriptor. Created by update(), executed by runtime.

    Users should only create Cmd values via the static methods.
    """

    _kind: str = "none"
    _data: Any = None

    @staticmethod
    def none() -> Cmd:
        """No side effect."""
        return Cmd()

    @staticmethod
    def quit() -> Cmd:
        """Exit the application."""
        return Cmd(_kind="quit")

    @staticmethod
    def batch(*cmds: Cmd) -> Cmd:
        """Run multiple commands."""
        # Flatten: skip nones, unwrap single
        real = [c for c in cmds if c._kind != "none"]
        if not real:
            return Cmd.none()
        if len(real) == 1:
            return real[0]
        return Cmd(_kind="batch", _data=tuple(real))

    @staticmethod
    def task(fn: Callable[[], object]) -> Cmd:
        """Run fn() in a background thread. Result is sent as a message.

        fn must be safe to call from a thread. The return value becomes
        the next message dispatched to update().
        """
        return Cmd(_kind="task", _data=fn)


# ---------------------------------------------------------------------------
# Sub: subscription descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sub:
    """Subscription descriptor. Long-running sources of messages.

    Users should only create Sub values via the static methods.
    """

    _kind: str = "none"
    _data: Any = None

    @staticmethod
    def none() -> Sub:
        """No subscription."""
        return Sub()

    @staticmethod
    def every(interval_sec: float, msg_fn: Callable[[], object]) -> Sub:
        """Call msg_fn() every interval_sec seconds, send result as message."""
        return Sub(_kind="every", _data=(interval_sec, msg_fn))

    @staticmethod
    def file_tail(path: str, msg_fn: Callable[[str], object]) -> Sub:
        """Tail a file. msg_fn(line) called for each new line appended."""
        return Sub(_kind="file_tail", _data=(path, msg_fn))

    @staticmethod
    def batch(*subs: Sub) -> Sub:
        """Combine multiple subscriptions."""
        real = [s for s in subs if s._kind != "none"]
        if not real:
            return Sub.none()
        if len(real) == 1:
            return real[0]
        return Sub(_kind="batch", _data=tuple(real))


# ---------------------------------------------------------------------------
# Internal: subscription identity for lifecycle management
# ---------------------------------------------------------------------------


def _sub_key(sub: Sub) -> tuple:
    """Identity key for a subscription (for start/stop diffing)."""
    if sub._kind == "every":
        interval, fn = sub._data
        return ("every", interval, id(fn))
    elif sub._kind == "file_tail":
        path, fn = sub._data
        return ("file_tail", path, id(fn))
    elif sub._kind == "batch":
        return ("batch", tuple(_sub_key(s) for s in sub._data))
    return ("none",)


def _flatten_subs(sub: Sub) -> list[Sub]:
    """Flatten batch subs into a flat list of leaf subs."""
    if sub._kind == "batch":
        result = []
        for s in sub._data:
            result.extend(_flatten_subs(s))
        return result
    elif sub._kind == "none":
        return []
    return [sub]


# ---------------------------------------------------------------------------
# Internal: subscription runners (threads)
# ---------------------------------------------------------------------------


@dataclass
class _SubRunner:
    """Running subscription thread + stop event."""

    key: tuple
    stop: threading.Event
    thread: threading.Thread


def _run_every(
    interval: float,
    msg_fn: Callable[[], object],
    msg_queue: queue.Queue,
    stop: threading.Event,
) -> None:
    """Thread target for Sub.every."""
    while not stop.is_set():
        stop.wait(interval)
        if not stop.is_set():
            msg_queue.put(msg_fn())


def _run_file_tail(
    path: str,
    msg_fn: Callable[[str], object],
    msg_queue: queue.Queue,
    stop: threading.Event,
) -> None:
    """Thread target for Sub.file_tail.

    Waits for file to exist, reads existing content, then tails for new lines.
    """
    _log("file_tail_start", path=path)
    p = Path(path)

    # Wait for file to exist
    wait_count = 0
    while not p.exists() and not stop.is_set():
        wait_count += 1
        if wait_count % 10 == 1:  # Log every 5 seconds
            _log("file_tail_waiting", path=path, wait_seconds=wait_count * 0.5)
        stop.wait(0.5)

    if stop.is_set():
        _log("file_tail_stopped_before_open", path=path)
        return

    _log("file_tail_opened", path=path, size=p.stat().st_size)
    line_count = 0

    with open(p) as f:
        # Read from beginning (existing content + new lines)
        while not stop.is_set():
            line = f.readline()
            if line:
                stripped = line.rstrip("\n")
                if stripped:  # Skip blank lines
                    line_count += 1
                    msg_queue.put(msg_fn(stripped))
                    if line_count <= 5 or line_count % 100 == 0:
                        _log("file_tail_line", path=path, line_num=line_count, preview=stripped[:80])
            else:
                stop.wait(0.1)

    _log("file_tail_stopped", path=path, total_lines=line_count)


# ---------------------------------------------------------------------------
# App: the runtime
# ---------------------------------------------------------------------------

# Type aliases for user-provided functions
UpdateFn = Callable[[Any, object], tuple[Any, Cmd]]
ViewFn = Callable[[Any, int, int], list[str]]
SubsFn = Callable[[Any], Sub]


class App:
    """Elm-style TUI application runtime.

    Owns the terminal, runs the main loop, executes commands,
    manages subscriptions, and calls view.

    Args:
        init: (initial_model, initial_cmd) tuple.
        update: Pure function (model, msg) -> (model, cmd).
        view: Pure function (model, width, height) -> list[str].
        subscriptions: Optional function (model) -> Sub.
        alternate_screen: Use alternate screen buffer (monitor-style apps).
        bracketed_paste: Enable bracketed paste mode (editor-style apps).
        fps: Target frames per second for the render loop.
        debug_log: Path to debug log file. If set, logs subscriptions, messages,
            and app lifecycle events to this file as JSONL.
        debug_fn: Optional callback (model, width, height, frame_count) -> None.
            Called every debug_frame_interval rendered frames. For dumping
            layout snapshots, model state, etc. to a debug log file.
        debug_frame_interval: How often to call debug_fn (every N frames).
    """

    def __init__(
        self,
        *,
        init: tuple[Any, Cmd],
        update: UpdateFn,
        view: ViewFn,
        subscriptions: SubsFn | None = None,
        alternate_screen: bool = True,
        bracketed_paste: bool = False,
        fps: int = 30,
        debug_log: str | Path | None = None,
        debug_fn: Callable[[Any, int, int, int], None] | None = None,
        debug_frame_interval: int = 100,
    ) -> None:
        self._init = init
        self._update_fn = update
        self._view_fn = view
        self._subs_fn = subscriptions
        self._alternate_screen = alternate_screen
        self._bracketed_paste = bracketed_paste
        self._fps = fps
        self._debug_fn = debug_fn
        self._debug_frame_interval = debug_frame_interval

        self._model: Any = None
        self._running = False
        self._frame_count: int = 0
        self._msg_queue: queue.Queue = queue.Queue()
        self._terminal: Terminal | None = None

        # Set up debug logging
        global _DEBUG_LOG
        if debug_log is not None:
            _DEBUG_LOG = Path(debug_log)
            # Truncate on startup
            _DEBUG_LOG.write_text("")
            os.chmod(_DEBUG_LOG, 0o644)
            _log("app_init", fps=fps, alternate_screen=alternate_screen)
        else:
            _DEBUG_LOG = None
        self._render_state = RenderState()
        self._active_subs: dict[tuple, _SubRunner] = {}

    def send(self, msg: object) -> None:
        """Send a message from outside the update loop (thread-safe)."""
        self._msg_queue.put(msg)

    def run(self) -> None:
        """Run the application. Blocks until quit."""
        terminal = Terminal(
            alternate_screen=self._alternate_screen,
            bracketed_paste=self._bracketed_paste,
        )
        self._terminal = terminal

        # We don't use the on_input callback - we poll with read_input instead.
        # But we need resize to enqueue a message.
        def on_input(data: str) -> None:
            pass  # Unused, we poll

        def on_resize() -> None:
            self._msg_queue.put(
                Resize(
                    width=terminal.columns,
                    height=terminal.rows,
                )
            )

        terminal.start(on_input=on_input, on_resize=on_resize)
        terminal.hide_cursor()

        try:
            # Initialize
            self._model, init_cmd = self._init
            self._running = True
            self._execute_cmd(init_cmd)

            # Initial subscription setup
            if self._subs_fn:
                self._sync_subs(self._subs_fn(self._model))

            # Initial render
            self._render()

            # Main loop
            sleep_time = 1.0 / self._fps
            while self._running:
                dirty = False

                # 1. Poll keyboard input
                key = terminal.read_input()
                if key is not None:
                    dirty = True
                    self._dispatch(KeyPress(key=key))

                # 2. Drain message queue (from Cmd.task threads, subs, resize)
                while not self._msg_queue.empty():
                    dirty = True
                    try:
                        msg = self._msg_queue.get_nowait()
                    except queue.Empty:
                        break
                    self._dispatch(msg)

                # 3. Re-sync subscriptions if model changed
                if dirty and self._subs_fn:
                    self._sync_subs(self._subs_fn(self._model))

                # 4. Render if anything changed
                if dirty:
                    self._render()

                time.sleep(sleep_time)

        finally:
            # Stop all subscriptions
            self._stop_all_subs()
            terminal.show_cursor()
            terminal.stop()
            self._terminal = None

    def _dispatch(self, msg: object) -> None:
        """Send message through update, execute resulting command."""
        if not self._running:
            return
        msg_type = type(msg).__name__
        _log("dispatch", msg_type=msg_type)
        self._model, cmd = self._update_fn(self._model, msg)
        self._execute_cmd(cmd)

    def _execute_cmd(self, cmd: Cmd) -> None:
        """Execute a command."""
        if cmd._kind == "none":
            return
        elif cmd._kind == "quit":
            self._running = False
        elif cmd._kind == "batch":
            for c in cmd._data:
                self._execute_cmd(c)
        elif cmd._kind == "task":
            fn = cmd._data
            q = self._msg_queue

            def run() -> None:
                result = fn()
                q.put(result)

            t = threading.Thread(target=run, daemon=True)
            t.start()

    def _render(self) -> None:
        """Render current model to terminal."""
        if self._terminal is None:
            return
        width = self._terminal.columns
        height = self._terminal.rows
        lines = self._view_fn(self._model, width, height)
        diff_render(self._terminal, lines, self._render_state)

        self._frame_count += 1
        if self._debug_fn is not None and self._frame_count % self._debug_frame_interval == 0:
            self._debug_fn(self._model, width, height, self._frame_count)

    # ---------------------------------------------------------------------------
    # Subscription lifecycle
    # ---------------------------------------------------------------------------

    def _sync_subs(self, wanted: Sub) -> None:
        """Start/stop subscriptions to match what's wanted."""
        wanted_flat = _flatten_subs(wanted)
        wanted_keys = {_sub_key(s): s for s in wanted_flat}

        # Stop subs that are no longer wanted
        to_stop = [k for k in self._active_subs if k not in wanted_keys]
        for k in to_stop:
            runner = self._active_subs.pop(k)
            runner.stop.set()

        # Start subs that are new
        for k, sub in wanted_keys.items():
            if k not in self._active_subs:
                self._start_sub(k, sub)

    def _start_sub(self, key: tuple, sub: Sub) -> None:
        """Start a subscription thread."""
        stop = threading.Event()

        if sub._kind == "every":
            interval, msg_fn = sub._data
            _log("sub_start", kind="every", interval=interval)
            t = threading.Thread(
                target=_run_every,
                args=(interval, msg_fn, self._msg_queue, stop),
                daemon=True,
            )
        elif sub._kind == "file_tail":
            path, msg_fn = sub._data
            _log("sub_start", kind="file_tail", path=path)
            t = threading.Thread(
                target=_run_file_tail,
                args=(path, msg_fn, self._msg_queue, stop),
                daemon=True,
            )
        else:
            return

        t.start()
        self._active_subs[key] = _SubRunner(key=key, stop=stop, thread=t)

    def _stop_all_subs(self) -> None:
        """Stop all running subscriptions."""
        for runner in self._active_subs.values():
            runner.stop.set()
        self._active_subs.clear()
