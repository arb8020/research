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
import select
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .input import FocusEvent, KeyPress, MouseEvent, PasteEvent  # noqa: F401
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

    def _stable_callable_key(fn: Callable[..., object]) -> tuple:
        """Return a stable, hashable identity for callables.

        Subscriptions are recomputed on every update(). Users often write
        `lambda ...` inline inside subscriptions(), which creates a new
        function object each call. Keying on `id(fn)` causes subscriptions
        to churn (stop/restart) on every message, which breaks file tailing
        and floods the UI with duplicate log lines.

        Prefer code-location-based keys when available; fall back to object
        identity for non-function callables.
        """
        # Bound method: stabilize on (underlying func location, instance id)
        func = getattr(fn, "__func__", None)
        self_obj = getattr(fn, "__self__", None)
        if func is not None and hasattr(func, "__code__"):
            code = func.__code__
            return ("method", code.co_filename, code.co_firstlineno, code.co_name, id(self_obj))

        # Plain function / lambda: stabilize on code location (and closure contents)
        code = getattr(fn, "__code__", None)
        if code is not None:
            closure = getattr(fn, "__closure__", None) or ()
            closure_key = []
            for cell in closure:
                try:
                    val = cell.cell_contents
                except ValueError:
                    val = None
                if val is None or isinstance(val, (bool, int, float, str)):
                    closure_key.append(("lit", val))
                else:
                    closure_key.append(("id", id(val)))
            return (
                "func",
                code.co_filename,
                code.co_firstlineno,
                code.co_name,
                tuple(closure_key),
            )

        return ("id", id(fn))

    if sub._kind == "every":
        interval, fn = sub._data
        return ("every", interval, _stable_callable_key(fn))
    elif sub._kind == "file_tail":
        path, fn = sub._data
        return ("file_tail", path, _stable_callable_key(fn))
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


class _WakeQueue:
    """Queue that writes a byte to a wakeup pipe on put().

    Background threads enqueue messages here. The main loop select()s
    on the wakeup fd alongside stdin, so it wakes immediately instead
    of polling on a timer.
    """

    __slots__ = ("_q", "_wakeup_w")

    def __init__(self, wakeup_w: int) -> None:
        self._q: queue.Queue = queue.Queue()
        self._wakeup_w = wakeup_w

    def put(self, msg: object) -> None:
        self._q.put(msg)
        try:
            os.write(self._wakeup_w, b"\x00")
        except OSError:
            pass  # Pipe full or closed — main loop will drain queue anyway

    def get_nowait(self) -> object:
        return self._q.get_nowait()

    def empty(self) -> bool:
        return self._q.empty()


def _run_every(
    interval: float,
    msg_fn: Callable[[], object],
    msg_queue: _WakeQueue,
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
    msg_queue: _WakeQueue,
    stop: threading.Event,
) -> None:
    """Thread target for Sub.file_tail.

    Waits for file to exist, reads existing content, then tails for new lines.
    """
    p = Path(path)

    # Wait for file to exist
    while not p.exists() and not stop.is_set():
        stop.wait(0.5)

    if stop.is_set():
        return

    # Use errors="replace" so a single bad byte can't kill the tail thread.
    with open(p, encoding="utf-8", errors="replace") as f:
        # Read from beginning (existing content + new lines).
        # Keep a buffer so we don't emit partial lines when the writer flushes
        # without a trailing newline.
        pending = ""
        while not stop.is_set():
            chunk = f.readline()
            if not chunk:
                stop.wait(0.1)
                continue

            if chunk.endswith("\n"):
                full = pending + chunk
                pending = ""
                stripped = full.rstrip("\n")
                if stripped:  # Skip blank lines
                    msg_queue.put(msg_fn(stripped))
            else:
                pending += chunk


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
        mouse: Enable mouse tracking (wheel scroll, clicks).
        fps: Deprecated — kept for backwards compat. Sets min_frame_interval
            to 1/fps. Prefer min_frame_interval directly.
        min_frame_interval: Minimum seconds between renders. Batches rapid
            events (e.g. file tail spew) into fewer frames. Default 0.016
            (~60fps cap). Set to 0 for unlimited.
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
        mouse: bool = False,
        fps: int | None = None,
        min_frame_interval: float = 0.016,
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
        self._mouse = mouse
        self._min_frame_interval = 1.0 / fps if fps is not None else min_frame_interval
        self._debug_fn = debug_fn
        self._debug_frame_interval = debug_frame_interval

        self._model: Any = None
        self._running = False
        self._frame_count: int = 0

        # Wakeup pipe: background threads write a byte here to wake select()
        self._wakeup_r, self._wakeup_w = os.pipe()
        os.set_blocking(self._wakeup_r, False)
        os.set_blocking(self._wakeup_w, False)

        self._msg_queue = _WakeQueue(self._wakeup_w)
        self._terminal: Terminal | None = None

        # Set up debug logging
        global _DEBUG_LOG
        if debug_log is not None:
            _DEBUG_LOG = Path(debug_log)
            # Truncate on startup
            _DEBUG_LOG.write_text("")
            os.chmod(_DEBUG_LOG, 0o644)
            _log(
                "app_init",
                min_frame_interval=self._min_frame_interval,
                alternate_screen=alternate_screen,
            )
        else:
            _DEBUG_LOG = None
        self._render_state = RenderState(log_fn=_log if _DEBUG_LOG is not None else None)
        self._active_subs: dict[tuple, _SubRunner] = {}

    def send(self, msg: object) -> None:
        """Send a message from outside the update loop (thread-safe).

        Wakes the main loop immediately via the wakeup pipe.
        """
        self._msg_queue.put(msg)

    def _drain_wakeup(self) -> None:
        """Drain all bytes from the wakeup pipe (non-blocking)."""
        try:
            while os.read(self._wakeup_r, 4096):
                pass
        except (BlockingIOError, OSError):
            pass

    def run(self) -> None:
        """Run the application. Blocks until quit.

        Event-driven: blocks on select() until stdin has input or a
        background thread enqueues a message (via wakeup pipe).
        Zero CPU when idle.
        """
        terminal = Terminal(
            alternate_screen=self._alternate_screen,
            bracketed_paste=self._bracketed_paste,
            mouse=self._mouse,
        )
        self._terminal = terminal

        # We don't use the on_input callback — we select() on the tty fd.
        # Resize writes to the wakeup pipe so select() wakes up.
        def on_input(data: str) -> None:
            pass  # Unused, we read from tty fd directly

        def on_resize() -> None:
            self._msg_queue.put(
                Resize(
                    width=terminal.columns,
                    height=terminal.rows,
                )
            )

        terminal.start(on_input=on_input, on_resize=on_resize)
        terminal.hide_cursor()

        tty_fd = terminal._tty_fd
        wakeup_r = self._wakeup_r
        wait_fds = [wakeup_r]
        if tty_fd is not None:
            wait_fds.append(tty_fd)

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

            last_render = 0.0  # Force immediate first-event render
            min_interval = self._min_frame_interval
            pending_dirty = False
            msgs_since_render = 0
            msg_types_since_render: dict[str, int] = {}

            # Main loop — event-driven, blocks on select()
            while self._running:
                # Calculate select timeout:
                # - If we have pending dirty state from rapid events,
                #   wait only until min_interval elapses, then render.
                # - Otherwise block indefinitely until something happens.
                if pending_dirty:
                    remaining = min_interval - (time.monotonic() - last_render)
                    timeout = max(0.0, remaining)
                else:
                    timeout = None  # Block indefinitely

                try:
                    select.select(wait_fds, [], [], timeout)
                except (InterruptedError, OSError):
                    # SIGWINCH can interrupt select — that's fine,
                    # the resize handler already enqueued a message.
                    pass

                # Drain wakeup pipe
                self._drain_wakeup()

                dirty = False

                # 1. Drain all available keyboard/mouse input
                while True:
                    msg = terminal.read_message()
                    if msg is None:
                        break
                    dirty = True
                    msgs_since_render += 1
                    t = type(msg).__name__
                    msg_types_since_render[t] = msg_types_since_render.get(t, 0) + 1
                    self._dispatch(msg)

                # 2. Drain message queue (from Cmd.task threads, subs, resize)
                while not self._msg_queue.empty():
                    dirty = True
                    try:
                        msg = self._msg_queue.get_nowait()
                    except queue.Empty:
                        break
                    msgs_since_render += 1
                    t = type(msg).__name__
                    msg_types_since_render[t] = msg_types_since_render.get(t, 0) + 1
                    self._dispatch(msg)

                # 3. Re-sync subscriptions if model changed this iteration
                if dirty and self._subs_fn:
                    self._sync_subs(self._subs_fn(self._model))

                if dirty:
                    pending_dirty = True

                # 4. Render if dirty and enough time has passed since last render
                now = time.monotonic()
                if pending_dirty and now - last_render >= min_interval:
                    self._render_state.msgs_batched = msgs_since_render
                    self._render_state.msg_types_batched = dict(msg_types_since_render)
                    self._render()
                    last_render = now
                    pending_dirty = False
                    msgs_since_render = 0
                    msg_types_since_render.clear()

        finally:
            # Stop all subscriptions
            self._stop_all_subs()
            terminal.show_cursor()
            terminal.stop()
            self._terminal = None
            # Close wakeup pipe
            try:
                os.close(self._wakeup_r)
                os.close(self._wakeup_w)
            except OSError:
                pass

    def _dispatch(self, msg: object) -> None:
        """Send message through update, execute resulting command."""
        if not self._running:
            return
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
            t = threading.Thread(
                target=_run_every,
                args=(interval, msg_fn, self._msg_queue, stop),
                daemon=True,
            )
        elif sub._kind == "file_tail":
            path, msg_fn = sub._data
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
