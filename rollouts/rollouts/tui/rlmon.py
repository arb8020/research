"""Experiment monitor — btop-style TUI for RL, SFT, eval, and generic runs.

Elm architecture on pytui. Auto-detects experiment type from files present
in the output directory:

  metrics.jsonl + rollouts.jsonl  → RL mode
  metrics.jsonl (no rollouts)     → SFT mode
  events.jsonl                    → Eval mode
  fallback                        → Generic (tail whatever exists)

Layout (RL mode):
  ╭─ Metrics ─────────────────────────────╮╭─ Config ──────────╮
  │ reward  ▁▂▃▅▆▇█▇▆▅  0.73            ││ model  Qwen3-0.6B │
  │ loss    █▇▆▅▃▂▁▁▁▁  0.12            ││ lr     1e-6       │
  │ entropy ▅▅▅▄▄▃▃▃▃▂  3.21            ││ step   42/100     │
  ╰───────────────────────────────────────╯╰────────────────────╯
  ╭─ Training ────────────────────────────────────────────────────╮
  │ Step 42/100  reward=0.73  loss=0.12                           │
  ╰───────────────────────────────────────────────────────────────╯
  ╭─ SGLang ──────────────────────────────────────────────────────╮
  │ Request /v1/completions received                              │
  ╰───────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path

from pytui import RESET, App, Cmd, KeyPress, Sub, hex_to_fg
from pytui.text import slice_ansi, truncate_to_width, visible_width

# ─── Experiment type detection ───────────────────────────────────────────


class ExperimentType(Enum):
    RL = auto()  # metrics.jsonl + rollouts.jsonl
    SFT = auto()  # metrics.jsonl, no rollouts
    EVAL = auto()  # events.jsonl
    GENERIC = auto()  # fallback


def detect_experiment_type(watch_dir: str) -> ExperimentType:
    """Detect experiment type from files present in the output directory."""
    d = Path(watch_dir)
    has_metrics = (d / "metrics.jsonl").exists()
    has_rollouts = (d / "rollouts.jsonl").exists()
    has_events = (d / "events.jsonl").exists()

    if has_metrics and has_rollouts:
        return ExperimentType.RL
    if has_events:
        return ExperimentType.EVAL
    if has_metrics:
        return ExperimentType.SFT
    return ExperimentType.GENERIC


# ─── Colors (btop-inspired) ──────────────────────────────────────────────

C_BORDER = hex_to_fg("#555555")
C_BORDER_ACCENT = hex_to_fg("#8abeb7")
C_TITLE = hex_to_fg("#8abeb7")
C_DIM = hex_to_fg("#555555")
C_TEXT = hex_to_fg("#b0b0b0")
C_BRIGHT = hex_to_fg("#e0e0e0")
C_VALUE = hex_to_fg("#f0c674")
C_GOOD = hex_to_fg("#b5bd68")
C_WARN = hex_to_fg("#f0c674")
C_BAD = hex_to_fg("#cc6666")
C_LABEL = hex_to_fg("#81a2be")
C_SPARK = hex_to_fg("#8abeb7")

# Box drawing
TL = "╭"
TR = "╮"
BL = "╰"
BR = "╯"
H = "─"
V = "│"

SPARK_CHARS = " ▁▂▃▄▅▆▇█"


# ─── Messages ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricsLine:
    line: str


@dataclass(frozen=True)
class TrainingLine:
    line: str


@dataclass(frozen=True)
class SglangLine:
    line: str


@dataclass(frozen=True)
class RolloutLine:
    line: str


@dataclass(frozen=True)
class EventLine:
    line: str


@dataclass(frozen=True)
class GenericLogLine:
    line: str


@dataclass(frozen=True)
class ConfigLoaded:
    config: dict


@dataclass(frozen=True)
class ExperimentDetected:
    experiment_type: ExperimentType


# ─── Model ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricSeries:
    """Immutable metric timeseries."""

    name: str
    values: tuple[float, ...] = ()

    def append(self, value: float, max_len: int = 200) -> MetricSeries:
        new = self.values + (value,)
        if len(new) > max_len:
            new = new[-max_len:]
        return MetricSeries(name=self.name, values=new)


@dataclass(frozen=True)
class Model:
    watch_dir: str
    experiment_type: ExperimentType = ExperimentType.GENERIC
    # Metrics
    metrics: tuple[MetricSeries, ...] = ()
    current_step: int = 0
    total_steps: int = 0
    # Logs
    training_lines: tuple[str, ...] = ()
    sglang_lines: tuple[str, ...] = ()
    event_lines: tuple[str, ...] = ()
    generic_lines: tuple[str, ...] = ()
    # RL rollout stats
    rollout_count: int = 0
    last_mean_reward: float = 0.0
    reward_history: tuple[float, ...] = ()
    # Eval stats
    eval_total: int = 0
    eval_completed: int = 0
    eval_scores: tuple[float, ...] = ()
    # Config
    config: dict = field(default_factory=dict)
    # UI state
    active_panel: int = 0
    scroll: int = 0  # Vertical scroll offset (line number)
    x_offset: int = 0  # Horizontal scroll offset (column number)
    auto_scroll: bool = True

    @property
    def panel_names(self) -> list[str]:
        match self.experiment_type:
            case ExperimentType.RL:
                return ["Metrics", "Training", "SGLang"]
            case ExperimentType.SFT:
                return ["Metrics", "Training"]
            case ExperimentType.EVAL:
                return ["Metrics", "Events"]
            case ExperimentType.GENERIC:
                return ["Metrics", "Logs"]


def _append_log(lines: tuple[str, ...], line: str, max_len: int = 5000) -> tuple[str, ...]:
    new = lines + (line,)
    if len(new) > max_len:
        new = new[-max_len:]
    return new


def _clamp_scroll(scroll: int, total_lines: int, viewport_height: int = 20) -> int:
    """Clamp scroll to valid range [0, max_scroll].

    Args:
        scroll: Current scroll position
        total_lines: Total number of lines in content
        viewport_height: Approximate viewport height (lines visible)

    Returns:
        Clamped scroll value
    """
    max_scroll = max(0, total_lines - viewport_height)
    return max(0, min(scroll, max_scroll))


def _get_active_lines(model: Model) -> tuple[str, ...]:
    """Get the log lines for the currently active panel."""
    match model.experiment_type:
        case ExperimentType.RL:
            if model.active_panel == 1:
                return model.training_lines
            elif model.active_panel == 2:
                return model.sglang_lines
        case ExperimentType.SFT:
            if model.active_panel == 1:
                return model.training_lines
        case ExperimentType.EVAL:
            if model.active_panel == 1:
                return model.event_lines
        case ExperimentType.GENERIC:
            if model.active_panel == 1:
                return model.generic_lines or model.training_lines or model.event_lines
    return ()


def _scroll_down(model: Model, delta: int) -> Model:
    """Scroll down by delta lines, transitioning from auto-scroll if needed."""
    lines = _get_active_lines(model)
    total = len(lines)

    if model.auto_scroll:
        # Transitioning from auto-scroll: start at bottom, then move up by 1
        # (user pressed j to scroll down, but we were following, so go back 1)
        max_scroll = max(0, total - 20)  # Approximate viewport
        new_scroll = max(0, max_scroll - 1)
    else:
        new_scroll = _clamp_scroll(model.scroll + delta, total)

    return replace(model, scroll=new_scroll, auto_scroll=False)


def _scroll_up(model: Model, delta: int) -> Model:
    """Scroll up by delta lines, transitioning from auto-scroll if needed."""
    lines = _get_active_lines(model)
    total = len(lines)

    if model.auto_scroll:
        # Transitioning from auto-scroll: start at bottom, then move up
        max_scroll = max(0, total - 20)  # Approximate viewport
        new_scroll = max(0, max_scroll - delta)
    else:
        new_scroll = _clamp_scroll(model.scroll - delta, total)

    return replace(model, scroll=new_scroll, auto_scroll=False)


def _extract_log_message(raw: str) -> str:
    """Extract human-readable message from a log line.

    Handles:
    - JSON logs: {"level": "INFO", "message": "..."} -> "..."
    - SGLang prefixed: "[2026-01-29 01:02:03] Server ready" -> "Server ready"
    - Script headers: "Script started on..." -> skip
    - Plain text: pass through
    """
    # Skip script command headers
    if raw.startswith("Script ") or raw.startswith("[COMMAND_EXIT_CODE"):
        return ""

    # Try JSON parsing
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            msg = data.get("message", "")
            if msg:
                return msg
            # No message field - skip internal logs
            return ""
        # Not a dict (could be int, list, etc.) - fall through to plain text handling
    except json.JSONDecodeError:
        pass

    # Strip SGLang timestamp prefix: [2026-01-29 01:02:03] message
    if raw.startswith("[") and "] " in raw[:30]:
        idx = raw.index("] ")
        return raw[idx + 2 :]

    # Plain text - pass through (but skip empty/whitespace)
    stripped = raw.strip()
    return stripped if stripped else ""


def _parse_metrics(raw: str, model: Model) -> Model:
    """Parse a metrics.jsonl line and update model."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return model

    step = data.get("step", model.current_step)
    total = data.get("total_steps", model.total_steps)

    metrics_dict = {m.name: m for m in model.metrics}
    for key, value in data.items():
        if key in ("step", "timestamp", "total_steps"):
            continue
        if isinstance(value, (int, float)):
            if key in metrics_dict:
                metrics_dict[key] = metrics_dict[key].append(float(value))
            else:
                metrics_dict[key] = MetricSeries(name=key, values=(float(value),))

    return replace(
        model,
        metrics=tuple(metrics_dict.values()),
        current_step=max(step, model.current_step),
        total_steps=total if total else model.total_steps,
    )


def _parse_rollout(raw: str, model: Model) -> Model:
    """Parse a rollouts.jsonl line and update model."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return model

    reward = data.get("reward")
    if reward is not None:
        rh = model.reward_history + (float(reward),)
        if len(rh) > 200:
            rh = rh[-200:]
        return replace(
            model,
            rollout_count=model.rollout_count + 1,
            last_mean_reward=float(reward),
            reward_history=rh,
        )
    return replace(model, rollout_count=model.rollout_count + 1)


def _parse_event(raw: str, model: Model) -> Model:
    """Parse an events.jsonl line and update model."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return model

    event_type = data.get("type", "")
    summary = f"[{event_type}]"

    match event_type:
        case "eval_start":
            total = data.get("total", 0)
            name = data.get("name", "")
            summary = f"Eval started: {name} ({total} samples)"
            return replace(
                model,
                event_lines=_append_log(model.event_lines, summary),
                eval_total=total,
            )
        case "sample_end":
            score = data.get("score")
            sample_id = data.get("id", "?")
            time_sec = data.get("time_sec", 0)
            summary = (
                f"  {sample_id}: score={score:.3f} ({time_sec:.1f}s)"
                if score is not None
                else f"  {sample_id}: done ({time_sec:.1f}s)"
            )
            scores = model.eval_scores
            if score is not None:
                scores = scores + (float(score),)
                if len(scores) > 500:
                    scores = scores[-500:]
            return replace(
                model,
                event_lines=_append_log(model.event_lines, summary),
                eval_completed=model.eval_completed + 1,
                eval_scores=scores,
            )
        case "sample_start":
            name = data.get("name", data.get("id", "?"))
            summary = f"  {name}: started"
        case "rl_step":
            step = data.get("step", "?")
            reward = data.get("reward", "?")
            summary = f"  step {step}: reward={reward}"
        case "log" | "error":
            summary = data.get("message", raw)
        case _:
            summary = f"[{event_type}] {json.dumps({k: v for k, v in data.items() if k not in ('type', 'timestamp')})}"

    return replace(model, event_lines=_append_log(model.event_lines, summary))


# ─── Update ───────────────────────────────────────────────────────────────


def update(model: Model, msg: object) -> tuple[Model, Cmd]:
    match msg:
        case KeyPress(key="q"):
            return model, Cmd.quit()

        case KeyPress(key="1"):
            return replace(model, active_panel=0), Cmd.none()
        case KeyPress(key="2"):
            return replace(model, active_panel=1, scroll=0, auto_scroll=True), Cmd.none()
        case KeyPress(key="3"):
            if len(model.panel_names) > 2:
                return replace(model, active_panel=2, scroll=0, auto_scroll=True), Cmd.none()

        # Vertical scroll: j/k or arrow keys (single line)
        case KeyPress(key="j" | "\x1b[B"):
            return _scroll_down(model, 1), Cmd.none()
        case KeyPress(key="k" | "\x1b[A"):
            return _scroll_up(model, 1), Cmd.none()

        # Half-page scroll: Ctrl-D/Ctrl-U (vim style)
        case KeyPress(key="\x04"):  # Ctrl-D
            return _scroll_down(model, 10), Cmd.none()
        case KeyPress(key="\x15"):  # Ctrl-U
            return _scroll_up(model, 10), Cmd.none()

        # Full page scroll: Page Down/Page Up, Space/b
        case KeyPress(key="\x1b[6~" | " "):  # Page Down or Space
            return _scroll_down(model, 20), Cmd.none()
        case KeyPress(key="\x1b[5~" | "b"):  # Page Up or 'b'
            return _scroll_up(model, 20), Cmd.none()

        # Horizontal scroll: h/l or left/right arrows
        case KeyPress(key="l" | "\x1b[C"):  # Right
            return replace(model, x_offset=model.x_offset + 10), Cmd.none()
        case KeyPress(key="h" | "\x1b[D"):  # Left
            return replace(model, x_offset=max(0, model.x_offset - 10)), Cmd.none()
        case KeyPress(key="0"):  # Go to start of line
            return replace(model, x_offset=0), Cmd.none()
        case KeyPress(key="$"):  # Go to end of line (will be clamped in view)
            return replace(model, x_offset=9999), Cmd.none()

        # Go to top/bottom: g/G
        case KeyPress(key="G"):
            return replace(model, auto_scroll=True), Cmd.none()
        case KeyPress(key="g"):
            return replace(model, scroll=0, auto_scroll=False), Cmd.none()

        case ExperimentDetected(experiment_type=et):
            return replace(model, experiment_type=et), Cmd.none()

        case MetricsLine(line=raw):
            return _parse_metrics(raw, model), Cmd.none()

        case TrainingLine(line=raw):
            msg_text = _extract_log_message(raw)
            if not msg_text:
                return model, Cmd.none()
            new_lines = _append_log(model.training_lines, msg_text)
            # Don't update scroll here - auto_scroll mode ignores it anyway,
            # and manual mode should preserve user's position
            return replace(model, training_lines=new_lines), Cmd.none()

        case SglangLine(line=raw):
            msg_text = _extract_log_message(raw)
            if not msg_text:
                return model, Cmd.none()
            new_lines = _append_log(model.sglang_lines, msg_text)
            return replace(model, sglang_lines=new_lines), Cmd.none()

        case RolloutLine(line=raw):
            return _parse_rollout(raw, model), Cmd.none()

        case EventLine(line=raw):
            return _parse_event(raw, model), Cmd.none()

        case GenericLogLine(line=raw):
            msg_text = _extract_log_message(raw)
            if not msg_text:
                return model, Cmd.none()
            new_lines = _append_log(model.generic_lines, msg_text)
            return replace(model, generic_lines=new_lines), Cmd.none()

        case ConfigLoaded(config=cfg):
            total = cfg.get("num_steps", model.total_steps)
            return replace(model, config=cfg, total_steps=total), Cmd.none()

    return model, Cmd.none()


# ─── Subscriptions ────────────────────────────────────────────────────────


def subscriptions(model: Model) -> Sub:
    d = model.watch_dir
    subs: list[Sub] = []

    match model.experiment_type:
        case ExperimentType.RL:
            subs.append(Sub.file_tail(f"{d}/metrics.jsonl", lambda line: MetricsLine(line=line)))
            subs.append(Sub.file_tail(f"{d}/rollouts.jsonl", lambda line: RolloutLine(line=line)))
            subs.append(Sub.file_tail(f"{d}/error_log.jsonl", lambda line: TrainingLine(line=line)))
            subs.append(Sub.file_tail(f"{d}/training.log", lambda line: TrainingLine(line=line)))
            subs.append(Sub.file_tail(f"{d}/sglang.log", lambda line: SglangLine(line=line)))

        case ExperimentType.SFT:
            subs.append(Sub.file_tail(f"{d}/metrics.jsonl", lambda line: MetricsLine(line=line)))
            subs.append(Sub.file_tail(f"{d}/error_log.jsonl", lambda line: TrainingLine(line=line)))
            subs.append(Sub.file_tail(f"{d}/training.log", lambda line: TrainingLine(line=line)))
            subs.append(Sub.file_tail(f"{d}/sglang.log", lambda line: SglangLine(line=line)))

        case ExperimentType.EVAL:
            subs.append(Sub.file_tail(f"{d}/metrics.jsonl", lambda line: MetricsLine(line=line)))
            subs.append(Sub.file_tail(f"{d}/events.jsonl", lambda line: EventLine(line=line)))

        case ExperimentType.GENERIC:
            # Tail everything we can find
            subs.append(Sub.file_tail(f"{d}/metrics.jsonl", lambda line: MetricsLine(line=line)))
            subs.append(Sub.file_tail(f"{d}/events.jsonl", lambda line: EventLine(line=line)))
            subs.append(Sub.file_tail(f"{d}/training.log", lambda line: GenericLogLine(line=line)))
            subs.append(
                Sub.file_tail(f"{d}/error_log.jsonl", lambda line: GenericLogLine(line=line))
            )
            subs.append(Sub.file_tail(f"{d}/sglang.log", lambda line: GenericLogLine(line=line)))

    return Sub.batch(*subs)


# ─── View helpers ─────────────────────────────────────────────────────────


def _box(title: str, content: list[str], width: int, active: bool = False) -> list[str]:
    """Draw a btop-style box with rounded corners.

    Invariants:
    - Every returned line has visible_width == width
    - Returns len(content) + 2 lines (top + content + bottom)
    """
    assert width >= 4, f"Box width must be >= 4, got {width}"
    inner_w = width - 2

    border_c = C_BORDER_ACCENT if active else C_BORDER
    title_c = C_TITLE if active else C_DIM

    if title:
        title_text = f" {title} "
        title_vis_len = len(title) + 2
        # TL + H + title_text + H*remaining + TR = width
        # 1  + 1 + title_vis_len + remaining + 1 = width
        remaining = inner_w - title_vis_len - 1
        top = f"{border_c}{TL}{H}{title_c}{title_text}{border_c}{H * max(0, remaining)}{TR}{RESET}"
    else:
        top = f"{border_c}{TL}{H * inner_w}{TR}{RESET}"

    bottom = f"{border_c}{BL}{H * inner_w}{BR}{RESET}"

    lines = [top]
    for row in content:
        vis_len = visible_width(row)
        pad = max(0, inner_w - vis_len)
        line = f"{border_c}{V}{RESET}{row}{' ' * pad}{border_c}{V}{RESET}"
        lines.append(line)
    lines.append(bottom)

    # Assert invariants
    assert len(lines) == len(content) + 2, (
        f"Box line count mismatch: {len(lines)} != {len(content) + 2}"
    )
    for i, line in enumerate(lines):
        line_w = visible_width(line)
        assert line_w == width, (
            f"Box line {i} width {line_w} != expected {width}. Line: {repr(line[:100])}"
        )

    return lines


def _sparkline(values: tuple[float, ...] | list[float], width: int) -> str:
    """Render a sparkline."""
    if not values:
        return C_DIM + "·" * width + RESET

    vals = list(values[-width:])
    mn, mx = min(vals), max(vals)
    rng = mx - mn

    result = []
    for v in vals:
        if rng == 0:
            idx = 4
        else:
            idx = min(8, int(((v - mn) / rng) * 8))
        result.append(SPARK_CHARS[idx])

    spark_str = "".join(result)
    pad = width - len(vals)
    if pad > 0:
        spark_str = C_DIM + "·" * pad + RESET + C_SPARK + spark_str
    else:
        spark_str = C_SPARK + spark_str
    return spark_str + RESET


def _side_by_side(left: list[str], right: list[str], left_w: int, right_w: int) -> list[str]:
    """Place two sets of lines side by side.

    Expects left and right to be pre-rendered boxes where each line has the
    correct visible width. This function pads the left side to left_w if needed.

    Invariants:
        - Left lines are padded to exactly left_w visible chars
        - Returns max(len(left), len(right)) lines
    """
    max_h = max(len(left), len(right))
    result = []
    for i in range(max_h):
        l = left[i] if i < len(left) else ""
        r = right[i] if i < len(right) else ""
        # Pad left line to exact width using visible_width
        l_vis = visible_width(l)
        l_pad = max(0, left_w - l_vis)
        combined = l + " " * l_pad + r
        result.append(combined)

        # Assert left padding is correct
        padded_left_w = l_vis + l_pad
        assert padded_left_w == left_w, (
            f"Side-by-side line {i}: padded left width {padded_left_w} != left_w {left_w}. "
            f"l_vis={l_vis}, l_pad={l_pad}"
        )

    assert len(result) == max_h, f"Side-by-side returned {len(result)} lines, expected {max_h}"
    return result


def _render_log_box(
    title: str,
    lines: tuple[str, ...],
    width: int,
    height: int,
    active: bool,
    scroll: int,
    auto_scroll: bool,
    x_offset: int = 0,
    color: str = C_TEXT,
) -> list[str]:
    """Render a scrollable log box with horizontal and vertical scrolling.

    Args:
        title: Box title
        lines: Log lines to display
        width: Total box width
        height: Total box height
        active: Whether this panel is active (affects border color)
        scroll: Vertical scroll offset (line number)
        auto_scroll: Whether to auto-scroll to bottom
        x_offset: Horizontal scroll offset (column number)
        color: ANSI color for text

    Invariants:
        - Returns exactly `height` lines
        - Each content line fits within inner_w (width - 4)
        - scroll is clamped to valid range [0, max_scroll]
        - x_offset >= 0
    """
    assert width >= 6, f"Log box width must be >= 6, got {width}"
    assert height >= 3, f"Log box height must be >= 3, got {height}"
    assert x_offset >= 0, f"x_offset must be >= 0, got {x_offset}"
    assert scroll >= 0, f"scroll must be >= 0, got {scroll}"

    content_h = height - 2
    total_lines = len(lines)

    # Vertical scrolling
    if active and not auto_scroll:
        # Clamp scroll to valid range
        max_scroll = max(0, total_lines - content_h)
        start = min(scroll, max_scroll)
        assert 0 <= start <= max(0, total_lines - 1), (
            f"start {start} out of range for {total_lines} lines"
        )
        visible = lines[start : start + content_h]
    else:
        visible = lines[-content_h:] if lines else ()

    content = []
    inner_w = width - 4  # 2 for box borders, 2 for padding

    for i, line in enumerate(visible):
        original_line = line
        line_width = visible_width(line)

        # Apply horizontal scrolling if needed
        if x_offset > 0 and line_width > 0:
            # Slice from x_offset to x_offset + inner_w
            line = slice_ansi(line, x_offset, x_offset + inner_w)
        elif line_width > inner_w:
            # No horizontal offset but line is too long - truncate
            line = truncate_to_width(line, inner_w)

        # Assert content fits
        final_width = visible_width(line)
        assert final_width <= inner_w, (
            f"Log line {i} width {final_width} > inner_w {inner_w}. "
            f"x_offset={x_offset}, original_width={line_width}, line={repr(original_line[:80])}"
        )

        content.append(f" {color}{line}{RESET}")
    while len(content) < content_h:
        content.append("")

    # Add scroll position indicator to title if scrolling is active
    if total_lines > content_h:
        if auto_scroll:
            scroll_indicator = " [FOLLOW]"
        else:
            # Calculate scroll percentage
            max_scroll = total_lines - content_h
            if max_scroll > 0:
                pct = min(100, int((scroll / max_scroll) * 100))
                scroll_indicator = f" {pct}%"
            else:
                scroll_indicator = " 100%"
        title = title + scroll_indicator

    return _box(title, content, width, active=active)


# ─── View ─────────────────────────────────────────────────────────────────


def _view_config_box(model: Model, config_w: int) -> list[str]:
    """Render the config sidebar. Adapts to experiment type."""
    config_content: list[str] = []
    cfg = model.config

    if cfg:
        # Show all config keys that fit, prioritizing known ones
        known_keys = [
            ("model", "model_name"),
            ("lr", "lr"),
            ("batch", "batch_size"),
            ("samples", "n_samples_per_prompt"),
            ("max_tok", "max_tokens"),
            ("lora", "use_lora"),
        ]
        for label, key in known_keys:
            val = cfg.get(key)
            if val is not None:
                val_str = str(val)
                if isinstance(val, float) and val < 0.01:
                    val_str = f"{val:.1e}"
                if len(val_str) > config_w - 12:
                    val_str = val_str[: config_w - 15] + "..."
                config_content.append(f" {C_DIM}{label:>8}{RESET} {C_BRIGHT}{val_str}{RESET}")
    else:
        config_content.append(f" {C_DIM}no config.json{RESET}")

    # RL: reward summary
    if model.experiment_type == ExperimentType.RL and model.reward_history:
        config_content.append(f" {C_DIM}{'':>8}{RESET}")
        avg = sum(model.reward_history[-50:]) / len(model.reward_history[-50:])
        config_content.append(f" {C_DIM}{'reward':>8}{RESET} {C_GOOD}{avg:.3f}{RESET} avg")
        config_content.append(
            f" {C_DIM}{'rollouts':>8}{RESET} {C_BRIGHT}{model.rollout_count}{RESET}"
        )

    # Eval: score summary
    if model.experiment_type == ExperimentType.EVAL and model.eval_scores:
        config_content.append(f" {C_DIM}{'':>8}{RESET}")
        scores = model.eval_scores
        avg = sum(scores) / len(scores)
        config_content.append(f" {C_DIM}{'score':>8}{RESET} {C_GOOD}{avg:.3f}{RESET} avg")
        config_content.append(
            f" {C_DIM}{'done':>8}{RESET} {C_BRIGHT}{model.eval_completed}/{model.eval_total}{RESET}"
        )

    return config_content


def view(model: Model, width: int, height: int) -> list[str]:
    lines: list[str] = []

    # ─── Header ───
    step_text = f"step {model.current_step}"
    if model.total_steps:
        pct = model.current_step / model.total_steps * 100
        step_text += f"/{model.total_steps} ({pct:.0f}%)"

    dir_name = Path(model.watch_dir).name
    type_tag = model.experiment_type.name.lower()
    header = f" {C_TITLE}monitor{RESET}  {C_DIM}{dir_name}{RESET}  {C_DIM}[{type_tag}]{RESET}  {C_BRIGHT}{step_text}{RESET}"
    lines.append(header)

    # ─── Top row: Metrics box + Config box ───
    config_w = min(28, width // 3)
    metrics_w = width - config_w

    # Metrics content
    metrics_content: list[str] = []
    spark_w = max(10, metrics_w - 24)
    if model.metrics:
        for m in model.metrics:
            if not m.values:
                continue
            current = m.values[-1]
            label = f"{C_LABEL}{m.name:>12}{RESET}"
            spark = _sparkline(m.values, spark_w)
            val = f"{C_VALUE}{current:>8.4f}{RESET}"
            metrics_content.append(f" {label} {spark} {val}")
    else:
        metrics_content.append(f" {C_DIM}waiting for metrics...{RESET}")

    config_content = _view_config_box(model, config_w)

    # Equalize heights
    top_h = max(len(metrics_content), len(config_content), 3)
    while len(metrics_content) < top_h:
        metrics_content.append("")
    while len(config_content) < top_h:
        config_content.append("")

    metrics_box = _box("Metrics", metrics_content, metrics_w, active=(model.active_panel == 0))
    config_box = _box("Config", config_content, config_w)

    lines.extend(_side_by_side(metrics_box, config_box, metrics_w, config_w))

    # ─── Remaining space: log boxes adapted to experiment type ───
    used = len(lines)
    remaining = height - used - 1  # -1 for footer

    panels = model.panel_names

    match model.experiment_type:
        case ExperimentType.RL:
            training_h = max(3, int(remaining * 0.6))
            sglang_h = max(3, remaining - training_h)

            lines.extend(
                _render_log_box(
                    f"Training ({len(model.training_lines)})",
                    model.training_lines,
                    width,
                    training_h,
                    active=(model.active_panel == 1),
                    scroll=model.scroll,
                    auto_scroll=model.auto_scroll,
                    x_offset=model.x_offset,
                )
            )
            lines.extend(
                _render_log_box(
                    f"SGLang ({len(model.sglang_lines)})",
                    model.sglang_lines,
                    width,
                    sglang_h,
                    active=(model.active_panel == 2),
                    scroll=model.scroll,
                    auto_scroll=model.auto_scroll,
                    x_offset=model.x_offset,
                    color=C_DIM,
                )
            )

        case ExperimentType.SFT:
            lines.extend(
                _render_log_box(
                    f"Training ({len(model.training_lines)})",
                    model.training_lines,
                    width,
                    remaining,
                    active=(model.active_panel == 1),
                    scroll=model.scroll,
                    auto_scroll=model.auto_scroll,
                    x_offset=model.x_offset,
                )
            )

        case ExperimentType.EVAL:
            lines.extend(
                _render_log_box(
                    f"Events ({len(model.event_lines)})",
                    model.event_lines,
                    width,
                    remaining,
                    active=(model.active_panel == 1),
                    scroll=model.scroll,
                    auto_scroll=model.auto_scroll,
                    x_offset=model.x_offset,
                )
            )

        case ExperimentType.GENERIC:
            # Show whatever logs we have
            all_lines = model.generic_lines or model.training_lines or model.event_lines
            lines.extend(
                _render_log_box(
                    f"Logs ({len(all_lines)})",
                    all_lines,
                    width,
                    remaining,
                    active=(model.active_panel == 1),
                    scroll=model.scroll,
                    auto_scroll=model.auto_scroll,
                    x_offset=model.x_offset,
                )
            )

    # ─── Footer ───
    tabs = []
    for i, name in enumerate(panels):
        if i == model.active_panel:
            tabs.append(f"{C_BRIGHT}[{i + 1}]{name}{RESET}")
        else:
            tabs.append(f"{C_DIM}[{i + 1}]{name}{RESET}")
    tab_str = " ".join(tabs)

    scroll_hint = ""
    if model.active_panel > 0:
        if model.auto_scroll:
            scroll_hint = f"  {C_DIM}[FOLLOW]{RESET}"
        else:
            scroll_hint = f"  {C_DIM}j/k:line ^d/^u:page h/l:pan G:follow{RESET}"

    footer = f" {tab_str}{scroll_hint}  {C_DIM}q:quit{RESET}"
    lines.append(footer)

    while len(lines) < height:
        lines.append("")
    lines = lines[:height]

    # Assert view invariants
    assert len(lines) == height, f"View returned {len(lines)} lines, expected {height}"
    # Note: lines wider than screen are truncated by the renderer, no warning needed

    return lines


# ─── Init ─────────────────────────────────────────────────────────────────


def _load_init(watch_dir: str) -> Callable:
    """Return a Cmd.batch that loads config and detects experiment type."""

    def _load() -> ConfigLoaded:
        config_path = Path(watch_dir) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return ConfigLoaded(config=json.load(f))
        return ConfigLoaded(config={})

    return _load


def _detect_type(watch_dir: str) -> Callable:
    def _detect() -> ExperimentDetected:
        return ExperimentDetected(experiment_type=detect_experiment_type(watch_dir))

    return _detect


def frame_debug_snapshot(model: Model, width: int, height: int) -> dict:
    """Pure function: compute a wide event describing the current frame layout.

    Returns a dict suitable for JSON serialization. Contains terminal dims,
    computed panel sizes, model state summary, and truncation info.
    """
    config_w = min(28, width // 3)
    metrics_w = width - config_w

    # Recompute layout heights (mirrors view())
    metrics_content_h = max(len(model.metrics), 1)
    config_content_h = len(_view_config_box(model, config_w))
    top_h = max(metrics_content_h, config_content_h, 3)
    top_box_h = top_h + 2  # +2 for border top/bottom

    header_lines = 1
    footer_lines = 1
    remaining = height - header_lines - top_box_h - footer_lines

    log_boxes: list[dict] = []
    match model.experiment_type:
        case ExperimentType.RL:
            training_h = max(3, int(remaining * 0.6))
            sglang_h = max(3, remaining - training_h)
            log_boxes.append({
                "name": "Training",
                "width": width,
                "height": training_h,
                "lines": len(model.training_lines),
            })
            log_boxes.append({
                "name": "SGLang",
                "width": width,
                "height": sglang_h,
                "lines": len(model.sglang_lines),
            })
        case ExperimentType.SFT:
            log_boxes.append({
                "name": "Training",
                "width": width,
                "height": remaining,
                "lines": len(model.training_lines),
            })
        case ExperimentType.EVAL:
            log_boxes.append({
                "name": "Events",
                "width": width,
                "height": remaining,
                "lines": len(model.event_lines),
            })
        case ExperimentType.GENERIC:
            all_lines = model.generic_lines or model.training_lines or model.event_lines
            log_boxes.append({
                "name": "Logs",
                "width": width,
                "height": remaining,
                "lines": len(all_lines),
            })

    return {
        "terminal": {"width": width, "height": height},
        "layout": {
            "header_lines": header_lines,
            "metrics_box": {"width": metrics_w, "height": top_box_h},
            "config_box": {"width": config_w, "height": top_box_h},
            "log_boxes": log_boxes,
            "footer_lines": footer_lines,
            "remaining_for_logs": remaining,
        },
        "model": {
            "experiment_type": model.experiment_type.name,
            "step": f"{model.current_step}/{model.total_steps}"
            if model.total_steps
            else str(model.current_step),
            "metric_names": [m.name for m in model.metrics],
            "metric_counts": [len(m.values) for m in model.metrics],
            "training_lines": len(model.training_lines),
            "sglang_lines": len(model.sglang_lines),
            "event_lines": len(model.event_lines),
            "generic_lines": len(model.generic_lines),
            "active_panel": model.active_panel,
            "scroll": model.scroll,
            "auto_scroll": model.auto_scroll,
            "config_keys": list(model.config.keys()) if model.config else [],
        },
    }


DEBUG_LOG_PATH = "/tmp/rlmon-debug.jsonl"


def _make_debug_fn(debug_path: str = DEBUG_LOG_PATH) -> Callable:
    """Create a debug callback that appends frame snapshots to a JSONL file."""
    import os

    def _debug(model: Model, width: int, height: int, frame_count: int) -> None:
        snapshot = frame_debug_snapshot(model, width, height)
        snapshot["frame"] = frame_count
        with open(debug_path, "a") as f:
            f.write(json.dumps(snapshot) + "\n")

    # Truncate on startup so we only see this session's frames
    with open(debug_path, "w") as f:
        pass
    os.chmod(debug_path, 0o644)

    return _debug


def make_app(watch_dir: str, debug: bool = False, debug_frame_interval: int = 100) -> App:
    """Create the monitor App for a given output directory.

    Args:
        watch_dir: Path to the experiment output directory to watch.
        debug: If True, dump frame layout snapshots to /tmp/rlmon-debug.jsonl.
        debug_frame_interval: Dump every N rendered frames (default 100 = ~5s at 20fps).

    Debug output is always written to {watch_dir}/monitor.jsonl for observability.
    The `debug` flag controls additional frame snapshots.
    """
    # Detect type eagerly for initial subscriptions (before first Cmd runs)
    experiment_type = detect_experiment_type(watch_dir)
    init_model = Model(watch_dir=watch_dir, experiment_type=experiment_type)
    init_cmd = Cmd.task(_load_init(watch_dir))

    debug_fn = _make_debug_fn() if debug else None

    # Always log to run directory for observability
    debug_log = Path(watch_dir) / "monitor.jsonl"

    return App(
        init=(init_model, init_cmd),
        update=update,
        view=view,
        subscriptions=subscriptions,
        alternate_screen=True,
        fps=20,
        debug_log=debug_log,
        debug_fn=debug_fn,
        debug_frame_interval=debug_frame_interval,
    )
