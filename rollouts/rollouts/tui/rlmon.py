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
    # RL rollout data (for rollouts pane)
    rollout_records: tuple[dict, ...] = ()  # raw parsed JSON records from rollouts.jsonl
    # Eval stats
    eval_total: int = 0
    eval_completed: int = 0
    eval_scores: tuple[float, ...] = ()
    # Config
    config: dict = field(default_factory=dict)
    # UI state — fullscreen pane switching
    active_pane: int = 0  # 0=Summary, 1=Charts, 2=Training, 3=SGLang, 4=Rollouts
    pane_scroll: tuple[int, ...] = (0, 0, 0, 0, 0)
    pane_x_offset: tuple[int, ...] = (0, 0, 0, 0, 0)
    # Charts pane
    selected_metric: int = 0
    # Rollouts pane
    rollout_cursor: int = 0
    rollout_view: str = "list"  # "list" or "detail"

    @property
    def scroll(self) -> int:
        """Scroll offset for the active pane."""
        return self.pane_scroll[self.active_pane]

    @property
    def x_offset(self) -> int:
        """Horizontal offset for the active pane."""
        return self.pane_x_offset[self.active_pane]

    @property
    def pane_names(self) -> list[str]:
        match self.experiment_type:
            case ExperimentType.RL:
                return ["Summary", "Charts", "Training", "SGLang", "Rollouts"]
            case ExperimentType.SFT:
                return ["Summary", "Charts", "Training"]
            case ExperimentType.EVAL:
                return ["Summary", "Charts", "Events"]
            case ExperimentType.GENERIC:
                return ["Summary", "Charts", "Logs"]


LOG_MAX_LINES = 5000


def _append_log(
    lines: tuple[str, ...], line: str, max_len: int = LOG_MAX_LINES
) -> tuple[tuple[str, ...], int]:
    """Append a line, trim to max_len. Returns (new_lines, trimmed_count).

    trimmed_count is how many lines were dropped from the front.
    Callers should subtract this from scroll to keep the viewport stable.
    """
    new = lines + (line,)
    overflow = len(new) - max_len
    if overflow > 0:
        new = new[-max_len:]
        return new, overflow
    return new, 0


def _line_count_label(lines: tuple[str, ...]) -> str:
    """Format line count for box titles. Shows '5000+' when at the cap."""
    n = len(lines)
    if n >= LOG_MAX_LINES:
        return f"{n:,}+"
    return f"{n:,}"


# Approximate viewport height for scroll calculations in update().
# Used until we wire Resize messages into the Model (open thread #3).
_VIEWPORT_HEIGHT = 20


def _max_scroll(total_lines: int, viewport_height: int = _VIEWPORT_HEIGHT) -> int:
    """Maximum scroll offset for given content length."""
    return max(0, total_lines - viewport_height)


def _set_pane_scroll(model: Model, scroll: int) -> Model:
    """Set scroll offset for the active pane."""
    lst = list(model.pane_scroll)
    lst[model.active_pane] = scroll
    return replace(model, pane_scroll=tuple(lst))


def _set_pane_x_offset(model: Model, x_offset: int) -> Model:
    """Set horizontal offset for the active pane."""
    lst = list(model.pane_x_offset)
    lst[model.active_pane] = x_offset
    return replace(model, pane_x_offset=tuple(lst))


def _get_active_lines(model: Model) -> tuple[str, ...]:
    """Get the log lines for the currently active pane."""
    match model.active_pane:
        case 2:  # Training
            return model.training_lines
        case 3:  # SGLang
            return model.sglang_lines
        case _:
            # For summary pane, use the experiment-type default
            match model.experiment_type:
                case ExperimentType.RL:
                    return model.training_lines
                case ExperimentType.SFT:
                    return model.training_lines
                case ExperimentType.EVAL:
                    return model.event_lines
                case ExperimentType.GENERIC:
                    return model.generic_lines or model.training_lines or model.event_lines
    return ()


def _at_bottom(model: Model) -> bool:
    """Whether the active panel is scrolled to the bottom (like bubbles AtBottom).

    When at bottom, new content should auto-follow (caller calls _goto_bottom).
    """
    lines = _get_active_lines(model)
    return model.scroll >= _max_scroll(len(lines))


def _goto_bottom(model: Model) -> Model:
    """Scroll to the bottom of the active pane (like bubbles GotoBottom)."""
    lines = _get_active_lines(model)
    return _set_pane_scroll(model, _max_scroll(len(lines)))


def _scroll_down(model: Model, delta: int) -> Model:
    """Scroll down by delta lines. Clamps to [0, max_scroll]."""
    assert delta > 0, f"delta must be positive, got {delta}"
    lines = _get_active_lines(model)
    ms = _max_scroll(len(lines))
    clamped = min(model.scroll, ms)
    new_scroll = min(clamped + delta, ms)
    return _set_pane_scroll(model, new_scroll)


def _scroll_up(model: Model, delta: int) -> Model:
    """Scroll up by delta lines. Clamps to [0, max_scroll]."""
    assert delta > 0, f"delta must be positive, got {delta}"
    lines = _get_active_lines(model)
    ms = _max_scroll(len(lines))
    clamped = min(model.scroll, ms)
    new_scroll = max(0, clamped - delta)
    return _set_pane_scroll(model, new_scroll)


def _adjust_pane_scroll_after_append(model: Model, pane: int, trimmed: int) -> Model:
    """After appending to a log pane, adjust scroll and auto-follow if at bottom.

    If the pane is currently active and at bottom, follow. Otherwise just
    compensate for trimmed lines so the viewport stays stable.
    """
    lst = list(model.pane_scroll)
    lst[pane] = max(0, lst[pane] - trimmed)
    new_model = replace(model, pane_scroll=tuple(lst))
    # Auto-follow if this is the active pane and we were at bottom
    if model.active_pane == pane and _at_bottom(model):
        new_model = _goto_bottom(new_model)
    return new_model


def _pane_j(model: Model) -> Model:
    """Handle j/down in current pane. Charts: next metric. Rollouts: next item. Others: scroll."""
    if model.active_pane == 1:  # Charts
        n = len(model.metrics)
        if n > 0:
            return replace(model, selected_metric=min(model.selected_metric + 1, n - 1))
        return model
    if model.active_pane == 4:  # Rollouts
        if model.rollout_view == "list":
            n = len(model.rollout_records)
            if n > 0:
                return replace(model, rollout_cursor=min(model.rollout_cursor + 1, n - 1))
            return model
        else:
            return _scroll_down(model, 1)
    return _scroll_down(model, 1)


def _pane_k(model: Model) -> Model:
    """Handle k/up in current pane. Charts: prev metric. Rollouts: prev item. Others: scroll."""
    if model.active_pane == 1:  # Charts
        return replace(model, selected_metric=max(model.selected_metric - 1, 0))
    if model.active_pane == 4:  # Rollouts
        if model.rollout_view == "list":
            return replace(model, rollout_cursor=max(model.rollout_cursor - 1, 0))
        else:
            return _scroll_up(model, 1)
    return _scroll_up(model, 1)


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
    """Parse an events.jsonl line and update model.

    Expects logging format: {"message": "sample_end", "sample_id": "001", ...}
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return model

    event_type = data.get("message", "")
    summary = f"[{event_type}]"

    match event_type:
        case "eval_start":
            total = data.get("total", 0)
            name = data.get("eval_name", "")
            summary = f"Eval started: {name} ({total} samples)"
            new_lines, trimmed = _append_log(model.event_lines, summary)
            return replace(
                model,
                event_lines=new_lines,
                scroll=max(0, model.scroll - trimmed),
                eval_total=total,
            )
        case "sample_end":
            score = data.get("score")
            sample_id = data.get("sample_id", "?")
            summary = (
                f"  {sample_id}: score={score:.3f}"
                if score is not None
                else f"  {sample_id}: done"
            )
            scores = model.eval_scores
            if score is not None:
                scores = scores + (float(score),)
                if len(scores) > 500:
                    scores = scores[-500:]
            new_lines, trimmed = _append_log(model.event_lines, summary)
            return replace(
                model,
                event_lines=new_lines,
                scroll=max(0, model.scroll - trimmed),
                eval_completed=model.eval_completed + 1,
                eval_scores=scores,
            )
        case "sample_start":
            name = data.get("sample_name") or data.get("sample_id", "?")
            summary = f"  {name}: started"
        case "rl_step":
            step = data.get("step", "?")
            reward = data.get("reward", "?")
            summary = f"  step {step}: reward={reward}"
        case _:
            # Filter out common logging fields for cleaner display
            skip_keys = {"message", "timestamp", "logger", "level"}
            extras = {k: v for k, v in data.items() if k not in skip_keys}
            summary = f"[{event_type}] {json.dumps(extras)}" if extras else f"[{event_type}]"

    new_lines, trimmed = _append_log(model.event_lines, summary)
    return replace(model, event_lines=new_lines, scroll=max(0, model.scroll - trimmed))


# ─── Update ───────────────────────────────────────────────────────────────


def update(model: Model, msg: object) -> tuple[Model, Cmd]:
    match msg:
        case KeyPress(key="q"):
            # In rollout detail view, go back to list
            if model.active_pane == 4 and model.rollout_view == "detail":
                return replace(model, rollout_view="list"), Cmd.none()
            return model, Cmd.quit()

        # Pane switching: 1-5
        case KeyPress(key="1"):
            return replace(model, active_pane=0), Cmd.none()
        case KeyPress(key="2"):
            if len(model.pane_names) > 1:
                return replace(model, active_pane=1), Cmd.none()
        case KeyPress(key="3"):
            if len(model.pane_names) > 2:
                return replace(model, active_pane=2), Cmd.none()
        case KeyPress(key="4"):
            if len(model.pane_names) > 3:
                return replace(model, active_pane=3), Cmd.none()
        case KeyPress(key="5"):
            if len(model.pane_names) > 4:
                return replace(model, active_pane=4), Cmd.none()

        # Tab cycles panes
        case KeyPress(key="\t"):
            n = len(model.pane_names)
            return replace(model, active_pane=(model.active_pane + 1) % n), Cmd.none()

        # j/k — pane-specific behavior
        case KeyPress(key="j" | "\x1b[B"):
            return _pane_j(model), Cmd.none()
        case KeyPress(key="k" | "\x1b[A"):
            return _pane_k(model), Cmd.none()

        # Enter — pane-specific
        case KeyPress(key="\r" | "\n"):
            if model.active_pane == 4 and model.rollout_view == "list" and model.rollout_records:
                return replace(model, rollout_view="detail"), Cmd.none()

        # Half-page scroll: Ctrl-D/Ctrl-U
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
            return _set_pane_x_offset(model, model.x_offset + 10), Cmd.none()
        case KeyPress(key="h" | "\x1b[D"):  # Left
            return _set_pane_x_offset(model, max(0, model.x_offset - 10)), Cmd.none()
        case KeyPress(key="0"):  # Go to start of line
            return _set_pane_x_offset(model, 0), Cmd.none()
        case KeyPress(key="$"):  # Go to end of line
            return _set_pane_x_offset(model, 9999), Cmd.none()

        # Go to top/bottom: g/G
        case KeyPress(key="G"):
            return _goto_bottom(model), Cmd.none()
        case KeyPress(key="g"):
            return _set_pane_scroll(model, 0), Cmd.none()

        case ExperimentDetected(experiment_type=et):
            return replace(model, experiment_type=et), Cmd.none()

        case MetricsLine(line=raw):
            return _parse_metrics(raw, model), Cmd.none()

        case TrainingLine(line=raw):
            msg_text = _extract_log_message(raw)
            if not msg_text:
                return model, Cmd.none()
            new_lines, trimmed = _append_log(model.training_lines, msg_text)
            new_model = replace(model, training_lines=new_lines)
            new_model = _adjust_pane_scroll_after_append(new_model, 2, trimmed)
            return new_model, Cmd.none()

        case SglangLine(line=raw):
            msg_text = _extract_log_message(raw)
            if not msg_text:
                return model, Cmd.none()
            new_lines, trimmed = _append_log(model.sglang_lines, msg_text)
            new_model = replace(model, sglang_lines=new_lines)
            new_model = _adjust_pane_scroll_after_append(new_model, 3, trimmed)
            return new_model, Cmd.none()

        case RolloutLine(line=raw):
            new_model = _parse_rollout(raw, model)
            # Also store the raw record for the rollouts pane
            try:
                record = json.loads(raw)
                records = new_model.rollout_records + (record,)
                if len(records) > 5000:
                    records = records[-5000:]
                new_model = replace(new_model, rollout_records=records)
            except json.JSONDecodeError:
                pass
            return new_model, Cmd.none()

        case EventLine(line=raw):
            new_model = _parse_event(raw, model)
            # Events go to event_lines, shown in pane 2 for EVAL type
            return new_model, Cmd.none()

        case GenericLogLine(line=raw):
            msg_text = _extract_log_message(raw)
            if not msg_text:
                return model, Cmd.none()
            new_lines, trimmed = _append_log(model.generic_lines, msg_text)
            new_model = replace(model, generic_lines=new_lines)
            new_model = _adjust_pane_scroll_after_append(new_model, 2, trimmed)
            return new_model, Cmd.none()

        case ConfigLoaded(config=cfg):
            total = cfg.get("num_steps", model.total_steps)
            return replace(model, config=cfg, total_steps=total), Cmd.none()

    return model, Cmd.none()


# ─── Subscriptions ────────────────────────────────────────────────────────


EXPERIMENT_REDETECT_INTERVAL_SEC = 2.0


def subscriptions(model: Model) -> Sub:
    d = model.watch_dir
    subs: list[Sub] = []

    # Re-detect experiment type until we upgrade from GENERIC.
    # In --attach mode, files arrive via remote sync after the app starts,
    # so initial detection sees an empty directory. Once the type changes,
    # this sub disappears and the correct file tails start below.
    if model.experiment_type == ExperimentType.GENERIC:
        subs.append(
            Sub.every(
                EXPERIMENT_REDETECT_INTERVAL_SEC,
                lambda: ExperimentDetected(experiment_type=detect_experiment_type(d)),
            )
        )

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

    # Vertical scrolling — always use scroll offset (like bubbles YOffset)
    max_scroll_val = max(0, total_lines - content_h)
    start = min(scroll, max_scroll_val) if active else max_scroll_val
    start = max(0, start)
    visible = lines[start : start + content_h] if lines else ()
    is_at_bottom = start >= max_scroll_val

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
        if is_at_bottom:
            scroll_indicator = " [FOLLOW]"
        else:
            pct = min(100, int((start / max(1, max_scroll_val)) * 100))
            scroll_indicator = f" {pct}%"
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


def _view_header(model: Model) -> str:
    """Shared header line across all panes."""
    step_text = f"step {model.current_step}"
    if model.total_steps:
        pct = model.current_step / model.total_steps * 100
        step_text += f"/{model.total_steps} ({pct:.0f}%)"
    dir_name = Path(model.watch_dir).name
    type_tag = model.experiment_type.name.lower()
    return f" {C_TITLE}monitor{RESET}  {C_DIM}{dir_name}{RESET}  {C_DIM}[{type_tag}]{RESET}  {C_BRIGHT}{step_text}{RESET}"


def _view_footer(model: Model) -> str:
    """Shared footer with pane tabs and context hints."""
    panes = model.pane_names
    tabs = []
    for i, name in enumerate(panes):
        if i == model.active_pane:
            tabs.append(f"{C_BRIGHT}[{i + 1}]{name}{RESET}")
        else:
            tabs.append(f"{C_DIM}[{i + 1}]{name}{RESET}")
    tab_str = " ".join(tabs)

    # Context-specific hints
    hint = ""
    match model.active_pane:
        case 0:  # Summary
            hint = ""
        case 1:  # Charts
            if model.metrics:
                n = len(model.metrics)
                hint = f"  {C_DIM}j/k:metric ({model.selected_metric + 1}/{n}){RESET}"
            else:
                hint = f"  {C_DIM}waiting for metrics...{RESET}"
        case 2 | 3:  # Log panes
            if _at_bottom(model):
                hint = f"  {C_DIM}[FOLLOW]{RESET}"
            else:
                hint = f"  {C_DIM}j/k:line ^d/^u:page h/l:pan G:follow{RESET}"
        case 4:  # Rollouts
            if model.rollout_view == "list":
                n = len(model.rollout_records)
                hint = (
                    f"  {C_DIM}j/k:select Enter:detail ({model.rollout_cursor + 1}/{n}){RESET}"
                    if n
                    else ""
                )
            else:
                hint = f"  {C_DIM}j/k:scroll q:back{RESET}"

    return f" {tab_str}{hint}  {C_DIM}Tab:next q:quit{RESET}"


def _pad_to_height(lines: list[str], height: int) -> list[str]:
    """Pad or trim lines to exactly height."""
    while len(lines) < height:
        lines.append("")
    return lines[:height]


# ─── Pane: Summary ────────────────────────────────────────────────────────


def _view_summary(model: Model, width: int, height: int) -> list[str]:
    """Summary pane — metrics sparklines + config + condensed log tails."""
    lines: list[str] = [_view_header(model)]

    # Top row: Metrics box + Config box
    config_w = min(28, width // 3)
    metrics_w = width - config_w

    metrics_content: list[str] = []
    spark_w = max(10, metrics_w - 25)
    if model.metrics:
        for m in model.metrics:
            if not m.values:
                continue
            current = m.values[-1]
            label = f"{C_LABEL}{m.name[:12]:>12}{RESET}"
            spark = _sparkline(m.values, spark_w)
            val = f"{C_VALUE}{current:>8.4f}{RESET}"
            metrics_content.append(f" {label} {spark} {val}")
    else:
        metrics_content.append(f" {C_DIM}waiting for metrics...{RESET}")

    config_content = _view_config_box(model, config_w)

    top_h = max(len(metrics_content), len(config_content), 3)
    while len(metrics_content) < top_h:
        metrics_content.append("")
    while len(config_content) < top_h:
        config_content.append("")

    metrics_box = _box("Metrics", metrics_content, metrics_w)
    config_box = _box("Config", config_content, config_w)
    lines.extend(_side_by_side(metrics_box, config_box, metrics_w, config_w))

    # Remaining space: condensed log tails (not scrollable — just last N lines)
    used = len(lines)
    remaining = height - used - 1  # -1 for footer

    match model.experiment_type:
        case ExperimentType.RL:
            training_h = max(3, int(remaining * 0.6))
            sglang_h = max(3, remaining - training_h)
            lines.extend(
                _render_log_box(
                    f"Training ({_line_count_label(model.training_lines)})",
                    model.training_lines,
                    width,
                    training_h,
                    active=False,
                    scroll=_max_scroll(len(model.training_lines)),
                )
            )
            lines.extend(
                _render_log_box(
                    f"SGLang ({_line_count_label(model.sglang_lines)})",
                    model.sglang_lines,
                    width,
                    sglang_h,
                    active=False,
                    scroll=_max_scroll(len(model.sglang_lines)),
                    color=C_DIM,
                )
            )
        case ExperimentType.SFT:
            lines.extend(
                _render_log_box(
                    f"Training ({_line_count_label(model.training_lines)})",
                    model.training_lines,
                    width,
                    remaining,
                    active=False,
                    scroll=_max_scroll(len(model.training_lines)),
                )
            )
        case ExperimentType.EVAL:
            lines.extend(
                _render_log_box(
                    f"Events ({_line_count_label(model.event_lines)})",
                    model.event_lines,
                    width,
                    remaining,
                    active=False,
                    scroll=_max_scroll(len(model.event_lines)),
                )
            )
        case ExperimentType.GENERIC:
            all_lines = model.generic_lines or model.training_lines or model.event_lines
            lines.extend(
                _render_log_box(
                    f"Logs ({_line_count_label(all_lines)})",
                    all_lines,
                    width,
                    remaining,
                    active=False,
                    scroll=_max_scroll(len(all_lines)),
                )
            )

    lines.append(_view_footer(model))
    return _pad_to_height(lines, height)


# ─── Pane: Charts ─────────────────────────────────────────────────────────


def _view_charts(model: Model, width: int, height: int) -> list[str]:
    """Charts pane — one plotext braille chart at a time, j/k to cycle."""
    lines: list[str] = [_view_header(model)]

    if not model.metrics:
        lines.append(f" {C_DIM}waiting for metrics...{RESET}")
        lines.append(_view_footer(model))
        return _pad_to_height(lines, height)

    metric_idx = min(model.selected_metric, len(model.metrics) - 1)
    metric = model.metrics[metric_idx]

    # Chart header
    current_val = metric.values[-1] if metric.values else 0
    chart_title = f" {C_LABEL}{metric.name}{RESET}  {C_VALUE}{current_val:.4f}{RESET}  {C_DIM}({len(metric.values)} pts){RESET}"
    lines.append(chart_title)
    lines.append("")

    chart_h = height - 5  # header + chart_title + blank + footer + padding

    try:
        import plotext as plt

        plt.clf()
        plt.plot(list(metric.values), marker="braille")
        plt.title(f"{metric.name}: {current_val:.4f}")
        plt.xlabel(f"Step (latest: {model.current_step})")
        plt.plotsize(width - 4, max(5, chart_h))
        plt.theme("dark")

        chart_str = plt.build()
        for chart_line in chart_str.split("\n"):
            lines.append(f"  {chart_line}")
    except ImportError:
        lines.append(f"  {C_DIM}plotext not installed — run: pip install plotext{RESET}")
    except Exception as e:
        lines.append(f"  {C_BAD}chart error: {e}{RESET}")

    lines.append(_view_footer(model))
    return _pad_to_height(lines, height)


# ─── Pane: Fullscreen Logs ────────────────────────────────────────────────


def _view_fullscreen_log(
    model: Model,
    title: str,
    log_lines: tuple[str, ...],
    width: int,
    height: int,
) -> list[str]:
    """Fullscreen scrollable log pane."""
    lines: list[str] = [_view_header(model)]
    log_h = height - 2  # header + footer
    lines.extend(
        _render_log_box(
            f"{title} ({_line_count_label(log_lines)})",
            log_lines,
            width,
            log_h,
            active=True,
            scroll=model.scroll,
            x_offset=model.x_offset,
        )
    )
    lines.append(_view_footer(model))
    return _pad_to_height(lines, height)


def _view_training_logs(model: Model, width: int, height: int) -> list[str]:
    return _view_fullscreen_log(model, "Training", model.training_lines, width, height)


def _view_sglang_logs(model: Model, width: int, height: int) -> list[str]:
    return _view_fullscreen_log(model, "SGLang", model.sglang_lines, width, height)


# ─── Pane: Rollouts ──────────────────────────────────────────────────────


def _view_rollouts(model: Model, width: int, height: int) -> list[str]:
    """Rollouts pane — browsable list of rollout records with detail view."""
    lines: list[str] = [_view_header(model)]

    if not model.rollout_records:
        lines.append(f" {C_DIM}waiting for rollouts...{RESET}")
        lines.append(_view_footer(model))
        return _pad_to_height(lines, height)

    if model.rollout_view == "detail":
        return _view_rollout_detail(model, width, height, lines)

    # ─── List view ───
    content_h = height - 3  # header + column header + footer
    records = model.rollout_records
    cursor = min(model.rollout_cursor, len(records) - 1)

    # Column header
    lines.append(f" {C_DIM}{'step':>5}  {'group':>5}  {'reward':>8}  {'status':>10}  prompt{RESET}")

    # Scroll so cursor is visible
    if cursor >= content_h:
        start = cursor - content_h + 1
    else:
        start = 0
    visible = records[start : start + content_h]

    for i, rec in enumerate(visible):
        idx = start + i
        step = rec.get("step", "?")
        group = rec.get("group_index", "?")
        reward = rec.get("reward")
        status = rec.get("status", "?")
        prompt = rec.get("prompt", "")[:50].replace("\n", " ")

        reward_str = f"{reward:>8.3f}" if reward is not None else "     n/a"
        if reward is not None:
            if reward > 0.5:
                reward_str = f"{C_GOOD}{reward_str}{RESET}"
            elif reward > 0:
                reward_str = f"{C_WARN}{reward_str}{RESET}"
            else:
                reward_str = f"{C_BAD}{reward_str}{RESET}"

        is_selected = idx == cursor
        prefix = f"{C_BRIGHT}>{RESET}" if is_selected else " "
        dim = "" if is_selected else C_DIM
        dim_r = "" if is_selected else RESET

        line = f"{prefix}{dim}{step:>5}  {group:>5}{dim_r}  {reward_str}  {dim}{status:>10}  {prompt}{dim_r}"
        lines.append(line)

    lines.append(_view_footer(model))
    return _pad_to_height(lines, height)


def _view_rollout_detail(model: Model, width: int, height: int, lines: list[str]) -> list[str]:
    """Detail view for a single rollout record."""
    records = model.rollout_records
    cursor = min(model.rollout_cursor, len(records) - 1)
    rec = records[cursor]

    step = rec.get("step", "?")
    group = rec.get("group_index", "?")
    reward = rec.get("reward", 0)
    status = rec.get("status", "?")

    reward_color = C_GOOD if reward and reward > 0.5 else C_WARN if reward and reward > 0 else C_BAD
    lines.append(
        f" {C_LABEL}step{RESET} {step}  {C_LABEL}group{RESET} {group}  {C_LABEL}reward{RESET} {reward_color}{reward:.3f}{RESET}  {C_LABEL}status{RESET} {status}"
    )
    lines.append("")

    # Prompt
    prompt = rec.get("prompt", "")
    lines.append(f" {C_LABEL}prompt:{RESET}")
    for pl in prompt.split("\n"):
        lines.append(f"   {C_TEXT}{pl}{RESET}")
    lines.append("")

    # Response
    response = rec.get("response", "")
    lines.append(f" {C_LABEL}response:{RESET}")
    for rl in response.split("\n"):
        lines.append(f"   {C_TEXT}{rl}{RESET}")
    lines.append("")

    # Messages (if present)
    messages = rec.get("messages") or rec.get("metadata", {}).get("messages", [])
    if messages:
        lines.append(f" {C_LABEL}messages ({len(messages)}):{RESET}")
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            role_color = C_LABEL if role == "user" else C_GOOD if role == "assistant" else C_DIM
            lines.append(f"   {role_color}[{role}]{RESET}")
            for ml in content.split("\n")[:20]:
                lines.append(f"     {C_TEXT}{ml}{RESET}")
            if len(content.split("\n")) > 20:
                lines.append(f"     {C_DIM}... ({len(content.split(chr(10)))} lines total){RESET}")

    # Metadata
    meta = rec.get("metadata", {})
    if meta:
        lines.append("")
        lines.append(f" {C_LABEL}metadata:{RESET}")
        for k, v in list(meta.items())[:10]:
            lines.append(f"   {C_DIM}{k}:{RESET} {C_TEXT}{str(v)[:60]}{RESET}")

    lines.append(_view_footer(model))
    return _pad_to_height(lines, height)


# ─── View dispatch ────────────────────────────────────────────────────────


def view(model: Model, width: int, height: int) -> list[str]:
    match model.active_pane:
        case 0:
            result = _view_summary(model, width, height)
        case 1:
            result = _view_charts(model, width, height)
        case 2:
            # Training for RL/SFT, Events for EVAL, Logs for GENERIC
            match model.experiment_type:
                case ExperimentType.EVAL:
                    result = _view_fullscreen_log(model, "Events", model.event_lines, width, height)
                case ExperimentType.GENERIC:
                    all_lines = model.generic_lines or model.training_lines or model.event_lines
                    result = _view_fullscreen_log(model, "Logs", all_lines, width, height)
                case _:
                    result = _view_training_logs(model, width, height)
        case 3:
            result = _view_sglang_logs(model, width, height)
        case 4:
            result = _view_rollouts(model, width, height)
        case _:
            result = _view_summary(model, width, height)

    assert len(result) == height, f"View returned {len(result)} lines, expected {height}"
    return result


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
            "active_pane": model.active_pane,
            "pane_scroll": list(model.pane_scroll),
            "at_bottom": _at_bottom(model),
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
    init_model = Model(watch_dir=watch_dir)
    init_cmd = Cmd.batch(
        Cmd.task(_load_init(watch_dir)),
        Cmd.task(_detect_type(watch_dir)),
    )

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
