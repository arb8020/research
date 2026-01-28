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
    scroll: int = 0
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

        case KeyPress(key="j" | "\x1b[B"):
            return replace(model, scroll=model.scroll + 1, auto_scroll=False), Cmd.none()
        case KeyPress(key="k" | "\x1b[A"):
            return replace(model, scroll=max(0, model.scroll - 1), auto_scroll=False), Cmd.none()
        case KeyPress(key="G"):
            return replace(model, auto_scroll=True), Cmd.none()
        case KeyPress(key="g"):
            return replace(model, scroll=0, auto_scroll=False), Cmd.none()

        case ExperimentDetected(experiment_type=et):
            return replace(model, experiment_type=et), Cmd.none()

        case MetricsLine(line=raw):
            return _parse_metrics(raw, model), Cmd.none()

        case TrainingLine(line=raw):
            msg_text = raw
            try:
                data = json.loads(raw)
                msg_text = data.get("message", raw)
            except json.JSONDecodeError:
                pass
            new_lines = _append_log(model.training_lines, msg_text)
            new_scroll = model.scroll
            if model.auto_scroll and model.active_panel == 1:
                new_scroll = max(0, len(new_lines) - 1)
            return replace(model, training_lines=new_lines, scroll=new_scroll), Cmd.none()

        case SglangLine(line=raw):
            new_lines = _append_log(model.sglang_lines, raw)
            new_scroll = model.scroll
            if model.auto_scroll and model.active_panel == 2:
                new_scroll = max(0, len(new_lines) - 1)
            return replace(model, sglang_lines=new_lines, scroll=new_scroll), Cmd.none()

        case RolloutLine(line=raw):
            return _parse_rollout(raw, model), Cmd.none()

        case EventLine(line=raw):
            return _parse_event(raw, model), Cmd.none()

        case GenericLogLine(line=raw):
            new_lines = _append_log(model.generic_lines, raw)
            new_scroll = model.scroll
            if model.auto_scroll and model.active_panel == 1:
                new_scroll = max(0, len(new_lines) - 1)
            return replace(model, generic_lines=new_lines, scroll=new_scroll), Cmd.none()

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
    """Draw a btop-style box with rounded corners."""
    inner_w = width - 2

    border_c = C_BORDER_ACCENT if active else C_BORDER
    title_c = C_TITLE if active else C_DIM

    if title:
        title_text = f" {title} "
        title_vis_len = len(title) + 2
        remaining = inner_w - title_vis_len
        top = f"{border_c}{TL}{H}{title_c}{title_text}{border_c}{H * max(0, remaining)}{TR}{RESET}"
    else:
        top = f"{border_c}{TL}{H * inner_w}{TR}{RESET}"

    bottom = f"{border_c}{BL}{H * inner_w}{BR}{RESET}"

    lines = [top]
    for row in content:
        raw_len = len(row)
        pad = max(0, inner_w - raw_len)
        lines.append(f"{border_c}{V}{RESET}{row}{' ' * pad}{border_c}{V}{RESET}")
    lines.append(bottom)
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
    """Place two sets of lines side by side."""
    max_h = max(len(left), len(right))
    result = []
    for i in range(max_h):
        l = left[i] if i < len(left) else " " * left_w
        r = right[i] if i < len(right) else " " * right_w
        result.append(l + r)
    return result


def _render_log_box(
    title: str,
    lines: tuple[str, ...],
    width: int,
    height: int,
    active: bool,
    scroll: int,
    auto_scroll: bool,
    color: str = C_TEXT,
) -> list[str]:
    """Render a scrollable log box."""
    content_h = height - 2
    if active and not auto_scroll:
        start = min(scroll, max(0, len(lines) - content_h))
        visible = lines[start : start + content_h]
    else:
        visible = lines[-content_h:] if lines else ()

    content = []
    for line in visible:
        if len(line) > width - 4:
            line = line[: width - 7] + "..."
        content.append(f" {color}{line}{RESET}")
    while len(content) < content_h:
        content.append("")

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
            scroll_hint = f"  {C_DIM}j/k:scroll G:follow{RESET}"

    footer = f" {tab_str}{scroll_hint}  {C_DIM}q:quit{RESET}"
    lines.append(footer)

    while len(lines) < height:
        lines.append("")
    lines = lines[:height]

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


def make_app(watch_dir: str) -> App:
    """Create the monitor App for a given output directory."""
    # Detect type eagerly for initial subscriptions (before first Cmd runs)
    experiment_type = detect_experiment_type(watch_dir)
    init_model = Model(watch_dir=watch_dir, experiment_type=experiment_type)
    init_cmd = Cmd.task(_load_init(watch_dir))

    return App(
        init=(init_model, init_cmd),
        update=update,
        view=view,
        subscriptions=subscriptions,
        alternate_screen=True,
        fps=20,
    )
