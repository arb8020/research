"""rlmon - btop-style RL training monitor.

Elm architecture on pytui. Watches an output directory for:
- metrics.jsonl  (training metrics per step)
- rollouts.jsonl (generated rollouts with rewards)
- sglang.log     (inference server logs)
- config.json    (training config, read once)

Layout:
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
from dataclasses import dataclass, field, replace
from pathlib import Path

from pytui import RESET, App, Cmd, KeyPress, Sub, hex_to_fg

# ─── Colors (btop-inspired) ──────────────────────────────────────────────

# Muted, cool palette
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
C_MUTED_SPARK = hex_to_fg("#5f7a76")

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
class ConfigLoaded:
    config: dict


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
    # Metrics
    metrics: tuple[MetricSeries, ...] = ()
    current_step: int = 0
    total_steps: int = 0
    # Logs
    training_lines: tuple[str, ...] = ()
    sglang_lines: tuple[str, ...] = ()
    # Rollout stats
    rollout_count: int = 0
    last_mean_reward: float = 0.0
    reward_history: tuple[float, ...] = ()
    # Config
    config: dict = field(default_factory=dict)
    # UI state
    active_panel: int = 0  # 0=metrics, 1=training, 2=sglang
    scroll: int = 0
    auto_scroll: bool = True

    @property
    def max_log_lines(self) -> int:
        return 5000


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

    # Update each metric series
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
            return replace(model, active_panel=2, scroll=0, auto_scroll=True), Cmd.none()

        case KeyPress(key="j" | "\x1b[B"):
            return replace(model, scroll=model.scroll + 1, auto_scroll=False), Cmd.none()
        case KeyPress(key="k" | "\x1b[A"):
            return replace(model, scroll=max(0, model.scroll - 1), auto_scroll=False), Cmd.none()
        case KeyPress(key="G"):
            return replace(model, auto_scroll=True), Cmd.none()
        case KeyPress(key="g"):
            return replace(model, scroll=0, auto_scroll=False), Cmd.none()

        case MetricsLine(line=raw):
            return _parse_metrics(raw, model), Cmd.none()

        case TrainingLine(line=raw):
            # Parse JSONL to extract message, or use raw
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

        case ConfigLoaded(config=cfg):
            total = cfg.get("num_steps", model.total_steps)
            return replace(model, config=cfg, total_steps=total), Cmd.none()

    return model, Cmd.none()


# ─── Subscriptions ────────────────────────────────────────────────────────


def subscriptions(model: Model) -> Sub:
    d = model.watch_dir
    subs = [
        Sub.file_tail(f"{d}/metrics.jsonl", lambda line: MetricsLine(line=line)),
        Sub.file_tail(f"{d}/rollouts.jsonl", lambda line: RolloutLine(line=line)),
    ]

    # Training logs - try error_log.jsonl first (JSONL format), fall back to looking
    # for any .log files
    subs.append(Sub.file_tail(f"{d}/error_log.jsonl", lambda line: TrainingLine(line=line)))
    subs.append(Sub.file_tail(f"{d}/sglang.log", lambda line: SglangLine(line=line)))

    return Sub.batch(*subs)


# ─── View helpers ─────────────────────────────────────────────────────────


def _box(title: str, content: list[str], width: int, active: bool = False) -> list[str]:
    """Draw a btop-style box with rounded corners.

    Returns lines including border.
    """
    inner_w = width - 2  # for V borders

    border_c = C_BORDER_ACCENT if active else C_BORDER
    title_c = C_TITLE if active else C_DIM

    # Title in top border
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
        # Pad/truncate to inner width
        # We need visible width for proper padding but keep it simple
        raw_len = len(row)  # approximate
        pad = max(0, inner_w - raw_len)
        lines.append(f"{border_c}{V}{RESET}{row}{' ' * pad}{border_c}{V}{RESET}")
    lines.append(bottom)
    return lines


def _sparkline(values: tuple[float, ...] | list[float], width: int) -> str:
    """Render a braille-style sparkline."""
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
    # Pad if not enough values
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


# ─── View ─────────────────────────────────────────────────────────────────


def view(model: Model, width: int, height: int) -> list[str]:
    lines: list[str] = []

    # ─── Header ───
    step_text = f"step {model.current_step}"
    if model.total_steps:
        pct = model.current_step / model.total_steps * 100
        step_text += f"/{model.total_steps} ({pct:.0f}%)"

    dir_name = Path(model.watch_dir).name
    header = f" {C_TITLE}rlmon{RESET}  {C_DIM}{dir_name}{RESET}  {C_BRIGHT}{step_text}{RESET}"
    lines.append(header)

    # ─── Top row: Metrics box + Config box ───
    config_w = min(28, width // 3)
    metrics_w = width - config_w

    # Metrics content
    metrics_content: list[str] = []
    spark_w = max(10, metrics_w - 24)  # room for label + value + spark
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

    # Config content
    config_content: list[str] = []
    cfg = model.config
    if cfg:
        show_keys = [
            ("model", "model_name"),
            ("lr", "lr"),
            ("batch", "batch_size"),
            ("samples", "n_samples_per_prompt"),
            ("max_tok", "max_tokens"),
            ("lora", "use_lora"),
        ]
        for label, key in show_keys:
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

    # Reward summary in config box
    if model.reward_history:
        config_content.append(f" {C_DIM}{'':>8}{RESET}")
        avg = sum(model.reward_history[-50:]) / len(model.reward_history[-50:])
        config_content.append(f" {C_DIM}{'reward':>8}{RESET} {C_GOOD}{avg:.3f}{RESET} avg")
        config_content.append(
            f" {C_DIM}{'rollouts':>8}{RESET} {C_BRIGHT}{model.rollout_count}{RESET}"
        )

    # Equalize heights
    top_h = max(len(metrics_content), len(config_content), 3)
    while len(metrics_content) < top_h:
        metrics_content.append("")
    while len(config_content) < top_h:
        config_content.append("")

    metrics_box = _box("Metrics", metrics_content, metrics_w, active=(model.active_panel == 0))
    config_box = _box("Config", config_content, config_w)

    lines.extend(_side_by_side(metrics_box, config_box, metrics_w, config_w))

    # ─── Remaining space: Training + SGLang log boxes ───
    used = len(lines)
    remaining = height - used - 1  # -1 for footer

    # Split remaining between training and sglang (60/40)
    training_h = max(3, int(remaining * 0.6))
    sglang_h = max(3, remaining - training_h)

    # Training log box
    train_content_h = training_h - 2  # minus borders
    train_lines = model.training_lines
    if model.active_panel == 1 and not model.auto_scroll:
        start = min(model.scroll, max(0, len(train_lines) - train_content_h))
        visible = train_lines[start : start + train_content_h]
    else:
        visible = train_lines[-train_content_h:] if train_lines else ()

    train_content = []
    for tl in visible:
        # Truncate to fit
        if len(tl) > width - 4:
            tl = tl[: width - 7] + "..."
        train_content.append(f" {C_TEXT}{tl}{RESET}")
    while len(train_content) < train_content_h:
        train_content.append("")

    train_label = f"Training ({len(model.training_lines)})"
    lines.extend(_box(train_label, train_content, width, active=(model.active_panel == 1)))

    # SGLang log box
    sg_content_h = sglang_h - 2
    sg_lines = model.sglang_lines
    if model.active_panel == 2 and not model.auto_scroll:
        start = min(model.scroll, max(0, len(sg_lines) - sg_content_h))
        visible_sg = sg_lines[start : start + sg_content_h]
    else:
        visible_sg = sg_lines[-sg_content_h:] if sg_lines else ()

    sg_content = []
    for sl in visible_sg:
        if len(sl) > width - 4:
            sl = sl[: width - 7] + "..."
        sg_content.append(f" {C_DIM}{sl}{RESET}")
    while len(sg_content) < sg_content_h:
        sg_content.append("")

    sg_label = f"SGLang ({len(model.sglang_lines)})"
    lines.extend(_box(sg_label, sg_content, width, active=(model.active_panel == 2)))

    # ─── Footer ───
    panel_names = ["Metrics", "Training", "SGLang"]
    tabs = []
    for i, name in enumerate(panel_names):
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

    # Pad / truncate to exact height
    while len(lines) < height:
        lines.append("")
    lines = lines[:height]

    return lines


# ─── Init ─────────────────────────────────────────────────────────────────


def load_config(watch_dir: str) -> Callable:
    """Return a Cmd.task function that loads config.json."""

    def _load() -> ConfigLoaded:
        config_path = Path(watch_dir) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return ConfigLoaded(config=json.load(f))
        return ConfigLoaded(config={})

    return _load


def make_app(watch_dir: str) -> App:
    """Create the rlmon App for a given output directory."""
    init_model = Model(watch_dir=watch_dir)
    init_cmd = Cmd.task(load_config(watch_dir))

    return App(
        init=(init_model, init_cmd),
        update=update,
        view=view,
        subscriptions=subscriptions,
        alternate_screen=True,
        fps=20,
    )


# Need this import at module level for load_config return type
from collections.abc import Callable
