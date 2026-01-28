"""File tail example - watches a file and displays new lines.

Run: python pytui/examples/tail.py <file>

Demonstrates Sub.file_tail subscription.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

from pytui import RESET, App, Cmd, KeyPress, Sub, hex_to_fg


@dataclass(frozen=True)
class FileLine:
    line: str


@dataclass(frozen=True)
class Model:
    path: str
    lines: tuple[str, ...] = ()
    scroll: int = 0
    auto_scroll: bool = True
    max_lines: int = 10000


def update(model: Model, msg: object) -> tuple[Model, Cmd]:
    match msg:
        case KeyPress(key="q"):
            return model, Cmd.quit()
        case KeyPress(key="j"):
            return replace(model, scroll=model.scroll + 1, auto_scroll=False), Cmd.none()
        case KeyPress(key="k"):
            return replace(model, scroll=max(0, model.scroll - 1), auto_scroll=False), Cmd.none()
        case KeyPress(key="G"):
            return replace(model, auto_scroll=True), Cmd.none()
        case KeyPress(key="g"):
            return replace(model, scroll=0, auto_scroll=False), Cmd.none()
        case FileLine(line=line):
            new_lines = model.lines + (line,)
            # Trim to max
            if len(new_lines) > model.max_lines:
                new_lines = new_lines[-model.max_lines :]
            new_scroll = model.scroll
            if model.auto_scroll:
                new_scroll = max(0, len(new_lines) - 1)
            return replace(model, lines=new_lines, scroll=new_scroll), Cmd.none()
    return model, Cmd.none()


DIM = hex_to_fg("#666666")
WHITE = hex_to_fg("#cccccc")
CYAN = hex_to_fg("#8abeb7")
YELLOW = hex_to_fg("#f0c674")


def view(model: Model, width: int, height: int) -> list[str]:
    output = []

    # Header
    auto = f" {CYAN}[auto-scroll]{RESET}" if model.auto_scroll else ""
    output.append(f" {WHITE}{model.path}{RESET}  {DIM}{len(model.lines)} lines{RESET}{auto}")

    # Content area
    content_height = height - 2  # header + footer
    visible = model.lines[model.scroll : model.scroll + content_height]
    for line in visible:
        output.append(f" {line}")

    # Pad
    while len(output) < height - 1:
        output.append("")

    # Footer
    pos = f"{model.scroll + 1}/{len(model.lines)}" if model.lines else "0/0"
    output.append(f" {DIM}j/k:scroll  g/G:top/bottom  q:quit{RESET}  {YELLOW}{pos}{RESET}")

    return output


def subscriptions(model: Model) -> Sub:
    return Sub.file_tail(model.path, lambda line: FileLine(line=line))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tail.py <file>")
        sys.exit(1)

    App(
        init=(Model(path=sys.argv[1]), Cmd.none()),
        update=update,
        view=view,
        subscriptions=subscriptions,
        alternate_screen=True,
    ).run()
