"""Minimal counter example - verifies the Elm loop works.

Run: python -m pytui.examples.counter
  or: python pytui/examples/counter.py
"""

from dataclasses import dataclass, replace

from pytui import RESET, App, Cmd, KeyPress, hex_to_fg


@dataclass(frozen=True)
class Model:
    count: int = 0
    message: str = ""


def update(model: Model, msg: object) -> tuple[Model, Cmd]:
    match msg:
        case KeyPress(key="q"):
            return model, Cmd.quit()
        case KeyPress(key="j" | "k"):
            delta = 1 if msg.key == "j" else -1
            return replace(model, count=model.count + delta, message=""), Cmd.none()
        case KeyPress(key="r"):
            return replace(model, count=0, message="reset!"), Cmd.none()
        case KeyPress(key=k):
            return replace(model, message=f"unknown key: {k!r}"), Cmd.none()
    return model, Cmd.none()


CYAN = hex_to_fg("#8abeb7")
DIM = hex_to_fg("#666666")
WHITE = hex_to_fg("#cccccc")
YELLOW = hex_to_fg("#f0c674")


def view(model: Model, width: int, height: int) -> list[str]:
    lines = []
    lines.append("")
    lines.append(f"  {CYAN}pytui counter{RESET}")
    lines.append("")
    lines.append(f"  {WHITE}Count: {YELLOW}{model.count}{RESET}")
    lines.append("")
    if model.message:
        lines.append(f"  {DIM}{model.message}{RESET}")
        lines.append("")
    lines.append(f"  {DIM}j/k: increment/decrement  r: reset  q: quit{RESET}")

    # Pad to fill screen
    while len(lines) < height:
        lines.append("")

    return lines


if __name__ == "__main__":
    App(
        init=(Model(), Cmd.none()),
        update=update,
        view=view,
        alternate_screen=True,
    ).run()
