#!/usr/bin/env python3
"""
Demo script to test TUI rendering.

Run with: python -m frontends.tui.demo
"""

import time

from .components import Spacer, Text
from .terminal import ProcessTerminal
from .tui import TUI, Container


# ANSI color helpers
def bold(text: str) -> str:
    return f"\x1b[1m{text}\x1b[0m"


def cyan(text: str) -> str:
    return f"\x1b[36m{text}\x1b[0m"


def dim(text: str) -> str:
    return f"\x1b[2m{text}\x1b[0m"


def green(text: str) -> str:
    return f"\x1b[32m{text}\x1b[0m"


def yellow(text: str) -> str:
    return f"\x1b[33m{text}\x1b[0m"


def main():
    """Run a simple TUI demo."""
    terminal = ProcessTerminal()
    ui = TUI(terminal)

    # Build UI
    ui.add_child(Spacer(1))
    ui.add_child(Text(bold(cyan("TUI Demo")) + dim(" v0.1"), padding_x=1, padding_y=0))
    ui.add_child(Spacer(1))

    # Add some content
    content = Container()
    content.add_child(
        Text(
            "This is a " + bold("simple demo") + " of the TUI system.\n"
            "It supports "
            + cyan("ANSI colors")
            + ", "
            + bold("bold text")
            + ", and word wrapping.\n\n"
            "The differential rendering engine only redraws lines that have changed, "
            "making updates efficient and flicker-free.",
            padding_x=1,
            padding_y=0,
        )
    )
    ui.add_child(content)

    ui.add_child(Spacer(1))
    ui.add_child(Text(dim("Press Ctrl+C to exit"), padding_x=1, padding_y=0))
    ui.add_child(Spacer(1))

    # Counter component for demonstrating updates
    counter_text = Text("", padding_x=1, padding_y=0)
    ui.add_child(counter_text)

    try:
        ui.start()

        # Animate counter to show differential rendering
        for i in range(10):
            counter_text.set_text(f"Counter: {green(str(i))} " + yellow("(updating every 0.5s)"))
            ui.request_render()
            time.sleep(0.5)

        counter_text.set_text(green("Done!") + " " + dim("Exiting in 2 seconds..."))
        ui.request_render()
        time.sleep(2)

    except KeyboardInterrupt:
        pass
    finally:
        ui.stop()
        print("\nDemo complete!")


if __name__ == "__main__":
    main()
