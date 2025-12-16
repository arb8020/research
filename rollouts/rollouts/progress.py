"""Lightweight progress bar implementation.

Adapted from tinygrad/helpers.py - zero dependencies, ~40 lines.
"""

import math
import shutil
import sys
import time
from typing import Generic, TypeVar

T = TypeVar("T")


class tqdm(Generic[T]):
    """Minimal tqdm-compatible progress bar.

    Supports:
    - ETA calculation
    - Adaptive refresh rate
    - SI unit scaling
    - set_postfix() for custom metrics
    - Context manager and manual update

    Does not support:
    - Jupyter widgets
    - Nested bars
    - Threading
    - Custom formatters
    """

    def __init__(
        self,
        total: int | None = None,
        desc: str = "",
        disable: bool = False,
        unit: str = "it",
        unit_scale: bool = False,
        rate: int = 100,
        bar_format: str | None = None,
    ):
        self.disable = disable
        self.unit = unit
        self.unit_scale = unit_scale
        self.rate = rate
        self.t = total
        self.bar_format = bar_format
        # Check if bar_format requests inverse rate (s/it instead of it/s)
        self.use_inverse_rate = (
            bar_format and "{rate_inv_fmt}" in bar_format if bar_format else False
        )

        # Timing and counters
        self.st = time.perf_counter()  # start time
        self.i = -1  # update call count
        self.n = 0  # completed items
        self.skip = 1  # adaptive refresh rate

        # Postfix for custom metrics
        self.postfix_dict = {}

        self.set_description(desc)
        self.update(0)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.update(close=True)

    def set_description(self, desc: str):
        """Set the description prefix."""
        self.desc = f"{desc}: " if desc else ""

    def set_postfix(self, postfix_dict: dict):
        """Set custom metrics to display (e.g., {'loss': '0.123'})."""
        self.postfix_dict = postfix_dict

    def update(self, n: int = 0, close: bool = False):
        """Update progress by n items."""
        self.n += n
        self.i += 1

        # Skip rendering if disabled or not at refresh interval
        if self.disable or (not close and self.i % self.skip != 0):
            return

        # Calculate progress
        prog = self.n / self.t if self.t else 0
        elapsed = time.perf_counter() - self.st
        ncols = shutil.get_terminal_size().columns

        # Adaptive refresh rate: adjust skip to target ~100 updates/sec
        if elapsed and self.i / elapsed > self.rate and self.i:
            self.skip = max(int(self.i / elapsed) // self.rate, 1)

        # Format helpers
        def HMS(t):
            """Format seconds as H:MM:SS."""
            return ":".join(
                f"{x:02d}" if i else str(x)
                for i, x in enumerate([int(t) // 3600, int(t) % 3600 // 60, int(t) % 60])
                if i or x
            )

        def SI(x):
            """Format number with SI prefix (k, M, G, etc)."""
            if not x:
                return "0.00"
            g = round(math.log(x, 1000), 6)
            scaled = x / 1000 ** int(g)
            precision = int(3 - 3 * math.fmod(g, 1))
            return f"{scaled:.{precision}f}"[:4].rstrip(".") + " kMGTPEZY"[int(g)].strip()

        # Build progress text
        if self.unit_scale:
            prog_text = f"{SI(self.n)}{f'/{SI(self.t)}' if self.t else self.unit}"
        else:
            prog_text = f"{self.n}{f'/{self.t}' if self.t else self.unit}"

        # ETA text
        if self.t:
            est_text = f"<{HMS(elapsed / prog - elapsed) if self.n else '?'}"
        else:
            est_text = ""

        # Iteration rate - show as it/s or s/it depending on use_inverse_rate
        if self.n:
            if self.use_inverse_rate:
                # Show s/unit (e.g., "111.3s/sample") for slow operations
                inv_rate = elapsed / self.n
                it_text = f"{inv_rate:5.2f}s/{self.unit}"
            else:
                # Show unit/s (e.g., "0.01sample/s") - standard format
                it_text = SI(self.n / elapsed) if self.unit_scale else f"{self.n / elapsed:5.2f}"
                it_text = f"{it_text}{self.unit}/s"
        else:
            it_text = f"?{self.unit}/s" if not self.use_inverse_rate else f"?s/{self.unit}"

        # Postfix (custom metrics like last_reward)
        postfix_str = ""
        if self.postfix_dict:
            postfix_str = ", " + ", ".join(f"{k}={v}" for k, v in self.postfix_dict.items())

        # Build suffix: [elapsed<eta, rate, postfix]
        suf = f"{prog_text} [{HMS(elapsed)}{est_text}, {it_text}{postfix_str}]"

        # Build progress bar
        sz = max(ncols - len(self.desc) - 3 - 2 - 2 - len(suf), 1)
        if self.t:
            # Calculate bar position (with sub-character precision)
            num = sz * prog
            full_blocks = int(num)
            partial_block_idx = int(8 * num) % 8
            partial_block = " ▏▎▍▌▋▊▉"[partial_block_idx].strip()
            bar_content = ("█" * full_blocks + partial_block).ljust(sz, " ")
            bar = f"\r{self.desc}{100 * prog:3.0f}%|{bar_content}| {suf}"
        else:
            bar = f"\r{self.desc}{suf}"

        # Print (truncate to terminal width)
        print(bar[: ncols + 1], flush=True, end="\n" * close, file=sys.stderr)

    def close(self):
        """Finalize the progress bar."""
        self.update(close=True)
