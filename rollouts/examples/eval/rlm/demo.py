#!/usr/bin/env python3
"""Interactive RLM demo - needle in a haystack.

Generates a haystack with a hidden magic number and launches the RLM
environment interactively so you can watch it search.

Usage:
    python -m examples.eval.rlm.demo           # 1000 lines (~64KB)
    python -m examples.eval.rlm.demo --lines 10000   # 10K lines (~640KB)
    python -m examples.eval.rlm.demo --print   # Non-interactive, print result
"""

import argparse
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def generate_haystack(num_lines: int = 1000, seed: int = 42) -> tuple[str, int]:
    """Generate haystack with hidden needle. Returns (haystack, magic_number)."""
    random.seed(seed)
    words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]

    lines = []
    for i in range(num_lines):
        line = " ".join(random.choice(words) for _ in range(8))
        lines.append(f"[{i:08d}] {line}")

    # Hide the needle in the middle third
    magic_number = random.randint(100000, 999999)
    needle_pos = random.randint(num_lines // 3, 2 * num_lines // 3)
    lines[needle_pos] = f"[{needle_pos:08d}] SECRET: The magic number is {magic_number}. Remember this."

    return "\n".join(lines), magic_number


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive RLM demo")
    parser.add_argument("--lines", type=int, default=1000, help="Lines in haystack (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--print", "-p", dest="print_mode", action="store_true", help="Non-interactive mode")
    parser.add_argument("--preset", type=str, default=None, help="Use a preset (default: none, uses --env repl)")
    args = parser.parse_args()

    # Generate haystack
    haystack, magic_number = generate_haystack(args.lines, args.seed)

    print(f"Generated {len(haystack):,} char haystack with {args.lines:,} lines")
    print(f"Magic number: {magic_number} (hidden somewhere in the middle)")
    print()

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(haystack)
        context_file = f.name

    # Build rollouts command
    cmd = ["python", "-m", "rollouts"]

    if args.preset:
        cmd.extend(["--preset", args.preset])
    else:
        cmd.extend(["--env", "repl"])

    cmd.extend(["--context-file", context_file])

    if args.print_mode:
        cmd.extend(["-p", "Find the magic number hidden in the context. Use the repl tool to search."])

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    print()

    try:
        result = subprocess.run(cmd)
        return result.returncode
    finally:
        Path(context_file).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
