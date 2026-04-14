"""Argus control-plane CLI.

This is the public control-plane entrypoint. For now it reuses existing
Rollouts launch/monitor implementation details. Over time, detached run truth
and transport-specific monitoring should move fully behind Argus.
"""

from __future__ import annotations

import argparse
import sys

from .monitor import monitor_main
from .run import run_main
from .tail import tail_main


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "run":
        return run_main(argv[1:])

    if argv and argv[0] == "monitor":
        return monitor_main(argv[1:])

    if argv and argv[0] == "tail":
        return tail_main(argv[1:])

    parser = argparse.ArgumentParser(prog="argus", description="Argus control plane")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Launch a run from a workload config",
        description="Launch a training/eval/benchmark run via the Argus control plane.",
    )
    run_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments for the Argus run entrypoint.",
    )

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Observe, attach to, or manage runs",
        description="Monitor runs via the Argus control plane.",
    )
    monitor_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments for the Argus monitor entrypoint.",
    )

    tail_parser = subparsers.add_parser(
        "tail",
        help="Stream events from a run directory to stdout",
        description="Stream run events to stdout without launching the TUI.",
    )
    tail_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments for the Argus tail entrypoint.",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        return run_main(args.args)

    if args.command == "monitor":
        return monitor_main(args.args)

    if args.command == "tail":
        return tail_main(args.args)

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1
