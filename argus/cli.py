"""Argus control-plane CLI.

This is the public control-plane entrypoint. For now it reuses existing
Rollouts launch/monitor implementation details. Over time, detached run truth
and transport-specific monitoring should move fully behind Argus.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
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
        help="Arguments forwarded to the underlying run launcher (for now: rollouts.run).",
    )

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Observe, attach to, or manage runs",
        description="Monitor runs via the Argus control plane.",
    )
    monitor_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the current monitor implementation.",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        from rollouts.run import main as rollouts_run_main

        # TODO: replace this with native Argus launch submission once the
        # supervisor owns run/attempt/allocation orchestration directly.
        return rollouts_run_main(args.args)

    if args.command == "monitor":
        from rollouts.tui.monitor_cli import monitor_main

        # TODO: replace this with snapshot + subscribe over Argus journals
        # instead of reconstructing state from rollouts-side logs and tmux.
        return monitor_main(args.args)

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1
