"""Kerbal package - Remote execution helpers.

Like Kerbal Space Program, this package helps you orchestrate complex remote operations
with modular, composable components. Launch your code into production!

This package provides generic utilities for remote execution:
- tmux: Process management (sessions, detached execution)
- gpu: Hardware checking (nvidia-smi queries, availability)
- env: Environment variable helpers (export building)
- transfer: File transfer (push, sync)

These are orthogonal concerns that can be composed together.

Separation of concerns:
- midas/ - Python environment setup (the Midas touch)
- kerbal/ - Generic remote execution helpers (this package)
- deploy/ - High-level composition of midas + kerbal
"""

from kerbal.tmux import start_tmux_session
from kerbal.gpu import check_gpus_available, wait_for_gpus
from kerbal.env import build_env_prefix
from kerbal.transfer import push_code, sync_results

__all__ = [
    "start_tmux_session",
    "check_gpus_available",
    "wait_for_gpus",
    "build_env_prefix",
    "push_code",
    "sync_results",
]
