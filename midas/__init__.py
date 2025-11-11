"""Midas package - Python environment setup on remote machines.

The Midas touch: turning bare machines into golden Python environments.

This package handles getting from "unknown state" to "working Python environment".
It's purely about environment bootstrapping, nothing else.

Separation of concerns:
- midas/ - Just environment setup (this package)
- remote/ - Generic remote execution helpers (tmux, gpu, env vars, transfer)
- deploy/ - High-level composition of midas + remote

Available backends:
- UvBackend: UV-based Python environment (production ready)
- NixBackend: Nix-based reproducible environment (future)
- DockerBackend: Docker-based environment (future)
"""

from midas.protocol import EnvBackend, CommandResult
from midas.backends.uv import UvBackend

__all__ = ["EnvBackend", "CommandResult", "UvBackend"]
