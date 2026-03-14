"""Megatron-Core training backend.

Keep this package import-light.

The coordinator imports `rollouts.training.backends.megatron.remote_backend`
before forking miniray workers. Eagerly importing `.initialize` or `.model`
here makes the pre-fork boundary dishonest by dragging in the heavy Megatron
stack in the parent process. Child workers should import those modules
directly after fork.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["init_megatron", "setup_megatron_model", "MegatronModelConfig"]

if TYPE_CHECKING:
    from .initialize import init_megatron
    from .model import MegatronModelConfig, setup_megatron_model


def __getattr__(name: str) -> Any:
    if name == "init_megatron":
        from .initialize import init_megatron

        return init_megatron
    if name in {"MegatronModelConfig", "setup_megatron_model"}:
        from .model import MegatronModelConfig, setup_megatron_model

        exports = {
            "MegatronModelConfig": MegatronModelConfig,
            "setup_megatron_model": setup_megatron_model,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
