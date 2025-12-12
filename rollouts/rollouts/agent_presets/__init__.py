"""Agent preset configurations for rollouts.

Presets bundle (model, environment, system_prompt) for quick iteration on agent behavior.

Usage:
    >>> from rollouts.agent_presets import load_preset
    >>> preset = load_preset("fast_coder_01_01")
    >>> cli_args = preset.to_cli_args()
"""

from .base_preset import AgentPresetConfig
from .loader import load_preset, list_presets

__all__ = [
    "AgentPresetConfig",
    "load_preset",
    "list_presets",
]
