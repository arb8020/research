"""Base preset configuration for rollouts agents.

Design follows experiment_config.md:
- Pythonic + hierarchical + serializable
- Frozen dataclasses for immutability
- Explicit parameters, no magic
- Composition over inheritance

A preset bundles (model, environment, system_prompt) for quick iteration.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentPresetConfig:
    """Configuration preset for an agent.

    Bundles model, environment, and system prompt into a single reusable config.

    Example:
        >>> preset = AgentPresetConfig(
        ...     name="fast_coder",
        ...     model="anthropic/claude-3-5-haiku-20241022",
        ...     env="coding",
        ...     system_prompt="You are a fast coding assistant...",
        ... )
    """

    # Identity
    name: str  # e.g., "fast_coder", "careful_coder"

    # Core parameters (match CLI args)
    model: str  # "provider/model" format
    env: str  # "none", "calculator", "coding", "git"
    system_prompt: str  # Full system prompt text

    # Optional: Model behavior
    thinking: bool = True  # Enable extended thinking (Anthropic models)
    temperature: float | None = None  # Override model temperature

    # Optional: Environment settings
    working_dir: Path | None = None  # Working directory for coding/git envs

    def save(self, path: Path) -> None:
        """Save this preset for reproducibility."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "AgentPresetConfig":
        """Load a saved preset."""
        with open(path) as f:
            data = json.load(f)
        # Convert string paths back to Path objects
        if "working_dir" in data and data["working_dir"] is not None:
            data["working_dir"] = Path(data["working_dir"])
        return cls(**data)

    def to_cli_args(self) -> dict[str, Any]:
        """Convert to CLI argument dict for easy consumption.

        Returns:
            Dict with keys matching argparse argument names.
        """
        args = {
            "model": self.model,
            "env": self.env,
            "system_prompt": self.system_prompt,
            "thinking": self.thinking,
        }

        if self.temperature is not None:
            args["temperature"] = self.temperature

        if self.working_dir is not None:
            args["working_dir"] = self.working_dir

        return args
