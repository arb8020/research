"""Base template configuration for rollouts agents.

Templates define constrained agents for specific tasks:
- Specific toolset (subset of coding env tools)
- Bash allowlist (prefix-matching)
- Optional model override
- System prompt with variable interpolation
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TemplateConfig:
    """Configuration for a task-specific agent template.

    Templates are constrained agents designed for headless execution.
    They bundle (system_prompt, tools, bash_allowlist) into a reusable config.

    Example:
        >>> template = TemplateConfig(
        ...     name="ask-docs",
        ...     system_prompt="You analyze documentation to answer questions...",
        ...     tools=["read", "glob", "grep", "bash"],
        ...     bash_allowlist=["uv run pytest", "jq", "python -c"],
        ... )
    """

    # Identity
    name: str

    # Core config
    system_prompt: str
    tools: list[str] = field(default_factory=lambda: ["read", "glob", "grep", "bash"])

    # Bash constraints (prefix matching)
    bash_allowlist: list[str] | None = None

    # Optional model override (if not set, uses CLI default)
    model: str | None = None
    thinking: bool | None = None

    # Template variables and their defaults
    # Example: {"corpus": "./docs/", "format": "markdown"}
    defaults: dict[str, str] = field(default_factory=dict)

    # Description for --list-templates
    description: str = ""

    def interpolate_prompt(self, args: dict[str, str] | None = None) -> str:
        """Interpolate template variables into the system prompt.

        Args:
            args: Variable values from --args. Example: {"corpus": "./my-docs/"}

        Returns:
            System prompt with $variables replaced.

        Raises:
            ValueError: If required variables are missing.
        """
        params = dict(self.defaults)
        if args:
            params.update(args)

        # Find all variables in the template
        template = string.Template(self.system_prompt)
        required_vars = self._extract_vars(template)

        # Check for missing required variables
        missing = [v for v in required_vars if v not in params]
        if missing:
            raise ValueError(
                f"Template '{self.name}' requires variables: {', '.join(missing)}. "
                f"Provide them with --args {missing[0]}=value"
            )

        return template.safe_substitute(**params)

    def get_variables(self) -> list[str]:
        """Get list of variables used in the system prompt."""
        template = string.Template(self.system_prompt)
        return self._extract_vars(template)

    @staticmethod
    def _extract_vars(template: string.Template) -> list[str]:
        """Extract variable names from a string.Template."""
        return [
            match.group("named") or match.group("braced")
            for match in template.pattern.finditer(template.template)
            if match.group("named") or match.group("braced")
        ]

    def to_cli_args(self) -> dict[str, Any]:
        """Convert to CLI argument dict for consumption by runner."""
        args: dict[str, Any] = {
            "tools": self.tools,
            "bash_allowlist": self.bash_allowlist,
        }

        if self.model is not None:
            args["model"] = self.model

        if self.thinking is not None:
            args["thinking"] = self.thinking

        return args
