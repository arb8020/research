"""Template system for constrained, task-specific agents.

Templates are (system_prompt, tools, bash_allowlist, model?) tuples that define
constrained agents for specific tasks. Unlike presets (which are full agent configs
for interactive use), templates are designed for headless/detached execution.

Usage:
    rollouts -t ask-docs "How do bank conflicts occur?"
    rollouts -t trace-analyze --args trace=./profile.ncu-rep "What's the bottleneck?"
    rollouts -t ask-docs -i "..."  # -i to attach TUI

Template discovery (project > user > global):
    ./rollouts/templates/       # Project templates
    ~/.rollouts/templates/      # User templates
    rollouts/templates/         # Built-in templates
"""

from .base import TemplateConfig
from .loader import list_templates, load_template

__all__ = ["TemplateConfig", "load_template", "list_templates"]
