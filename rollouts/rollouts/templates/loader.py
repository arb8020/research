"""Template loader utilities.

Loads templates from Python files with discovery order:
    1. Project templates: ./rollouts/templates/ or ./.rollouts/templates/
    2. User templates: ~/.rollouts/templates/
    3. Global templates: rollouts/templates/ (built-in)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .base import TemplateConfig

# Search paths for templates (in priority order)
TEMPLATE_SEARCH_PATHS = [
    # Project templates (not inside rollouts/ to avoid Python package shadowing)
    lambda: Path.cwd() / ".rollouts" / "templates",
    lambda: Path.cwd() / "templates",  # Simple alternative
    # User templates
    lambda: Path.home() / ".rollouts" / "templates",
    # Global templates (this package)
    lambda: Path(__file__).parent,
]


def _get_search_paths() -> list[Path]:
    """Get all template search paths that exist."""
    paths = []
    for path_fn in TEMPLATE_SEARCH_PATHS:
        path = path_fn()
        if path.exists() and path.is_dir():
            paths.append(path)
    return paths


def load_template(
    template_name: str,
    search_paths: list[Path] | None = None,
) -> TemplateConfig:
    """Load a template by name or from a file path.

    Args:
        template_name: Either:
            - Template name (e.g., "ask-docs")
            - Path to a template file (e.g., "~/my-templates/custom.py")
        search_paths: Override search paths (default: project > user > global)

    Returns:
        TemplateConfig instance

    Raises:
        FileNotFoundError: If template not found
        AttributeError: If template file doesn't export 'template'
        ValueError: If template is not a TemplateConfig
    """
    # Check if template_name is a file path
    template_path = Path(template_name).expanduser()
    if template_path.exists() and template_path.is_file():
        return _load_template_file(template_path)

    # Search in template directories
    if search_paths is None:
        search_paths = _get_search_paths()

    # Normalize name: "ask-docs" -> "ask_docs" for Python module names
    module_name = template_name.replace("-", "_")

    for search_dir in search_paths:
        template_file = search_dir / f"{module_name}.py"
        if template_file.exists():
            return _load_template_file(template_file)

        # Also try original name
        template_file = search_dir / f"{template_name}.py"
        if template_file.exists():
            return _load_template_file(template_file)

    # Not found
    available = list_templates(search_paths)
    raise FileNotFoundError(
        f"Template '{template_name}' not found.\n"
        f"Searched: {', '.join(str(p) for p in search_paths)}\n"
        f"Available templates: {', '.join(available) if available else '(none)'}"
    )


def _load_template_file(path: Path) -> TemplateConfig:
    """Load template from Python file.

    Expects file to export 'template' variable of type TemplateConfig.
    """
    spec = importlib.util.spec_from_file_location(f"template_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load template from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "template"):
        raise AttributeError(
            f"Template file {path} must export 'template' variable.\n"
            f"Example: template = TemplateConfig(name='my-template', ...)"
        )

    config = module.template

    if not isinstance(config, TemplateConfig):
        raise ValueError(f"Template must be TemplateConfig, got {type(config)}")

    return config


def list_templates(search_paths: list[Path] | None = None) -> list[str]:
    """List available template names.

    Args:
        search_paths: Override search paths (default: project > user > global)

    Returns:
        List of template names (without .py extension), deduplicated.
        Project templates shadow user/global templates with the same name.
    """
    if search_paths is None:
        search_paths = _get_search_paths()

    seen: set[str] = set()
    templates: list[str] = []

    # Files to skip
    skip = {"__init__.py", "base.py", "loader.py"}

    for search_dir in search_paths:
        for file in sorted(search_dir.glob("*.py")):
            if file.name in skip:
                continue

            name = file.stem
            # Normalize: ask_docs -> ask-docs for display
            display_name = name.replace("_", "-")

            if display_name not in seen:
                seen.add(display_name)
                templates.append(display_name)

    return templates
