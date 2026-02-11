"""Interactive model/driver picker for rollouts CLI.

Allows users to interactively select:
1. Driver mode (rollouts, claude-code, codex)
2. Model to use

Usage:
    from rollouts.picker import pick_model_interactive

    result = pick_model_interactive()
    # result = {"driver": "rollouts", "model": "anthropic/claude-sonnet-4-20250514"}
    # or result = {"driver": "claude-code", "model": "sonnet"}
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from typing import Literal

from .models import MODELS, Provider


DriverType = Literal["rollouts", "claude-code", "codex"]


@dataclass
class PickerResult:
    """Result from the interactive picker."""
    driver: DriverType
    model: str  # For rollouts: "provider/model", for claude-code: "sonnet", etc.
    provider: str | None = None  # Only set for rollouts driver


# Driver definitions
DRIVERS: dict[DriverType, dict] = {
    "rollouts": {
        "name": "Rollouts Agent",
        "description": "Built-in agent with file/shell tools",
        "available": True,  # Always available
    },
    "claude-code": {
        "name": "Claude Code",
        "description": "Anthropic's Claude Code CLI (requires `claude` installed)",
        "available": shutil.which("claude") is not None,
        "models": ["opus", "sonnet", "haiku"],  # Claude Code model shortcuts
    },
    "codex": {
        "name": "OpenAI Codex CLI",
        "description": "OpenAI's Codex CLI (requires `codex` installed)",
        "available": shutil.which("codex") is not None,
        "models": ["o3", "o4-mini", "gpt-4.1"],  # Codex model options
    },
}


def _print_menu(title: str, options: list[tuple[str, str, bool]], default_idx: int = 0) -> int | None:
    """Print a menu and get user selection.
    
    Args:
        title: Menu title
        options: List of (key, description, available) tuples
        default_idx: Default selection index (0-based)
    
    Returns:
        Selected index (0-based) or None if cancelled
    """
    print(f"\n{title}\n")
    
    for i, (key, desc, available) in enumerate(options):
        marker = "→" if i == default_idx else " "
        status = "" if available else " (not installed)"
        default_str = " [default]" if i == default_idx else ""
        print(f"  {marker} [{i + 1}] {key}: {desc}{status}{default_str}")
    
    print(f"\n  [0] Cancel")
    print()
    
    while True:
        try:
            prompt = f"Select [1-{len(options)}, default={default_idx + 1}]: "
            choice = input(prompt).strip()
            
            if not choice:
                return default_idx
            
            num = int(choice)
            if num == 0:
                return None
            if 1 <= num <= len(options):
                # Check if available
                if not options[num - 1][2]:
                    print(f"  ⚠️  {options[num - 1][0]} is not installed")
                    continue
                return num - 1
            
            print(f"  Please enter 0-{len(options)}")
        except ValueError:
            print("  Please enter a number")
        except (KeyboardInterrupt, EOFError):
            print()
            return None


def pick_driver() -> DriverType | None:
    """Interactively pick a driver."""
    options = [
        (info["name"], info["description"], info["available"])
        for info in DRIVERS.values()
    ]
    
    idx = _print_menu("Select agent driver:", options, default_idx=0)
    if idx is None:
        return None
    
    return list(DRIVERS.keys())[idx]


def pick_model_for_rollouts() -> tuple[str, str] | None:
    """Pick a provider and model for the rollouts driver.
    
    Returns:
        (provider, model) tuple or None if cancelled
    """
    # First, pick provider
    providers = [p for p in MODELS.keys() if MODELS[p]]  # Only providers with models
    
    # Sort to put anthropic first (most common)
    provider_order = ["anthropic", "openai", "cerebras", "groq", "google"]
    providers = sorted(providers, key=lambda p: provider_order.index(p) if p in provider_order else 99)
    
    provider_options = []
    for p in providers:
        model_count = len(MODELS[p])
        provider_options.append((p, f"{model_count} models", True))
    
    idx = _print_menu("Select provider:", provider_options, default_idx=0)
    if idx is None:
        return None
    
    provider = providers[idx]
    
    # Then, pick model from that provider
    models = list(MODELS[provider].values())
    
    # Sort by name, put reasoning models first
    models.sort(key=lambda m: (not m.reasoning, m.name))
    
    model_options = []
    for m in models:
        features = []
        if m.reasoning:
            features.append("reasoning")
        features.append(f"{m.context_window // 1000}K ctx")
        features.append(f"${m.cost.input:.2f}/${m.cost.output:.2f}")
        desc = ", ".join(features)
        model_options.append((m.name, desc, True))
    
    # Default to first reasoning model if available
    default_idx = 0
    for i, m in enumerate(models):
        if m.reasoning:
            default_idx = i
            break
    
    idx = _print_menu(f"Select {provider} model:", model_options, default_idx=default_idx)
    if idx is None:
        return None
    
    return provider, models[idx].id


def pick_model_for_claude_code() -> str | None:
    """Pick a model for Claude Code driver."""
    models = DRIVERS["claude-code"]["models"]
    
    options = [
        ("opus", "Most capable, best for complex tasks", True),
        ("sonnet", "Balanced speed and capability [recommended]", True),
        ("haiku", "Fastest, good for simple tasks", True),
    ]
    
    idx = _print_menu("Select Claude Code model:", options, default_idx=1)  # Default to sonnet
    if idx is None:
        return None
    
    return models[idx]


def pick_model_for_codex() -> str | None:
    """Pick a model for Codex driver."""
    options = [
        ("o3", "Most capable reasoning model", True),
        ("o4-mini", "Fast reasoning model [recommended]", True),
        ("gpt-4.1", "Standard GPT-4.1", True),
    ]
    
    idx = _print_menu("Select Codex model:", options, default_idx=1)  # Default to o4-mini
    if idx is None:
        return None
    
    return DRIVERS["codex"]["models"][idx]


def pick_model_interactive(skip_driver: bool = False, driver: DriverType | None = None) -> PickerResult | None:
    """Full interactive model picker.
    
    Args:
        skip_driver: If True, skip driver selection and use 'rollouts'
        driver: If provided, use this driver and skip driver selection
    
    Returns:
        PickerResult with driver and model, or None if cancelled
    """
    # Pick driver
    if driver is not None:
        selected_driver = driver
    elif skip_driver:
        selected_driver = "rollouts"
    else:
        selected_driver = pick_driver()
        if selected_driver is None:
            return None
    
    # Pick model based on driver
    if selected_driver == "rollouts":
        result = pick_model_for_rollouts()
        if result is None:
            return None
        provider, model = result
        return PickerResult(
            driver="rollouts",
            model=f"{provider}/{model}",
            provider=provider,
        )
    
    elif selected_driver == "claude-code":
        model = pick_model_for_claude_code()
        if model is None:
            return None
        return PickerResult(driver="claude-code", model=model)
    
    elif selected_driver == "codex":
        model = pick_model_for_codex()
        if model is None:
            return None
        return PickerResult(driver="codex", model=model)
    
    return None


def main():
    """CLI entry point for testing the picker."""
    result = pick_model_interactive()
    if result:
        print(f"\nSelected: driver={result.driver}, model={result.model}")
    else:
        print("\nCancelled")


if __name__ == "__main__":
    main()
