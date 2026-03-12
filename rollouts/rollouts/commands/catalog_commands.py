from __future__ import annotations


def cmd_list_models(search_pattern: str | None = None) -> int:
    """Handle --list-models command with optional fuzzy search."""
    from ..fuzzy import fuzzy_filter
    from ..models import MODELS, ModelMetadata

    def format_tokens(count: int) -> str:
        """Format token count as human-readable (e.g., 200K, 1M)."""
        if count >= 1_000_000:
            millions = count / 1_000_000
            return f"{millions:.0f}M" if millions == int(millions) else f"{millions:.1f}M"
        if count >= 1_000:
            thousands = count / 1_000
            return f"{thousands:.0f}K" if thousands == int(thousands) else f"{thousands:.1f}K"
        return str(count)

    all_models: list[tuple[str, str, ModelMetadata]] = []
    for provider, models in MODELS.items():
        for model_id, meta in models.items():
            all_models.append((provider, model_id, meta))

    if not all_models:
        print("No models in registry.")
        return 0

    if search_pattern:
        filtered = fuzzy_filter(
            all_models,
            search_pattern,
            lambda x: f"{x[0]} {x[1]}",
        )
        if not filtered:
            print(f'No models matching "{search_pattern}"')
            return 0
        all_models = filtered

    all_models.sort(key=lambda x: (x[0], x[1]))

    rows = []
    for provider, model_id, meta in all_models:
        rows.append({
            "provider": provider,
            "model": model_id,
            "context": format_tokens(meta.context_window),
            "max_out": format_tokens(meta.max_tokens),
            "thinking": "yes" if meta.reasoning else "no",
            "cost": f"${meta.cost.input:.2f}/${meta.cost.output:.2f}",
        })

    headers = {
        "provider": "provider",
        "model": "model",
        "context": "context",
        "max_out": "max-out",
        "thinking": "thinking",
        "cost": "cost (in/out)",
    }

    widths = {
        key: max(len(headers[key]), max(len(str(row[key])) for row in rows)) for key in headers
    }

    header_line = "  ".join(headers[k].ljust(widths[k]) for k in headers)
    print(header_line)

    for row in rows:
        line = "  ".join(str(row[k]).ljust(widths[k]) for k in headers)
        print(line)

    if search_pattern:
        print(f'\n{len(rows)} model(s) matching "{search_pattern}"')
    else:
        print(f"\n{len(rows)} model(s) total")

    return 0


def cmd_sync_models(write: bool = False) -> int:
    """Handle --sync-models command."""
    import os

    import trio

    from ..models import (
        ModelDiff,
        fetch_anthropic_docs,
        sync_anthropic_models,
        update_models_file,
        write_models_to_disk,
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY required for --sync-models")
        return 1

    print("Fetching models from Anthropic API and docs...")

    async def do_sync() -> tuple[ModelDiff, dict[str, dict]]:
        diff = await sync_anthropic_models(api_key)
        docs = await fetch_anthropic_docs()
        return diff, docs

    diff, docs = trio.run(do_sync)

    if diff.missing:
        print(f"\nNew models ({len(diff.missing)}):")
        for model_id in diff.missing:
            if model_id in docs:
                d = docs[model_id]
                print(
                    f"  + {model_id}: ${d['input_cost']:.2f}/${d['output_cost']:.2f}, {d['context_window'] // 1000}K ctx"
                )
            else:
                print(f"  + {model_id}: (no metadata available)")

    if diff.updated:
        print(f"\nUpdated models ({len(diff.updated)}):")
        for model_id, changes in diff.updated.items():
            print(f"  ~ {model_id}:")
            for field, (old, new) in changes.items():
                print(f"      {field}: {old} -> {new}")

    if diff.extra:
        print(f"\nDeprecated/unlisted ({len(diff.extra)}):")
        for model_id in diff.extra:
            print(f"  - {model_id}")

    if not diff.missing and not diff.updated:
        print("\nRegistry is already up to date.")
        return 0

    result = update_models_file(diff, docs)
    print(f"\n{result}")

    if write:
        models_path = write_models_to_disk()
        print(f"\nWrote updated models to {models_path}")
    else:
        print("\nNote: Changes applied to runtime registry only.")
        print("Use --sync-models --write to persist to models.py")

    return 0


def cmd_list_presets() -> int:
    """Handle --list-presets command."""
    from ..agent_presets import list_presets

    presets = list_presets()
    if not presets:
        print("No presets found in rollouts/agent_presets/")
        return 0

    print("Available agent presets:")
    for preset_name in presets:
        print(f"  - {preset_name}")

    print("\nUsage: rollouts --preset <name>")
    print("Example: rollouts --preset sonnet_4")
    return 0


def cmd_list_templates() -> int:
    """Handle --list-templates command."""
    from ..templates import list_templates

    templates = list_templates()
    if not templates:
        print("No templates found.")
        print("Templates are searched in:")
        print("  - ./rollouts/templates/ (project)")
        print("  - ~/.rollouts/templates/ (user)")
        print("  - rollouts/templates/ (built-in)")
        return 0

    print("Available templates:")
    for template_name in templates:
        print(f"  - {template_name}")

    print('\nUsage: rollouts -t <template> [--args key=value] "prompt"')
    print('Example: rollouts -t ask-docs "How do bank conflicts occur?"')
    return 0
