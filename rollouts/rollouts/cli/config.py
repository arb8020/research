from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..core import Endpoint, Trajectory
from ..environments.factory import infer_environment_name
from ..store import FileSessionStore

if TYPE_CHECKING:
    from ..environments.base import Environment


DEFAULT_THINKING_BUDGET = 10000
MAX_TOKENS_WITH_THINKING = 16384
MAX_TOKENS_DEFAULT = 8192

PARSER_DEFAULTS = {
    "model": "anthropic/claude-opus-4-5-20251101",
    "env": "none",
    "thinking": "enabled",
}


@dataclass
class CLIConfig:
    """Parsed CLI configuration."""

    model: str = PARSER_DEFAULTS["model"]
    api_base: str | None = None
    api_key: str | None = None
    thinking: str = PARSER_DEFAULTS["thinking"]

    env: str = PARSER_DEFAULTS["env"]
    tools: str | None = None
    cwd: str | None = None
    confirm_tools: bool = False
    context: str | None = None
    tbench_task_id: str | None = None
    tbench_dataset_name: str = "terminal-bench-core"
    tbench_dataset_version: str = "head"
    tbench_surface: str = "terminal"
    tbench_logging_dir: str | None = None
    tbench_rebuild: bool = False
    tbench_no_cleanup: bool = False

    continue_session: bool = False
    session: str | None = None
    no_session: bool = False

    print_mode: str | None = None
    stream_json: bool = False
    quiet: bool = False
    bootstrap_input: str | None = None

    frontend: str = "tui"
    theme: str = "minimal"
    debug: bool = False
    debug_layout: bool = False
    log_file: str | None = None

    driver: str = "sdk"
    cursor_api_key: str | None = None

    preset: str | None = None
    system_prompt: str | None = None

    template: str | None = None
    template_args: dict[str, str] | None = None
    interactive: bool = False
    list_templates: bool = False
    _template_config: object | None = None
    _bash_allowlist: list[str] | None = None

    pick: bool = False

    list_models: str | None = None
    sync_models: bool = False
    write_models: bool = False
    list_presets: bool = False
    login_claude: bool = False
    logout_claude: bool = False
    list_claude_profiles: bool = False
    set_default_profile: str | None = None
    profile: str | None = None
    export_md: str | None = None
    export_html: str | None = None
    handoff: str | None = None
    fast_handoff: bool = False
    slice: str | None = None
    slice_goal: str | None = None
    doctor: bool = False
    trim: int | None = None
    fix: bool = False

    send: tuple[str, str] | None = None
    send_file: tuple[str, str] | None = None
    attach: str | None = None
    status: str | None = None
    ls: bool = False
    ls_all: bool = False
    detached: bool = False

    working_dir: Path = field(default_factory=Path.cwd)
    endpoint: Endpoint | None = None
    environment: Environment | None = None
    session_store: FileSessionStore | None = None
    trajectory: Trajectory | None = None


def parse_model_string(model_str: str) -> tuple[str, str]:
    if "/" not in model_str:
        raise ValueError(
            f'Model must be in "provider/model" format (e.g., "anthropic/claude-sonnet-4-5"). '
            f'Got: "{model_str}"'
        )

    provider, model = model_str.split("/", 1)
    return provider, model


def create_endpoint(
    model_str: str,
    api_base: str | None = None,
    api_key: str | None = None,
    thinking: str = "enabled",
    quiet: bool = False,
    profile: str = "default",
    driver: str = "sdk",
) -> Endpoint:
    del profile
    del driver

    from typing import cast

    from ..models import MODELS, Provider, get_api_type, get_model

    provider, model = parse_model_string(model_str)
    model_metadata = get_model(cast(Provider, provider), model)
    if model_metadata is None:
        from ..fuzzy import fuzzy_filter

        provider_models = MODELS.get(cast(Provider, provider), {})
        all_model_ids = list(provider_models.keys())
        suggestions = fuzzy_filter(all_model_ids, model, lambda x: x)[:3]

        error_msg = f"Model '{model}' not found for provider '{provider}'."
        if suggestions:
            error_msg += "\n\nDid you mean one of these?\n"
            for suggestion in suggestions:
                error_msg += f"  - {provider}/{suggestion}\n"
        error_msg += f"\nSee available models: rollouts --list-models {provider}"
        raise ValueError(error_msg)

    if thinking == "enabled" and not model_metadata.reasoning:
        print(
            f"Model '{model}' doesn't support extended thinking, disabling.",
            file=sys.stderr,
        )
        thinking = "disabled"

    if api_base is None:
        if model_metadata.base_url:
            base_url = model_metadata.base_url
        else:
            default_urls = {
                "anthropic": "https://api.anthropic.com/v1",
                "openai": "https://api.openai.com/v1",
                "google": "https://generativelanguage.googleapis.com/v1beta",
                "openrouter": "https://openrouter.ai/api/v1",
                "groq": "https://api.groq.com/openai/v1",
                "fireworks": "https://api.fireworks.ai/inference/v1",
                "together": "https://api.together.xyz/v1",
                "cerebras": "https://api.cerebras.ai/v1",
                "xai": "https://api.x.ai/v1",
            }
            base_url = default_urls.get(provider, "https://api.openai.com/v1")
    else:
        base_url = api_base

    api_format = model_metadata.api if model_metadata else get_api_type(provider, model)

    oauth_token = ""
    is_claude_code_api_key = False
    if api_key is not None and not quiet:
        print("Using API key (--api-key flag)", file=sys.stderr)

    if api_key is None:
        from ..credentials import get_api_key

        api_key = get_api_key(provider) or ""

    thinking_config = None
    if api_format == "anthropic-messages" and thinking == "enabled":
        thinking_config = {"type": "enabled", "budget_tokens": DEFAULT_THINKING_BUDGET}

    max_tokens = MAX_TOKENS_WITH_THINKING if thinking_config else MAX_TOKENS_DEFAULT
    actual_model_id = model_metadata.id if model_metadata else model
    model_string = f"{provider}/{actual_model_id}"

    return Endpoint(
        model=model_string,
        base_url=base_url,
        api_format=api_format,
        api_key=api_key,
        oauth_token=oauth_token,
        is_claude_code_api_key=is_claude_code_api_key,
        thinking=thinking_config,
        max_tokens=max_tokens,
    )


def apply_preset(config: CLIConfig) -> bool:
    if not config.preset:
        return True

    from ..agent_presets import load_preset

    try:
        preset = load_preset(config.preset)
    except Exception as e:
        print(f"Error loading preset '{config.preset}': {e}", file=sys.stderr)
        return False

    if config.model == PARSER_DEFAULTS["model"]:
        config.model = preset.model
    if config.env == PARSER_DEFAULTS["env"]:
        config.env = preset.env
    if config.system_prompt is None:
        config.system_prompt = preset.system_prompt
    if preset.thinking:
        config.thinking = preset.thinking
    if preset.working_dir and config.cwd is None:
        config.cwd = str(preset.working_dir)
    return True


def apply_template(config: CLIConfig) -> bool:
    if not config.template:
        return True

    from ..templates import TemplateConfig, load_template

    try:
        template: TemplateConfig = load_template(config.template)
    except Exception as e:
        print(f"Error loading template '{config.template}': {e}", file=sys.stderr)
        return False

    config._template_config = template
    if template.model is not None and config.model == PARSER_DEFAULTS["model"]:
        config.model = template.model
    if template.thinking is not None and config.thinking == PARSER_DEFAULTS["thinking"]:
        config.thinking = "enabled" if template.thinking else "disabled"

    config.env = "coding"
    config.tools = ",".join(template.tools)
    config._bash_allowlist = template.bash_allowlist

    try:
        config.system_prompt = template.interpolate_prompt(config.template_args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

    if not config.interactive:
        config.detached = True
    return True


def apply_session_config(config: CLIConfig) -> bool:
    session_id_for_config: str | None = None
    if config.session and config.session != "":
        session_id_for_config = config.session
    elif config.continue_session:
        session_store = FileSessionStore()
        session_id_for_config = session_store.get_latest_id_sync()

    if not session_id_for_config:
        return True

    session_store = FileSessionStore()
    session_config, err = session_store.get_config_sync(session_id_for_config)
    if err:
        print(f"Error loading session config: {err}", file=sys.stderr)
        return False

    if not session_config:
        return True

    if config.model == PARSER_DEFAULTS["model"]:
        endpoint_config = session_config.get("endpoint", {})
        if endpoint_config.get("model"):
            model_str = endpoint_config["model"]
            if "/" in model_str:
                provider = model_str.split("/")[0]
            else:
                provider = endpoint_config.get("provider", "anthropic")
            if config.driver == "claude" and provider != "anthropic":
                config.model = "anthropic/claude-sonnet-4-5-20250929"
            elif "/" in model_str:
                config.model = model_str
            else:
                config.model = f"{provider}/{model_str}"

    if config.env == PARSER_DEFAULTS["env"]:
        env_config = session_config.get("environment", {})
        env_type = env_config.get("type", "")
        inferred_env = infer_environment_name(env_type)
        if inferred_env is not None:
            config.env = inferred_env

    if config.thinking == PARSER_DEFAULTS["thinking"]:
        endpoint_config = session_config.get("endpoint", {})
        if endpoint_config.get("thinking") is False:
            config.thinking = "disabled"

    if not config.confirm_tools:
        env_config = session_config.get("environment", {})
        if env_config.get("config", {}).get("confirm_tools"):
            config.confirm_tools = True

    return True
