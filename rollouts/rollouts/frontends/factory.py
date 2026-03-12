from __future__ import annotations

import sys

from ..core import Endpoint, Environment
from .json_frontend import JsonFrontend
from .minimal import MinimalFrontend
from .none import NoneFrontend
from .protocol import Frontend
from .tui_frontend import TUIFrontend


def create_print_frontend(
    *,
    stream_json: bool,
    quiet: bool,
    frontend_name: str,
    environment: Environment | None,
    endpoint: Endpoint,
) -> Frontend:
    if stream_json:
        frontend = JsonFrontend(include_thinking=True)
        if environment:
            frontend.set_tools([tool.function.name for tool in environment.get_tools()])
        return frontend

    if quiet:
        return NoneFrontend(show_tool_calls=False, show_thinking=False)

    if frontend_name == "minimal":
        env_name = environment.__class__.__name__ if environment else None
        return MinimalFrontend(
            show_tool_calls=True,
            show_thinking=False,
            agent=env_name,
            model=endpoint.model,
        )

    return NoneFrontend(show_tool_calls=True, show_thinking=False)


def create_interactive_frontend(
    *,
    frontend_name: str,
    environment: Environment | None,
    endpoint: Endpoint,
    theme: str,
    debug: bool,
    debug_layout: bool,
    driver: str,
    detached: bool,
) -> Frontend:
    if detached:
        return NoneFrontend(show_tool_calls=True, show_thinking=False)

    env_name = environment.__class__.__name__ if environment else None

    if frontend_name == "none":
        return NoneFrontend(show_tool_calls=True, show_thinking=True)

    if frontend_name == "minimal":
        return MinimalFrontend(
            show_tool_calls=True,
            show_thinking=True,
            agent=env_name,
            model=endpoint.model,
        )

    if frontend_name == "textual":
        print("Textual frontend not yet implemented. Use --frontend=tui for now.", file=sys.stderr)
        raise ValueError("unsupported_frontend")

    return TUIFrontend(
        theme=theme,
        environment=environment,
        debug=debug,
        debug_layout=debug_layout,
        driver=driver,
    )
