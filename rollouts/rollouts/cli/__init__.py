"""CLI package and compatibility exports for the public rollouts CLI surface."""

from ..chat_runtime import (
    create_session_store as _create_session_store,
)
from ..chat_runtime import (
    finalize_tbench_run,
    get_tbench_agent_timeout_sec,
)
from ..store import FileSessionStore
from .config import (
    PARSER_DEFAULTS,
    CLIConfig,
    apply_preset,
    apply_session_config,
    apply_template,
    create_endpoint,
)
from .main import create_environment, main, run_agent
from .parser import create_parser


def create_session_store(config: CLIConfig) -> FileSessionStore | None:
    """Compatibility wrapper for the historical rollouts.cli surface."""
    return _create_session_store(
        no_session=config.no_session,
        env_name=config.env,
        environment=config.environment,
    )


__all__ = [
    "CLIConfig",
    "PARSER_DEFAULTS",
    "apply_preset",
    "apply_session_config",
    "apply_template",
    "create_endpoint",
    "create_environment",
    "create_parser",
    "create_session_store",
    "finalize_tbench_run",
    "get_tbench_agent_timeout_sec",
    "main",
    "run_agent",
]
