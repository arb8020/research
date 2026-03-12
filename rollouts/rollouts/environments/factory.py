from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trio

from ..core import Endpoint, Environment


@dataclass(frozen=True)
class EnvironmentBuildConfig:
    endpoint: Endpoint | None
    working_dir: Path
    tools: str | None = None
    bash_allowlist: list[str] | None = None
    context: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


EnvironmentBuilder = Callable[[EnvironmentBuildConfig], tuple[Environment | None, bool]]


@dataclass(frozen=True)
class EnvironmentFactory:
    name: str
    build: EnvironmentBuilder
    help_text: str = ""
    session_types: tuple[str, ...] = ()


_ENVIRONMENT_FACTORIES: dict[str, EnvironmentFactory] = {}
_DEFAULT_REGISTERED = False


def register_environment_factory(factory: EnvironmentFactory) -> None:
    _ENVIRONMENT_FACTORIES[factory.name] = factory


def _warn_missing_context() -> None:
    print(
        "Warning: No context provided for REPL environment. "
        "Use --context or --context-file to provide input.",
        file=sys.stderr,
    )


def _build_ask_user(config: EnvironmentBuildConfig) -> tuple[Environment | None, bool]:
    del config
    from .ask_user import AskUserQuestionEnvironment

    return AskUserQuestionEnvironment(), True


def _build_calculator(config: EnvironmentBuildConfig) -> tuple[Environment | None, bool]:
    del config
    from .calculator import CalculatorEnvironment

    return CalculatorEnvironment(), True


def _build_coding(config: EnvironmentBuildConfig) -> tuple[Environment | None, bool]:
    from .coding import TOOL_PRESETS, LocalFilesystemEnvironment

    tools = config.tools or "full"

    if "," in tools:
        tools_list = [t.strip() for t in tools.split(",")]
        return LocalFilesystemEnvironment(
            working_dir=config.working_dir,
            tools=tools_list,
            bash_allowlist=config.bash_allowlist,
        ), True

    if tools not in TOOL_PRESETS:
        print(
            f"Unknown tool preset: {tools}. Available: {', '.join(TOOL_PRESETS.keys())}",
            file=sys.stderr,
        )
        return None, False

    return LocalFilesystemEnvironment(
        working_dir=config.working_dir,
        tools=tools,
        bash_allowlist=config.bash_allowlist,
    ), True


def _build_git(config: EnvironmentBuildConfig) -> tuple[Environment | None, bool]:
    from .git_worktree import GitWorktreeEnvironment

    return GitWorktreeEnvironment(working_dir=config.working_dir), True


def _build_orchestrate(config: EnvironmentBuildConfig) -> tuple[Environment | None, bool]:
    from .orchestrate import OrchestrateEnvironment

    return OrchestrateEnvironment(endpoint=config.endpoint, cwd=config.working_dir), True


def _build_repl(config: EnvironmentBuildConfig) -> tuple[Environment | None, bool]:
    from .repl import REPLEnvironment

    context = config.context or ""
    if not context:
        _warn_missing_context()
    return REPLEnvironment(context=context, sub_endpoint=config.endpoint), True


def _build_repl_blocks(config: EnvironmentBuildConfig) -> tuple[Environment | None, bool]:
    from .repl import MessageParsingREPLEnvironment

    context = config.context or ""
    if not context:
        _warn_missing_context()
    return MessageParsingREPLEnvironment(context=context, sub_endpoint=config.endpoint), True


def _build_tbench(config: EnvironmentBuildConfig) -> tuple[Environment | None, bool]:
    task_id = str(config.extra.get("tbench_task_id") or "")
    if not task_id:
        print("--tbench-task-id is required when using --env tbench", file=sys.stderr)
        return None, False

    from .terminal_bench import create_tbench_environment, create_tbench_resource

    dataset_name = str(config.extra.get("tbench_dataset_name", "terminal-bench-core"))
    dataset_version = str(config.extra.get("tbench_dataset_version", "head"))
    surface = str(config.extra.get("tbench_surface", "terminal"))
    logging_dir = config.extra.get("tbench_logging_dir")
    no_rebuild = not bool(config.extra.get("tbench_rebuild", False))
    cleanup = not bool(config.extra.get("tbench_no_cleanup", False))

    try:

        async def _create() -> Environment:
            resource = await create_tbench_resource(
                task_id=task_id,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                logging_dir=logging_dir,
                no_rebuild=no_rebuild,
                cleanup=cleanup,
            )
            return await create_tbench_environment(surface=surface, resource=resource)

        environment = trio.run(_create)
    except Exception as exc:
        print(f"Failed to create Terminal-Bench environment: {exc}", file=sys.stderr)
        return None, False
    return environment, True


def _ensure_default_factories() -> None:
    global _DEFAULT_REGISTERED
    if _DEFAULT_REGISTERED:
        return
    _DEFAULT_REGISTERED = True

    register_environment_factory(
        EnvironmentFactory(
            name="ask_user",
            build=_build_ask_user,
            help_text="Ask the user clarifying questions.",
            session_types=("AskUserQuestionEnvironment",),
        )
    )
    register_environment_factory(
        EnvironmentFactory(
            name="calculator",
            build=_build_calculator,
            help_text="Simple calculator tools.",
            session_types=("CalculatorEnvironment",),
        )
    )
    register_environment_factory(
        EnvironmentFactory(
            name="coding",
            build=_build_coding,
            help_text="Local filesystem coding tools.",
            session_types=("LocalFilesystemEnvironment", "CodingEnvironment"),
        )
    )
    register_environment_factory(
        EnvironmentFactory(
            name="git",
            build=_build_git,
            help_text="Coding tools with isolated git history.",
            session_types=("GitWorktreeEnvironment",),
        )
    )
    register_environment_factory(
        EnvironmentFactory(
            name="orchestrate",
            build=_build_orchestrate,
            help_text="Slate-style orchestration environment.",
            session_types=("OrchestrateEnvironment",),
        )
    )
    register_environment_factory(
        EnvironmentFactory(
            name="repl",
            build=_build_repl,
            help_text="REPL environment for large contexts.",
            session_types=("REPLEnvironment",),
        )
    )
    register_environment_factory(
        EnvironmentFactory(
            name="repl_blocks",
            build=_build_repl_blocks,
            help_text="Message-parsing REPL environment.",
            session_types=("MessageParsingREPLEnvironment",),
        )
    )
    register_environment_factory(
        EnvironmentFactory(
            name="tbench",
            build=_build_tbench,
            help_text="Terminal-Bench environment.",
            session_types=("TerminalBenchEnvironment",),
        )
    )


def get_environment_factories() -> dict[str, EnvironmentFactory]:
    _ensure_default_factories()
    return dict(_ENVIRONMENT_FACTORIES)


def get_environment_names() -> list[str]:
    return sorted(get_environment_factories())


def infer_environment_name(env_type: str) -> str | None:
    for name, factory in get_environment_factories().items():
        if env_type in factory.session_types:
            return name
    return None


def build_environment(
    config: EnvironmentBuildConfig,
    env_name: str,
) -> tuple[Environment | None, bool]:
    factories = get_environment_factories()

    if "+" in env_name:
        from .compose import compose

        built_envs: list[Environment] = []
        for part in env_name.split("+"):
            env, ok = build_environment(config, part)
            if not ok:
                return None, False
            assert env is not None
            built_envs.append(env)
        return compose(*built_envs), True

    if env_name == "none":
        return None, True

    factory = factories.get(env_name)
    if factory is None:
        print(f"Unknown environment: {env_name}", file=sys.stderr)
        return None, False
    return factory.build(config)
