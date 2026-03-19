#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import trio

from ..chat_runtime import (
    ChatRunRequest,
    cleanup_environment,
    create_session_store,
    run_interactive_mode,
    run_print_mode,
)
from ..commands import (
    auth_main,
    cmd_attach,
    cmd_doctor,
    cmd_export,
    cmd_handoff,
    cmd_list_models,
    cmd_list_presets,
    cmd_list_profiles,
    cmd_list_templates,
    cmd_ls,
    cmd_oauth,
    cmd_set_default_profile,
    cmd_slice,
    cmd_status,
    cmd_sync_models,
    pick_session_async,
)
from ..core import Message, Trajectory
from ..environments.factory import EnvironmentBuildConfig, build_environment
from ..store import FileSessionStore
from .config import (
    CLIConfig,
    apply_preset,
    apply_session_config,
    apply_template,
    create_endpoint,
)
from .parser import create_parser

SYSTEM_PROMPTS = {
    "none": "You are a helpful assistant.",
    "calculator": """You are a calculator assistant with access to math tools.

Available tools: add, subtract, multiply, divide, clear, complete_task.
Each tool operates on a running total (starts at 0).

For calculations:
1. Break down the problem into steps
2. Use tools to compute each step
3. Use complete_task when done

Example: For "(5 + 3) * 2", first add(5), then add(3), then multiply(2).""",
    "coding": """You are a coding assistant with access to file and shell tools.

Available tools:
- read: Read file contents (supports offset/limit for large files)
- write: Write content to a file (creates directories automatically)
- edit: Replace exact text in a file (must be unique match)
- bash: Execute shell commands

When working on code:
1. First read relevant files to understand context
2. Make precise edits using the edit tool
3. Use bash to run tests, linting, etc.
4. Prefer small, focused changes over large rewrites""",
    "git": """You are a coding assistant with access to file and shell tools.

All file changes are automatically tracked in an isolated git history.
This gives you full undo capability - every write/edit/bash creates a commit.

Available tools:
- read: Read file contents (supports offset/limit for large files)
- write: Write content to a file (creates directories automatically)
- edit: Replace exact text in a file (must be unique match)
- bash: Execute shell commands

When working on code:
1. First read relevant files to understand context
2. Make precise edits using the edit tool
3. Use bash to run tests, linting, etc.
4. Prefer small, focused changes over large rewrites""",
    "repl": """You are an assistant with access to a REPL environment for processing large contexts.

The input context is stored in a Python variable called `context`. You explore it programmatically.

Available tools:
- repl: Execute Python code (context variable available, plus re module)
- llm_query: Query a sub-LLM for semantic tasks on text chunks
- final_answer: Submit your final answer

Strategy:
1. Peek first: context[:1000], len(context)
2. Search: re.findall(pattern, context), list comprehensions
3. Chunk for semantics: llm_query("Classify: " + chunk)
4. Answer: final_answer(your_result)""",
    "repl_blocks": """You are an assistant with access to a REPL environment for processing large contexts.

The input context is stored in a Python variable called `context`. You explore it programmatically.

Write code in ```repl or ```python blocks to execute. Use FINAL(answer) when done.

Example:
```repl
print(len(context))
matches = [l for l in context.split('\\n') if 'keyword' in l]
print(matches[:5])
```

When you have the answer: FINAL(42)""",
    "orchestrate": """You are an orchestration agent. Delegate bounded work to child threads instead of doing implementation directly.

Use the orchestrate tool to write async Python that calls:
- system.thread(...) for subagents
- system.query(...) for tool-less questions
- system.fromId(...) to reuse prior child results
- system.allocate(...) / shared docs for cross-thread artifacts
- system.log(...) for progress

Prefer short, purpose-built child tasks. Reuse aliases when continuing a workstream.""",
}


def create_environment(config: CLIConfig) -> tuple[object | None, bool]:
    build_config = EnvironmentBuildConfig(
        endpoint=config.endpoint,
        working_dir=config.working_dir,
        tools=config.tools,
        bash_allowlist=config._bash_allowlist,
        context=config.context,
        extra={
            "tbench_task_id": config.tbench_task_id,
            "tbench_dataset_name": config.tbench_dataset_name,
            "tbench_dataset_version": config.tbench_dataset_version,
            "tbench_surface": config.tbench_surface,
            "tbench_logging_dir": config.tbench_logging_dir,
            "tbench_rebuild": config.tbench_rebuild,
            "tbench_no_cleanup": config.tbench_no_cleanup,
        },
    )
    return build_environment(build_config, config.env)


def _build_chat_run_request(config: CLIConfig) -> ChatRunRequest:
    assert config.endpoint is not None
    return ChatRunRequest(
        endpoint=config.endpoint,
        environment=config.environment,
        working_dir=config.working_dir,
        cwd=config.cwd,
        driver=config.driver,
        cursor_api_key=config.cursor_api_key,
        session_store=config.session_store,
        frontend=config.frontend,
        theme=config.theme,
        debug=config.debug,
        debug_layout=config.debug_layout,
        stream_json=config.stream_json,
        quiet=config.quiet,
        confirm_tools=config.confirm_tools,
        detached=config.detached,
    )


async def run_agent(config: CLIConfig) -> int:
    from ..agents import resume_session

    try:
        session_store = config.session_store
        session_id: str | None = None
        trajectory: Trajectory

        if session_store is not None:
            if config.session is not None:
                if config.session == "":
                    session = await pick_session_async(session_store)
                    if session is None:
                        return 0
                    session_id = session.session_id
                else:
                    session_id = config.session
            elif config.continue_session:
                session, _err = await session_store.get_latest()
                if session:
                    session_id = session.session_id
                else:
                    print("No previous session found, starting new session")

        parent_session_id: str | None = None
        branch_point: int | None = None

        if config.system_prompt:
            system_prompt = config.system_prompt
        elif config.environment:
            from ..prompt import build_system_prompt

            env_system_prompt = None
            if hasattr(config.environment, "get_system_prompt"):
                env_system_prompt = config.environment.get_system_prompt()

            system_prompt = build_system_prompt(
                env_name=config.env,
                tools=config.environment.get_tools(),
                cwd=config.working_dir,
                env_system_prompt=env_system_prompt,
            )
        else:
            system_prompt = SYSTEM_PROMPTS.get(config.env, SYSTEM_PROMPTS["none"])

        if session_id and session_store:
            try:
                assert config.endpoint is not None
                state = await resume_session(
                    session_id, session_store, config.endpoint, config.environment
                )
                trajectory = state.actor.trajectory

                parent_session, _ = await session_store.get(session_id)
                if parent_session:
                    current_env_type = (
                        type(config.environment).__name__ if config.environment else "none"
                    )
                    parent_env_type = (
                        parent_session.environment_config().type
                        if parent_session.environment
                        else "none"
                    )
                    parent_confirm_tools = (
                        parent_session.environment_config().config.get("confirm_tools", False)
                        if parent_session.environment
                        else False
                    )

                    if config.driver == "claude":
                        config_differs = False
                    else:
                        config_differs = (
                            config.endpoint.model != parent_session.endpoint.model
                            or config.endpoint.provider != parent_session.endpoint.provider
                            or current_env_type != parent_env_type
                            or config.confirm_tools != parent_confirm_tools
                        )

                    if config_differs:
                        parent_session_id = session_id
                        branch_point = len(trajectory.messages)
                        session_id = None
                        print(f"Forking from session: {parent_session_id}")
                        print(
                            f"  Config changed: model={config.endpoint.model}, env={current_env_type}"
                        )
                        print(f"  Branch point: {branch_point} messages")
                    else:
                        print(f"Resuming session: {parent_session.session_id}")
                        print(f"  {len(trajectory.messages)} messages")
                else:
                    print(f"Resuming session: {session_id}")
                    print(f"  {len(trajectory.messages)} messages")
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

            if not trajectory.messages or trajectory.messages[0].role != "system":
                trajectory = Trajectory(
                    messages=[Message(role="system", content=system_prompt)]
                    + list(trajectory.messages)
                )
        else:
            trajectory = Trajectory(messages=[Message(role="system", content=system_prompt)])

        bootstrap_input = config.bootstrap_input
        if bootstrap_input is None and not sys.stdin.isatty():
            bootstrap_input = sys.stdin.read().strip() or None

        if config.print_mode is not None:
            return await _run_print_mode(config, trajectory, session_id, bootstrap_input)

        return await _run_interactive_mode(
            config, trajectory, session_id, parent_session_id, branch_point, bootstrap_input
        )
    finally:
        await cleanup_environment(config.environment)


async def _run_print_mode(
    config: CLIConfig,
    trajectory: Trajectory,
    session_id: str | None,
    bootstrap_input: str | None,
) -> int:
    assert config.endpoint is not None
    return await run_print_mode(
        _build_chat_run_request(config),
        trajectory=trajectory,
        session_id=session_id,
        bootstrap_input=bootstrap_input,
        query=config.print_mode,
    )


async def _run_interactive_mode(
    config: CLIConfig,
    trajectory: Trajectory,
    session_id: str | None,
    parent_session_id: str | None,
    branch_point: int | None,
    bootstrap_input: str | None,
) -> int:
    assert config.endpoint is not None
    return await run_interactive_mode(
        _build_chat_run_request(config),
        trajectory=trajectory,
        session_id=session_id,
        parent_session_id=parent_session_id,
        branch_point=branch_point,
        bootstrap_input=bootstrap_input,
    )


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "auth":
        return auth_main(sys.argv[2:])

    if len(sys.argv) > 1 and sys.argv[1] == "webui":
        if "--dev" in sys.argv[2:]:
            from ..frontend.dev import main as frontend_dev_main

            forwarded = [arg for arg in sys.argv[2:] if arg != "--dev"]
            return frontend_dev_main(forwarded)

        from ..frontend.server import main as frontend_server_main

        sys.argv = [f"{sys.argv[0]} webui", *sys.argv[2:]]
        frontend_server_main()
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        print("The 'agent' subcommand has been replaced with --driver:", file=sys.stderr)
        print("  rollouts --driver claude    # Start with Claude Code", file=sys.stderr)
        print("  rollouts --driver codex     # Start with Codex", file=sys.stderr)
        print("  rollouts                    # Start with SDK (default)", file=sys.stderr)
        print()
        print("You can also swap mid-session with /swap claude or /swap rollouts", file=sys.stderr)
        return 1

    from dotenv import load_dotenv

    load_dotenv()

    parser = create_parser()
    args = parser.parse_args()

    if args.debug or args.log_file:
        from .._logging import setup_logging

        setup_logging(
            level="DEBUG" if args.debug else "INFO",
            log_file=args.log_file,
            use_color=True,
            logger_levels={
                "httpx": "WARNING",
                "httpcore": "WARNING",
                "anthropic": "DEBUG" if args.debug else "WARNING",
            },
        )

    context = args.context
    if args.context_file:
        try:
            context = Path(args.context_file).read_text()
        except Exception as e:
            print(f"Error reading context file: {e}", file=sys.stderr)
            return 1

    template_args: dict[str, str] | None = None
    if args.args:
        template_args = {}
        for arg in args.args:
            if "=" not in arg:
                print(f"Invalid --args format: {arg!r}. Use KEY=VALUE", file=sys.stderr)
                return 1
            key, value = arg.split("=", 1)
            template_args[key] = value

    config = CLIConfig(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        thinking=args.thinking,
        env=args.env,
        tools=args.tools,
        cwd=args.cwd,
        confirm_tools=args.confirm_tools,
        context=context,
        continue_session=args.continue_session,
        session=args.session,
        no_session=args.no_session,
        tbench_surface=args.tbench_surface,
        print_mode=args.print_mode,
        stream_json=args.stream_json,
        quiet=args.quiet,
        frontend=args.frontend,
        theme=args.theme,
        debug=args.debug,
        debug_layout=args.debug_layout,
        log_file=args.log_file,
        driver=args.driver,
        cursor_api_key=getattr(args, "cursor_api_key", None),
        preset=args.preset,
        system_prompt=args.system_prompt,
        pick=args.pick,
        list_models=args.list_models,
        sync_models=args.sync_models,
        write_models=args.write,
        list_presets=args.list_presets,
        login_claude=args.login_claude,
        logout_claude=args.logout_claude,
        list_claude_profiles=args.list_claude_profiles,
        set_default_profile=args.set_default_profile,
        profile=args.profile,
        export_md=args.export_md,
        export_html=args.export_html,
        handoff=args.handoff,
        fast_handoff=args.fast,
        slice=args.slice,
        slice_goal=args.slice_goal,
        doctor=args.doctor,
        trim=args.trim,
        fix=args.fix,
        attach=args.attach,
        status=args.status,
        ls=args.ls,
        ls_all=args.ls_all,
        detached=args.detached,
        template=args.template,
        template_args=template_args,
        interactive=args.interactive,
        list_templates=args.list_templates,
        tbench_task_id=args.tbench_task_id,
        tbench_dataset_name=args.tbench_dataset_name,
        tbench_dataset_version=args.tbench_dataset_version,
        tbench_logging_dir=args.tbench_logging_dir,
        tbench_rebuild=args.tbench_rebuild,
        tbench_no_cleanup=args.tbench_no_cleanup,
    )

    if config.list_models is not None:
        return cmd_list_models(config.list_models or None)
    if config.sync_models:
        return cmd_sync_models(write=config.write_models)
    if config.list_presets:
        return cmd_list_presets()
    if config.list_templates:
        return cmd_list_templates()

    import os

    profile = config.profile or os.environ.get("ROLLOUTS_PROFILE", "default")

    if config.list_claude_profiles:
        return cmd_list_profiles()
    if config.set_default_profile is not None:
        return cmd_set_default_profile(config.set_default_profile)
    if config.login_claude or config.logout_claude:
        return cmd_oauth(login=config.login_claude, profile=profile)
    if config.export_md is not None or config.export_html is not None:
        return cmd_export(config, FileSessionStore())
    if config.doctor or config.trim is not None or config.fix:
        return cmd_doctor(config, FileSessionStore())

    if config.ls or config.ls_all:
        return cmd_ls(FileSessionStore(), include_all=config.ls_all)
    if config.status is not None:
        if config.status == "":
            return cmd_ls(FileSessionStore(), include_all=False)
        return cmd_status(FileSessionStore(), config.status)
    if config.attach:
        result = cmd_attach(config, config.attach)
        if result != -1:
            return result

    if not apply_preset(config):
        return 1
    if not apply_template(config):
        return 1
    if not apply_session_config(config):
        return 1

    config.working_dir = Path(config.cwd) if config.cwd else Path.cwd()

    try:
        config.endpoint = create_endpoint(
            config.model,
            config.api_base,
            config.api_key,
            config.thinking,
            config.quiet,
            profile,
            driver=config.driver,
        )
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    if config.driver == "sdk" and not config.endpoint.api_key and not config.endpoint.oauth_token:
        provider = config.endpoint.provider
        print(f"❌ No API key found for {provider}.", file=sys.stderr)
        print(f"   Run: rollouts auth login {provider}", file=sys.stderr)
        return 1

    if config.handoff:
        return cmd_handoff(config, FileSessionStore())
    if config.slice:
        return cmd_slice(config, FileSessionStore())

    environment, ok = create_environment(config)
    if not ok:
        return 1
    config.environment = environment

    config.session_store = create_session_store(
        no_session=config.no_session,
        env_name=config.env,
        environment=config.environment,
    )

    try:
        return trio.run(run_agent, config)
    except BaseException as e:
        from ..providers.base import AuthenticationError

        if isinstance(e, AuthenticationError):
            print(f"\n❌ {e}", file=sys.stderr)
            return 1
        if hasattr(e, "exceptions"):
            auth_errors = [exc for exc in e.exceptions if isinstance(exc, AuthenticationError)]
            if auth_errors:
                print(f"\n❌ {auth_errors[0]}", file=sys.stderr)
                return 1
        print(f"\n\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
