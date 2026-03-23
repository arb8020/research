#!/usr/bin/env python3
"""Unified evaluation runner.

Runs evals against API endpoints or SGLang/vLLM servers.

Usage:
    # Against API (reads ANTHROPIC_API_KEY from env)
    python -m rollouts.eval.run --config examples/eval/reverse_text/smoke.py

    # Against local SGLang server (must be running)
    python -m rollouts.eval.run --config examples/eval/reverse_text/sglang.py

    # Launch one sample interactively in an external runtime
    python -m rollouts.eval.run launch --config examples/eval/reverse_text/smoke.py --sample 0 --runtime codex

Config files should export:
    - eval_task: EvalTaskSpec (preferred)

Legacy shape still supported:
    - endpoint: EndpointConfig (optional when attempt_executor supplies execution)
    - run: EvalRunConfig (optional, defaults provided)
    - output: EvalOutputConfig (optional, defaults provided)
    - hardware: HardwareConfig (optional, for SGLang provisioning)
    - server: InferenceServerConfig (optional, for SGLang settings)

    - tasks: list[dict] OR tasks_path: Path (one required)
    - prepare_messages: Callable[[dict], list[Message]]
    - attempt_executor: Callable[[dict, str, Environment | None, RunConfig], AttemptResult] (optional)
    - scorer: explicit scoring stage over AttemptResult
    - make_environment: Callable[[], Environment] (optional)
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import signal
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

import trio_asyncio

if TYPE_CHECKING:
    from rollouts.core import Environment

from ..config_contracts import validate_eval_config_module

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent


def _find_config_project_root(config_path: Path) -> Path:
    search_roots = [config_path.parent, *config_path.parents]
    for candidate in search_roots:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return config_path.parent


def _resolve_output_dir(
    *,
    config_path: Path,
    output_config: Any,
) -> Path:
    project_root = _find_config_project_root(config_path)
    configured_output_dir = output_config.output_dir
    if configured_output_dir is not None:
        if configured_output_dir.is_absolute():
            return configured_output_dir
        return (project_root / configured_output_dir).resolve()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return project_root / "results" / f"{output_config.experiment_name}_{timestamp}"


def _lower_eval_stop_handler(stop_handler: Any) -> Any:
    from rollouts.agents import (
        handle_stop_cost_budget,
        handle_stop_max_turns,
        handle_stop_token_budget,
        handle_stop_wall_clock_budget,
    )
    from rollouts.eval.configs import CostBudgetStop, MaxTurnsStop, TokenBudgetStop, WallClockStop

    if callable(stop_handler):
        return stop_handler
    if isinstance(stop_handler, MaxTurnsStop):
        return handle_stop_max_turns(stop_handler.max_turns)
    if isinstance(stop_handler, TokenBudgetStop):
        return handle_stop_token_budget(stop_handler.max_tokens)
    if isinstance(stop_handler, CostBudgetStop):
        return handle_stop_cost_budget(stop_handler.max_cost_usd)
    if isinstance(stop_handler, WallClockStop):
        return handle_stop_wall_clock_budget(stop_handler.max_seconds)
    raise ValueError(f"Unsupported eval stop handler: {stop_handler!r}")


def _build_stop_handler(run_config: Any) -> Any:
    return _lower_eval_stop_handler(run_config.resolved_stop_handler())


def load_config_module(config_path: Path) -> Any:
    """Load a config module from path."""
    spec = importlib.util.spec_from_file_location("_eval_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_eval_config"] = module
    spec.loader.exec_module(module)
    return module


def load_tasks_from_module(config_module: Any) -> list[dict[str, Any]]:
    """Load task rows from an eval config module."""
    from .configs import resolve_eval_task_spec

    eval_task = resolve_eval_task_spec(config_module)
    if eval_task.tasks is not None:
        tasks = eval_task.tasks
    elif eval_task.tasks_path is not None:
        import json

        tasks_path = eval_task.tasks_path
        data = json.loads(tasks_path.read_text())
        tasks = data if isinstance(data, list) else data.get("tasks", [])
    else:
        raise ValueError("Config must define 'tasks' or 'tasks_path'")

    if not isinstance(tasks, list):
        raise ValueError("Eval config tasks must be a list")
    return tasks


async def run_with_api(
    config_module: Any,
    endpoint_config: Any,
    run_config: Any,
    output_config: Any,
    cancel_scope: Any | None = None,
) -> dict[str, Any]:
    """Run eval against an API endpoint."""
    from rollouts.agents import RunConfig as AgentRunConfig
    from rollouts.core import EvalConfig
    from rollouts.eval import evaluate
    from rollouts.eval.configs import materialize_endpoint, resolve_eval_task_spec

    eval_task = resolve_eval_task_spec(config_module)

    endpoint = materialize_endpoint(endpoint_config) if endpoint_config is not None else None

    # Load tasks
    tasks = load_tasks_from_module(config_module)

    if run_config.max_samples:
        tasks = tasks[: run_config.max_samples]

    logger.info(f"Loaded {len(tasks)} tasks")

    run_spec = eval_task.run_spec
    prepare_messages = run_spec.prepare_messages
    attempt_executor = run_spec.attempt_executor
    scorer = eval_task.scorer
    environment: Environment | None = run_spec.environment
    environment_factory = run_spec.environment_factory

    # Build agent run config
    async def silent_on_chunk(_: object) -> None:
        pass

    from rollouts.agents import AgentState, StopReason

    async def stop_on_no_tool(state: AgentState, _run_config: AgentRunConfig) -> AgentState:
        return replace(state, stop=StopReason.TASK_COMPLETED)

    handle_stop = _build_stop_handler(run_config)
    if run_spec is not None and run_spec.stop_handler is not None:
        handle_stop = _lower_eval_stop_handler(run_spec.stop_handler)

    handle_no_tool = (
        run_spec.handle_no_tool
        if run_spec is not None and run_spec.handle_no_tool is not None
        else stop_on_no_tool
    )

    agent_run_config = AgentRunConfig(
        on_chunk=silent_on_chunk,
        handle_stop=handle_stop,
        handle_no_tool=handle_no_tool,
        cancel_scope=cancel_scope,
    )

    output_dir = output_config.output_dir
    assert output_dir is not None, "output_dir should be resolved before execution"

    # Build EvalConfig
    eval_config = EvalConfig(
        endpoint=endpoint,
        scorer=scorer,
        prepare_messages=prepare_messages,
        environment=environment,
        environment_factory=environment_factory,
        attempt_executor=attempt_executor,
        run_config=agent_run_config,
        max_samples=len(tasks),
        max_concurrent=run_config.max_concurrent,
        max_api_concurrent=run_config.max_api_concurrent,
        verbose=run_config.verbose,
        output_dir=output_dir,
        eval_name=output_config.experiment_name,
        show_progress=run_config.show_progress,
    )

    # Run evaluation
    report = await evaluate(iter(tasks), eval_config)

    return {
        "total": report.total_samples,
        **report.summary_metrics,
    }


async def run_with_sglang_local(
    config_module: Any,
    endpoint_config: Any,
    run_config: Any,
    output_config: Any,
    cancel_scope: Any | None = None,
) -> dict[str, Any]:
    """Run eval against a local SGLang server (already running)."""
    # Same as API but with SGLang endpoint
    return await run_with_api(
        config_module,
        endpoint_config,
        run_config,
        output_config,
        cancel_scope=cancel_scope,
    )


async def run_with_sglang_provision(
    config_module: Any,
    endpoint_config: Any,
    run_config: Any,
    output_config: Any,
    hardware_config: Any,
    server_config: Any,
) -> dict[str, Any]:
    """Provision GPU, launch SGLang server, run eval, cleanup."""
    # This would use bifrost to provision, similar to rollouts/run.py
    # For now, raise NotImplementedError
    raise NotImplementedError(
        "SGLang provisioning not yet implemented. "
        "Start SGLang server manually and set base_url in endpoint config."
    )


def main() -> int:
    import trio

    from .configs import (
        EndpointConfig,
        resolve_eval_task_spec,
    )

    class _EvalInterrupted(Exception):
        pass

    parser = argparse.ArgumentParser(
        description="Run evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Against Anthropic API
    python -m rollouts.eval.run --config examples/eval/reverse_text/smoke.py

    # Against local SGLang server
    python -m rollouts.eval.run --config examples/eval/reverse_text/sglang.py

    # Launch one sample interactively in Codex
    python -m rollouts.eval.run launch --config examples/eval/reverse_text/smoke.py --sample 0 --runtime codex
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run batch evaluation")
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch one eval sample in an external runtime",
    )

    for subparser in (run_parser, launch_parser):
        subparser.add_argument("--config", type=Path, required=True, help="Config file path")

    # Batch run flags
    run_parser.add_argument("--quiet", action="store_true")

    # Launch flags
    launch_parser.add_argument(
        "--sample",
        required=True,
        help="Sample selector: zero-based index or id/problem_id/name",
    )
    launch_parser.add_argument(
        "--runtime",
        required=True,
        choices=["claude_code", "codex"],
        help="External runtime to launch",
    )
    launch_parser.add_argument("--quiet", action="store_true")

    argv = sys.argv[1:]
    if argv and argv[0].startswith("-"):
        argv = ["run", *argv]
    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "run"

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    config_module = load_config_module(config_path)
    try:
        validate_eval_config_module(config_module, config_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    eval_task = resolve_eval_task_spec(config_module)

    # Get configs with defaults
    endpoint_config = eval_task.run_spec.endpoint
    if endpoint_config is None and eval_task.run_spec.attempt_executor is None:
        endpoint_config = EndpointConfig()
    run_config = eval_task.run
    output_config = eval_task.output
    hardware_config = eval_task.hardware
    server_config = eval_task.server

    resolved_output_dir = _resolve_output_dir(
        config_path=config_path,
        output_config=output_config,
    )
    output_config = replace(output_config, output_dir=resolved_output_dir)

    print(f"Config: {config_path}")
    if args.command == "launch":
        print(f"Launch sample: {args.sample}")
        print(f"Runtime: {args.runtime}")
        print("Control mode: interactive")
    else:
        if endpoint_config is None:
            print("Endpoint: direct-attempt executor")
        else:
            print(f"Endpoint: {endpoint_config.provider}/{endpoint_config.model}")
            if endpoint_config.base_url:
                print(f"Base URL: {endpoint_config.base_url}")
        print(f"Max concurrent: {run_config.max_concurrent}")
        print(f"Output dir: {output_config.output_dir}")

    scope_holder: dict[str, trio.CancelScope | None] = {"scope": None}
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        active_scope = scope_holder["scope"]
        if active_scope is not None:
            logger.info("SIGINT received, cancelling evaluation...")
            active_scope.cancel()
            return
        raise KeyboardInterrupt

    async def _run() -> dict[str, Any]:
        with trio.CancelScope() as cancel_scope:
            scope_holder["scope"] = cancel_scope
            try:
                if args.command == "launch":
                    from .launch import launch_sample

                    attempt, session_id = await launch_sample(
                        config_module=config_module,
                        config_path=config_path,
                        sample_selector=args.sample,
                        runtime=args.runtime,
                        run_config=run_config,
                        output_config=output_config,
                    )
                    results: dict[str, Any] = {
                        "sample_id": attempt.id,
                        "session_id": session_id,
                        "runtime": args.runtime,
                        "control_mode": "interactive",
                        "message_count": len(attempt.trajectory.messages)
                        if attempt.trajectory
                        else 0,
                    }
                    if attempt.score is not None:
                        results["reward"] = attempt.reward
                    return results

                if endpoint_config is not None and endpoint_config.provider in ("sglang", "vllm"):
                    if endpoint_config.base_url:
                        return await run_with_sglang_local(
                            config_module,
                            endpoint_config,
                            run_config,
                            output_config,
                            cancel_scope=cancel_scope,
                        )
                    if hardware_config is not None:
                        return await run_with_sglang_provision(
                            config_module,
                            endpoint_config,
                            run_config,
                            output_config,
                            hardware_config,
                            server_config,
                        )

                    endpoint_with_url = replace(
                        endpoint_config,
                        base_url=endpoint_config.get_base_url(),
                    )
                    return await run_with_sglang_local(
                        config_module,
                        endpoint_with_url,
                        run_config,
                        output_config,
                        cancel_scope=cancel_scope,
                    )

                return await run_with_api(
                    config_module,
                    endpoint_config,
                    run_config,
                    output_config,
                    cancel_scope=cancel_scope,
                )
            finally:
                scope_holder["scope"] = None

        raise _EvalInterrupted

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        results = trio_asyncio.run(_run)
    except _EvalInterrupted:
        logger.info("Evaluation interrupted")
        return 130
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

    class _EvalInterrupted(Exception):
        pass
