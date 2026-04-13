#!/usr/bin/env python3
"""Unified evaluation runner.

Runs evals against API endpoints or SGLang/vLLM servers.

Preferred usage (fire-and-forget, monitor via jsonl):
    argus run --config examples/eval/reverse_text/smoke.py
    argus run --config inference.eval_skeleton_server

    # Then monitor:
    tail -f results/eval/<run>/events.jsonl | jq .
    jq 'select(.message == "eval_end")' results/eval/<run>/events.jsonl

Direct usage (interactive, all output to terminal - useful for debugging):
    python -m rollouts.eval.run --config examples/eval/reverse_text/smoke.py
    python -m rollouts.eval.run --config examples/eval/reverse_text/sglang.py
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
    - attempt_executor: Callable[[dict, str, Environment | None, RunConfig], RowAttempt] (optional)
    - scorer: explicit scoring stage over RowAttempt
    - make_environment: Callable[[], Environment] (optional)
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
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
from .endpoint_realization import realize_worker_backed_endpoint

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent

_INFERENCE_EVALS_ROOT = REPO_ROOT / "examples" / "inference" / "evals" / "configs"


def _resolve_config_path(config: Path) -> Path:
    """Resolve --config to an absolute path.

    Accepts:
    - Absolute path
    - Relative path (resolved from REPO_ROOT)
    - Dotted inference config id: e.g. "eval_skeleton_server" or
      "inference.eval_skeleton_server" (both resolve under examples/inference/evals/configs/)
    """
    raw = str(config)

    # Absolute or explicitly relative path: take as-is
    if config.is_absolute() or raw.startswith("."):
        return config.resolve()

    # Dotted id: strip leading "inference." prefix if present, then look up in configs dir
    dotted = raw.removeprefix("inference.")
    candidate = (_INFERENCE_EVALS_ROOT / Path(*dotted.split("."))).with_suffix(".py")
    if candidate.is_file():
        return candidate

    # Fall back: resolve relative to REPO_ROOT
    return (REPO_ROOT / config).resolve()


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
    cli_output_dir: Path | None = None,
) -> Path:
    project_root = _find_config_project_root(config_path)
    explicit_output_dir = os.environ.get("ROLLOUTS_OUTPUT_DIR")
    if explicit_output_dir:
        return Path(explicit_output_dir)

    if cli_output_dir is not None:
        if cli_output_dir.is_absolute():
            return cli_output_dir
        return (project_root / cli_output_dir).resolve()

    configured_output_dir = output_config.output_dir
    if configured_output_dir is not None:
        if configured_output_dir.is_absolute():
            return configured_output_dir
        return (project_root / configured_output_dir).resolve()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return project_root / "results" / f"{output_config.experiment_name}_{timestamp}"


def _apply_endpoint_env_overrides(endpoint_config: Any) -> Any:
    if endpoint_config is None:
        return None

    base_url_override = os.environ.get("ROLLOUTS_ENDPOINT_BASE_URL")
    if not base_url_override:
        return endpoint_config
    from rollouts.eval.configs import ExternalEndpoint, OwnedEndpoint

    if isinstance(endpoint_config, OwnedEndpoint):
        return ExternalEndpoint(
            url=base_url_override,
            model=endpoint_config.model,
            provider=endpoint_config.provider,
            temperature=endpoint_config.temperature,
            max_tokens=endpoint_config.max_tokens,
            extra_params=endpoint_config.extra_params,
        )
    if isinstance(endpoint_config, ExternalEndpoint):
        return replace(endpoint_config, url=base_url_override)
    return replace(endpoint_config, base_url=base_url_override)


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
    """Realize a worker-backed endpoint locally, run eval, then tear it down."""
    worker = None
    worker_topology = getattr(config_module, "worker_topology", None)
    if worker_topology is not None:
        worker = worker_topology.get_worker_for_role("actor")

    async with realize_worker_backed_endpoint(
        endpoint_config=endpoint_config,
        output_dir=output_config.output_dir,
        hardware_config=hardware_config,
        server_config=server_config,
        worker=worker,
    ) as realized:
        return await run_with_api(
            config_module,
            realized.endpoint_config,
            run_config,
            output_config,
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
    # Preferred: fire-and-forget via argus (monitor via events.jsonl)
    argus run --config inference.eval_skeleton_server
    argus run --config examples/eval/reverse_text/smoke.py

    # Direct: interactive, all output to terminal (useful for debugging)
    python -m rollouts.eval.run --config examples/eval/reverse_text/smoke.py
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
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit output directory. ROLLOUTS_OUTPUT_DIR takes precedence when set.",
    )

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

    config_path = _resolve_config_path(args.config)

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
    endpoint_config = _apply_endpoint_env_overrides(endpoint_config)
    run_config = eval_task.run
    output_config = eval_task.output
    hardware_config = eval_task.hardware
    server_config = eval_task.server

    resolved_output_dir = _resolve_output_dir(
        config_path=config_path,
        output_config=output_config,
        cli_output_dir=getattr(args, "output_dir", None),
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
            if not endpoint_config.requires_server:
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
                    if not endpoint_config.requires_server:
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

                    from rollouts.eval.configs import ExternalEndpoint, OwnedEndpoint

                    if isinstance(endpoint_config, OwnedEndpoint):
                        endpoint_with_url = ExternalEndpoint(
                            url=endpoint_config.base_url,
                            model=endpoint_config.model,
                            provider=endpoint_config.provider,
                            temperature=endpoint_config.temperature,
                            max_tokens=endpoint_config.max_tokens,
                        )
                    else:
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
