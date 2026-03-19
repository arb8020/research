#!/usr/bin/env python3
"""Unified evaluation runner.

Runs evals against API endpoints or SGLang/vLLM servers.

Usage:
    # Against API (reads ANTHROPIC_API_KEY from env)
    python -m rollouts.eval.run --config examples/eval/reverse_text/smoke.py

    # Against local SGLang server (must be running)
    python -m rollouts.eval.run --config examples/eval/reverse_text/smoke.py

    # Provision GPU and launch SGLang server
    python -m rollouts.eval.run --config examples/eval/reverse_text/sglang.py --provider runpod
    python -m rollouts.eval.run --config examples/eval/reverse_text/sglang.py --provider modal

    # Override endpoint from CLI
    python -m rollouts.eval.run --config ... --model claude-opus-4-20250514
    python -m rollouts.eval.run --config ... --endpoint sglang --model Qwen/Qwen2.5-7B-Instruct

Config files should export:
    - endpoint: EndpointConfig (optional when attempt_executor supplies execution)
    - run: EvalRunConfig (optional, defaults provided)
    - output: EvalOutputConfig (optional, defaults provided)
    - hardware: HardwareConfig (optional, for SGLang provisioning)
    - server: InferenceServerConfig (optional, for SGLang settings)

    - tasks: list[dict] OR tasks_path: Path (one required)
    - prepare_messages: Callable[[dict], list[Message]]
    - attempt_executor: Callable[[dict, str, Environment | None, RunConfig], AttemptResult] (optional)
    - score_fn: Callable[[AttemptResult], Score] or sample_scorer
    - make_environment: Callable[[], Environment] (optional)
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    cli_output_dir: Path | None = None,
) -> Path:
    if cli_output_dir is not None:
        return (
            cli_output_dir.resolve()
            if cli_output_dir.is_absolute()
            else (Path.cwd() / cli_output_dir).resolve()
        )

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


def _resolve_endpoint_metadata(provider: str, model: str) -> tuple[str | None, str | None]:
    """Resolve provider/model against the local registry when available.

    Returns:
        (base_url, api_format), where either may be None if the provider/model
        should be treated as a custom runtime endpoint (for example sglang/vllm).
    """
    from difflib import get_close_matches
    from typing import cast

    from rollouts.fuzzy import fuzzy_filter
    from rollouts.models import MODELS, Provider, get_model

    if provider not in MODELS:
        return None, None

    provider_models = MODELS[cast("Provider", provider)]
    if not provider_models:
        return None, None

    metadata = get_model(cast("Provider", provider), model)
    if metadata is not None:
        return metadata.base_url, metadata.api

    model_ids = list(provider_models.keys())
    suggestions = fuzzy_filter(model_ids, model, lambda x: x)[:3]
    if not suggestions:
        suggestions = get_close_matches(model, model_ids, n=3, cutoff=0.5)
    error_msg = f"Model '{model}' not found for provider '{provider}'."
    if suggestions:
        error_msg += "\n\nDid you mean one of these?\n"
        for suggestion in suggestions:
            error_msg += f"  - {provider}/{suggestion}\n"
    error_msg += f"\nSee available models: rollouts --list-models {provider}"
    raise ValueError(error_msg)


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
    if hasattr(config_module, "tasks"):
        tasks = config_module.tasks
    elif hasattr(config_module, "tasks_path"):
        import json

        tasks_path = Path(config_module.tasks_path)
        data = json.loads(tasks_path.read_text())
        tasks = data if isinstance(data, list) else data.get("tasks", [])
    else:
        raise ValueError("Config must define 'tasks' or 'tasks_path'")

    if not isinstance(tasks, list):
        raise ValueError("Eval config tasks must be a list")
    return tasks


def get_api_key(provider: str) -> str:
    """Get API key from environment for provider."""
    env_vars = {
        "anthropic": ["ANTHROPIC_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    }

    for var in env_vars.get(provider, []):
        key = os.environ.get(var)
        if key:
            return key

    return ""


async def run_with_api(
    config_module: Any,
    endpoint_config: Any,
    run_config: Any,
    output_config: Any,
) -> dict[str, Any]:
    """Run eval against an API endpoint."""
    from rollouts.agents import RunConfig as AgentRunConfig
    from rollouts.core import Endpoint, EvalConfig
    from rollouts.eval import evaluate

    endpoint = None
    if endpoint_config is not None:
        api_key = endpoint_config.api_key or get_api_key(endpoint_config.provider)
        if not api_key and endpoint_config.provider in ("anthropic", "openai", "google"):
            raise ValueError(
                f"No API key found for {endpoint_config.provider}. "
                f"Set {endpoint_config.provider.upper()}_API_KEY in environment."
            )

        resolved_base_url, resolved_api_format = _resolve_endpoint_metadata(
            endpoint_config.provider,
            endpoint_config.model,
        )

        endpoint = Endpoint(
            model=f"{endpoint_config.provider}/{endpoint_config.model}",
            base_url=endpoint_config.base_url
            or resolved_base_url
            or endpoint_config.get_base_url(),
            api_format=resolved_api_format or endpoint_config.get_api_format(),
            api_key=api_key,
            temperature=endpoint_config.temperature,
            max_tokens=endpoint_config.max_tokens,
        )

    # Load tasks
    tasks = load_tasks_from_module(config_module)

    if run_config.max_samples:
        tasks = tasks[: run_config.max_samples]

    logger.info(f"Loaded {len(tasks)} tasks")

    run_spec = getattr(config_module, "run_spec", None)

    # Get eval functions from config
    prepare_messages = (
        run_spec.prepare_messages
        if run_spec is not None
        else getattr(config_module, "prepare_messages", None)
    )
    attempt_executor = (
        run_spec.attempt_executor
        if run_spec is not None
        else getattr(config_module, "attempt_executor", None)
    )
    score_fn = getattr(config_module, "score_fn", None)
    sample_scorer = getattr(config_module, "sample_scorer", None)

    # Environment (optional)
    environment: Environment | None = run_spec.environment if run_spec is not None else None
    environment_factory = run_spec.environment_factory if run_spec is not None else None
    if run_spec is None and hasattr(config_module, "make_environment"):
        make_env = config_module.make_environment
        if (
            hasattr(config_module, "per_sample_environment")
            and config_module.per_sample_environment
        ):
            environment_factory = make_env
        else:
            environment = make_env()

    if (
        score_fn is None
        and sample_scorer is None
        and environment is None
        and environment_factory is None
    ):
        raise ValueError(
            "Eval configs must define score_fn, sample_scorer, or an environment path that can own scoring"
        )

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
    )

    output_dir = output_config.output_dir
    assert output_dir is not None, "output_dir should be resolved before execution"

    # Build EvalConfig
    eval_config = EvalConfig(
        endpoint=endpoint,
        score_fn=score_fn,
        sample_scorer=sample_scorer,
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
) -> dict[str, Any]:
    """Run eval against a local SGLang server (already running)."""
    # Same as API but with SGLang endpoint
    return await run_with_api(config_module, endpoint_config, run_config, output_config)


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
        EvalOutputConfig,
        EvalRunConfig,
        HardwareConfig,
        InferenceServerConfig,
    )

    parser = argparse.ArgumentParser(
        description="Run evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Against Anthropic API
    python -m rollouts.eval.run --config examples/eval/reverse_text/smoke.py

    # Against local SGLang server
    python -m rollouts.eval.run --config examples/eval/reverse_text/sglang.py

    # Override model
    python -m rollouts.eval.run --config ... --model claude-opus-4-20250514
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
    run_parser.add_argument(
        "--provider", choices=["anthropic", "openai", "google", "sglang", "vllm"]
    )
    run_parser.add_argument("--model", help="Override model")
    run_parser.add_argument("--base-url", help="Override base URL (for SGLang/vLLM)")
    run_parser.add_argument("--limit", type=int, help="Limit number of samples")
    run_parser.add_argument("--max-concurrent", type=int, help="Max parallel samples")
    run_parser.add_argument("--max-turns", type=int, help="Max conversation turns")
    run_parser.add_argument("--provision", action="store_true", help="Provision GPU for SGLang")
    run_parser.add_argument("--gpu-type", default="A100", help="GPU type for provisioning")
    run_parser.add_argument(
        "--hardware-provider", choices=["modal", "runpod", "lambdalabs", "vast"]
    )
    run_parser.add_argument("--output-dir", type=Path, help="Override output directory")
    run_parser.add_argument("--verbose", action="store_true", default=True)
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
    launch_parser.add_argument(
        "--control-mode",
        default="autonomous",
        choices=["interactive", "autonomous"],
        help="Who controls the runtime session",
    )
    launch_parser.add_argument("--model", help="Override external runtime model")
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

    # Get configs with defaults
    run_spec = getattr(config_module, "run_spec", None)
    top_level_attempt_executor = getattr(config_module, "attempt_executor", None)
    if run_spec is not None:
        endpoint_config = run_spec.endpoint
    elif hasattr(config_module, "endpoint"):
        endpoint_config = config_module.endpoint
    elif top_level_attempt_executor is not None:
        endpoint_config = None
    else:
        endpoint_config = EndpointConfig()
    run_config = getattr(config_module, "run", EvalRunConfig())
    output_config = getattr(config_module, "output", EvalOutputConfig())
    hardware_config = getattr(config_module, "hardware", None)
    server_config = getattr(config_module, "server", InferenceServerConfig())

    # Apply CLI overrides
    if endpoint_config is None:
        if args.command == "run" and (
            args.provider or args.model or args.base_url or args.provision or args.hardware_provider
        ):
            raise ValueError(
                "Endpoint overrides and provisioning flags are invalid for attempt-executor-only evals."
            )
    else:
        if args.command == "run" and args.provider:
            endpoint_config = replace(endpoint_config, provider=args.provider)
        if args.command == "run" and args.model:
            endpoint_config = replace(endpoint_config, model=args.model)
        if args.command == "run" and args.base_url:
            endpoint_config = replace(endpoint_config, base_url=args.base_url)

    if args.command == "run" and args.limit:
        run_config = replace(run_config, max_samples=args.limit)
    if args.command == "run" and args.max_concurrent:
        run_config = replace(run_config, max_concurrent=args.max_concurrent)
    if args.command == "run" and args.max_turns:
        run_config = replace(run_config, max_turns=args.max_turns)

    resolved_output_dir = _resolve_output_dir(
        config_path=config_path,
        output_config=output_config,
        cli_output_dir=args.output_dir if args.command == "run" else None,
    )
    output_config = replace(output_config, output_dir=resolved_output_dir)

    if (
        args.command == "run"
        and endpoint_config is not None
        and (args.provision or args.hardware_provider)
    ):
        if hardware_config is None:
            hardware_config = HardwareConfig()
        if args.gpu_type:
            hardware_config = replace(hardware_config, gpu_type=args.gpu_type)
        if args.hardware_provider:
            hardware_config = replace(hardware_config, provider=args.hardware_provider)

    print(f"Config: {config_path}")
    if args.command == "launch":
        print(f"Launch sample: {args.sample}")
        print(f"Runtime: {args.runtime}")
        print(f"Control mode: {args.control_mode}")
    else:
        if endpoint_config is None:
            print("Endpoint: direct-attempt executor")
        else:
            print(f"Endpoint: {endpoint_config.provider}/{endpoint_config.model}")
            if endpoint_config.base_url:
                print(f"Base URL: {endpoint_config.base_url}")
        print(f"Max concurrent: {run_config.max_concurrent}")
        print(f"Output dir: {output_config.output_dir}")

    async def _run() -> dict[str, Any]:
        if args.command == "launch":
            from .launch import launch_sample

            attempt, session_id = await launch_sample(
                config_module=config_module,
                config_path=config_path,
                sample_selector=args.sample,
                runtime=args.runtime,
                control_mode=args.control_mode,
                run_config=run_config,
                output_config=output_config,
                model=args.model,
            )
            results: dict[str, Any] = {
                "sample_id": attempt.id,
                "session_id": session_id,
                "runtime": args.runtime,
                "control_mode": args.control_mode,
                "message_count": len(attempt.trajectory.messages) if attempt.trajectory else 0,
            }
            if attempt.score is not None:
                results["reward"] = attempt.reward
            return results

        if endpoint_config is not None and endpoint_config.provider in ("sglang", "vllm"):
            if endpoint_config.base_url:
                return await run_with_sglang_local(
                    config_module, endpoint_config, run_config, output_config
                )
            elif args.provision or hardware_config:
                return await run_with_sglang_provision(
                    config_module,
                    endpoint_config,
                    run_config,
                    output_config,
                    hardware_config,
                    server_config,
                )
            else:
                endpoint_with_url = replace(
                    endpoint_config,
                    base_url=endpoint_config.get_base_url(),
                )
                return await run_with_sglang_local(
                    config_module, endpoint_with_url, run_config, output_config
                )
        else:
            return await run_with_api(config_module, endpoint_config, run_config, output_config)

    try:
        results = trio.run(_run)
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1

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
