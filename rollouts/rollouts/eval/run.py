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
    - endpoint: EndpointConfig (required)
    - run: EvalRunConfig (optional, defaults provided)
    - output: EvalOutputConfig (optional, defaults provided)
    - hardware: HardwareConfig (optional, for SGLang provisioning)
    - server: InferenceServerConfig (optional, for SGLang settings)

    - tasks: list[dict] OR tasks_path: Path (one required)
    - prepare_messages: Callable[[dict], list[Message]]
    - score_fn: Callable[[Sample], Score] or sample_scorer
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

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent


def load_config_module(config_path: Path) -> Any:
    """Load a config module from path."""
    spec = importlib.util.spec_from_file_location("_eval_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_eval_config"] = module
    spec.loader.exec_module(module)
    return module


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
    from rollouts.agents.handlers import handle_stop_max_turns
    from rollouts.core import Endpoint, EvalConfig
    from rollouts.eval import evaluate

    # Build endpoint
    api_key = endpoint_config.api_key or get_api_key(endpoint_config.provider)
    if not api_key and endpoint_config.provider in ("anthropic", "openai", "google"):
        raise ValueError(
            f"No API key found for {endpoint_config.provider}. "
            f"Set {endpoint_config.provider.upper()}_API_KEY in environment."
        )

    endpoint = Endpoint(
        model=f"{endpoint_config.provider}/{endpoint_config.model}",
        base_url=endpoint_config.get_base_url(),
        api_format=endpoint_config.get_api_format(),
        api_key=api_key,
        temperature=endpoint_config.temperature,
        max_tokens=endpoint_config.max_tokens,
    )

    # Load tasks
    if hasattr(config_module, "tasks"):
        tasks = config_module.tasks
    elif hasattr(config_module, "tasks_path"):
        import json

        tasks_path = Path(config_module.tasks_path)
        data = json.loads(tasks_path.read_text())
        tasks = data if isinstance(data, list) else data.get("tasks", [])
    else:
        raise ValueError("Config must define 'tasks' or 'tasks_path'")

    if run_config.max_samples:
        tasks = tasks[: run_config.max_samples]

    logger.info(f"Loaded {len(tasks)} tasks")

    # Get eval functions from config
    prepare_messages = config_module.prepare_messages
    score_fn = getattr(config_module, "score_fn", None)
    sample_scorer = getattr(config_module, "sample_scorer", None)
    if score_fn is None and sample_scorer is None:
        raise ValueError("Config must define 'score_fn' or 'sample_scorer'")

    # Environment (optional)
    environment: Environment | None = None
    environment_factory = None
    if hasattr(config_module, "make_environment"):
        make_env = config_module.make_environment
        if (
            hasattr(config_module, "per_sample_environment")
            and config_module.per_sample_environment
        ):
            environment_factory = make_env
        else:
            environment = make_env()

    # Build agent run config
    async def silent_on_chunk(_: object) -> None:
        pass

    from rollouts.agents import AgentState, StopReason

    async def stop_on_no_tool(state: AgentState, _run_config: AgentRunConfig) -> AgentState:
        return replace(state, stop=StopReason.TASK_COMPLETED)

    agent_run_config = AgentRunConfig(
        on_chunk=silent_on_chunk,
        handle_stop=handle_stop_max_turns(run_config.max_turns),
        handle_no_tool=stop_on_no_tool,
    )

    # Output directory
    output_dir = output_config.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("results") / f"{output_config.experiment_name}_{timestamp}"

    # Build EvalConfig
    eval_config = EvalConfig(
        endpoint=endpoint,
        score_fn=score_fn,
        sample_scorer=sample_scorer,
        prepare_messages=prepare_messages,
        environment=environment,
        environment_factory=environment_factory,
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

    parser.add_argument("--config", type=Path, required=True, help="Config file path")

    # Endpoint overrides
    parser.add_argument("--provider", choices=["anthropic", "openai", "google", "sglang", "vllm"])
    parser.add_argument("--model", help="Override model")
    parser.add_argument("--base-url", help="Override base URL (for SGLang/vLLM)")

    # Run overrides
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--max-concurrent", type=int, help="Max parallel samples")
    parser.add_argument("--max-turns", type=int, help="Max conversation turns")

    # Hardware (for SGLang provisioning)
    parser.add_argument("--provision", action="store_true", help="Provision GPU for SGLang")
    parser.add_argument("--gpu-type", default="A100", help="GPU type for provisioning")
    parser.add_argument("--hardware-provider", choices=["modal", "runpod", "lambdalabs", "vast"])

    # Output
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    config_module = load_config_module(config_path)

    # Get configs with defaults
    endpoint_config = getattr(config_module, "endpoint", EndpointConfig())
    run_config = getattr(config_module, "run", EvalRunConfig())
    output_config = getattr(config_module, "output", EvalOutputConfig())
    hardware_config = getattr(config_module, "hardware", None)
    server_config = getattr(config_module, "server", InferenceServerConfig())

    # Apply CLI overrides
    if args.provider:
        endpoint_config = replace(endpoint_config, provider=args.provider)
    if args.model:
        endpoint_config = replace(endpoint_config, model=args.model)
    if args.base_url:
        endpoint_config = replace(endpoint_config, base_url=args.base_url)

    if args.limit:
        run_config = replace(run_config, max_samples=args.limit)
    if args.max_concurrent:
        run_config = replace(run_config, max_concurrent=args.max_concurrent)
    if args.max_turns:
        run_config = replace(run_config, max_turns=args.max_turns)

    if args.output_dir:
        output_config = replace(output_config, output_dir=args.output_dir)

    if args.provision or args.hardware_provider:
        if hardware_config is None:
            hardware_config = HardwareConfig()
        if args.gpu_type:
            hardware_config = replace(hardware_config, gpu_type=args.gpu_type)
        if args.hardware_provider:
            hardware_config = replace(hardware_config, provider=args.hardware_provider)

    # Print config
    print(f"Config: {config_path}")
    print(f"Endpoint: {endpoint_config.provider}/{endpoint_config.model}")
    if endpoint_config.base_url:
        print(f"Base URL: {endpoint_config.base_url}")
    print(f"Max concurrent: {run_config.max_concurrent}")

    # Dispatch based on endpoint type
    async def _run() -> dict[str, Any]:
        if endpoint_config.provider in ("sglang", "vllm"):
            if endpoint_config.base_url:
                # Connect to existing server
                return await run_with_sglang_local(
                    config_module, endpoint_config, run_config, output_config
                )
            elif args.provision or hardware_config:
                # Provision and launch server
                return await run_with_sglang_provision(
                    config_module,
                    endpoint_config,
                    run_config,
                    output_config,
                    hardware_config,
                    server_config,
                )
            else:
                # Assume local server on default port
                endpoint_with_url = replace(
                    endpoint_config,
                    base_url=endpoint_config.get_base_url(),
                )
                return await run_with_sglang_local(
                    config_module, endpoint_with_url, run_config, output_config
                )
        else:
            # API endpoint
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
