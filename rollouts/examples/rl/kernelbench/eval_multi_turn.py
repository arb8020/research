"""Evaluate multi-turn KernelBench via the shared eval pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, cast

import trio

from examples.rl.kernelbench.dataset import load_kernelbench_prompts
from rollouts.agents import RunConfig, handle_stop_max_turns
from rollouts.core import Endpoint, EvalConfig
from rollouts.credentials import get_api_key
from rollouts.eval import evaluate
from rollouts.models import MODELS

from .resources import KernelBenchRolloutResources, KernelBenchScoringResources
from .scoring import KEVIN_MULTI_TURN_REWARD_WEIGHTS


def _sample_to_result(sample: Any) -> dict[str, Any]:
    metadata = sample.metadata
    return {
        "problem_id": metadata.get("problem_id", sample.id),
        "name": metadata.get("name", metadata.get("problem_name", "unknown")),
        "level": metadata.get("level", "unknown"),
        "turns_used": metadata.get("turns_used", 0),
        "best_speedup": metadata.get("best_speedup", 0.0),
        "has_correct_kernel": metadata.get("has_correct_kernel", False),
        "turn_history": metadata.get("turn_history", []),
        "status": metadata.get("status", "unknown"),
        "reward": sample.reward,
        "error": metadata.get("error"),
    }


def _make_endpoint(
    *,
    model: str,
    provider: str,
    endpoint_url: str | None,
) -> Endpoint:
    if provider == "sglang" or endpoint_url:
        if not endpoint_url:
            endpoint_url = "http://localhost:30000/v1"
        return Endpoint(
            model=f"openai/{model}",
            base_url=endpoint_url,
            api_format="openai-completions",
            api_key="dummy",
            temperature=0.7,
            max_tokens=4096,
        )

    provider_models = cast("dict[str, Any]", MODELS)
    if provider not in provider_models:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(MODELS.keys())}")
    if model not in provider_models[provider]:
        available = list(provider_models[provider].keys())
        raise ValueError(f"Unknown model: {model} for provider {provider}. Available: {available}")

    model_meta = provider_models[provider][model]
    api_key = None
    if provider == "opencode":
        api_key = os.environ.get("OPENCODE_API_KEY")
    elif provider == "moonshot":
        api_key = os.environ.get("MOONSHOT_API_KEY")

    if not api_key:
        api_key = get_api_key(provider)
    if not api_key:
        raise ValueError(f"No API key found for {provider}. Set {provider.upper()}_API_KEY.")

    return Endpoint(
        model=f"{provider}/{model_meta.id}",
        base_url=model_meta.base_url,
        api_format=model_meta.api,
        api_key=api_key,
        temperature=0.7,
        max_tokens=min(4096, model_meta.max_tokens),
    )


async def evaluate_multi_turn(
    model: str = "kimi-k2.5",
    provider: str = "opencode",
    endpoint_url: str | None = None,
    num_problems: int = 5,
    levels: list[int] | None = None,
    max_turns: int = 8,
    backend: str = "cuda",
    output_file: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run multi-turn evaluation on KernelBench problems."""
    if levels is None:
        levels = [1]

    problems = load_kernelbench_prompts(
        levels=levels,
        max_samples=num_problems,
        backend=backend,
    )
    endpoint = _make_endpoint(model=model, provider=provider, endpoint_url=endpoint_url)
    rollout_resources = KernelBenchRolloutResources.from_sandbox_configs(
        backend=backend,
        max_turns=max_turns,
    )
    scoring_resources = KernelBenchScoringResources.metadata_only(
        reward_weights=KEVIN_MULTI_TURN_REWARD_WEIGHTS,
    )

    def prepare_messages(sample_data: dict[str, Any]) -> list[dict[str, Any]]:
        return sample_data["messages"]

    async def silent_handler(_: object) -> None:
        await trio.lowlevel.checkpoint()

    run_config = RunConfig(
        on_chunk=silent_handler,
        handle_stop=handle_stop_max_turns(max_turns),
        show_progress=verbose,
    )

    eval_config = EvalConfig(
        endpoint=endpoint,
        prepare_messages=prepare_messages,
        sample_scorer=scoring_resources.scorer,
        environment_factory=rollout_resources,
        max_samples=len(problems),
        max_concurrent=1,
        verbose=verbose,
        run_config=run_config,
        eval_name="kernelbench_multi_turn_eval",
        metadata={"model": model, "provider": provider, "levels": levels},
    )

    await rollout_resources.start()
    try:
        report = await evaluate(iter(problems), eval_config)
    finally:
        await rollout_resources.stop()

    results = [_sample_to_result(sample) for sample in report.sample_results]
    successful = [r for r in results if r.get("status") == "success"]
    correct = [r for r in successful if r.get("has_correct_kernel")]
    speedups = [r.get("best_speedup", 0.0) for r in correct]

    summary = {
        "total": len(results),
        "successful": len(successful),
        "correct": len(correct),
        "correct_rate": len(correct) / len(successful) if successful else 0.0,
        "avg_speedup": sum(speedups) / len(speedups) if speedups else 0.0,
        "max_speedup": max(speedups) if speedups else 0.0,
        "avg_turns": sum(r.get("turns_used", 0) for r in successful) / len(successful)
        if successful
        else 0.0,
        "summary_metrics": report.summary_metrics,
        "results": results,
    }

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate KernelBench multi-turn environment")
    parser.add_argument("--model", type=str, default="kimi-k2.5")
    parser.add_argument(
        "--provider",
        type=str,
        default="opencode",
        choices=["opencode", "moonshot", "sglang", "openai", "anthropic"],
    )
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--num-problems", type=int, default=5)
    parser.add_argument("--levels", type=int, nargs="+", default=[1])
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--backend", type=str, default="cuda", choices=["cuda", "hip"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        summary = asyncio.run(
            evaluate_multi_turn(
                model=args.model,
                provider=args.provider,
                endpoint_url=args.endpoint,
                num_problems=args.num_problems,
                levels=args.levels,
                max_turns=args.max_turns,
                backend=args.backend,
                output_file=args.output,
                verbose=not args.quiet,
            )
        )
        if summary.get("correct", 0) > 0:
            print("Evaluation successful")
            return 0
        print("Evaluation produced no correct kernels")
        return 1
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
