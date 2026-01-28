"""End-to-end experiment runner.

Orchestrates: dataset prep → training → deploy → evaluate → analyze.

Usage:
    python -m examples.inoculation.run_experiment --experiment ablations
"""

import argparse
import sys

import trio

from rollouts.dtypes import Endpoint
from rollouts.evaluation import evaluate

from .analysis import aggregate_results, save_results_csv
from .config import ExperimentConfig
from .train import train_all


async def run_experiment(experiment: ExperimentConfig) -> None:
    """Run full experiment pipeline."""

    print(f"=== Experiment: {experiment.name} ===")
    print(f"Base model: {experiment.base_model}")
    print(f"Conditions: {len(experiment.conditions)}")
    print(f"Seeds: {experiment.seeds}")
    print()

    # 1. Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(experiment.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Train all conditions
    print("--- Training ---")
    checkpoints = await train_all(experiment, tokenizer)
    print()

    # 3. Evaluate each trained model
    print("--- Evaluation ---")
    from .evals.emergent_misalignment import make_emergent_misalignment_eval_config

    all_results: dict[str, list] = {}

    for group_name, seed_checkpoints in checkpoints.items():
        all_results[group_name] = []

        for seed, ckpt_dir in seed_checkpoints.items():
            print(f"Evaluating {group_name} seed={seed}")

            # Create endpoint pointing to the trained model
            # For local models, this would be an SGLang endpoint.
            # For now, assumes the model is served somewhere.
            endpoint = Endpoint(
                provider="sglang",
                model=str(ckpt_dir),
                api_base="http://localhost:30000/v1",
                temperature=experiment.eval.temperature,
            )

            eval_config, dataset = make_emergent_misalignment_eval_config(
                endpoint=endpoint,
                n_samples=experiment.eval.n_samples_per_prompt,
                output_dir=experiment.results_dir / group_name / f"seed{seed}",
            )

            report = await evaluate(iter(dataset), eval_config)

            for sample in report.sample_results:
                sample.metadata["seed"] = seed
                sample.metadata["group"] = group_name

            all_results[group_name].extend(report.sample_results)

    # 4. Analyze
    print()
    print("--- Analysis ---")
    group_results = aggregate_results(
        all_results,
        eval_name="emergent-misalignment",
        metric_name="misaligned",
    )

    save_results_csv(group_results, experiment.results_dir / "summary.csv")

    print()
    print("Results:")
    print(f"{'Group':<25} {'Mean':>8} {'CI':>20} {'N':>6}")
    print("-" * 65)
    for r in sorted(group_results, key=lambda x: x.mean, reverse=True):
        ci_str = f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]"
        print(f"{r.group_name:<25} {r.mean:>8.3f} {ci_str:>20} {r.n:>6}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inoculation experiment")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["ablations", "backdoored"],
        help="Which experiment to run",
    )
    args = parser.parse_args()

    if args.experiment == "ablations":
        from .experiments.ablations import make_ablations_experiment

        experiment = make_ablations_experiment()
    elif args.experiment == "backdoored":
        from .experiments.backdoored import make_backdoored_experiment

        experiment = make_backdoored_experiment()
    else:
        print(f"Unknown experiment: {args.experiment}")
        return 1

    trio.run(run_experiment, experiment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
