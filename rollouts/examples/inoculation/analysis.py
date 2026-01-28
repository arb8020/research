"""Score aggregation and confidence interval computation.

Takes raw eval results, groups by experimental condition,
computes means and CIs, saves CSVs.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from rollouts.training.types import Sample


@dataclass(frozen=True)
class GroupResult:
    """Aggregated result for one experimental group."""

    group_name: str
    eval_name: str
    mean: float
    ci_lower: float
    ci_upper: float
    n: int


def compute_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Compute mean and confidence interval using t-distribution.

    Args:
        values: Sample values
        confidence: Confidence level (default 0.95)

    Returns:
        (mean, ci_lower, ci_upper)
    """
    n = len(values)
    assert n > 0, "Cannot compute CI for empty list"

    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std_err = math.sqrt(variance / n)

    # t-distribution critical value approximation for 95% CI
    # For small n, use scipy if available, otherwise approximate
    try:
        from scipy.stats import t

        t_crit = t.ppf((1 + confidence) / 2, df=n - 1)
    except ImportError:
        # Rough approximation for small n
        t_crit = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}.get(n - 1, 1.96)

    margin = t_crit * std_err
    return mean, mean - margin, mean + margin


def aggregate_results(
    results_by_group: dict[str, list[Sample]],
    eval_name: str,
    metric_name: str = "misaligned",
) -> list[GroupResult]:
    """Aggregate evaluation results by experimental group.

    For each group, computes the mean score and CI across samples.
    If samples have seeds in metadata, computes CI over seed-level means.

    Args:
        results_by_group: {group_name: [Sample, ...]}
        eval_name: Name of the evaluation task
        metric_name: Which metric to aggregate

    Returns:
        List of GroupResult objects
    """
    group_results = []

    for group_name, samples in results_by_group.items():
        # Extract metric values
        values = []
        for s in samples:
            if s.score is None:
                continue
            for m in s.score.metrics:
                if m.name == metric_name:
                    values.append(m.value)
                    break

        if not values:
            continue

        # Check if we have seed-level grouping
        seeds: dict[int, list[float]] = {}
        for s, v in zip(samples, values, strict=False):
            seed = s.metadata.get("seed", 0)
            seeds.setdefault(seed, []).append(v)

        if len(seeds) > 1:
            # Compute CI over seed-level means
            seed_means = [sum(vs) / len(vs) for vs in seeds.values()]
            mean, ci_lower, ci_upper = compute_ci(seed_means)
        else:
            mean, ci_lower, ci_upper = compute_ci(values)

        group_results.append(
            GroupResult(
                group_name=group_name,
                eval_name=eval_name,
                mean=mean,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n=len(values),
            )
        )

    return group_results


def save_results_csv(
    group_results: list[GroupResult],
    output_path: Path,
) -> None:
    """Save aggregated results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "eval", "mean", "ci_lower", "ci_upper", "n"])
        for r in group_results:
            writer.writerow([
                r.group_name,
                r.eval_name,
                f"{r.mean:.4f}",
                f"{r.ci_lower:.4f}",
                f"{r.ci_upper:.4f}",
                r.n,
            ])

    print(f"Saved results to {output_path}")
