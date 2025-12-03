#!/usr/bin/env python3
"""Generate perplexity analysis plots (replicating Dettmers Figure 3b).

Combines perplexity results with outlier analysis to plot:
- Perplexity vs % layers affected
- Perplexity vs % sequence positions affected
- Perplexity vs number of outlier dimensions

Usage:
    python generate_perplexity_plots.py                        # Save PNG plots
    python generate_perplexity_plots.py --terminal              # Display plots in terminal
    python generate_perplexity_plots.py --terminal --format csv    # Output as CSV
    python generate_perplexity_plots.py --terminal --format json   # Output as JSON
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Dense (non-MoE) models
DENSE_MODELS = {
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
}


def is_dense_model(model_name: str) -> bool:
    """Check if a model is a dense (non-MoE) model.

    Args:
        model_name: Full model name

    Returns:
        True if model is dense, False if MoE
    """
    return model_name in DENSE_MODELS


# Model metadata with total and active parameter counts
MODEL_METADATA = {
    "allenai/OLMoE-1B-7B-0125": {
        "name": "OLMoE-1B-7B",
        "total_params_b": 7.0,
        "active_params_b": 1.3,
    },
    "openai/gpt-oss-20b": {
        "name": "GPT-OSS-20B",
        "total_params_b": 21.0,
        "active_params_b": 3.6,
    },
    "Qwen/Qwen3-30B-A3B": {
        "name": "Qwen3-30B",
        "total_params_b": 30.5,
        "active_params_b": 3.3,
    },
    "mistralai/Mixtral-8x7B-v0.1": {
        "name": "Mixtral-8x7B",
        "total_params_b": 47.0,
        "active_params_b": 12.9,
    },
    "Qwen/Qwen3-Next-80B-A3B-Instruct": {
        "name": "Qwen3-Next-80B",
        "total_params_b": 80.0,
        "active_params_b": 3.0,
    },
    "zai-org/GLM-4.5-Air": {
        "name": "GLM-4.5-Air",
        "total_params_b": 106.0,
        "active_params_b": 12.0,
    },
    "openai/gpt-oss-120b": {
        "name": "GPT-OSS-120B",
        "total_params_b": 117.0,
        "active_params_b": 5.1,
    },
    "Qwen/Qwen3-0.6B": {
        "name": "Qwen3-0.6B",
        "total_params_b": 0.6,
        "active_params_b": 0.6,
    },
    "Qwen/Qwen3-1.7B": {
        "name": "Qwen3-1.7B",
        "total_params_b": 1.7,
        "active_params_b": 1.7,
    },
    "Qwen/Qwen3-4B": {
        "name": "Qwen3-4B",
        "total_params_b": 4.0,
        "active_params_b": 4.0,
    },
    "Qwen/Qwen3-8B": {
        "name": "Qwen3-8B",
        "total_params_b": 8.0,
        "active_params_b": 8.0,
    },
    "Qwen/Qwen3-14B": {
        "name": "Qwen3-14B",
        "total_params_b": 14.0,
        "active_params_b": 14.0,
    },
}


def load_perplexity_result(perplexity_dir: Path) -> dict | None:
    """Load perplexity result from a directory.

    Args:
        perplexity_dir: Path to directory containing perplexity_results.json

    Returns:
        Dict with perplexity and model info, or None if not found
    """
    assert perplexity_dir is not None, "perplexity_dir must not be None"
    assert perplexity_dir.exists(), f"Directory does not exist: {perplexity_dir}"

    json_file = perplexity_dir / "perplexity_results.json"
    if not json_file.exists():
        return None

    with open(json_file) as f:
        data = json.load(f)

    model_name = data.get('model')
    perplexity = data.get('perplexity')
    assert model_name is not None, f"Missing model in {perplexity_dir.name}"
    assert perplexity is not None, f"Missing perplexity in {perplexity_dir.name}"

    return {
        'model_name': model_name,
        'perplexity': perplexity,
        'num_sequences': data.get('num_sequences', 0),
        'total_tokens': data.get('total_tokens', 0),
    }


def extract_outliers_from_json(file_path: Path) -> list[dict] | None:
    """Extract all_systematic_outliers from JSON file.

    Args:
        file_path: Path to final_analysis_results.json

    Returns:
        List of outlier dicts, or None if extraction failed
    """
    assert file_path is not None, "file_path must not be None"
    assert file_path.exists(), f"File does not exist: {file_path}"

    # Read first 50MB to avoid memory issues
    with open(file_path) as f:
        content = f.read(50_000_000)

    # Extract all_systematic_outliers array using regex
    match = re.search(
        r'"all_systematic_outliers":\s*\[(.*?)(?:\],\s*"batch_results")',
        content,
        re.DOTALL
    )

    if not match:
        return None

    outliers_json = '[' + match.group(1) + ']'
    outliers = json.loads(outliers_json)
    assert isinstance(outliers, list), f"Expected list, got {type(outliers)}"

    return outliers


def calculate_outlier_metrics(outliers: list[dict]) -> dict:
    """Calculate aggregate metrics from outliers list.

    Args:
        outliers: List of outlier dicts

    Returns:
        Dict with outlier metrics
    """
    assert outliers is not None, "outliers must not be None"

    if not outliers:
        return {
            'num_outliers': 0,
            'mean_layer_pct': 0.0,
            'median_layer_pct': 0.0,
            'mean_seq_pct': 0.0,
            'median_seq_pct': 0.0,
        }

    layer_pcts = [o['layer_percentage'] * 100 for o in outliers]
    seq_pcts = [o['seq_percentage'] * 100 for o in outliers]

    return {
        'num_outliers': len(outliers),
        'mean_layer_pct': float(np.mean(layer_pcts)),
        'median_layer_pct': float(np.median(layer_pcts)),
        'mean_seq_pct': float(np.mean(seq_pcts)),
        'median_seq_pct': float(np.median(seq_pcts)),
    }


def extract_model_name_from_result(result_dir: Path) -> str | None:
    """Extract model name from result directory.

    Args:
        result_dir: Path to result directory

    Returns:
        Model name or None if not found
    """
    assert result_dir is not None, "result_dir must not be None"
    assert result_dir.exists(), f"Directory does not exist: {result_dir}"

    # Try config.json first
    config_file = result_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            model_name = config.get('model', {}).get('name')
            if model_name:
                return model_name

    # Try final_analysis_results.json
    outlier_file = result_dir / "final_analysis_results.json"
    if outlier_file.exists():
        with open(outlier_file) as f:
            content = f.read(10000)
        match = re.search(r'"model":\s*"([^"]+)"', content)
        if match:
            return match.group(1)

    return None


def load_all_perplexity_results(perplexity_dir: Path) -> dict[str, dict]:
    """Load all perplexity results from directory.

    Args:
        perplexity_dir: Directory containing perplexity results

    Returns:
        Dict mapping model_name to perplexity data
    """
    assert perplexity_dir is not None, "perplexity_dir must not be None"

    results = {}
    if not perplexity_dir.exists():
        return results

    for result_dir in sorted(perplexity_dir.iterdir()):
        if not result_dir.is_dir():
            continue

        ppl_data = load_perplexity_result(result_dir)
        if ppl_data:
            model_name = ppl_data['model_name']
            results[model_name] = ppl_data
            print(f"✅ Loaded perplexity: {model_name} = {ppl_data['perplexity']:.2f}")

    return results


def match_single_outlier_result(
    result_dir: Path,
    perplexity_results: dict[str, dict]
) -> dict | None:
    """Match a single outlier result with perplexity data.

    Args:
        result_dir: Directory with outlier results
        perplexity_results: Dict of perplexity data by model name

    Returns:
        Combined result dict or None if no match
    """
    assert result_dir is not None, "result_dir must not be None"
    assert perplexity_results is not None, "perplexity_results must not be None"

    model_name = extract_model_name_from_result(result_dir)
    if not model_name:
        return None

    # Load outlier data
    outlier_file = result_dir / "final_analysis_results.json"
    if not outlier_file.exists():
        return None

    outliers = extract_outliers_from_json(outlier_file)
    if outliers is None:
        return None

    outlier_metrics = calculate_outlier_metrics(outliers)

    # Match with perplexity
    if model_name not in perplexity_results:
        return None

    # Get metadata
    if model_name not in MODEL_METADATA:
        return None

    metadata = MODEL_METADATA[model_name]
    ppl_data = perplexity_results[model_name]

    result = {
        'model_name': metadata['name'],
        'full_model_name': model_name,
        'total_params_b': metadata['total_params_b'],
        'active_params_b': metadata['active_params_b'],
        'perplexity': ppl_data['perplexity'],
        **outlier_metrics
    }

    print(f"✅ Matched: {metadata['name']} - PPL={ppl_data['perplexity']:.2f}, "
          f"Outliers={outlier_metrics['num_outliers']}")

    return result


def match_perplexity_to_outliers(
    perplexity_dir: Path,
    outlier_dir: Path
) -> list[dict]:
    """Match perplexity results with outlier analysis results.

    Args:
        perplexity_dir: Directory containing perplexity results
        outlier_dir: Directory containing outlier analysis results

    Returns:
        List of combined results
    """
    assert perplexity_dir is not None, "perplexity_dir must not be None"
    assert outlier_dir is not None, "outlier_dir must not be None"

    # Load all perplexity results
    perplexity_results = load_all_perplexity_results(perplexity_dir)

    # Match with outlier results
    combined = []
    if outlier_dir.exists():
        for result_dir in sorted(outlier_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            result = match_single_outlier_result(result_dir, perplexity_results)
            if result:
                combined.append(result)

    return combined


def create_plot_matplotlib(
    results: list[dict],
    y_key: str,
    output_path: Path,
    title: str,
    ylabel: str
):
    """Create perplexity plot using matplotlib.

    Args:
        results: List of result dicts
        y_key: Key for y-axis values (outlier metric)
        output_path: Where to save the plot
        title: Plot title
        ylabel: Y-axis label
    """
    assert results is not None, "results must not be None"
    assert len(results) > 0, "results must not be empty"
    assert y_key, "y_key must not be empty"
    assert output_path is not None, "output_path must not be None"

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Extract data
    x_values = [r['perplexity'] for r in results]
    y_values = [r[y_key] for r in results]
    labels = [r['model_name'] for r in results]

    # Create figure
    plt.figure(figsize=(10, 7))

    # Scatter plot
    plt.scatter(x_values, y_values, s=100, alpha=0.6, c='steelblue')

    # Add model labels
    for x, y, label in zip(x_values, y_values, labels):
        plt.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8
        )

    # Formatting
    plt.xlabel('Perplexity (FineWeb-Edu)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Use log scale for x-axis if perplexity range is large
    if max(x_values) / min(x_values) > 10:
        plt.xscale('log')

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {output_path.name}")


def create_plot_terminal(
    results: list[dict],
    y_key: str,
    title: str,
    ylabel: str
):
    """Create perplexity plot in terminal using plotext.

    Args:
        results: List of result dicts
        y_key: Key for y-axis values (outlier metric)
        title: Plot title
        ylabel: Y-axis label
    """
    assert results is not None, "results must not be None"
    assert len(results) > 0, "results must not be empty"

    import plotext as plt

    # Extract data
    x_values = [r['perplexity'] for r in results]
    y_values = [r[y_key] for r in results]
    labels = [r['model_name'] for r in results]

    # Clear previous plot
    plt.clear_figure()
    plt.theme('clear')

    # Colors for each model
    colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']

    # Plot each point separately with label
    for i, (x, y, label) in enumerate(zip(x_values, y_values, labels)):
        color = colors[i % len(colors)]
        plt.scatter([x], [y], marker="●", label=label, color=color)

    # Formatting
    plt.xlabel('Perplexity (FineWeb-Edu)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

    # Print legend
    print("Legend:")
    color_codes = {
        'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m',
        'blue': '\033[94m', 'magenta': '\033[95m', 'cyan': '\033[96m',
        'white': '\033[97m',
    }
    reset = '\033[0m'

    for i, (x, y, label) in enumerate(zip(x_values, y_values, labels)):
        color = colors[i % len(colors)]
        color_code = color_codes.get(color, '')
        print(f"  {color_code}●{reset} {label}: PPL={x:.2f}, {ylabel}={y:.1f}")
    print()


def generate_single_plot(
    results: list[dict],
    plot_spec: dict,
    output_dir: Path | None,
    terminal: bool
):
    """Generate a single plot.

    Args:
        results: List of result dicts
        plot_spec: Plot specification dict
        output_dir: Output directory (None if terminal)
        terminal: Whether to display in terminal
    """
    assert results is not None, "results must not be None"
    assert plot_spec is not None, "plot_spec must not be None"

    if terminal:
        create_plot_terminal(
            results=results,
            y_key=plot_spec['y_key'],
            title=plot_spec['title'],
            ylabel=plot_spec['ylabel'],
        )
    else:
        assert output_dir is not None, "output_dir required when not in terminal mode"
        create_plot_matplotlib(
            results=results,
            y_key=plot_spec['y_key'],
            output_path=output_dir / plot_spec['filename'],
            title=plot_spec['title'],
            ylabel=plot_spec['ylabel'],
        )


def get_plot_specs() -> list[dict]:
    """Get all plot specifications.

    Returns:
        List of plot specification dicts
    """
    return [
        {
            'y_key': 'mean_layer_pct',
            'filename': '01_perplexity_vs_mean_layer_pct.png',
            'title': 'Outlier Emergence vs Perplexity (Layers - Mean)',
            'ylabel': 'Mean % Layers with Outliers',
        },
        {
            'y_key': 'median_layer_pct',
            'filename': '02_perplexity_vs_median_layer_pct.png',
            'title': 'Outlier Emergence vs Perplexity (Layers - Median)',
            'ylabel': 'Median % Layers with Outliers',
        },
        {
            'y_key': 'mean_seq_pct',
            'filename': '03_perplexity_vs_mean_seq_pct.png',
            'title': 'Outlier Emergence vs Perplexity (Sequence - Mean)',
            'ylabel': 'Mean % Sequence Positions with Outliers',
        },
        {
            'y_key': 'median_seq_pct',
            'filename': '04_perplexity_vs_median_seq_pct.png',
            'title': 'Outlier Emergence vs Perplexity (Sequence - Median)',
            'ylabel': 'Median % Sequence Positions with Outliers',
        },
        {
            'y_key': 'num_outliers',
            'filename': '05_perplexity_vs_num_outliers.png',
            'title': 'Number of Outlier Dimensions vs Perplexity',
            'ylabel': 'Number of Systematic Outlier Dimensions',
        },
    ]


def generate_perplexity_plots(
    results: list[dict],
    output_dir: Path | None = None,
    terminal: bool = False
):
    """Generate perplexity analysis plots (Figure 3b replication).

    Args:
        results: List of result dicts
        output_dir: Directory to save plots (None if terminal mode)
        terminal: If True, display in terminal instead of saving files
    """
    assert results is not None, "results must not be None"
    assert len(results) > 0, "results must not be empty"

    if not terminal:
        assert output_dir is not None, "output_dir required when not in terminal mode"
        output_dir.mkdir(exist_ok=True, parents=True)
        print("\n📊 Generating perplexity plots...")
    else:
        print("\n📊 Displaying perplexity plots in terminal...")

    plots = get_plot_specs()

    # Generate each plot
    for i, plot_spec in enumerate(plots):
        generate_single_plot(results, plot_spec, output_dir, terminal)

        # Prompt to continue if in terminal and not last plot
        if terminal and i < len(plots) - 1:
            input("Press Enter for next plot...")


def print_summary_table(results: list[dict], use_rich: bool = False):
    """Print formatted summary table of all results.

    Args:
        results: List of result dicts
        use_rich: If True, use Rich library for formatting
    """
    assert results is not None, "results must not be None"

    if use_rich:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title="Perplexity Analysis Summary", show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Params (B)", justify="right", style="green")
        table.add_column("Perplexity", justify="right", style="yellow")
        table.add_column("Outliers", justify="right", style="red")
        table.add_column("Mean L%", justify="right", style="blue")
        table.add_column("Mean S%", justify="right", style="blue")

        # Sort by perplexity
        for r in sorted(results, key=lambda x: x['perplexity']):
            table.add_row(
                r['model_name'],
                f"{r['total_params_b']:.1f}",
                f"{r['perplexity']:.2f}",
                f"{r['num_outliers']}",
                f"{r['mean_layer_pct']:.1f}",
                f"{r['mean_seq_pct']:.1f}"
            )

        console.print("\n")
        console.print(table)
        console.print("\n[italic]L% = % Layers Affected | S% = % Sequence Positions Affected[/italic]\n")
        return

    # Plain text fallback
    print("\n" + "=" * 90)
    print("PERPLEXITY ANALYSIS SUMMARY")
    print("=" * 90)
    print(f"{'Model':<20} {'Params':<10} {'Perplexity':<12} {'Outliers':<10} "
          f"{'Mean L%':<10} {'Mean S%':<10}")
    print("-" * 90)

    # Sort by perplexity
    for r in sorted(results, key=lambda x: x['perplexity']):
        print(f"{r['model_name']:<20} "
              f"{r['total_params_b']:<10.1f} "
              f"{r['perplexity']:<12.2f} "
              f"{r['num_outliers']:<10} "
              f"{r['mean_layer_pct']:<10.1f} "
              f"{r['mean_seq_pct']:<10.1f}")

    print("=" * 90)
    print("Sorted by perplexity (lower = better) | L% = % Layers | S% = % Sequence Positions")
    print("=" * 90)


def output_csv(results: list[dict]):
    """Output results as CSV format."""
    assert results is not None, "results must not be None"

    import csv
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            'model_name', 'total_params_b', 'perplexity', 'num_outliers',
            'mean_layer_pct', 'median_layer_pct', 'mean_seq_pct', 'median_seq_pct'
        ]
    )
    writer.writeheader()
    for r in sorted(results, key=lambda x: x['perplexity']):
        writer.writerow(r)


def output_json(results: list[dict]):
    """Output results as JSON format."""
    assert results is not None, "results must not be None"
    print(json.dumps(results, indent=2))


def run_analysis(args) -> int:
    """Run perplexity analysis with given arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    assert args is not None, "args must not be None"

    print("🔍 Perplexity Analysis - Dettmers Figure 3b Replication")
    print("=" * 60)

    # Match perplexity with outlier results
    print("\n📂 Loading and matching results...")
    results = match_perplexity_to_outliers(args.perplexity_dir, args.outlier_dir)

    if not results:
        print("❌ No matched results found!")
        print("\nMake sure you have both:")
        print("  1. Perplexity results (perplexity_results.json)")
        print("  2. Outlier analysis results (final_analysis_results.json)")
        return 1

    print(f"\n✅ Matched {len(results)} models with both perplexity and outlier data")

    # Filter for dense models if requested
    if args.dense_only:
        original_count = len(results)
        results = [r for r in results if is_dense_model(r['full_model_name'])]
        print(f"🔍 Filtered to {len(results)} dense models (excluded {original_count - len(results)} MoE models)")

    if not results:
        print("❌ No results after filtering!")
        return 1

    # Handle different output formats
    output_dir = Path("perplexity_analysis")

    if args.terminal:
        if args.format == "csv":
            output_csv(results)
        elif args.format == "json":
            output_json(results)
        elif args.format == "table":
            print_summary_table(results, use_rich=True)
        else:  # plot
            print_summary_table(results, use_rich=True)
            generate_perplexity_plots(results, terminal=True)
            print("\n✅ Analysis complete! All plots displayed in terminal.")
    else:
        # Default: save plots and show summary
        print_summary_table(results, use_rich=True)
        generate_perplexity_plots(results, output_dir=output_dir, terminal=False)
        print(f"\n✅ Analysis complete! Plots saved to: {output_dir}")
        print("   Total plots generated: 5")

    return 0


def main():
    """Main entry point with error handling boundary."""
    try:
        parser = argparse.ArgumentParser(description="Generate perplexity analysis plots (Figure 3b)")
        parser.add_argument(
            "--perplexity-dir",
            type=Path,
            default=Path("remote_results"),
            help="Directory containing perplexity results (default: remote_results)"
        )
        parser.add_argument(
            "--outlier-dir",
            type=Path,
            default=Path("remote_results"),
            help="Directory containing outlier analysis results (default: remote_results)"
        )
        parser.add_argument(
            "--terminal",
            action="store_true",
            help="Display plots in terminal using plotext instead of saving to files"
        )
        parser.add_argument(
            "--format",
            choices=["plot", "table", "csv", "json"],
            default="plot",
            help="Output format for terminal mode: plot (default), table, csv, or json"
        )
        parser.add_argument(
            "--dense-only",
            action="store_true",
            help="Only include dense (non-MoE) models in the analysis"
        )
        args = parser.parse_args()

        return run_analysis(args)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
