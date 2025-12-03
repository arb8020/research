#!/usr/bin/env python3
"""Compare outlier features across dense models.

Generates plots comparing dense models by parameter count vs outlier metrics:
- X-axis: Total params
- Y-axis: % layers affected or % sequence affected
- Aggregation: Mean or Median

Usage:
    python compare_dense_results.py                        # Save PNG plots
    python compare_dense_results.py --terminal              # Display plots in terminal
    python compare_dense_results.py --terminal --format csv    # Output as CSV
    python compare_dense_results.py --terminal --format json   # Output as JSON
    python compare_dense_results.py --terminal --format table  # Output as table only
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Dense model metadata with parameter counts
MODEL_METADATA = {
    "Qwen/Qwen3-0.6B": {
        "name": "Qwen3-0.6B",
        "total_params_b": 0.6,
    },
    "Qwen/Qwen3-1.7B": {
        "name": "Qwen3-1.7B",
        "total_params_b": 1.7,
    },
    "Qwen/Qwen3-4B": {
        "name": "Qwen3-4B",
        "total_params_b": 4.0,
    },
    "Qwen/Qwen3-8B": {
        "name": "Qwen3-8B",
        "total_params_b": 8.0,
    },
    "Qwen/Qwen3-14B": {
        "name": "Qwen3-14B",
        "total_params_b": 14.0,
    },
    "google/gemma-2-2b": {
        "name": "Gemma3-270M",
        "total_params_b": 0.27,
    },
    "google/gemma-2-9b": {
        "name": "Gemma3-1B",
        "total_params_b": 1.0,
    },
    "google/gemma-2-27b": {
        "name": "Gemma3-4B",
        "total_params_b": 4.0,
    },
    "google/gemma-3-12b": {
        "name": "Gemma3-12B",
        "total_params_b": 12.0,
    },
    "google/gemma-3-27b": {
        "name": "Gemma3-27B",
        "total_params_b": 27.0,
    },
}


def extract_outliers_from_json(file_path: Path) -> list[dict] | None:
    """Extract all_systematic_outliers from JSON file.

    Handles truncated/incomplete JSON files by reading only the beginning
    where all_systematic_outliers is located.

    Args:
        file_path: Path to final_analysis_results.json

    Returns:
        List of outlier dicts with layer_percentage and seq_percentage,
        or None if extraction failed
    """
    try:
        # Read first 100MB to handle larger dense model results
        with open(file_path) as f:
            content = f.read(100_000_000)

        # Try to extract all_systematic_outliers array using regex
        # Pattern: "all_systematic_outliers": [...], "batch_results"
        match = re.search(
            r'"all_systematic_outliers":\s*\[',
            content,
            re.DOTALL
        )

        if not match:
            print(f"⚠️  Could not find all_systematic_outliers in {file_path.parent.name}")
            return None

        # Find the end of the array
        start_pos = match.end()
        end_match = re.search(r'\],\s*"batch_results"', content[start_pos:])

        if not end_match:
            print(f"⚠️  Could not find end of all_systematic_outliers in {file_path.parent.name}")
            return None

        # Parse just the outliers array
        outliers_json = '[' + content[start_pos:start_pos + end_match.start()] + ']'
        outliers = json.loads(outliers_json)

        return outliers

    except Exception as e:
        print(f"❌ Error extracting outliers from {file_path.parent.name}: {e}")
        return None


def extract_model_name_from_json(file_path: Path) -> str | None:
    """Extract model name from JSON file's run_config."""
    try:
        with open(file_path) as f:
            # Read just the first few KB which contains run_config
            content = f.read(10000)

        # Extract model name from run_config
        match = re.search(r'"model":\s*"([^"]+)"', content)
        if match:
            return match.group(1)

    except Exception as e:
        print(f"⚠️  Could not extract model name from {file_path.parent.name}: {e}")

    return None


def calculate_metrics(outliers: list[dict]) -> dict[str, float]:
    """Calculate aggregate metrics from outliers list.

    Args:
        outliers: List of outlier dicts with layer_percentage and seq_percentage

    Returns:
        Dict with mean and median for both metrics
    """
    if not outliers:
        return {
            'mean_layer_pct': 0.0,
            'median_layer_pct': 0.0,
            'mean_seq_pct': 0.0,
            'median_seq_pct': 0.0,
        }

    layer_pcts = [o['layer_percentage'] * 100 for o in outliers]  # Convert to percentage
    seq_pcts = [o['seq_percentage'] * 100 for o in outliers]

    return {
        'mean_layer_pct': float(np.mean(layer_pcts)),
        'median_layer_pct': float(np.median(layer_pcts)),
        'mean_seq_pct': float(np.mean(seq_pcts)),
        'median_seq_pct': float(np.median(seq_pcts)),
    }


def collect_all_results(results_dir: Path) -> list[dict]:
    """Collect metrics from all result directories.

    Args:
        results_dir: Path to remote_results directory

    Returns:
        List of dicts with model metadata and metrics
    """
    all_results = []

    # Find all final_analysis_results.json files
    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue

        json_file = result_dir / "final_analysis_results.json"
        if not json_file.exists():
            print(f"⚠️  No results file in {result_dir.name}")
            continue

        # Extract data
        model_name = extract_model_name_from_json(json_file)
        if not model_name or model_name not in MODEL_METADATA:
            print(f"⚠️  Unknown model in {result_dir.name}: {model_name}")
            continue

        outliers = extract_outliers_from_json(json_file)
        if outliers is None:
            continue

        metrics = calculate_metrics(outliers)
        metadata = MODEL_METADATA[model_name]

        result = {
            'model_name': metadata['name'],
            'total_params_b': metadata['total_params_b'],
            'num_outliers': len(outliers),
            **metrics
        }

        all_results.append(result)
        print(f"✅ {metadata['name']}: {len(outliers)} outliers, "
              f"mean layer {metrics['mean_layer_pct']:.1f}%, "
              f"mean seq {metrics['mean_seq_pct']:.1f}%")

    return all_results


def create_plot_matplotlib(
    results: list[dict],
    x_key: str,
    y_key: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str
):
    """Create a single comparison plot using matplotlib.

    Args:
        results: List of result dicts
        x_key: Key for x-axis values
        y_key: Key for y-axis values
        output_path: Where to save the plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')  # Non-interactive backend

    # Extract data
    x_values = [r[x_key] for r in results]
    y_values = [r[y_key] for r in results]
    labels = [r['model_name'] for r in results]

    # Create figure
    plt.figure(figsize=(10, 7))

    # Scatter plot with labels
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
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {output_path.name}")


def create_plot_terminal(
    results: list[dict],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str
):
    """Create a single comparison plot in terminal using plotext.

    Args:
        results: List of result dicts
        x_key: Key for x-axis values
        y_key: Key for y-axis values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    try:
        import plotext as plt
    except ImportError:
        print("⚠️  plotext not installed. Install with: uv add plotext")
        return

    # Extract data
    x_values = [r[x_key] for r in results]
    y_values = [r[y_key] for r in results]
    labels = [r['model_name'] for r in results]

    # Clear previous plot
    plt.clear_figure()

    # Use terminal's native colors (respects your terminal theme)
    plt.theme('clear')

    # Define colors for each model (using plotext color names)
    colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']

    # Create a scatter plot with labels and colors for each point
    # plotext doesn't have good annotation support, so we plot each point separately with a label
    for i, (x, y, label) in enumerate(zip(x_values, y_values, labels)):
        color = colors[i % len(colors)]
        plt.scatter([x], [y], marker="●", label=label, color=color)

    # Formatting
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Show
    plt.show()

    # Print colored legend separately below the plot for clarity
    print("Legend:")
    color_codes = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
    }
    reset = '\033[0m'

    for i, (x, y, label) in enumerate(zip(x_values, y_values, labels)):
        color = colors[i % len(colors)]
        color_code = color_codes.get(color, '')
        print(f"  {color_code}●{reset} {label}: ({x:.1f}, {y:.1f})")
    print()


def generate_all_plots(results: list[dict], output_dir: Path | None = None, terminal: bool = False):
    """Generate all 4 comparison plots for dense models.

    Args:
        results: List of result dicts
        output_dir: Directory to save plots (None if terminal mode)
        terminal: If True, display in terminal instead of saving files
    """
    if not terminal and output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        print("\n📊 Generating plots...")
    else:
        print("\n📊 Displaying plots in terminal...")

    # Define all plots (4 for dense models - only total params, no active params)
    plots = [
        # Total params x Layer % x Mean/Median
        {
            'x_key': 'total_params_b',
            'y_key': 'mean_layer_pct',
            'filename': '01_dense_params_vs_mean_layer_pct.png',
            'title': 'Outlier Layer Coverage vs Parameters (Mean) - Dense Models',
            'xlabel': 'Total Parameters (B)',
            'ylabel': 'Mean % Layers with Outliers',
        },
        {
            'x_key': 'total_params_b',
            'y_key': 'median_layer_pct',
            'filename': '02_dense_params_vs_median_layer_pct.png',
            'title': 'Outlier Layer Coverage vs Parameters (Median) - Dense Models',
            'xlabel': 'Total Parameters (B)',
            'ylabel': 'Median % Layers with Outliers',
        },

        # Total params x Seq % x Mean/Median
        {
            'x_key': 'total_params_b',
            'y_key': 'mean_seq_pct',
            'filename': '03_dense_params_vs_mean_seq_pct.png',
            'title': 'Outlier Sequence Coverage vs Parameters (Mean) - Dense Models',
            'xlabel': 'Total Parameters (B)',
            'ylabel': 'Mean % Sequence Positions with Outliers',
        },
        {
            'x_key': 'total_params_b',
            'y_key': 'median_seq_pct',
            'filename': '04_dense_params_vs_median_seq_pct.png',
            'title': 'Outlier Sequence Coverage vs Parameters (Median) - Dense Models',
            'xlabel': 'Total Parameters (B)',
            'ylabel': 'Median % Sequence Positions with Outliers',
        },
    ]

    # Generate each plot
    for plot_spec in plots:
        if terminal:
            create_plot_terminal(
                results=results,
                x_key=plot_spec['x_key'],
                y_key=plot_spec['y_key'],
                title=plot_spec['title'],
                xlabel=plot_spec['xlabel'],
                ylabel=plot_spec['ylabel'],
            )
            # Prompt to continue
            if plot_spec != plots[-1]:  # Not the last plot
                input("Press Enter for next plot...")
        else:
            assert output_dir is not None, "output_dir required when not in terminal mode"
            create_plot_matplotlib(
                results=results,
                x_key=plot_spec['x_key'],
                y_key=plot_spec['y_key'],
                output_path=output_dir / plot_spec['filename'],
                title=plot_spec['title'],
                xlabel=plot_spec['xlabel'],
                ylabel=plot_spec['ylabel'],
            )


def print_summary_table(results: list[dict], use_rich: bool = False):
    """Print formatted summary table of all results.

    Args:
        results: List of result dicts
        use_rich: If True, use Rich library for formatting
    """
    if use_rich:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()

            table = Table(title="Dense Models Comparison Summary", show_header=True, header_style="bold magenta")
            table.add_column("Model", style="cyan", no_wrap=True)
            table.add_column("Params (B)", justify="right", style="green")
            table.add_column("Outliers", justify="right", style="yellow")
            table.add_column("Mean L%", justify="right", style="blue")
            table.add_column("Med L%", justify="right", style="blue")
            table.add_column("Mean S%", justify="right", style="red")
            table.add_column("Med S%", justify="right", style="red")

            # Sort by total params
            for r in sorted(results, key=lambda x: x['total_params_b']):
                table.add_row(
                    r['model_name'],
                    f"{r['total_params_b']:.1f}",
                    f"{r['num_outliers']}",
                    f"{r['mean_layer_pct']:.1f}",
                    f"{r['median_layer_pct']:.1f}",
                    f"{r['mean_seq_pct']:.1f}",
                    f"{r['median_seq_pct']:.1f}"
                )

            console.print("\n")
            console.print(table)
            console.print("\n[italic]L% = % Layers Affected | S% = % Sequence Positions Affected[/italic]\n")
            return
        except ImportError:
            pass  # Fall back to plain text

    # Plain text fallback
    print("\n" + "=" * 90)
    print("DENSE MODELS COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Model':<15} {'Params':<10} {'Outliers':<10} "
          f"{'Mean L%':<10} {'Med L%':<10} {'Mean S%':<10} {'Med S%':<10}")
    print("-" * 90)

    # Sort by total params
    for r in sorted(results, key=lambda x: x['total_params_b']):
        print(f"{r['model_name']:<15} "
              f"{r['total_params_b']:<10.1f} "
              f"{r['num_outliers']:<10} "
              f"{r['mean_layer_pct']:<10.1f} "
              f"{r['median_layer_pct']:<10.1f} "
              f"{r['mean_seq_pct']:<10.1f} "
              f"{r['median_seq_pct']:<10.1f}")

    print("=" * 90)
    print("L% = % Layers Affected | S% = % Sequence Positions Affected")
    print("=" * 90)


def output_csv(results: list[dict]):
    """Output results as CSV format."""
    import csv
    import sys

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            'model_name', 'total_params_b', 'num_outliers',
            'mean_layer_pct', 'median_layer_pct', 'mean_seq_pct', 'median_seq_pct'
        ]
    )
    writer.writeheader()
    for r in sorted(results, key=lambda x: x['total_params_b']):
        writer.writerow(r)


def output_json(results: list[dict]):
    """Output results as JSON format."""
    import json
    print(json.dumps(results, indent=2))


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compare outlier features for dense models")
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
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "remote_results"
    output_dir = script_dir / "dense_analysis"

    print("🔍 Dense Models Outlier Features Analysis")
    print("=" * 60)

    # Collect all results
    print("\n📂 Processing result files...")
    results = collect_all_results(results_dir)

    if not results:
        print("❌ No results found!")
        return 1

    print(f"\n✅ Processed {len(results)} models")

    # Handle different output formats
    if args.terminal:
        if args.format == "csv":
            output_csv(results)
        elif args.format == "json":
            output_json(results)
        elif args.format == "table":
            print_summary_table(results, use_rich=True)
        else:  # plot
            print_summary_table(results, use_rich=True)
            generate_all_plots(results, terminal=True)
            print("\n✅ Analysis complete! All plots displayed in terminal.")
    else:
        # Default: save plots and show summary with Rich
        print_summary_table(results, use_rich=True)
        generate_all_plots(results, output_dir=output_dir, terminal=False)
        print(f"\n✅ Analysis complete! Plots saved to: {output_dir}")
        print("   Total plots generated: 4")

    return 0


if __name__ == "__main__":
    sys.exit(main())
