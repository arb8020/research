#!/usr/bin/env python3
"""Compare outlier features across all models in the sweep.

Generates 8 plots comparing models by parameter count vs outlier metrics:
- X-axis: Total params or Active params
- Y-axis: % layers affected or % sequence affected
- Aggregation: Mean or Median

Usage:
    python compare_sweep_results.py                        # Save PNG plots
    python compare_sweep_results.py --terminal              # Display plots in terminal
    python compare_sweep_results.py --terminal --format csv    # Output as CSV
    python compare_sweep_results.py --terminal --format json   # Output as JSON
    python compare_sweep_results.py --terminal --format table  # Output as table only
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


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
}


def extract_outliers_from_json(file_path: Path) -> Optional[List[Dict]]:
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
        # Read first 50MB to avoid memory issues with large files
        with open(file_path, 'r') as f:
            content = f.read(50_000_000)

        # Try to extract all_systematic_outliers array using regex
        # Pattern: "all_systematic_outliers": [...], "batch_results"
        match = re.search(
            r'"all_systematic_outliers":\s*\[(.*?)(?:\],\s*"batch_results")',
            content,
            re.DOTALL
        )

        if not match:
            print(f"⚠️  Could not find all_systematic_outliers in {file_path.parent.name}")
            return None

        # Parse just the outliers array
        outliers_json = '[' + match.group(1) + ']'
        outliers = json.loads(outliers_json)

        return outliers

    except Exception as e:
        print(f"❌ Error extracting outliers from {file_path.parent.name}: {e}")
        return None


def extract_model_name_from_json(file_path: Path) -> Optional[str]:
    """Extract model name from JSON file's run_config."""
    try:
        with open(file_path, 'r') as f:
            # Read just the first few KB which contains run_config
            content = f.read(10000)

        # Extract model name from run_config
        match = re.search(r'"model":\s*"([^"]+)"', content)
        if match:
            return match.group(1)

    except Exception as e:
        print(f"⚠️  Could not extract model name from {file_path.parent.name}: {e}")

    return None


def calculate_metrics(outliers: List[Dict]) -> Dict[str, float]:
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


def collect_all_results(results_dir: Path) -> List[Dict]:
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
            'active_params_b': metadata['active_params_b'],
            'num_outliers': len(outliers),
            **metrics
        }

        all_results.append(result)
        print(f"✅ {metadata['name']}: {len(outliers)} outliers, "
              f"mean layer {metrics['mean_layer_pct']:.1f}%, "
              f"mean seq {metrics['mean_seq_pct']:.1f}%")

    return all_results


def create_plot_matplotlib(
    results: List[Dict],
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
    import matplotlib.pyplot as plt
    import matplotlib
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
    results: List[Dict],
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


def generate_all_plots(results: List[Dict], output_dir: Optional[Path] = None, terminal: bool = False):
    """Generate all 8 comparison plots (2^3 combinations).

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

    # Define all combinations (2^3 = 8)
    plots = [
        # Total params x Layer % x Mean/Median
        {
            'x_key': 'total_params_b',
            'y_key': 'mean_layer_pct',
            'filename': '01_total_params_vs_mean_layer_pct.png',
            'title': 'Outlier Layer Coverage vs Total Parameters (Mean)',
            'xlabel': 'Total Parameters (B)',
            'ylabel': 'Mean % Layers with Outliers',
        },
        {
            'x_key': 'total_params_b',
            'y_key': 'median_layer_pct',
            'filename': '02_total_params_vs_median_layer_pct.png',
            'title': 'Outlier Layer Coverage vs Total Parameters (Median)',
            'xlabel': 'Total Parameters (B)',
            'ylabel': 'Median % Layers with Outliers',
        },

        # Total params x Seq % x Mean/Median
        {
            'x_key': 'total_params_b',
            'y_key': 'mean_seq_pct',
            'filename': '03_total_params_vs_mean_seq_pct.png',
            'title': 'Outlier Sequence Coverage vs Total Parameters (Mean)',
            'xlabel': 'Total Parameters (B)',
            'ylabel': 'Mean % Sequence Positions with Outliers',
        },
        {
            'x_key': 'total_params_b',
            'y_key': 'median_seq_pct',
            'filename': '04_total_params_vs_median_seq_pct.png',
            'title': 'Outlier Sequence Coverage vs Total Parameters (Median)',
            'xlabel': 'Total Parameters (B)',
            'ylabel': 'Median % Sequence Positions with Outliers',
        },

        # Active params x Layer % x Mean/Median
        {
            'x_key': 'active_params_b',
            'y_key': 'mean_layer_pct',
            'filename': '05_active_params_vs_mean_layer_pct.png',
            'title': 'Outlier Layer Coverage vs Active Parameters (Mean)',
            'xlabel': 'Active Parameters (B)',
            'ylabel': 'Mean % Layers with Outliers',
        },
        {
            'x_key': 'active_params_b',
            'y_key': 'median_layer_pct',
            'filename': '06_active_params_vs_median_layer_pct.png',
            'title': 'Outlier Layer Coverage vs Active Parameters (Median)',
            'xlabel': 'Active Parameters (B)',
            'ylabel': 'Median % Layers with Outliers',
        },

        # Active params x Seq % x Mean/Median
        {
            'x_key': 'active_params_b',
            'y_key': 'mean_seq_pct',
            'filename': '07_active_params_vs_mean_seq_pct.png',
            'title': 'Outlier Sequence Coverage vs Active Parameters (Mean)',
            'xlabel': 'Active Parameters (B)',
            'ylabel': 'Mean % Sequence Positions with Outliers',
        },
        {
            'x_key': 'active_params_b',
            'y_key': 'median_seq_pct',
            'filename': '08_active_params_vs_median_seq_pct.png',
            'title': 'Outlier Sequence Coverage vs Active Parameters (Median)',
            'xlabel': 'Active Parameters (B)',
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


def print_summary_table(results: List[Dict], use_rich: bool = False):
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

            table = Table(title="Sweep Comparison Summary", show_header=True, header_style="bold magenta")
            table.add_column("Model", style="cyan", no_wrap=True)
            table.add_column("Total (B)", justify="right", style="green")
            table.add_column("Active (B)", justify="right", style="green")
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
                    f"{r['active_params_b']:.1f}",
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
    print("\n" + "=" * 100)
    print("SWEEP COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Model':<20} {'Total':<10} {'Active':<10} {'Outliers':<10} "
          f"{'Mean L%':<10} {'Med L%':<10} {'Mean S%':<10} {'Med S%':<10}")
    print("-" * 100)

    # Sort by total params
    for r in sorted(results, key=lambda x: x['total_params_b']):
        print(f"{r['model_name']:<20} "
              f"{r['total_params_b']:<10.1f} "
              f"{r['active_params_b']:<10.1f} "
              f"{r['num_outliers']:<10} "
              f"{r['mean_layer_pct']:<10.1f} "
              f"{r['median_layer_pct']:<10.1f} "
              f"{r['mean_seq_pct']:<10.1f} "
              f"{r['median_seq_pct']:<10.1f}")

    print("=" * 100)
    print("L% = % Layers Affected | S% = % Sequence Positions Affected")
    print("=" * 100)


def output_csv(results: List[Dict]):
    """Output results as CSV format."""
    import csv
    import sys

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            'model_name', 'total_params_b', 'active_params_b', 'num_outliers',
            'mean_layer_pct', 'median_layer_pct', 'mean_seq_pct', 'median_seq_pct'
        ]
    )
    writer.writeheader()
    for r in sorted(results, key=lambda x: x['total_params_b']):
        writer.writerow(r)


def output_json(results: List[Dict]):
    """Output results as JSON format."""
    import json
    print(json.dumps(results, indent=2))


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compare outlier features sweep results")
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
    output_dir = script_dir / "sweep_analysis"

    print("🔍 Outlier Features Sweep Analysis")
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
        print(f"   Total plots generated: 8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
