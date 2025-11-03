#!/usr/bin/env python3
"""Generate dense models plot with params on x-axis and 0-100% y-axis."""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


DENSE_MODELS = {
    "Qwen/Qwen3-0.6B": {"name": "Qwen3-0.6B", "total_params_b": 0.6},
    "Qwen/Qwen3-1.7B": {"name": "Qwen3-1.7B", "total_params_b": 1.7},
    "Qwen/Qwen3-4B": {"name": "Qwen3-4B", "total_params_b": 4.0},
    "Qwen/Qwen3-8B": {"name": "Qwen3-8B", "total_params_b": 8.0},
    "Qwen/Qwen3-14B": {"name": "Qwen3-14B", "total_params_b": 14.0},
}


def extract_outliers_from_json(file_path: Path) -> Optional[List[Dict]]:
    """Extract all_systematic_outliers from JSON file."""
    with open(file_path, 'r') as f:
        content = f.read(50_000_000)

    match = re.search(
        r'"all_systematic_outliers":\s*\[(.*?)(?:\],\s*"batch_results")',
        content,
        re.DOTALL
    )

    if not match:
        return None

    outliers_json = '[' + match.group(1) + ']'
    outliers = json.loads(outliers_json)
    return outliers


def calculate_outlier_metrics(outliers: List[Dict]) -> Dict:
    """Calculate aggregate metrics from outliers list."""
    if not outliers:
        return {'mean_layer_pct': 0.0, 'mean_seq_pct': 0.0}

    layer_pcts = [o['layer_percentage'] * 100 for o in outliers]
    seq_pcts = [o['seq_percentage'] * 100 for o in outliers]

    return {
        'mean_layer_pct': float(np.mean(layer_pcts)),
        'mean_seq_pct': float(np.mean(seq_pcts)),
    }


def extract_model_name_from_result(result_dir: Path) -> Optional[str]:
    """Extract model name from result directory."""
    outlier_file = result_dir / "final_analysis_results.json"
    if outlier_file.exists():
        with open(outlier_file, 'r') as f:
            content = f.read(10000)
        match = re.search(r'"model":\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    return None


def main():
    """Generate dense params plot with 0-100% y-axis."""
    results_dir = Path("remote_results")

    results = []
    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue

        model_name = extract_model_name_from_result(result_dir)
        if not model_name or model_name not in DENSE_MODELS:
            continue

        outlier_file = result_dir / "final_analysis_results.json"
        if not outlier_file.exists():
            continue

        outliers = extract_outliers_from_json(outlier_file)
        if outliers is None:
            continue

        metrics = calculate_outlier_metrics(outliers)
        metadata = DENSE_MODELS[model_name]

        results.append({
            'name': metadata['name'],
            'params': metadata['total_params_b'],
            'mean_layer_pct': metrics['mean_layer_pct'],
        })
        print(f"✅ {metadata['name']}: {metrics['mean_layer_pct']:.1f}%")

    # Sort by params
    results.sort(key=lambda x: x['params'])

    # Create plot
    x_values = [r['params'] for r in results]
    y_values = [r['mean_layer_pct'] for r in results]
    labels = [r['name'] for r in results]

    plt.figure(figsize=(10, 7))
    plt.scatter(x_values, y_values, s=100, alpha=0.6, c='steelblue')

    for x, y, label in zip(x_values, y_values, labels):
        plt.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)

    plt.xlabel('Total Parameters (Billions)', fontsize=12)
    plt.ylabel('Mean % Layers with Outliers', fontsize=12)
    plt.title('Dense Models: Outlier Layer Coverage vs Parameters', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()

    output_path = Path("unified_plots/dense_params_vs_layer_pct.png")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
