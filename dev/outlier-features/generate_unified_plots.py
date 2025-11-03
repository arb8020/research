#!/usr/bin/env python3
"""Generate unified plots with full 0-100% y-axis ranges.

For dense models: Perplexity vs % layers/sequence affected (0-100%)
For MoE models: Total/Active params vs % layers/sequence affected (0-100%)

Usage:
    python generate_unified_plots.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


# Dense (non-MoE) models
DENSE_MODELS = {
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
}


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


def is_dense_model(model_name: str) -> bool:
    """Check if a model is dense (non-MoE)."""
    return model_name in DENSE_MODELS


def load_perplexity_result(perplexity_dir: Path) -> Optional[Dict]:
    """Load perplexity result from a directory."""
    json_file = perplexity_dir / "perplexity_results.json"
    if not json_file.exists():
        return None

    with open(json_file, 'r') as f:
        data = json.load(f)

    return {
        'model_name': data.get('model'),
        'perplexity': data.get('perplexity'),
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
        return {
            'num_outliers': 0,
            'mean_layer_pct': 0.0,
            'mean_seq_pct': 0.0,
        }

    layer_pcts = [o['layer_percentage'] * 100 for o in outliers]
    seq_pcts = [o['seq_percentage'] * 100 for o in outliers]

    return {
        'num_outliers': len(outliers),
        'mean_layer_pct': float(np.mean(layer_pcts)),
        'mean_seq_pct': float(np.mean(seq_pcts)),
    }


def extract_model_name_from_result(result_dir: Path) -> Optional[str]:
    """Extract model name from result directory."""
    config_file = result_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            model_name = config.get('model', {}).get('name')
            if model_name:
                return model_name

    outlier_file = result_dir / "final_analysis_results.json"
    if outlier_file.exists():
        with open(outlier_file, 'r') as f:
            content = f.read(10000)
        match = re.search(r'"model":\s*"([^"]+)"', content)
        if match:
            return match.group(1)

    return None


def load_all_results(results_dir: Path) -> List[Dict]:
    """Load all results (outlier + perplexity data)."""
    # First, load all perplexity results
    perplexity_map = {}
    if results_dir.exists():
        for result_dir in sorted(results_dir.iterdir()):
            if not result_dir.is_dir():
                continue
            ppl_data = load_perplexity_result(result_dir)
            if ppl_data and ppl_data['model_name']:
                perplexity_map[ppl_data['model_name']] = ppl_data['perplexity']

    # Now load outlier results and match with perplexity
    all_results = []
    if results_dir.exists():
        for result_dir in sorted(results_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            model_name = extract_model_name_from_result(result_dir)
            if not model_name or model_name not in MODEL_METADATA:
                continue

            outlier_file = result_dir / "final_analysis_results.json"
            if not outlier_file.exists():
                continue

            outliers = extract_outliers_from_json(outlier_file)
            if outliers is None:
                continue

            outlier_metrics = calculate_outlier_metrics(outliers)
            metadata = MODEL_METADATA[model_name]

            result = {
                'model_name': metadata['name'],
                'full_model_name': model_name,
                'total_params_b': metadata['total_params_b'],
                'active_params_b': metadata['active_params_b'],
                'perplexity': perplexity_map.get(model_name),
                'is_dense': is_dense_model(model_name),
                **outlier_metrics
            }

            all_results.append(result)
            print(f"✅ Loaded: {metadata['name']} - Dense={result['is_dense']}, "
                  f"PPL={result['perplexity']}, Outliers={outlier_metrics['num_outliers']}")

    return all_results


def create_plot(
    results: List[Dict],
    x_key: str,
    y_key: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    use_log_x: bool = False
):
    """Create a plot with 0-100% y-axis."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    x_values = [r[x_key] for r in results]
    y_values = [r[y_key] for r in results]
    labels = [r['model_name'] for r in results]

    plt.figure(figsize=(10, 7))
    plt.scatter(x_values, y_values, s=100, alpha=0.6, c='steelblue')

    for x, y, label in zip(x_values, y_values, labels):
        plt.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Set y-axis to 0-100%
    plt.ylim(0, 100)

    if use_log_x:
        plt.xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {output_path.name}")


def generate_all_plots(results: List[Dict], output_dir: Path):
    """Generate all plots with 0-100% y-axis."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Split into dense and MoE
    dense_results = [r for r in results if r['is_dense']]
    moe_results = [r for r in results if not r['is_dense']]

    print(f"\n📊 Generating plots with 0-100% y-axis...")
    print(f"   Dense models: {len(dense_results)}")
    print(f"   MoE models: {len(moe_results)}")

    # Dense models: Perplexity vs Layer/Seq % (only if we have perplexity data)
    dense_with_ppl = [r for r in dense_results if r['perplexity'] is not None]

    if dense_with_ppl:
        print("\n🔵 Dense Models - Perplexity Plots:")
        create_plot(
            dense_with_ppl,
            x_key='perplexity',
            y_key='mean_layer_pct',
            output_path=output_dir / 'dense_perplexity_vs_layer_pct.png',
            title='Dense Models: Outlier Layer Coverage vs Perplexity',
            xlabel='Perplexity (FineWeb-Edu)',
            ylabel='Mean % Layers with Outliers',
        )

        create_plot(
            dense_with_ppl,
            x_key='perplexity',
            y_key='mean_seq_pct',
            output_path=output_dir / 'dense_perplexity_vs_seq_pct.png',
            title='Dense Models: Outlier Sequence Coverage vs Perplexity',
            xlabel='Perplexity (FineWeb-Edu)',
            ylabel='Mean % Sequence Positions with Outliers',
        )

    # MoE models: Total Params vs Layer/Seq %
    if moe_results:
        print("\n🟣 MoE Models - Parameter Plots:")
        create_plot(
            moe_results,
            x_key='total_params_b',
            y_key='mean_layer_pct',
            output_path=output_dir / 'moe_total_params_vs_layer_pct.png',
            title='MoE Models: Outlier Layer Coverage vs Total Parameters',
            xlabel='Total Parameters (Billions)',
            ylabel='Mean % Layers with Outliers',
        )

        create_plot(
            moe_results,
            x_key='total_params_b',
            y_key='mean_seq_pct',
            output_path=output_dir / 'moe_total_params_vs_seq_pct.png',
            title='MoE Models: Outlier Sequence Coverage vs Total Parameters',
            xlabel='Total Parameters (Billions)',
            ylabel='Mean % Sequence Positions with Outliers',
        )

        create_plot(
            moe_results,
            x_key='active_params_b',
            y_key='mean_layer_pct',
            output_path=output_dir / 'moe_active_params_vs_layer_pct.png',
            title='MoE Models: Outlier Layer Coverage vs Active Parameters',
            xlabel='Active Parameters (Billions)',
            ylabel='Mean % Layers with Outliers',
        )

        create_plot(
            moe_results,
            x_key='active_params_b',
            y_key='mean_seq_pct',
            output_path=output_dir / 'moe_active_params_vs_seq_pct.png',
            title='MoE Models: Outlier Sequence Coverage vs Active Parameters',
            xlabel='Active Parameters (Billions)',
            ylabel='Mean % Sequence Positions with Outliers',
        )


def main():
    """Main entry point."""
    print("🔍 Unified Plot Generation (0-100% Y-Axis)")
    print("=" * 60)

    results_dir = Path("remote_results")
    output_dir = Path("unified_plots")

    print("\n📂 Loading results...")
    results = load_all_results(results_dir)

    if not results:
        print("❌ No results found!")
        return 1

    print(f"\n✅ Loaded {len(results)} models total")

    generate_all_plots(results, output_dir)

    print(f"\n✅ All plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
