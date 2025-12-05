#!/usr/bin/env python3
"""Generate dual-line plots with both Layer % and Sequence % (Dettmers style).

Creates plots showing both metrics on the same chart with 0-100% y-axis:
- Dense models: params/perplexity vs L%/S%
- MoE models: total/active params vs L%/S%
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')


DENSE_MODELS = {
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
}

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


def load_perplexity_result(perplexity_dir: Path) -> dict | None:
    """Load perplexity result from a directory."""
    json_file = perplexity_dir / "perplexity_results.json"
    if not json_file.exists():
        return None

    with open(json_file) as f:
        data = json.load(f)

    return {
        'model_name': data.get('model'),
        'perplexity': data.get('perplexity'),
    }


def extract_outliers_from_json(file_path: Path) -> list[dict] | None:
    """Extract all_systematic_outliers from JSON file."""
    with open(file_path) as f:
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


def calculate_outlier_metrics(outliers: list[dict]) -> dict:
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


def extract_model_name_from_result(result_dir: Path) -> str | None:
    """Extract model name from result directory."""
    config_file = result_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            model_name = config.get('model', {}).get('name')
            if model_name:
                return model_name

    outlier_file = result_dir / "final_analysis_results.json"
    if outlier_file.exists():
        with open(outlier_file) as f:
            content = f.read(10000)
        match = re.search(r'"model":\s*"([^"]+)"', content)
        if match:
            return match.group(1)

    return None


def load_all_results(results_dir: Path) -> list[dict]:
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
            print(f"✅ Loaded: {metadata['name']} - Dense={result['is_dense']}")

    return all_results


def create_dual_line_plot(
    results: list[dict],
    x_key: str,
    output_path: Path,
    title: str,
    xlabel: str,
    invert_x: bool = False,
):
    """Create dual-line plot with both L% and S% on same chart (Dettmers style)."""

    # Extract data
    x_values = [r[x_key] for r in results]
    layer_values = [r['mean_layer_pct'] for r in results]
    seq_values = [r['mean_seq_pct'] for r in results]
    labels = [r['model_name'] for r in results]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot both as scatter points (blue and orange like Dettmers)
    ax.scatter(x_values, layer_values, s=100, color='#1f77b4', marker='o',
               label='% Layers', alpha=0.7)
    ax.scatter(x_values, seq_values, s=100, color='#ff7f0e', marker='s',
               label='% Sequence Positions', alpha=0.7)

    # Add labels for each point
    for x, y_layer, y_seq, label in zip(x_values, layer_values, seq_values, labels):
        # Layer % label (above point)
        ax.annotate(label, (x, y_layer), xytext=(0, 8),
                   textcoords='offset points', fontsize=8,
                   ha='center', alpha=0.7, color='#1f77b4')
        # Seq % label (below point) - only if far enough apart
        if abs(y_layer - y_seq) > 10:
            ax.annotate(label, (x, y_seq), xytext=(0, -12),
                       textcoords='offset points', fontsize=8,
                       ha='center', alpha=0.7, color='#ff7f0e')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('% Affected', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Invert x-axis if requested (e.g., for perplexity to match Dettmers)
    if invert_x:
        ax.invert_xaxis()

    ax.legend(loc='best', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {output_path.name}")


def generate_all_dual_plots(results: list[dict], output_dir: Path):
    """Generate all dual-line plots."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Split into dense and MoE
    dense_results = [r for r in results if r['is_dense']]
    moe_results = [r for r in results if not r['is_dense']]

    print("\n📊 Generating dual-line plots...")
    print(f"   Dense models: {len(dense_results)}")
    print(f"   MoE models: {len(moe_results)}")

    # Dense models with perplexity
    dense_with_ppl = [r for r in dense_results if r['perplexity'] is not None]

    if dense_with_ppl:
        # Sort by x-axis value for line plots
        dense_params_sorted = sorted(dense_results, key=lambda x: x['total_params_b'])
        dense_ppl_sorted = sorted(dense_with_ppl, key=lambda x: x['perplexity'])

        print("\n🔵 Dense Models:")
        create_dual_line_plot(
            dense_params_sorted,
            x_key='total_params_b',
            output_path=output_dir / 'dense_params_dual.png',
            title='Dense Models: Layer & Sequence Coverage vs Parameters',
            xlabel='Total Parameters (Billions)',
        )

        create_dual_line_plot(
            dense_ppl_sorted,
            x_key='perplexity',
            output_path=output_dir / 'dense_perplexity_dual.png',
            title='Dense Models: Layer & Sequence Coverage vs Perplexity',
            xlabel='Perplexity (FineWeb-Edu)',
            invert_x=True,  # Match Dettmers: high perplexity left, low right
        )

    # MoE models
    if moe_results:
        moe_total_sorted = sorted(moe_results, key=lambda x: x['total_params_b'])
        moe_active_sorted = sorted(moe_results, key=lambda x: x['active_params_b'])

        print("\n🟣 MoE Models:")
        create_dual_line_plot(
            moe_total_sorted,
            x_key='total_params_b',
            output_path=output_dir / 'moe_total_params_dual.png',
            title='MoE Models: Layer & Sequence Coverage vs Total Parameters',
            xlabel='Total Parameters (Billions)',
        )

        create_dual_line_plot(
            moe_active_sorted,
            x_key='active_params_b',
            output_path=output_dir / 'moe_active_params_dual.png',
            title='MoE Models: Layer & Sequence Coverage vs Active Parameters',
            xlabel='Active Parameters (Billions)',
        )


def main():
    """Main entry point."""
    print("🔍 Dual-Line Plot Generation (Layer % + Sequence %)")
    print("=" * 60)

    results_dir = Path("remote_results")
    output_dir = Path("dual_plots")

    print("\n📂 Loading results...")
    results = load_all_results(results_dir)

    if not results:
        print("❌ No results found!")
        return 1

    print(f"\n✅ Loaded {len(results)} models total")

    generate_all_dual_plots(results, output_dir)

    print(f"\n✅ All dual-line plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
