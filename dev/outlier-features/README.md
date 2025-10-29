# Outlier Features Analysis

Systematic analysis of outlier features in Mixture-of-Experts (MoE) language models, following the methodology from [Dettmers et al. (2022)](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/).

## Quick Start

### Run MoE Sweep (7 models)
```bash
# Deploy all 7 MoE models to remote GPUs
python deploy_sweep.py

# Or deploy specific models
python deploy_sweep.py --models 01 03 06

# Or run single model
python deploy.py sweep_configs/01_olmoe_1b_7b.py
```

### Run Dense Baseline Sweep (5 Qwen models)
```bash
# Deploy all 5 dense models for comparison
python deploy_sweep.py --config-dir sweep_dense

# Or run single dense model
python deploy.py sweep_dense/01_qwen3_0.6b.py
```

### Generate Comparison Plots
```bash
# After sweeps complete, generate visualizations
python generate_comparison_plots.py

# Display in terminal
python generate_comparison_plots.py --terminal

# Export as CSV/JSON
python generate_comparison_plots.py --terminal --format csv > results.csv
python generate_comparison_plots.py --terminal --format json > results.json
```

**Requirements:**
- `RUNPOD_API_KEY` in `.env` file
- `HF_TOKEN` for model access
- `SSH_KEY_PATH` (defaults to `~/.ssh/id_ed25519`)

## Code Navigation

**Core Analysis Pipeline:**
- **Outlier detection**: `analyze_activations.py:63-134` (find_outliers_in_activations)
- **Systematic filtering**: `analyze_activations.py:136-209` (filter_systematic_outliers)
- **Full pipeline**: `run_full_analysis.py:358-449`
- **Architecture handling**: `extract_activations.py:get_model_layers()`

**Deployment & Comparison:**
- **Remote deployment**: `deploy.py` (single), `deploy_sweep.py` (batch)
- **Result aggregation**: `generate_comparison_plots.py:153-201` (collect_all_results)
- **Metrics calculation**: `generate_comparison_plots.py:125-150` (calculate_metrics)
- **Visualization**: `generate_comparison_plots.py:203-313`

**Configuration:**
- **Base classes**: `config.py` (ModelConfig, DatasetConfig, AnalysisConfig, DeploymentConfig)
- **MoE configs**: `sweep_configs/` (7 models, 7B-120B total params)
- **Dense configs**: `sweep_dense/` (5 Qwen models, 0.6B-14B)

## Models Analyzed

### MoE Models (N=7)
| Model | Total Params | Active Params | Systematic Outliers |
|-------|--------------|---------------|---------------------|
| OLMoE-1B-7B | 7B | 1.3B | 49 |
| GPT-OSS-20B | 21B | 3.6B | 1,465 |
| Qwen3-30B | 30.5B | 3.3B | 110 |
| Mixtral-8x7B | 47B | 12.9B | 4,635 |
| Qwen3-Next-80B | 80B | 3.0B | 504 |
| GLM-4.5-Air | 106B | 12.0B | 459 |
| GPT-OSS-120B | 117B | 5.1B | 1,695 |

### Dense Baseline Models (N=5)
| Model | Params | Systematic Outliers |
|-------|--------|---------------------|
| Qwen3-0.6B | 0.6B | 9,212 |
| Qwen3-1.7B | 1.7B | 16,563 |
| Qwen3-4B | 4.0B | 1,042 |
| Qwen3-8B | 8.0B | 777 |
| Qwen3-14B | 14.0B | 985 |

## Methodology

### Outlier Detection Pipeline

Based on [Dettmers et al. (2022)](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/):

**Step 1: Find Raw Outliers** (`analyze_activations.py:63-134`)
- Identify activation values with **magnitude ≥ 6.0**
- Group by feature dimension (h_i in d_model)
- Track which layers and sequence positions contain each outlier

**Step 2: Filter for Systematic Outliers** (`analyze_activations.py:136-209`)

A feature dimension is "systematic" if it meets ALL criteria:
- **Magnitude threshold**: ≥6.0 activation magnitude
- **Layer coverage**: Affects ≥25% of transformer layers
- **Sequence coverage**: Affects ≥6% of sequence positions

**Dataset:** 16 sequences × 2048 tokens from FineWeb-Edu

### Research Question

**Context**: Dettmers et al. (2022) found dense transformers exhibit a phase transition at ~6.7B parameters where outlier features shift from probabilistic (scattered across layers) to systematic (100% layer agreement).

**Question**: Does this extend to MoE architectures? Do modern dense models still show this behavior?

## Key Findings

**TL;DR:** Outlier feature coordination, a notable empirical observation in dense model residual streams that potentially explains the '7B intelligence step change', is no longer a clean indicator for similar thresholds in MoE models. Modern architectures and training data have led to more sophisticated models that no longer exhibit the 2022 behavior.

### MoE Models (7B-120B total parameters)
- **No phase transition observed** - No correlation between parameter count (total or active) and outlier layer coordination
- Layer coordination percentages vary inconsistently (29%-63%) across models
- The 6.7B parameter threshold identified by Dettmers does not apply to MoE architectures

### Dense Models (Qwen3 family, 0.6B-14B)
- **Stable layer coordination** - Consistently ~30% across all sizes
- Phase transition absent even in dense models post-2022
- Suggests architectural improvements or training data quality has obsoleted the outlier feature heuristic

### Implication
The lack of phase transition in both MoE and modern dense models indicates that this phenomenon may be specific to older training regimes and architectures. If we want empirical ways to study step changes in intelligence as a function of parameter count, we may need to explore alternative approaches (e.g., circuit analysis, per-parameter interpretability).

## Project Structure

```
sweep_configs/          # MoE model configs (7 models)
sweep_dense/           # Dense baseline configs (5 models)
remote_results/        # Analysis results (downloaded from GPUs)
sweep_analysis/        # Output plots from generate_comparison_plots.py
```
