# Outlier Features Analysis

Systematic analysis of outlier features in Mixture-of-Experts (MoE) language models, following the methodology from [Dettmers et al. (2022)](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/).

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

## Quick Start

### 1. Analyze Existing Results

**Note:** This requires `remote_results/` directory with analysis outputs. If you're coming from the blog post and don't have this data, skip to step 2 to run your own analysis.

If you already have results in `remote_results/`:

```bash
# Save plots as PNG files (default)
python generate_comparison_plots.py

# Display plots in terminal (requires plotext)
python generate_comparison_plots.py --terminal

# Output as CSV (pipe to file if needed)
python generate_comparison_plots.py --terminal --format csv > results.csv

# Output as JSON
python generate_comparison_plots.py --terminal --format json > results.json

# Output as table only (no plots)
python generate_comparison_plots.py --terminal --format table
```

**Output:**
- Summary table showing outlier counts and coverage metrics per model
- 8 plots comparing models by parameter count vs outlier characteristics
  - Default: Saves PNG files to `sweep_analysis/`
  - `--terminal --format plot`: Displays plots directly in terminal
  - `--terminal --format csv`: Outputs CSV format
  - `--terminal --format json`: Outputs JSON format
  - `--terminal --format table`: Outputs table only (no plots)

### 2. Run New Analysis

If you need to generate results from scratch:

```bash
# Run full sweep across all 7 models (deploys to remote GPUs via RunPod)
python deploy_sweep.py

# Or run a single model
python deploy.py sweep_configs/01_olmoe_1b_7b.py
```

**Requirements:**
- `RUNPOD_API_KEY` in `.env` file
- `HF_TOKEN` for model access
- `SSH_KEY_PATH` (defaults to `~/.ssh/id_ed25519`)

**Cost:** ~$20-25 for full sweep (7 models, ~4 hours total)

## Project Structure

```
sweep_configs/          # Production configs for 7 models
remote_results/         # Analysis results (3.3GB)
sweep_analysis/         # Output plots from compare_sweep_results.py
archive/                # Old development configs
```

## Models Analyzed

| Model | Total Params | Active Params | Systematic Outliers |
|-------|--------------|---------------|---------------------|
| OLMoE-1B-7B | 7B | 1.3B | 49 |
| GPT-OSS-20B | 21B | 3.6B | 1,465 |
| Qwen3-30B | 30.5B | 3.3B | 110 |
| Mixtral-8x7B | 47B | 12.9B | 4,635 |
| Qwen3-Next-80B | 80B | 3.0B | 504 |
| GLM-4.5-Air | 106B | 12.0B | 459 |
| GPT-OSS-120B | 117B | 5.1B | 1,695 |

## Methodology

Based on [Dettmers et al. (2022) - LLM.int8()](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/):

### Outlier Detection Pipeline

**Step 1: Find Raw Outliers** (`analyze_activations.py:63-134`)
- Identify all activation values with **magnitude ≥ 6.0**
- Group by feature dimension (the h_i dimension in d_model)
- Track which layers and sequence positions contain each outlier

**Step 2: Filter for Systematic Outliers** (`analyze_activations.py:136-209`)

A feature dimension is considered "systematic" only if it meets ALL criteria:
- **Magnitude threshold**: ≥6.0 activation magnitude
- **Layer coverage**: Affects ≥25% of transformer layers
- **Sequence coverage**: Affects ≥6% of sequence positions

**Dataset:** 16 sequences × 2048 tokens from FineWeb-Edu

### Understanding the Metrics

**Output Metrics Explained:**
- **Total Outliers**: Count of feature dimensions meeting all three systematic criteria
- **Mean L%**: Average percentage of layers affected by each outlier feature (higher = more pervasive across layers)
- **Mean S%**: Average percentage of sequence positions affected by each outlier feature (higher = more pervasive across tokens)

Example: GLM-4.5-Air shows 459 outliers with Mean L% = 63.3%, meaning each outlier feature appears in ~63% of the model's layers on average.

Calculation details: `compare_sweep_results.py:125-150`

### Research Context

**Key Finding from Dettmers:** Dense transformers exhibit a phase transition at ~6.7B parameters where outlier features shift from probabilistic (scattered across layers) to systematic (100% layer agreement on specific dimensions).

Dettmers claims: *"It is clear that transformers after the phase shift at 6.7B parameters behave very differently to transformers before the phase shift. As such, one should not try to generalize from <6.7B transformers to beyond 6.7B parameters."*

**Research Question:** Does this phase transition and the associated claim about behavioral differences extend to Mixture-of-Experts (MoE) architectures? This analysis tests whether MoE models show the same systematic outlier coordination patterns at scale.

**Analysis Workflow** (`run_full_analysis.py:358-449`):
1. Load model and FineWeb-Edu dataset
2. Extract activation tensors from ln_attn and ln_mlp for each layer
3. Process in batches, detecting outliers meeting the three criteria
4. Aggregate results across batches
5. Save to `final_analysis_results.json`
6. Compare across models using `generate_comparison_plots.py`

## Code Navigation

**Core Analysis Pipeline:**
- **Outlier detection logic**: `analyze_activations.py:63-134` (find_outliers_in_activations)
- **Systematic filtering**: `analyze_activations.py:136-209` (filter_systematic_outliers)
- **Full pipeline orchestration**: `run_full_analysis.py:358-449`
- **Model architecture handling**: `extract_activations.py:get_model_layers()` - normalizes layer access across different architectures

**Deployment & Comparison:**
- **Remote GPU deployment**: `deploy.py` (single model), `deploy_sweep.py` (batch deployment)
- **Result aggregation**: `generate_comparison_plots.py:153-201` (collect_all_results)
- **Metrics calculation**: `generate_comparison_plots.py:125-150` (calculate_metrics)
- **Visualization**: `generate_comparison_plots.py:203-313` (plotting functions)

**Configuration:**
- **Base config classes**: `config.py` (ModelConfig, DatasetConfig, AnalysisConfig, DeploymentConfig)
- **MoE model configs**: `sweep_configs/` (7 models from 7B-120B total params)
- **Dense baseline configs**: `sweep_dense/` (5 Qwen models from 0.6B-14B)

## Key Scripts

- `generate_comparison_plots.py` - Generate comparative analysis plots (renamed from `compare_sweep_results.py`)
- `deploy_sweep.py` - Run full 7-model sweep
- `deploy.py` - Deploy single model analysis
- `run_full_analysis.py` - Local analysis (if you have GPU)
- `scrape_model_params_from_readme.py` - Fetch model metadata from HuggingFace
