# Outlier Features Analysis

Systematic analysis of outlier features in Mixture-of-Experts (MoE) language models, following the methodology from [Dettmers et al. (2022)](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/).

## Quick Start

### 1. Analyze Existing Results

If you already have results in `remote_results/`:

```bash
# Save plots as PNG files (default)
python compare_sweep_results.py

# Display plots in terminal (requires plotext)
python compare_sweep_results.py --terminal

# Output as CSV (pipe to file if needed)
python compare_sweep_results.py --terminal --format csv > results.csv

# Output as JSON
python compare_sweep_results.py --terminal --format json > results.json

# Output as table only (no plots)
python compare_sweep_results.py --terminal --format table
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
6. Compare across models using `compare_sweep_results.py`

## Key Scripts

- `compare_sweep_results.py` - Generate comparative analysis plots
- `deploy_sweep.py` - Run full 7-model sweep
- `deploy.py` - Deploy single model analysis
- `run_full_analysis.py` - Local analysis (if you have GPU)
- `scrape_model_params_from_readme.py` - Fetch model metadata from HuggingFace
