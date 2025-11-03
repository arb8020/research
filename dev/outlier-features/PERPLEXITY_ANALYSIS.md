# Perplexity Analysis - Dettmers Figure 3b Replication

This document describes the workflow for computing perplexity and replicating Dettmers et al. (2022) Figure 3b, which shows outlier emergence as a function of perplexity rather than model size.

## Overview

The perplexity analysis pipeline:
1. Computes validation perplexity on FineWeb-Edu for each model
2. Matches perplexity data with existing outlier analysis results
3. Generates plots showing outlier emergence vs. perplexity

## Key Insight from Dettmers

While Figure 3a shows a **sudden phase shift** around 6-7B parameters, Figure 3b reveals that outlier emergence is actually **smooth and exponential** when plotted against perplexity. This suggests:
- Perplexity (model quality) better predicts outlier emergence than raw parameter count
- Emergence is gradual and predictable from smaller models
- Model size alone doesn't determine emergence

## Workflow

### 1. Compute Perplexity

**Quick smoke test:**
```bash
# Test on one model (10 sequences, ~5-10 min)
python deploy.py sweep_perplexity_smoke/01_olmoe_1b_7b.py --mode perplexity
```

**Full perplexity sweep:**
```bash
# After smoke tests pass, create full configs in sweep_perplexity/
# Recommended: 100-1000 sequences for robust estimates

# Run sweep across all models
python deploy_sweep.py --config-dir sweep_perplexity
```

### 2. Generate Plots

**After both perplexity and outlier analysis are complete:**

```bash
# Generate Figure 3b replication plots (saves to perplexity_analysis/)
python generate_perplexity_plots.py

# View in terminal instead
python generate_perplexity_plots.py --terminal

# Export data only
python generate_perplexity_plots.py --terminal --format csv
python generate_perplexity_plots.py --terminal --format json
```

**Custom directories:**
```bash
python generate_perplexity_plots.py \
  --perplexity-dir remote_results \
  --outlier-dir remote_results
```

## Outputs

### Perplexity Computation
Each run creates in `results/` or `remote_results/`:
- `perplexity_results.json` - Perplexity value and metrics
- `config.json` - Config used for reproducibility
- `perplexity_computation.log` - Full execution log

### Plots Generated
The `generate_perplexity_plots.py` script creates 5 plots in `perplexity_analysis/`:

1. **`01_perplexity_vs_mean_layer_pct.png`** - Mean % layers affected vs perplexity
2. **`02_perplexity_vs_median_layer_pct.png`** - Median % layers affected vs perplexity
3. **`03_perplexity_vs_mean_seq_pct.png`** - Mean % sequence positions vs perplexity
4. **`04_perplexity_vs_median_seq_pct.png`** - Median % sequence positions vs perplexity
5. **`05_perplexity_vs_num_outliers.png`** - Number of outlier dimensions vs perplexity

## Expected Results

Based on Dettmers' findings, you should see:
- **Exponential relationship**: As perplexity decreases (model quality improves), outlier presence increases exponentially
- **Smooth curves**: Unlike parameter count, perplexity shows gradual emergence
- **Monotonic**: Lower perplexity → more outliers (strict monotonic relationship)

## File Structure

```
sweep_perplexity_smoke/     # Smoke test configs (10 sequences each)
  01_olmoe_1b_7b.py
  02_qwen3_0.6b.py
  03_qwen3_1.7b.py

sweep_perplexity/           # Full configs (create after smoke tests)
  # Copy from smoke/ and increase num_sequences to 100-1000

remote_results/             # Results from deploy.py
  perplexity_smoke_*/       # Perplexity results
    perplexity_results.json
    config.json
  outlier_analysis_*/       # Outlier analysis results
    final_analysis_results.json
    config.json

perplexity_analysis/        # Generated plots
  01_perplexity_vs_*.png
  ...
```

## Scripts

- **`compute_perplexity.py`** - Standalone perplexity computation (no nnsight)
- **`deploy.py --mode perplexity`** - Deploy perplexity computation to GPU
- **`generate_perplexity_plots.py`** - Match results and generate Figure 3b plots

## Configuration

Perplexity configs should specify:
```python
# Dataset: More sequences = better perplexity estimate
config.dataset.num_sequences = 100  # 10 for smoke test, 100-1000 for production

# Analysis: Batch size for perplexity computation
config.analysis.batch_size = 2  # Adjust based on model size

# Deployment: Same as outlier analysis
config.deployment.min_vram = None  # Auto-estimate
```

## Tips

1. **Start with smoke tests** - Validate pipeline before full runs
2. **Use same dataset** - Keep FineWeb-Edu for consistency with outlier analysis
3. **Sufficient sequences** - Use 100+ sequences for stable perplexity estimates
4. **Match model sets** - Run perplexity for all models you have outlier data for
5. **Check results** - Lower perplexity should correlate with more outliers

## Troubleshooting

**No matched results:**
- Ensure model names match exactly between perplexity and outlier results
- Check that both `perplexity_results.json` and `final_analysis_results.json` exist
- Verify model is in `MODEL_METADATA` in `generate_perplexity_plots.py`

**Perplexity too high/low:**
- Check dataset split (should use 'train' for FineWeb-Edu)
- Verify sequence_length matches (should be 2048)
- Ensure model loaded correctly (check logs)

**Plot generation fails:**
- Install matplotlib: `uv add matplotlib`
- For terminal plots: `uv add plotext`
- For rich tables: `uv add rich`
