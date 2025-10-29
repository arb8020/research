# Model Sweep Deployment Guide

Parallel deployment system for running outlier analysis on all 7 MoE models.

## Quick Start

```bash
# Dry run (shows what will be deployed)
python deploy_sweep.py --dry-run

# Launch all 7 models
python deploy_sweep.py

# Launch specific models
python deploy_sweep.py --models 01 03 06

# Check status
python deploy_sweep.py --status
broker list
```

## Model Configuration (N=7)

All models use standardized parameters following reviewer feedback:
- **16 sequences** (up from 4) - for bootstrap CIs and cross-batch validation
- **2048 sequence length** - standard from Dettmers et al.
- **batch_size = 1** - memory optimization
- **Proper chunking** - varies by model size

| ID | Model | Size | Active | Experts | Runtime |
|----|-------|------|--------|---------|---------|
| 01 | OLMoE-1B-7B | 7B | 1.3B | 64 | 15-20m |
| 02 | GPT-OSS-20B | 21.5B | ? | ? | 25-30m |
| 03 | Qwen3-30B | 30.5B | 3.3B | 60 | 30-35m |
| 04 | Mixtral-8x7B | 46.7B | 12.9B | 8 | 30-40m |
| 05 | Qwen3-Next-80B | 80B | 3B | 512 | 40-50m |
| 06 | GLM-4.5-Air | 106B | 12B | 128 | 45-60m |
| 07 | GPT-OSS-120B | 117B | ? | ? | 50-70m |

**Total sweep**: ~4-6 hours wall time (parallel)

## Directory Structure

```
examples/outlier-features/
├── sweep_configs/              # NEW: Standardized 16-sequence configs
│   ├── 01_olmoe_1b_7b.py
│   ├── 02_gpt_oss_20b.py
│   ├── 03_qwen3_30b.py
│   ├── 04_mixtral_8x7b.py
│   ├── 05_qwen_next_80b.py
│   ├── 06_glm_45_air.py
│   └── 07_gpt_oss_120b.py
├── deploy_sweep.py             # NEW: Parallel launcher
├── logs/                       # NEW: Per-model deployment logs
│   └── deploy_01_*.log
└── results/                    # Output directory
    ├── olmoe_1b_7b_sweep/
    ├── gpt_oss_20b_sweep/
    └── ...
```

## Usage Examples

### Full Sweep (Recommended)
```bash
# Launch all 7 models with 30s delay between starts
python deploy_sweep.py

# Custom delay
python deploy_sweep.py --delay 60
```

### Selective Deployment
```bash
# Just re-run GLM and add new models
python deploy_sweep.py --models 04 05 06

# Only systematic models (GPT-OSS)
python deploy_sweep.py --models 02 07
```

### Monitoring
```bash
# Check running GPUs
broker list

# Check deployment logs
tail -f logs/deploy_01_*.log
ls -lt logs/

# Check results
ls -R results/
```

### Dry Run
```bash
python deploy_sweep.py --dry-run
```

## Post-Sweep Analysis

After all deployments complete:

1. **Run comparison analysis**:
```bash
python generate_comparison_plots.py  # Generate plots and summary table
```

2. **View results in different formats**:
```bash
# Display plots in terminal
python generate_comparison_plots.py --terminal

# Export as CSV
python generate_comparison_plots.py --terminal --format csv > results.csv

# Export as JSON
python generate_comparison_plots.py --terminal --format json > results.json
```

## Analysis Workflow

After sweep completes:
1. Run `generate_comparison_plots.py` for comparative analysis
2. Review plots in `sweep_analysis/` directory
3. Compare MoE results with dense baselines (see `sweep_dense/README.md`)
4. Validate methodology with replication sweep (see `sweep_validation/README.md`)
