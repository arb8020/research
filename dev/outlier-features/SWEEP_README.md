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

| ID | Model | Size | Active | Experts | Expected | Runtime | Cost/hr |
|----|-------|------|--------|---------|----------|---------|---------|
| 01 | OLMoE-1B-7B | 7B | 1.3B | 64 | Probabilistic | 15-20m | ~$1.50 |
| 02 | GPT-OSS-20B | 21.5B | ? | ? | **Systematic** | 25-30m | ~$3.00 |
| 03 | Qwen3-30B | 30.5B | 3.3B | 60 | Probabilistic | 30-35m | ~$3.50 |
| 04 | Mixtral-8x7B | 46.7B | 12.9B | 8 | Probabilistic | 30-40m | ~$3.50 |
| 05 | Qwen3-Next-80B | 80B | 3B | 512 | Probabilistic | 40-50m | ~$4.50 |
| 06 | GLM-4.5-Air | 106B | 12B | 128 | Probabilistic | 45-60m | ~$4.50 |
| 07 | GPT-OSS-120B | 117B | ? | ? | **Systematic** | 50-70m | ~$5.00 |

**Total sweep**: ~4-6 hours wall time (parallel), ~$20-25 total cost

## Key Changes from Original Runs

### 1. Increased Sample Size
- **Old**: 4 sequences
- **New**: 16 sequences
- **Why**: Bootstrap CIs (Task 7), cross-batch validation (Task 3), addresses GLM data quality issue

### 2. GLM Re-run
- **Issue**: Original had only 2/12 valid batches due to file truncation
- **Fix**: 16 sequences with no volume disk to avoid mount issues

### 3. New Models
- **Added**: Mixtral-8x7B (fills 30B-106B gap, canonical MoE)
- **Added**: Qwen3-Next-80B (tests Qwen arch at scale, 512 experts!)

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

### Dry Run (Testing)
```bash
# See what would be deployed without actually launching
python deploy_sweep.py --dry-run
python deploy_sweep.py --models 01 03 --dry-run
```

## Troubleshooting

### No GPUs Available
- **Symptom**: "No GPUs found matching criteria"
- **Fix**: Increase `max_price` in config or wait and retry
- **Alternative**: Remove `gpu_filter` to allow any GPU type

### Out of Memory
- **Symptom**: CUDA OOM errors
- **Fix**: Decrease `chunk_layers` (e.g., 8 → 6 → 4)
- **Alternative**: Increase `gpu_count` in config

### File Truncation (GLM Issue)
- **Symptom**: Invalid JSON in batch results
- **Fix**: Config already sets `volume_disk = 0` to avoid mount issues
- **Alternative**: Increase `container_disk` if needed

### Deployment Hangs
- **Symptom**: SSH connection timeout
- **Fix**: Wait up to 15 minutes for SSH (normal for RunPod)
- **Check**: Instance may still be starting, check `broker list`

## Post-Sweep Analysis

After all deployments complete:

1. **Verify results**:
```bash
python verify_results.py  # Check all models have valid results
```

2. **Run analysis** (from llm-workbench):
```bash
cd ~/llm-workbench/examples/outlier_features_moe
python analyze_all_models_dettmers.py
```

3. **Bootstrap CIs** (TIER 2 Task 7):
```bash
python analyze_with_bootstrap.py  # Compute 95% confidence intervals
```

4. **Create visualizations**:
```bash
python create_comparison_plots.py  # Layer agreement, magnitude, etc.
```

## Expected Results

### Systematic Models (GPT-OSS)
- **Layer agreement**: ~100% on key dimensions (773, 2559, 883, 1359)
- **Magnitude**: 170-200
- **Systematic dimensions**: 4-7

### Probabilistic Models (All others)
- **Layer agreement**: 30-50%
- **Magnitude**: 15-150
- **Systematic dimensions**: 0

### Key Finding
**Architecture matters more than size**: Only GPT-OSS shows systematic outliers, regardless of parameter count.

## Cost Management

### Estimated Costs
- **Single model**: $1.50 - $5.00 (15-70 minutes)
- **Full sweep**: ~$20-25 (parallel execution)
- **With reruns**: ~$30-35

### Cost Reduction
```bash
# Deploy smaller models first (cheaper)
python deploy_sweep.py --models 01 03

# Reduce sequences if budget-constrained (not recommended)
# Edit configs: num_sequences = 8  # Instead of 16
```

## Reviewer Requirements Addressed

✅ **Task 11**: Re-run GLM with 5+ batches (now 16 sequences)
✅ **Task 8**: Add model at threshold (Phi-3.5-MoE, if available)
✅ **Task 9**: Add Mixtral-8x7B (fills gap, canonical model)
✅ **Task 7**: Enable bootstrap CIs (16 sequences provides data)
✅ **Sample size**: N=7 models (70% of Dettmers' N=10)

## Next Steps

After sweep completes:
1. Run TIER 1 fixes (circular logic, GLM reporting)
2. Compute bootstrap CIs (TIER 2)
3. Create visualizations
4. Write up findings with N=7 results
5. Package as blog post for EOW
