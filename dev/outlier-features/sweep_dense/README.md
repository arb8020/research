# Dense Model Sweep Configuration

Baseline configs for dense (non-MoE) transformers to compare against MoE results in `sweep_configs/`.

## Research Motivation

**Dettmers et al. (2022)** found that dense transformers exhibit a **phase transition at ~6.7B parameters** where outlier features shift from:
- **Below 6.7B**: Probabilistic (scattered across layers, 30-50% layer agreement)
- **Above 6.7B**: Systematic (100% layer agreement on specific dimensions)

**Our Question**: Do MoE models show the same behavior, or does the sparse expert architecture change this pattern?

**Why We Need Dense Baselines**: The MoE results in `sweep_configs/` only tell us "MoE models have outliers." To answer whether MoE is *different*, we need dense model baselines at comparable parameter counts.

## Model Coverage (N=11)

### Qwen3 Dense Series (N=5)
Matches architectural family with Qwen3-30B MoE from main sweep.

| ID | Model | Params | Expected Behavior | Runtime |
|----|-------|--------|-------------------|---------|
| 01 | Qwen3-0.6B | 0.6B | Probabilistic | ~10-15m |
| 02 | Qwen3-1.7B | 1.7B | Probabilistic | ~15-20m |
| 03 | Qwen3-4B | 4B | Probabilistic | ~20-25m |
| 04 | **Qwen3-8B** | **8B** | **Systematic?** | ~25-30m |
| 05 | Qwen3-14B | 14B | Systematic? | ~30-35m |

### Gemma 3 Dense Series (N=6)
Second architectural family for generalization testing.

| ID | Model | Params | Expected Behavior | Runtime |
|----|-------|--------|-------------------|---------|
| 06 | Gemma-3-270M | 0.27B | Probabilistic | ~8-12m |
| 07 | Gemma-3-1B | 1B | Probabilistic | ~12-18m |
| 08 | Gemma-3-4B | 4B | Probabilistic | ~20-25m |
| 09 | **Gemma-3-7B** | **7B** | **Systematic?** | ~25-30m |
| 10 | Gemma-3-12B | 12B | Systematic? | ~30-40m |
| 11 | Gemma-3-27B | 27B | Systematic? | ~40-50m |

**Total sweep**: ~5-7 hours wall time (parallel), ~$25-30 total cost

## Key Comparisons Enabled

### 1. Phase Transition Test
- **Below 6.7B**: Qwen 0.6B/1.7B/4B, Gemma 270M/1B/4B
- **At threshold**: Qwen 8B, Gemma 7B
- **Above threshold**: Qwen 14B, Gemma 12B/27B

### 2. MoE vs Dense at Same Scale
- **~1B**: OLMoE-1B-7B (MoE) vs Qwen 0.6B/1.7B, Gemma 1B (dense)
- **~7-8B**: OLMoE-1B-7B total vs Qwen 8B, Gemma 7B (dense)
- **~30B**: Qwen3-30B-A3B (MoE) vs Gemma 27B (dense)

### 3. Architectural Generalization
- **Qwen family**: 5 dense sizes
- **Gemma family**: 6 dense sizes
- Tests if phase transition is architecture-specific or universal

## Usage

### Run Full Dense Sweep
```bash
# Deploy all 11 models in parallel
python deploy_sweep.py --config-dir sweep_dense

# Or deploy specific models (e.g., just the 6.7B threshold models)
python deploy.py sweep_dense/04_qwen3_8b.py
python deploy.py sweep_dense/09_gemma3_7b.py
```

### Run Individual Models
```bash
# Test smallest model first (cheapest)
python deploy.py sweep_dense/06_gemma3_270m.py

# Test threshold models
python deploy.py sweep_dense/04_qwen3_8b.py  # Qwen at 8B
python deploy.py sweep_dense/09_gemma3_7b.py  # Gemma at 7B
```

### Monitor Progress
```bash
broker list
tail -f logs/deploy_*.log
ls -R results/
```

## Expected Results

### Hypothesis 1: Dense models replicate Dettmers
- **0.6B-4B models**: Probabilistic outliers (~30-50% layer agreement)
- **7B-27B models**: Systematic outliers (~100% layer agreement)
- **Clear discontinuity** at 6.7B threshold

### Hypothesis 2: MoE shows different behavior
- **MoE models**: Probabilistic even at large total params (e.g., Qwen3-30B MoE has 110 outliers)
- **Dense models**: Systematic above 6.7B (e.g., Gemma 27B should show 100% layer agreement)
- **Conclusion**: Sparse expert routing prevents systematic coordination

### Hypothesis 3: Outlier count scales with params
- **Dense**: Should see increasing outlier counts from 0.6B → 27B
- **MoE**: Outlier counts vary (OLMoE 49, Qwen3-30B 110, Mixtral 4,635)
- **Compare**: Dense vs MoE scaling patterns

## Cost Estimates

| Size Range | Models | Runtime | Cost/Model | Subtotal |
|------------|--------|---------|------------|----------|
| 0.6B-4B | 6 models | 10-25m | $1.50-2.00 | ~$10 |
| 7B-14B | 4 models | 25-35m | $2.50-3.00 | ~$11 |
| 27B | 1 model | 40-50m | $3.50 | ~$4 |

**Total**: ~$25-30 for full 11-model sweep

## Deployment Configuration

All configs use standardized parameters:
- **16 sequences** - for bootstrap CIs and statistical power
- **2048 sequence length** - standard from Dettmers et al.
- **batch_size = 1** - memory optimization
- **threshold = 6.0** - Dettmers' magnitude threshold

### GPU Requirements
- **0.6B-4B**: 1xA100 (40GB VRAM)
- **7B-14B**: 1xA100 (40GB VRAM) with chunking
- **27B**: 2xA100 (80GB total) with balanced device_map

## Analysis Scripts

After sweep completes:

```bash
# 1. Compare dense vs MoE results
python compare_dense_vs_moe.py

# 2. Test phase transition hypothesis
python analyze_phase_transition.py

# 3. Generate comparison plots
python compare_sweep_results.py --include-dense

# 4. Statistical analysis
python analyze_with_bootstrap.py --include-dense
```

## Next Steps

1. **Dry run**: `python deploy_sweep.py --config-dir sweep_dense --dry-run`
2. **Deploy threshold models first**: Qwen 8B + Gemma 7B (~$5, answers key question)
3. **If phase transition confirmed**: Deploy remaining 9 models
4. **Analysis**: Compare against MoE results in `remote_results/`
5. **Write-up**: Blog post with dense vs MoE comparison

## References

- **Dettmers et al. (2022)**: [LLM.int8() and Emergent Features](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)
- **Main MoE sweep**: `../sweep_configs/` (7 MoE models, 7B-120B)
- **Methodology**: `../README.md` (outlier detection pipeline)
