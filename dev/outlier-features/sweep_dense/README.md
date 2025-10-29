# Dense Model Sweep Configuration

Baseline configs for dense (non-MoE) transformers to compare against MoE results in `sweep_configs/`.

## Motivation

Dettmers et al. (2022) found dense models show a phase transition at ~6.7B parameters (probabilistic → systematic outliers). These baselines test whether: (1) the phase transition still exists in modern dense models, and (2) MoE architectures behave differently.

## Model Coverage (N=11)

### Qwen3 Dense Series (N=5)
| ID | Model | Params |
|----|-------|--------|
| 01 | Qwen3-0.6B | 0.6B |
| 02 | Qwen3-1.7B | 1.7B |
| 03 | Qwen3-4B | 4B |
| 04 | Qwen3-8B | 8B |
| 05 | Qwen3-14B | 14B |

### Gemma 3 Dense Series (N=6)
| ID | Model | Params |
|----|-------|--------|
| 06 | Gemma-3-270M | 0.27B |
| 07 | Gemma-3-1B | 1B |
| 08 | Gemma-3-4B | 4B |
| 09 | Gemma-3-7B | 7B |
| 10 | Gemma-3-12B | 12B |
| 11 | Gemma-3-27B | 27B |

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

## Analysis Scripts

After sweep completes:

```bash
# 1. Compare dense vs MoE results
python compare_dense_vs_moe.py

# 2. Test phase transition hypothesis
python analyze_phase_transition.py

# 3. Generate comparison plots
python generate_comparison_plots.py --include-dense

# 4. Statistical analysis
python analyze_with_bootstrap.py --include-dense
```

## References

- **Dettmers et al. (2022)**: [LLM.int8() and Emergent Features](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)
- **Main MoE sweep**: `../sweep_configs/` (7 MoE models, 7B-120B)
- **Methodology**: `../README.md` (outlier detection pipeline)
