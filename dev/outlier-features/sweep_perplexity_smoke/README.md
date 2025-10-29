# Perplexity Smoke Tests

Quick validation configs for testing perplexity computation pipeline before running full-scale experiments.

## Purpose

These configs run perplexity computation on **10 sequences** (instead of 100+) to:
- Validate the pipeline works end-to-end
- Test model loading and inference
- Verify GPU provisioning and deployment
- Quick turnaround (~5-10 minutes per model)

## Models (12 total)

**MoE Models (7):**
1. **OLMoE-1B-7B** - 7B total, 1.3B active
2. **GPT-OSS-20B** - 21.5B total, 3.6B active
3. **Qwen3-30B** - 30.5B total, 3.3B active
4. **Mixtral-8x7B** - 46.7B total, 12.9B active
5. **Qwen3-Next-80B** - 80B total, 3B active
6. **GLM-4.5-Air** - 106B total, 12B active
7. **GPT-OSS-120B** - 117B total, 5.1B active

**Dense Models (5):**
1. **Qwen3-0.6B** - 0.6B params
2. **Qwen3-1.7B** - 1.7B params
3. **Qwen3-4B** - 4B params
4. **Qwen3-8B** - 8B params
5. **Qwen3-14B** - 14B params

## Usage

**Run a single smoke test:**
```bash
python deploy.py sweep_perplexity_smoke/01_olmoe_1b_7b.py --mode perplexity
```

**Run all smoke tests in parallel:**
```bash
python deploy_sweep.py --config-dir sweep_perplexity_smoke --mode perplexity
```

**Run specific models only:**
```bash
# Just the small models first
python deploy_sweep.py --config-dir sweep_perplexity_smoke --mode perplexity --models 01 02 03

# Then the large models
python deploy_sweep.py --config-dir sweep_perplexity_smoke --mode perplexity --models 04 05 06 07 08 09

# Dense models only
python deploy_sweep.py --config-dir sweep_perplexity_smoke --mode perplexity --models 10 11 12
```

## After Smoke Tests Pass

Create full-scale configs in `sweep_perplexity/` with:
- More sequences (100-1000 for robust perplexity estimates)
- All models you want to analyze
- Appropriate VRAM/GPU settings for larger models

## Expected Output

Each run creates:
- `perplexity_results.json` - Contains perplexity value and metrics
- `config.json` - Config used for reproducibility
- `perplexity_computation.log` - Full execution log

## Next Steps

1. Run smoke tests to validate pipeline
2. Create `sweep_perplexity/` directory
3. Copy and modify these configs for full runs
4. Run perplexity sweep across all models
5. Combine with outlier analysis results for Figure 3b
