# Outlier Features Analysis for MoE Models

Systematic analysis of outlier features in Mixture-of-Experts language models, replicating and extending [Dettmers et al. (2022)](https://arxiv.org/abs/2208.07339).

## Quick Start

```bash
# 1. Set up credentials
cp .env.example .env
# Add your API keys: RUNPOD_API_KEY, HF_TOKEN

# 2. Launch full 7-model sweep
cd examples/outlier-features
python deploy_sweep.py

# 3. Check status (after ~1 hour)
python deploy_sweep.py --status
broker list

# 4. Results appear in results/ as each model completes
ls -R results/
```

**Time**: 4-6 hours (all models run in parallel)
**Cost**: ~$20-25 total
**Output**: 7 models × 16 sequences × analysis results

## What This Does

Analyzes **systematic outlier features** in MoE models following Dettmers' methodology:

1. Extract activations from layer normalization outputs
2. Identify outliers with magnitude ≥6.0
3. Find systematic outliers appearing in ≥25% layers and ≥6% positions
4. Compare layer agreement across models

**Key Finding**: Only GPT-OSS models show systematic outliers (100% layer agreement on dimensions like 773, 2559). Other MoE models (OLMoE, Qwen, Mixtral, GLM) remain probabilistic despite larger size.

→ **Architecture matters more than size for systematic outliers**

## Models Analyzed (N=7)

| Model | Size | Active | Experts | Expected Result | Runtime |
|-------|------|--------|---------|-----------------|---------|
| OLMoE-1B-7B | 7B | 1.3B | 64 | Probabilistic | ~20m |
| GPT-OSS-20B | 21.5B | ? | ? | **Systematic** | ~30m |
| Qwen3-30B-A3B | 30.5B | 3.3B | 60 | Probabilistic | ~35m |
| Mixtral-8x7B | 46.7B | 12.9B | 8 | Probabilistic | ~40m |
| Qwen3-Next-80B | 80B | 3B | 512 | Probabilistic | ~50m |
| GLM-4.5-Air | 106B | 12B | 128 | Probabilistic | ~60m |
| GPT-OSS-120B | 117B | ? | ? | **Systematic** | ~70m |

All models use standardized parameters:
- **16 sequences** (not 4) - for bootstrap confidence intervals
- **2048 token length** - matching Dettmers
- **Dettmers criteria** - magnitude ≥6.0, ≥25% layers, ≥6% positions

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  deploy_sweep.py - Parallel Launcher                            │
│  • Launches 7 deploy.py subprocesses (30s stagger)              │
│  • Each writes to logs/deploy_XX_*.log                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │  deploy.py (per model)           │
              │  1. Estimate VRAM                │
              │  2. Provision GPU via broker     │
              │  3. Deploy code via bifrost      │
              │  4. Run analysis remotely        │
              │  5. Sync results back            │
              │  6. Terminate GPU                │
              └────────────────┬────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    ┌────▼────┐         ┌─────▼─────┐        ┌─────▼─────┐
    │ broker  │         │  bifrost  │        │  Remote   │
    │ (GPU    │────────▶│  (code    │───────▶│  GPU      │
    │  mgmt)  │         │   sync)   │        │  Instance │
    └─────────┘         └───────────┘        └─────┬─────┘
                                                    │
                                            ┌───────▼───────┐
                                            │ run_full_     │
                                            │   analysis.py │
                                            │ • Load model  │
                                            │ • Extract     │
                                            │ • Analyze     │
                                            │ • Save JSON   │
                                            └───────────────┘
```

### Component Flow

1. **deploy_sweep.py**: Orchestrator
   - Launches N parallel `deploy.py` processes
   - No coordination after launch (fire-and-forget)

2. **deploy.py**: Per-model deployer
   - Provisions GPU via `broker` (RunPod)
   - Deploys code via `bifrost` (SSH/rsync)
   - Runs `run_full_analysis.py` remotely
   - Syncs results back to `results/`
   - Auto-terminates GPU on completion

3. **run_full_analysis.py**: Analysis pipeline (runs on GPU)
   - Loads model with chunking
   - Extracts activations (16 sequences × N layers)
   - Identifies outliers (magnitude ≥6.0)
   - Finds systematic features (≥25% layers, ≥6% positions)
   - Saves results as JSON

4. **broker**: GPU provisioning client
   - Searches RunPod for available GPUs
   - Creates instances with SSH access
   - Monitors ready state

5. **bifrost**: Code deployment client
   - Syncs local code to remote via SSH
   - Handles environment setup
   - Manages execution and result sync

## File Structure

```
examples/outlier-features/
├── README.md                   # This file
├── SWEEP_README.md             # Detailed sweep guide
│
├── sweep_configs/              # 16-sequence standardized configs
│   ├── 01_olmoe_1b_7b.py
│   ├── 02_gpt_oss_20b.py
│   ├── 03_qwen3_30b.py
│   ├── 04_mixtral_8x7b.py
│   ├── 05_qwen_next_80b.py
│   ├── 06_glm_45_air.py
│   └── 07_gpt_oss_120b.py
│
├── deploy_sweep.py             # Parallel launcher
├── deploy.py                   # Per-model deployer
├── run_full_analysis.py        # Remote analysis script
│
├── config.py                   # Config dataclasses
├── extract_activations.py      # Activation extraction
├── analyze_activations.py      # Outlier detection
├── dataset_utils.py            # FineWeb-Edu loading
├── estimate_vram.py            # VRAM calculation
│
├── logs/                       # Deployment logs
│   └── deploy_01_*.log
│
└── results/                    # Output (synced from remote)
    ├── olmoe_1b_7b_sweep/
    │   ├── final_analysis_results.json
    │   ├── batch_001_results.json
    │   └── config.json
    └── ...
```

## Replication Steps

### 1. Environment Setup
```bash
# Install dependencies
pip install torch transformers datasets nnsight

# Install broker/bifrost (GPU management)
pip install -e broker/
pip install -e bifrost/

# Set credentials
export RUNPOD_API_KEY="your_key"
export HF_TOKEN="your_token"
```

### 2. Single Model Test
```bash
# Test locally on small model (requires GPU)
python run_full_analysis.py sweep_configs/01_olmoe_1b_7b.py

# Or deploy to remote GPU
python deploy.py sweep_configs/01_olmoe_1b_7b.py
```

### 3. Full Sweep
```bash
# Dry run (shows plan without executing)
python deploy_sweep.py --dry-run

# Launch all 7 models
python deploy_sweep.py

# Selective launch
python deploy_sweep.py --models 02 07  # Just GPT-OSS models
```

### 4. Monitor Progress
```bash
# Check GPU instances
broker list

# Check deployment logs
tail -f logs/deploy_01_*.log

# Check results
ls -R results/
```

### 5. Analyze Results
```bash
# After all complete, aggregate results
python analyze_all_results.py

# Create visualizations
python create_plots.py

# Generate summary report
python generate_report.py
```

## Expected Results

### Systematic Models (GPT-OSS only)
```json
{
  "feature_dim": 773,
  "layer_percentage": 1.0,      // 100% of layers
  "max_magnitude": 177.0,
  "classification": "SYSTEMATIC"
}
```

### Probabilistic Models (all others)
```json
{
  "feature_dim": 1461,
  "layer_percentage": 0.396,    // ~40% of layers
  "max_magnitude": 47.8,
  "classification": "PROBABILISTIC"
}
```

## Cost Management

- **Single model**: $1.50-5.00 (15-70 min)
- **Full sweep**: ~$20-25 (parallel)
- **GPUs used**: RunPod A100 80GB (~$1.50-2.50/hr)

Reduce costs:
```bash
# Deploy cheaper models first
python deploy_sweep.py --models 01 03

# Adjust max_price in configs
# Edit sweep_configs/*.py: config.deployment.max_price = 2.0
```

## Troubleshooting

**No GPUs available**: Increase `max_price` or remove `gpu_filter` in config

**Out of memory**: Decrease `chunk_layers` (8 → 6 → 4)

**SSH timeout**: Normal, wait up to 15 minutes for RunPod instances

**File truncation**: Already fixed (GLM config sets `volume_disk=0`)

## Citation

```bibtex
@article{dettmers2022llm,
  title={LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale},
  author={Dettmers, Tim and Lewis, Mike and Belkada, Younes and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2208.07339},
  year={2022}
}
```

## Related Work

- Original implementation: `llm-workbench/examples/outlier_features_moe`
- Methodology: Dettmers et al. (2022) LLM.int8()
- Models: OLMoE, GPT-OSS, Qwen3, Mixtral, GLM-4.5
