# Outlier Features Analysis for MoE Models

Detect systematic outlier features in large Mixture-of-Experts language models.

## Quick Start

### Local Testing (Fast)
```bash
cd examples/outlier-features
python run_full_analysis.py configs/01_olmoe_tiny.py
```
Expected: 2-3 minutes on local GPU

### Remote Deployment (Production)
```bash
python deploy.py configs/02_olmoe_baseline.py
```
Expected: Auto-provisions GPU, runs analysis, syncs results, cleans up

### Large Model (Multi-GPU)
```bash
python deploy.py configs/03_qwen_medium.py
```
Expected: 2xA100, 20-30 minutes, automatic chunking

## How to Run

**Both local and remote use the same command pattern:**
```bash
# Local execution
python run_full_analysis.py configs/<your_config>.py

# Remote execution (auto-provisions GPU)
python deploy.py configs/<your_config>.py
```

The only difference: `deploy.py` wraps `run_full_analysis.py` with GPU provisioning.

## Experiment Configs

All experiments defined as Python config files following `docs/code_style/experiment_config.md`:

**01_olmoe_tiny.py** - Quick validation (2-3 min local)
- Model: OLMoE-1B-7B
- Sequences: 2 × 512 tokens
- Layers: First 4 only
- Use for: Testing changes, fast iteration

**02_olmoe_baseline.py** - Standard analysis (10-15 min remote)
- Model: OLMoE-1B-7B
- Sequences: 4 × 2048 tokens
- Layers: All
- Use for: Baseline results, method validation

**03_qwen_medium.py** - Large model (20-30 min, 2xA100)
- Model: Qwen3-30B-A3B
- Sequences: 4 × 2048 tokens
- Layers: All (chunked processing)
- Use for: Production analysis, systematic outliers

### Creating Your Own Config

```python
# configs/04_my_experiment_05.py
from config import Config

config = Config()

# Model
config.model.name = "allenai/OLMoE-1B-7B-0125-Instruct"

# Dataset
config.dataset.num_sequences = 4
config.dataset.sequence_length = 2048

# Analysis (Dettmers et al. 2022 criteria)
config.analysis.magnitude_threshold = 6.0      # ≥6.0 activation
config.analysis.min_layer_percentage = 0.25   # ≥25% layers
config.analysis.min_seq_percentage = 0.06     # ≥6% positions

# Deployment (for remote only)
config.deployment.gpu_count = 1
config.deployment.gpu_filter = "A100"

# Output
config.output.experiment_name = "my_experiment"
```

Naming convention: `<number>_<description>_<next_number>.py` tracks experiment lineage.

## What It Does

Implements the outlier detection methodology from **Dettmers et al. (2022) "LLM.int8()"**:

1. **Extract activations** from transformer layer normalization outputs
2. **Identify outliers** with magnitude ≥6.0
3. **Find systematic outliers** appearing in:
   - ≥25% of transformer layers
   - ≥6% of sequence positions
4. **Report** outlier features with statistics

### Why This Matters

Systematic outliers affect:
- **Quantization**: int8 quantization degrades with outliers
- **Model behavior**: Outliers indicate phase transitions
- **Architecture design**: MoE vs dense models show different patterns

Key finding: Outliers are **architecture-dependent**, not size-dependent. Small MoE models can have systematic outliers that large dense models lack.

## Results

Analysis generates:

```
results/
├── final_analysis_results.json    # Main results
├── batch_001_results.json         # Per-batch breakdowns
├── config.json                    # Exact config used
└── outlier_analysis.log          # Full execution log
```

### Example Output

```json
{
  "feature_dim": 1461,
  "max_magnitude": 33.75,
  "layer_percentage": 0.354,        // 35.4% of layers
  "seq_percentage": 0.460,          // 46.0% of positions
  "total_occurrences": 5894,
  "layers_affected": ["layer_0_ln_mlp", "layer_5_ln_attn", ...],
  "example_values": [33.75, 28.12, 24.56, ...]
}
```

## Configuration Reference

### ModelConfig
- `name`: HuggingFace model identifier
- `device_map`: "auto" (single GPU) | "balanced" (multi-GPU)
- `torch_dtype`: "bfloat16" | "float16"

### DatasetConfig
- `name`: Dataset (default: "HuggingFaceFW/fineweb-edu")
- `num_sequences`: Number of text sequences
- `sequence_length`: Tokens per sequence
- `shuffle`, `seed`: For reproducibility

### AnalysisConfig
- `magnitude_threshold`: ≥6.0 (Dettmers criterion)
- `min_layer_percentage`: ≥0.25 (25% of layers)
- `min_seq_percentage`: ≥0.06 (6% of positions)
- `layers`: `None` for all, or `[0,1,2,3]` for specific
- `chunk_layers`: Process N layers at a time (memory optimization)

### DeploymentConfig (remote only)
- `min_vram`: Auto-estimated if `None`
- `gpu_count`: 1-8 GPUs
- `gpu_filter`: "A100", "H100", etc.
- `keep_running`: Keep instance after completion

### OutputConfig
- `save_dir`: Results directory
- `log_level`: "DEBUG" | "INFO" | "WARNING"
- `experiment_name`: Experiment identifier

## Architecture

### Design Philosophy

**Separation of Concerns:**
- `run_full_analysis.py` - Core analysis logic (runs locally OR remotely)
- `deploy.py` - Thin deployment wrapper (GPU provisioning only)

This differs from the original `llm-workbench` where `deploy_and_analyze.py` contained both. Benefits:
- **DRY**: Analysis logic in one place
- **Consistency**: Same code runs locally and remotely
- **Simplicity**: Deployment doesn't need to know analysis details
- **Config-driven**: Just copy config to remote and execute

### File Structure

```
examples/outlier-features/
├── config.py                    # Hierarchical dataclass config system
├── run_full_analysis.py         # Main orchestrator (local + remote)
├── deploy.py                    # Deployment wrapper (remote only)
├── extract_activations.py       # Activation extraction with nnsight
├── analyze_activations.py       # Outlier detection logic
├── dataset_utils.py            # Text sequence generation
├── estimate_vram.py            # VRAM estimation utility
└── configs/                    # Experiment configurations
    ├── 01_olmoe_tiny.py
    ├── 02_olmoe_baseline.py
    └── 03_qwen_medium.py
```

### Core Components

**extract_activations.py**
- Pure function: `extract_activations_batch(model, texts, layers)`
- Memory-optimized: `extract_activations_optimized(llm, texts, chunk_size)`
- Extracts layer normalization outputs (attention + MLP)
- Supports chunked processing for large models

**analyze_activations.py**
- `load_activations(run_dir)` - Load saved activations
- `find_outliers_in_activations(activations, threshold)` - Apply magnitude criterion
- `analyze_systematic_outliers(outlier_info)` - Apply layer/position criteria
- Tracks outliers by feature dimension across all layers

**dataset_utils.py**
- `load_streaming_dataset()` - Stream from HuggingFace
- `chunk_text_by_tokens()` - Token-based chunking with tokenizer
- `get_text_sequences()` - Extract N sequences of target length
- Supports shuffling, seeding, and sequence skipping (for sharding)

**run_full_analysis.py**
- `load_config_from_file()` - Dynamic config loading
- `load_dataset_sequences()` - Dataset + tokenizer setup
- `load_model_optimized()` - Model loading with GPU detection
- `process_single_batch()` - Extract → analyze → cleanup pipeline
- `aggregate_batch_results()` - Cross-batch aggregation
- All functions <70 lines (Tiger Style)

**deploy.py**
- `provision_gpu()` - Broker GPU provisioning
- `deploy_code()` - Bifrost code deployment
- `start_analysis()` - Upload config, launch remote execution
- `sync_results()` - Download results from remote
- `cleanup_instance()` - Terminate GPU
- All functions <70 lines (Tiger Style)

### Memory Optimization

**Chunked Layer Processing:**
```python
config.analysis.chunk_layers = 8  # Process 8 layers at a time
```
- Loads model once
- Processes layers in chunks
- Immediately saves to disk and frees GPU memory
- Essential for large models (30B+ parameters)

**Activation Cleanup:**
- Saves activations as `.pt` files
- Analyzes immediately
- Deletes `.pt` files to save disk
- Keeps only JSON results

## Code Style Compliance

Following `docs/code_style/` principles:

**Tiger Style (Safety):**
- ✅ All functions <70 lines
- ✅ High assertion density (≥2 per function)
- ✅ Explicit bounds on all loops
- ✅ Paired assertions (pre/post tensor shapes)
- ✅ No recursion, explicit control flow

**Casey Muratori (API Design):**
- ✅ Pure functions (`extract_activations_batch`, `chunk_text_by_tokens`)
- ✅ Transparent data (JSON metadata, not binary blobs)
- ✅ No hidden coupling (explicit config passing)
- ✅ Easy to remove abstractions

**Experiment Config:**
- ✅ Dataclass-based hierarchical config
- ✅ Experiment versioning: `01_name_02.py` pattern
- ✅ Config serialization (save/load from JSON)
- ✅ Version controlled in git

## Utilities

### VRAM Estimation
```bash
python estimate_vram.py --model "Qwen/Qwen3-30B-A3B"
```

Output:
- Estimated parameters (total + effective for MoE)
- VRAM breakdown (model + KV cache + activations)
- GPU recommendations (RTX 4090, A100, etc.)
- Auto-used by `deploy.py` if `min_vram=None`

### Manual Remote Execution
```bash
# SSH into instance
bifrost exec "user@instance" "cd ~/.bifrost/workspace/examples/outlier-features && bash"

# Monitor analysis
tail -f outlier_analysis.log

# Sync results manually
bifrost download "user@instance" "~/.bifrost/workspace/examples/outlier-features/results" ./results
```

## Troubleshooting

**Out of Memory**
- Reduce `num_sequences` or `sequence_length`
- Enable `chunk_layers` (e.g., 8 layers at a time)
- Increase `deployment.min_vram`

**No Outliers Found**
- Try longer sequences or more complex text
- Check different layers (early vs late)
- Lower `magnitude_threshold` (< 6.0)

**Deployment Failures**
- Check GPU availability: `broker instances list`
- Verify HF_TOKEN set: `echo $HF_TOKEN`
- Check disk space (increase `container_disk`)

**Import Errors**
- Ensure in research repo root
- Check `shared/` module is available
- Run from `examples/outlier-features/` directory

## References

**Methodology:**
- Dettmers et al. (2022) "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"
- With optimizations for Mixture-of-Experts architectures

**Original Implementation:**
- Ported from `llm-workbench/examples/outlier_features_moe`
- Refactored for config-based experiments and Tiger Style compliance

**Related Work:**
- See `../corpus-proximity/` for similar config-based experiment pattern

## Development

See `plan.md` for detailed implementation plan and function splitting strategy.

See `todo.md` for task list and current status.
