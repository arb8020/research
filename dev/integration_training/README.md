# Integration Training

Post-training orchestration using the `rollouts/` module with nanochat-inspired hyperparameters.

## Overview

Single config + deploy setup for SFT and RL training:
- **base_config.py**: Schema (Tiger Style: no defaults, all explicit)
- **deploy.py**: Orchestrator (uses `rollouts/` like outlier-features uses broker/bifrost)
- **configs/**: Experiment configs with explicit parameter specification

## Quick Start

```bash
# SFT + RL pipeline (debug mode)
python deploy.py configs/01_debug_sft_rl.py

# Runs:
# 1. SFT training: 100 steps on 1K SmolTalk samples
# 2. RL training: Auto-loads SFT checkpoint, trains on 500 GSM8K problems
# Total runtime: ~15-20 minutes on single GPU
```

Mode (sft|rl|sft+rl) is auto-detected from `config.output.mode`. No flags needed!

## Tiger Style: No Defaults, All Explicit

Following **qwen3_next pattern** - configs construct Config() with ALL parameters:

```python
# configs/01_debug_sft_rl.py
config = Config(
    target=TargetConfig(
        gpu_ranks=[0],  # Single GPU
        device_type="cuda",
    ),
    model=ModelConfig(
        name="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        compile=False,
    ),
    data=DataConfig(
        sft_mixture=[...],  # All explicit
        rl_mixture=[...],   # All explicit
        max_length=512,
        shuffle_seed=42,
    ),
    sft=SFTConfig(
        # ALL 11 parameters explicit
        num_epochs=1,
        num_iterations=100,
        batch_size=2,
        # ...
    ),
    rl=RLConfig(
        # ALL 13 parameters explicit
        num_epochs=1,
        examples_per_step=8,
        # ...
    ),
    output=OutputConfig(
        save_dir=Path("./results"),
        log_level="INFO",
        experiment_name="01_debug_sft_rl",
        use_wandb=False,
        wandb_project="integration_training",
        mode="sft+rl",  # Run both stages
        source_checkpoint=None,
    ),
)
```

**Why no defaults?** (Tiger Style safety)
- ✅ **No hidden defaults** - Every parameter visible in config file
- ✅ **Explicit construction** - `Config(target=..., model=..., sft=..., rl=...)`
- ✅ **Reproducible** - Config file shows EXACTLY what was used
- ✅ **No surprises** - If base_config.py changes, configs stay the same

## Data Mixtures

### Flexible TaskMixture-style configs

Configurable data mixes via `DatasetSpec`:

```python
config.data.sft_mixture = [
    DatasetSpec(
        name="allenai/ai2_arc",
        split="train",
        subset="ARC-Easy",
        max_samples=2300,
        repeat=1,
    ),
    DatasetSpec(
        name="HuggingFaceTB/smol-smoltalk",
        split="train",
        max_samples=10_000,
    ),
]
```

### nanochat Reference Mixtures

**SFT** (chat_sft.py):
- ARC-Easy: 2.3K | ARC-Challenge: 1.1K | GSM8K: 8K | SmolTalk: 10K | identity: 1K | spelling: 600
- **Total: ~23K rows**

**RL** (chat_rl.py):
- GSM8K main train: 8K problems

## GPU Configuration

Specify GPU ranks explicitly (Tiger Style):

```python
target=TargetConfig(
    gpu_ranks=[0],          # Single GPU
    # gpu_ranks=[0, 1],     # 2 GPUs
    # gpu_ranks=[0,1,2,3],  # 4 GPUs
    device_type="cuda",
)
```

Sets `CUDA_VISIBLE_DEVICES` automatically.

## File Structure

```
dev/integration_training/
├── base_config.py             # Schema (no defaults)
├── deploy.py                  # Orchestrator
├── configs/
│   └── 01_debug_sft_rl.py     # Debug: SFT+RL pipeline (~15-20 min)
└── README.md
```

## SFT+RL Pipeline

When `mode="sft+rl"`:

1. **SFT Training**
   - Trains on `config.data.sft_mixture`
   - Saves checkpoints to `results/{experiment_name}/checkpoints/`

2. **RL Training** (automatic)
   - Auto-loads latest SFT checkpoint
   - Trains on `config.data.rl_mixture`
   - Saves RL checkpoints to same directory

## Design Principles

Following:
- **tiger_style_safety.md**: Explicit options, no defaults, assertions
- **experiment_config.md**: Serializable, version-controlled
- **qwen3_next**: Explicit construction pattern
- **nanochat**: Proven hyperparameters, TaskMixture pattern
- **Casey Muratori**: Pure functions, explicit dependencies

## Comparison: nanochat vs integration_training

| Aspect | nanochat | integration_training |
|--------|----------|---------------------|
| Data mixes | Hardcoded in scripts | Configurable via DatasetSpec |
| Scripts | Separate (mid_train, chat_sft, chat_rl) | Single deploy.py |
| Mode | CLI (`torchrun -m scripts.chat_sft`) | Auto-detected (`mode="sft+rl"`) |
| Configs | CLI args + overrides | Explicit construction |
| GPU selection | Manual CUDA_VISIBLE_DEVICES | `gpu_ranks=[0,1,2,3]` |
| Pipeline | Manual (run SFT, then RL) | Automatic (`mode="sft+rl"`) |

## Next Steps

- [ ] Implement checkpoint loading for RL
- [ ] Add support for `custom_json` datasets
- [ ] Add distributed training support (multi-GPU)
- [ ] Add evaluation metrics during training
