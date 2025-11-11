# SLIME Hyperparameters Complete Index

This directory contains comprehensive documentation of all hyperparameters used in SLIME (Scalable LLM Instruction-following and RL training).

## Documentation Files

### 1. SLIME_HYPERPARAMETERS.md
**Detailed Reference (17 KB, 574 lines)**
- Complete explanation of all hyperparameters
- Organized by category (optimizer, gradient, training, RL, generation, etc.)
- Includes default values, ranges, and technical explanations
- Best for understanding what each parameter does

### 2. SLIME_HYPERPARAMETERS_QUICK_REFERENCE.md
**Quick Lookup Guide (6 KB)**
- Most important parameters at a glance
- Parameter relationships and formulas
- Common configuration examples (debug/small/large scale)
- Key insights from the codebase
- Best for quick answers and parameter tuning

### 3. SLIME_HYPERPARAMETERS.json
**Structured Reference (7 KB)**
- Machine-readable format
- All parameters with type information
- Default values and value ranges
- Descriptions for each parameter
- Best for tooling and programmatic access

## Quick Navigation

### By Category

**Optimizer Parameters** (Learning & Regularization)
- Learning rate: `lr` (1e-6 for megatron, 2e-5 for FSDP)
- Warmup: `warmup_ratio` (0.03 default)
- Adam betas: `adam_beta1` (0.9), `adam_beta2` (0.95)
- L2 regularization: `weight_decay` (0.0 default)
- Gradient clipping: `--clip-grad` (1.0 default)

**Training Configuration**
- Batch sizes: `rollout_batch_size`, `global_batch_size`, `micro_batch_size`
- Sampling: `n_samples_per_prompt`
- Checkpointing: `save_interval`, `load`, `save`
- Reproducibility: `seed` (1234 default)

**RL/PPO Parameters**
- Clipping: `eps_clip` (0.2), `value_clip` (0.2)
- KL penalty: `kl_coef` (reward shaping), `kl_loss_coef` (loss term)
- Advantage: `advantage_estimator` (grpo/ppo/reinforce_plus_plus)
- Discounting: `gamma` (1.0), `lambd` (1.0)

**Generation & Sampling**
- Temperature: `rollout_temperature` (1.0)
- Top-p: `rollout_top_p` (1.0)
- Top-k: `rollout_top_k` (-1 = disabled)
- Max length: `rollout_max_response_len` (1024)

**Distributed Training**
- Nodes: `actor_num_nodes` (1), `critic_num_nodes`
- GPUs: `actor_num_gpus_per_node` (8), `rollout_num_gpus_per_engine`
- Backend: `train_backend` (megatron/fsdp)
- Communication: `distributed_backend` (nccl), `distributed_timeout_minutes` (10)

**Data Configuration**
- Dataset: `prompt_data` (JSONL path)
- Keys: `input_key`, `label_key`, `metadata_key`
- Batching: `use_dynamic_batch_size`, `max_tokens_per_gpu`
- Evaluation: `eval_interval`, `eval_prompt_data`

## Source Files Analyzed

All parameters extracted from SLIME codebase:
- `/references/slime/slime/utils/arguments.py` - Main SLIME arguments (1448 lines)
- `/references/slime/slime/backends/fsdp_utils/arguments.py` - FSDP backend (78 lines)
- `/references/slime/slime/backends/megatron_utils/arguments.py` - Megatron backend (27 lines)
- `/references/slime/slime/backends/sglang_utils/arguments.py` - Inference engine (126 lines)

## Total Parameters Documented

**100+ hyperparameters** across 12 major categories:

1. Optimizer Parameters (8 parameters)
2. Gradient Parameters (4 parameters)
3. Training Parameters (6 parameters)
4. RL Parameters (20+ parameters)
5. Generation Parameters (7 parameters)
6. Distributed Parameters (10 parameters)
7. Megatron Defaults (4 parameters)
8. FSDP Defaults (2 parameters)
9. Data Parameters (10 parameters)
10. Evaluation Parameters (8 parameters)
11. Logging Parameters (9 parameters)
12. Other Specialized Parameters (15+ parameters)

## Key Findings

### Important Design Choices
1. **Megatron LR is 20x smaller** than FSDP (1e-6 vs 2e-5)
2. **Adam beta2 is aggressive** (0.95 vs standard 0.999) for faster convergence
3. **Gradient clipping always enabled** (default 1.0) for stability
4. **No L2 regularization by default** (weight_decay=0.0)
5. **Reference model required** for KL penalties
6. **FSDP is experimental** - Megatron is production-ready

### Critical Parameters for Tuning
- `rollout_batch_size` - REQUIRED, controls data throughput
- `global_batch_size` - Controls effective learning rate scaling
- `warmup_ratio` - Prevents early divergence
- `clip_grad` - Prevents gradient explosions
- `eps_clip` - Controls PPO trust region
- `kl_coef` - Balance between policy and reference model

## How to Use These Docs

**Quick Setup?** Start with `SLIME_HYPERPARAMETERS_QUICK_REFERENCE.md`

**Understanding a Parameter?** Look it up in `SLIME_HYPERPARAMETERS.md`

**Writing Config Code?** Parse `SLIME_HYPERPARAMETERS.json`

**Comparing Configurations?** Use the default values table in quick reference

## Common Configurations

### Single GPU Debug
```
rollout_batch_size: 8
global_batch_size: 8
lr: 2e-5
warmup_ratio: 0.03
```

### Multi-GPU (4 GPUs)
```
rollout_batch_size: 32
global_batch_size: 32
lr: 1e-5
actor_num_gpus_per_node: 4
```

### Large Scale (64 GPUs)
```
rollout_batch_size: 256
global_batch_size: 256
lr: 5e-6
actor_num_nodes: 8
actor_num_gpus_per_node: 8
train_backend: megatron
```

## Notes

- Defaults are from actual SLIME source code
- Parameter names use hyphenated CLI format (e.g., `--clip-grad`)
- In Python code, names are underscored (e.g., `args.clip_grad`)
- All float defaults are based on FSDP backend unless noted
- Megatron has 20x smaller learning rate by default
- Some parameters have dependencies (e.g., `eps_clip_high` depends on `eps_clip`)

---

**Last Updated:** November 11, 2025
**Source:** SLIME Repository (references/slime/)
**Scope:** SFT, PPO/RL, and advanced training configurations
